"""
This script provides an example to wrap TencentPretrain for image to text with prompt.
"""
import sys
import os
import psutil
import datetime
import random
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dest
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import imghdr
import random
import deepspeed
import torch.distributed as dist

from argparse import Namespace

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.decoders import *
from tencentpretrain.targets import *
from tencentpretrain.utils.vocab import Vocab
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.optimizers import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.seed import set_seed
from tencentpretrain.utils.logging import init_logger
from tencentpretrain.utils.misc import pooling, ZeroOneNormalize
from tencentpretrain.model_saver import save_model
from tencentpretrain.opts import finetune_opts, tokenizer_opts, adv_opts, deepspeed_opts
from tencentpretrain.layers.layer_norm import LayerNorm


class VitTower(nn.Module):
    def __init__(self, args):
        super(VitTower, self).__init__()
        args_vit = vars(args)
        args_vit.update(args.vit_config)
        args_vit = Namespace(**args_vit)
        self.args = args_vit
        self.embedding = Embedding(args_vit)
        for embedding_name in args_vit.embedding:
            tmp_emb = str2embedding[embedding_name](args_vit, None)
            self.embedding.update(tmp_emb, embedding_name)

        self.encoder = str2encoder[args_vit.encoder](args_vit)

        for name, param in self.embedding.named_parameters():
            param.requires_grad = False
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

    def forward(self, src, seg):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        """
        with torch.no_grad():
            emb = self.embedding(src, seg)
            out = self.encoder(emb, seg)

        return out


class LLaVa(nn.Module):
    def __init__(self, args):
        super(LLaVa, self).__init__()
        self.args = args
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)

        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling

        self.target = Target()
        print("tokenizer vocab nums:", len(args.tokenizer.vocab))
        for target_name in args.target:
            tmp_target = str2target[target_name](args, len(args.tokenizer.vocab))
        self.target.update(tmp_target, target_name)

        self.remove_embedding_combine_layernorm = args.remove_embedding_combine_layernorm
        if not self.remove_embedding_combine_layernorm:
            self.combine_layer_norm = LayerNorm(args.emb_size)

        # vit model should be built after LLM
        self.vit_model = VitTower(args)

        connector_modules = [nn.Linear(args.vit_config["emb_size"], args.connector_config["mlp_hidden_size"])]
        for _ in range(1, args.connector_config["num_mlp_layer"]):
            connector_modules.append(nn.GELU())
            connector_modules.append(nn.Linear(args.connector_config["mlp_hidden_size"], args.connector_config["mlp_hidden_size"]))
        self.connector = nn.Sequential(*connector_modules)

        self.num_image_tokens = int(args.image_width / args.patch_size) * int(args.image_height / args.patch_size) 
        if args.stage == "pretrain":
            for name, param in self.embedding.named_parameters():
                param.requires_grad = False
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        elif args.stage == "finetune":
            for name, param in self.embedding.named_parameters():
                param.requires_grad = True
            for name, param in self.encoder.named_parameters():
                param.requires_grad = True

    def forward(self, src_text, seg_text, tgt, tgt_seg, src_image, seg_image, length_before):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        text_combine_emb = self.embedding(src_text, seg_text)
        if src_image is not None:
            image_embeds = self.vit_model(src_image, seg_image)
            image_embeds = self.connector(image_embeds)
        # Encoder.
        # assume text_before_image has the same length
        if text_combine_emb.shape[0] == 1:
            emb_cat = torch.cat((text_combine_emb[:,:length_before[0],:], image_embeds, text_combine_emb[:,length_before[0]:,:]), 1)
        else:
            emb_cat = torch.cat((text_combine_emb[0,:length_before[0],:], image_embeds[0], text_combine_emb[0,length_before[0]:,:]), 0).unsqueeze(0)
            for i in range(1, text_combine_emb.shape[0]):
                tmp = torch.cat((text_combine_emb[i,:length_before[i],:], image_embeds[i], text_combine_emb[i,length_before[i]:,:]), 0).unsqueeze(0)
                emb_cat = torch.cat((emb_cat, tmp), 0)
        seg_cat = torch.cat((seg_image, seg_text), 1)
        if not self.remove_embedding_combine_layernorm:
            emb_cat = self.combine_layer_norm(emb_cat)
        # emb_cat = self.dropout(emb_cat)
        # encoder
        output = self.encoder(emb_cat, seg_cat)
        # Target.
        loss, correct, denominator = self.target(output, tgt, tgt_seg)
        return loss, correct, denominator.to(loss.device)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        args.logger.info("loading model from {0}".format(args.pretrained_model_path))
        keys_info = model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"), strict=False)
        args.logger.info("missing_keys: {0}".format(keys_info.missing_keys))
        args.logger.info("unexpected_keys: {0}".format(keys_info.unexpected_keys))
        if args.vit_model_path is not None:
            args.logger.info("loading model from {0}".format(args.vit_model_path))
            keys_info = model.vit_model.load_state_dict(torch.load(args.vit_model_path, map_location="cpu"), strict=False)
            args.logger.info("missing_keys: {0}".format(keys_info.missing_keys))
            args.logger.info("unexpected_keys: {0}".format(keys_info.unexpected_keys))
        if args.connector_model_path is not None:
            args.logger.info("loading model from {0}".format(args.connector_model_path))
            keys_info = model.connector.load_state_dict(torch.load(args.connector_model_path, map_location="cpu"), strict=False)
            args.logger.info("missing_keys: {0}".format(keys_info.missing_keys))
            args.logger.info("unexpected_keys: {0}".format(keys_info.unexpected_keys))
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src_text, seg_text, tgt, tgt_seg, image_paths, length_before):
    instances_num = src_text.size()[0]
    for i in range(instances_num // batch_size):
        src_text_batch = src_text[i * batch_size : (i + 1) * batch_size, :]
        seg_text_batch = seg_text[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size, :]
        tgt_seg_batch = tgt_seg[i * batch_size : (i + 1) * batch_size, :]
        image_path_batch = image_paths[i * batch_size : (i + 1) * batch_size]
        length_before_batch = length_before[i * batch_size : (i + 1) * batch_size]
        yield src_text_batch, seg_text_batch, tgt_batch, tgt_seg_batch, image_path_batch, length_before_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_text_batch = src_text[instances_num // batch_size * batch_size :, :]
        seg_text_batch = seg_text[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :, :]
        tgt_seg_batch = tgt_seg[instances_num // batch_size * batch_size :, :]
        image_path_batch = image_paths[instances_num // batch_size * batch_size :]
        length_before_batch = length_before[instances_num // batch_size * batch_size :]
        yield src_text_batch, seg_text_batch, tgt_batch, tgt_seg_batch, image_path_batch, length_before_batch


def read_dataset(args, path, split):
    transform = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        ZeroOneNormalize()
    ])
    prompt_template = {
        "llama2": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n",
        "vicuna": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"}
    num_image_tokens = int(args.image_width / args.patch_size) * int(args.image_height / args.patch_size) + 1 # 336/14-14 --> 576 dim
    seq_text = args.seq_length - num_image_tokens # 576
    dataset, columns = [], {}

    if split:
        for i in range(args.world_size):
            dataset.append([])
        index = 0
    args.logger.info("{} begin read dataset".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            if args.prompt_template == "llama2":
                prompt_overall = prompt_template["llama2"]
            elif args.prompt_template == "vicuna":
                prompt_overall = prompt_template["vicuna"]
            else:
                args.logger.info("unsupported prompt template!")
                continue
            prompt = line[columns["prompt"]].replace("\\n","\n")
            prompt_before_image = prompt_overall + " USER:"
            prompt_after_image = "\nASSISTANT:"
            if prompt[:7] == "<image>":
                prompt_after_image = prompt[7:] + prompt_after_image
            elif prompt[-7:] == "<image>":
                prompt_before_image = prompt_before_image + prompt[:-7]
            else:
                prompt_after_image = "\n" + prompt + prompt_after_image

            # args.logger.info("prompt_before_image: {}".format(prompt_before_image))
            # args.logger.info("prompt_after_image: {}".format(prompt_after_image))
            prompt_before_image_id = args.tokenizer.convert_tokens_to_ids(
                args.tokenizer.tokenize(prompt_before_image)
            )
            prompt_after_image_id = args.tokenizer.convert_tokens_to_ids(
                args.tokenizer.tokenize(prompt_after_image)
            )
            seg_before_image_id = [1] * len(prompt_before_image_id)
            seg_after_image_id = [1] * len(prompt_after_image_id)
            if len(prompt_before_image_id) + len(prompt_after_image_id) > seq_text:
                args.logger.info("promt too long, jump for now")
                continue
            if "image_path" in columns:  # Sentence-pair and images classification.
                image_path = "/apdcephfs_qy3/share_300998916/janinezhao/data/llava/" + line[columns["image_path"]]
                if imghdr.what(image_path) != 'jpeg' and imghdr.what(image_path) != 'png':
                    continue
                try:
                    image = read_image(image_path, ImageReadMode.RGB)
                except:
                    continue
            else: 
                args.logger.info("image_path is missing!")
                continue
            tgt = line[columns["answer"]]
            tgt_id = args.tokenizer.convert_tokens_to_ids(
                    args.tokenizer.tokenize(tgt) + [SEP_TOKEN]
            )
            tgt_seg = [1] * len(tgt_id)
            
            # assert len(prompt_before_image_id) + len(prompt_after_image_id) < seq_text, "promt too long"
            seq_tgt = seq_text - len(prompt_before_image_id) - len(prompt_after_image_id)
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            
            if len(tgt_id) > seq_tgt+1:
                tgt_id = tgt_id[:seq_tgt+1]
                tgt_seg = tgt_seg[:seq_tgt+1]
            pad_num = seq_tgt+1-len(tgt_id)
            tgt_id = tgt_id + [PAD_ID] * pad_num
            tgt_seg = tgt_seg + [0] * pad_num

            tgt_in = tgt_id[:-1]
            tgt_in_seg = tgt_seg[:-1]

            text_combine = prompt_before_image_id + prompt_after_image_id + tgt_in
            text_combine_seg = seg_before_image_id + seg_after_image_id + tgt_in_seg

            emptys = [PAD_ID] * (args.seq_length - seq_tgt - 1 )
            emptys_seg = [0] * (args.seq_length - seq_tgt - 1 )
            tgt_out = emptys + tgt_id
            tgt_out_seg = emptys_seg + tgt_seg

            if split:
                # dataset[index].append((text_combine, text_combine_seg, tgt_out, tgt_out_seg, src_image.tolist(), seg_image, len(prompt_before_image_id)))
                dataset[index].append((text_combine, text_combine_seg, tgt_out, tgt_out_seg, image_path, len(prompt_before_image_id)))
                index += 1
                if index == args.world_size:
                    index = 0
            else:
                # dataset.append((text_combine, text_combine_seg, tgt_out, tgt_out_seg, src_image.tolist(), seg_image, len(prompt_before_image_id)))
                dataset.append((text_combine, text_combine_seg, tgt_out, tgt_out_seg, image_path, len(prompt_before_image_id)))

        if split:
            max_data_num_rank_index = 0
            max_data_num = len(dataset[0])
            for i in range(args.world_size):
                if len(dataset[i]) > max_data_num:
                    max_data_num_rank_index = i
                    max_data_num = len(dataset[i])
            for i in range(args.world_size):
                if len(dataset[i]) < max_data_num:
                    dataset[i].append(dataset[max_data_num_rank_index][-1])
        if (line_id + 1 )% 100 == 0:
            args.logger.info("当前进程的内存使用：{} GB".format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    args.logger.info("num of dataset 0: {}".format(len(dataset[0])))

    return dataset


def train_model(args, model, optimizer, scheduler, src_text_batch, seg_text_batch, tgt_batch, tgt_seg_batch, src_image_batch, seg_image_batch, length_before):
    model.zero_grad()

    src_text_batch = src_text_batch.to(args.device)
    seg_text_batch = seg_text_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    tgt_seg_batch = tgt_seg_batch.to(args.device)
    src_image_batch = src_image_batch.to(args.device).half()
    seg_image_batch = seg_image_batch.to(args.device)
    length_before = length_before.to(args.device)

    loss, correct, denominator = model(src_text_batch, seg_text_batch, tgt_batch, tgt_seg_batch, src_image_batch, seg_image_batch, length_before)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss) / args.accumulation_steps

    model.backward(loss)

    if args.cur_step % args.accumulation_steps == 0:
        model.step()

    return loss, correct, denominator


def evaluate(args, dataset):
    src_text = torch.LongTensor([sample[0] for sample in dataset])
    seg_text = torch.LongTensor([sample[1] for sample in dataset])
    tgt = torch.LongTensor([sample[2] for sample in dataset])
    tgt_seg = torch.LongTensor([sample[3] for sample in dataset])
    image_paths = [sample[4] for sample in dataset]
    # length_before = torch.LongTensor([dataset[0][5]])
    length_before = torch.LongTensor([sample[5] for sample in dataset])
    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        ZeroOneNormalize()
    ])
    image_seg_length = (args.image_height // args.patch_size) * (args.image_width // args.patch_size) + 1
    correct = 0
    denominator = 0

    args.model.eval()

    for i, (src_text_batch, seg_text_batch, tgt_batch, tgt_seg_batch, image_path_batch, length_before_batch) in \
    enumerate(batch_loader(batch_size, src_text, seg_text, tgt, tgt_seg, image_paths, length_before)):
        # args.logger.info("{}: evaluate: batch {} start".format(args.rank,i))
        src_image_batch = None
        for j, image_path in enumerate(image_path_batch):
            image = read_image(image_path, ImageReadMode.RGB)
            image = image.to(args.device)
            src_image = transform(image)
            # args.logger.info("{}: evaluate: batch {} read image {}: {}".format(args.rank, i,j,image_path))

            if src_image_batch is not None:
                src_image_batch = torch.stack([src_image_batch,src_image])
            else:
                src_image_batch = src_image
        if src_image_batch == None:
            # args.logger.info("{} evaluate: batch {} src_image_batch is None!".format(args.rank, i))
            continue
        if len(src_image_batch.shape) == 3:
            src_image_batch = torch.unsqueeze(src_image_batch, 0)
        # args.logger.info("{} evaluate: batch {} prep data".format(args.rank, i))
        src_image_batch = src_image_batch.to(args.device).half()
        seg_image_batch = torch.ones(src_image_batch.shape[0],image_seg_length).to(args.device)
        src_text_batch = src_text_batch.to(args.device)
        seg_text_batch = seg_text_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        tgt_seg_batch = tgt_seg_batch.to(args.device)
        length_before_batch = length_before_batch.to(args.device)
        # args.logger.info("{} evaluate: batch {} infer start".format(args.rank, i))
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            _, correct_i, denominator_i = args.model(src_text_batch, seg_text_batch, tgt_batch, tgt_seg_batch, src_image_batch, seg_image_batch, length_before_batch)
            # args.logger.info("evaluate: batch {} infer done".format(i))
        correct += correct_i.item()
        denominator += denominator_i.item()

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / denominator, correct, denominator))
    return correct / denominator


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)
    parser.add_argument("--world_size", type=int, default=1,
                    help="Total number of processes (GPUs) for training.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                    help="Specific steps to accumulate gradient.")               

    parser.add_argument("--vit_model_path", type=str,
                        help="Pretrained model of Vit.")
    parser.add_argument("--connector_model_path", type=str,
                        help="Pretrained model of Connector.")
    parser.add_argument("--save_steps", type=int,
                        help="Save model every N steps.", default=100000)
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune"],
                        help="Training stage", default="pretrain")
    parser.add_argument("--prompt_template", type=str, choices=["llama2", "vicuna"],
                        help="give the llm type to choose a prompt", default="llama2")
    adv_opts(parser)

    deepspeed_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)

    args.logger = init_logger(args)

    # 统计耗时
    # from pyinstrument import Profiler
    # profiler = Profiler()
    # profiler.start()

    # Load or initialize parameters.
    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = LLaVa(args)
            if args.pretrained_model_path:
                model = _load_state_dict_into_model(model, args.pretrained_model_path)
            if args.vit_model_path is not None:
                model.vit_model = _load_state_dict_into_model(model.vit_model, args.vit_model_path)
            if args.connector_model_path is not None:
                model.qformer = _load_state_dict_into_model(model.connector, args.connector_model_path)
    else:
        model = LLaVa(args)
        # Load or initialize parameters.
        load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    deepspeed.init_distributed()
    rank = dist.get_rank()
    args.rank = rank

    args.model = model
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training phase.
    trainset = read_dataset(args, args.train_path, split=True)[args.rank]
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    custom_optimizer, custom_scheduler = build_optimizer(args, model)

    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_grouped_parameters,
        args=args,
        optimizer=custom_optimizer,
        lr_scheduler=custom_scheduler,
        mpu=None,
        dist_init_required=False)

    transform = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        ZeroOneNormalize()
    ])
    image_seg_length = (args.image_height // args.patch_size) * (args.image_width // args.patch_size) + 1

    src_text = torch.LongTensor([sample[0] for sample in trainset])
    seg_text = torch.LongTensor([sample[1] for sample in trainset])
    tgt = torch.LongTensor([sample[2] for sample in trainset])
    tgt_seg = torch.LongTensor([sample[3] for sample in trainset])
    image_paths = [sample[4] for sample in trainset]
    # length_before = torch.LongTensor([trainset[0][5]])
    length_before = torch.LongTensor([sample[5] for sample in trainset])

    total_loss, result, best_result, best_epoch = 0.0, 0.0, 0.0, 0
    total_correct, total_denominator = 0.0, 0

    result_tensor = torch.tensor(result).to(args.device)
    if args.rank == 0:
        args.logger.info("Batch size: {}".format(batch_size))
        args.logger.info("The number of training instances: {}".format(instances_num))
        args.logger.info("Start training.")

    args.cur_step = 0
    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_text_batch, seg_text_batch, tgt_batch, tgt_seg_batch, image_path_batch, length_before_batch) in \
enumerate(batch_loader(batch_size, src_text, seg_text, tgt, tgt_seg, image_paths, length_before)):
            src_image_batch = None
            for j, image_path in enumerate(image_path_batch):
                image = read_image(image_path, ImageReadMode.RGB)
                image = image.to(args.device)
                src_image = transform(image)

                if src_image_batch is not None:
                    src_image_batch = torch.stack([src_image_batch,src_image])
                else:
                    src_image_batch = src_image

            if src_image_batch == None:
                continue
            if len(src_image_batch.shape) == 3:
                src_image_batch = torch.unsqueeze(src_image_batch, 0)
            seg_image_batch = torch.ones(src_image_batch.shape[0],image_seg_length)

            loss, correct, denominator = train_model(args, model, optimizer, scheduler,
                src_text_batch, seg_text_batch,
                tgt_batch, tgt_seg_batch, src_image_batch, seg_image_batch, length_before_batch)
            args.cur_step += 1

            total_loss += loss.item()
            total_correct += correct.item()
            total_denominator += denominator.item()
            if (i + 1) % args.report_steps == 0 and args.rank == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg acc: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps, total_correct / total_denominator ))
                total_loss = 0.0
                total_correct, total_denominator = 0.0, 0
            if (i + 1) % args.save_steps == 0:
                if args.stage == "finetune" or (args.stage == "pretrain" and args.rank == 0):
                    result = evaluate(args, read_dataset(args, args.dev_path, split=False))
                    model.train()
                    result_tensor = torch.tensor(result).to(args.device)
                dist.broadcast(result_tensor, 0, async_op=False)
                model.save_checkpoint(args.output_model_path+"-"+str(i+1), str(epoch))
        args.logger.info("Epoch: {} done in rank: {}".format(epoch, args.rank))
        if args.stage == "finetune" or (args.stage == "pretrain" and args.rank == 0):
            args.logger.info("begin evaluate in rank: {}".format(args.rank))
            result = evaluate(args, read_dataset(args, args.dev_path, split=False))
            args.logger.info("evaluate done in rank: {}".format(args.rank))
            result_tensor = torch.tensor(result).to(args.device)

        dist.broadcast(result_tensor, 0, async_op=False)
        if result_tensor.float() >= best_result:
            best_result = result_tensor.float().item()
            best_epoch = epoch
        model.save_checkpoint(args.output_model_path, str(epoch))
    
    # profiler.stop()
    # profiler.print()

    # Evaluation phase.
    if args.test_path is not None and args.rank == 0:
        args.logger.info("Test set evaluation.")
        model.load_checkpoint(args.output_model_path, str(best_epoch))
        evaluate(args, read_dataset(args, args.test_path, split=False))


if __name__ == "__main__":
    main()
