"""
  This script provides an exmaple to wrap TencentPretrain for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import imghdr
import deepspeed

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.targets import *
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.model_loader import *
from tencentpretrain.opts import infer_opts, tokenizer_opts, log_opts
from tencentpretrain.opts import deepspeed_opts
from finetune.run_llava_mem_deepspeed import *


class LLaVaGenerate(nn.Module):
    def __init__(self, args):
        super(LLaVaGenerate, self).__init__()
        self.args = args
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)

        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling

        self.target = Target()
        self.target.update(LmTarget(args, len(args.tokenizer.vocab)), "lm")
        print("tokenizer vocab nums:", len(args.tokenizer.vocab))

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

        

    def forward(self, src_text, seg_text, src_image, seg_image, length_before):
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
        emb_cat = torch.cat((text_combine_emb[:,:length_before[0],:], image_embeds, text_combine_emb[:,length_before[0]:,:]), 1)
        seg_cat = torch.cat((seg_image, seg_text), 1)

        if not self.remove_embedding_combine_layernorm:
            emb_cat = self.combine_layer_norm(emb_cat)
        # emb_cat = self.dropout(emb_cat)

        # encoder
        output = self.encoder(emb_cat, seg_cat)
        # # Target.
        output = self.target.lm.output_layer(output)
        return output


def top_k_top_p_filtering(logits, top_k, top_p):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--top_k", type=int, default=70)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vit_model_path", type=str, default=None,
                    help="Pretrained model of Vit.")
    parser.add_argument("--connector_model_path", type=str, default=None,
                    help="Pretrained model of Connector.")
    parser.add_argument("--prompt_template", type=str, choices=["llama2", "vicuna"],
                        help="give the llm type to choose a prompt", default="llama2")

    tokenizer_opts(parser)

    deepspeed_opts(parser)

    log_opts(parser)

    args = parser.parse_args()

    args.target = "lm"
    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    args.logger = init_logger(args)

    args.pretrained_model_path = args.load_model_path

    # Load or initialize parameters.
    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = LLaVaGenerate(args)
            if args.pretrained_model_path:
                model = _load_state_dict_into_model(model, args.pretrained_model_path)
            if args.vit_model_path is not None:
                model.vit_model = _load_state_dict_into_model(model.vit_model, args.vit_model_path)
            if args.connector_model_path is not None:
                model.qformer = _load_state_dict_into_model(model.connector, args.connector_model_path)
    else:
        model = LLaVaGenerate(args)
        # Load or initialize parameters.
        load_or_initialize_parameters(args, model)

    deepspeed.init_distributed()
    model = deepspeed.initialize(model=model,config_params=args.deepspeed_config)[0]

    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        ZeroOneNormalize()
    ])
    prompt_template = {
        "llama2": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n",
        "vicuna": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"}
    num_image_tokens = int(args.image_width / args.patch_size) * int(args.image_height / args.patch_size) + 1
    seq_text = args.seq_length - num_image_tokens
    outf = open(args.prediction_path, mode="w", encoding="utf-8")
    columns = {}
    with open(args.test_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f.readlines()):
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
                print("unsupported prompt template!")
                continue
            prompt_before_image = prompt_overall + " USER:"
            prompt_after_image = "\n" + line[columns["prompt"]].replace("\\n","\n") + "\nASSISTANT:"
            prompt_before_image_id = args.tokenizer.convert_tokens_to_ids(
                args.tokenizer.tokenize(prompt_before_image)
            )
            prompt_after_image_id = args.tokenizer.convert_tokens_to_ids(
                args.tokenizer.tokenize(prompt_after_image)
            )
            seg_before_image_id = [1] * len(prompt_before_image_id)
            seg_after_image_id = [1] * len(prompt_after_image_id)

            if len(prompt_before_image_id) + len(prompt_after_image_id) > seq_text:
                print("promt too long, jump for now")
                continue

            if "image_path" in columns:  # Sentence-pair and images classification.
                image_path = "/apdcephfs_qy3/share_300998916/janinezhao/data/llava/" + line[columns["image_path"]]
                assert imghdr.what(image_path) == 'jpeg' or imghdr.what(image_path) == 'png', 'image type should be jpeg or png!'
                try:
                    image = read_image(image_path, ImageReadMode.RGB)
                    image = image.to(device)
                    src_image = transform(image)
                except:
                    print("Image: {} has problem! skipped".format(image_path))
                    continue
            else: 
                print("image_path is missing!")
                continue
            ground_truth = line[columns["answer"]]

            text_combine = prompt_before_image_id + prompt_after_image_id
            text_combine_seg = seg_before_image_id + seg_after_image_id
            length_before = len(prompt_before_image_id)
            
            beginning_length = len(text_combine) + num_image_tokens

            text_tensor, text_seg_tensor = torch.LongTensor([text_combine]).to(device), torch.LongTensor([text_combine_seg]).to(device)
            image_tensor = torch.unsqueeze(src_image, 0).half()
            image_seg_tensor = torch.ones(1, num_image_tokens).to(device)
            length_before = torch.LongTensor([length_before]).to(device)
            SEP_ID = args.tokenizer.convert_tokens_to_ids([SEP_TOKEN])

            for i in range(args.seq_length - beginning_length):
                output = model(text_tensor, text_seg_tensor, image_tensor, image_seg_tensor, length_before)
                next_token_logits = output[0][-1] / args.temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                text_tensor = torch.cat([text_tensor, next_token.view(1, 1)], dim=1)
                text_seg_tensor = torch.cat([text_seg_tensor, torch.tensor([[1]]).to(device)], dim=1)

                if next_token.cpu().tolist() == SEP_ID:
                    break 
            if rank == 0:
                # outf.write("\t".join(line)+"\n")
                tokens = [token_id.item() for token_id in text_tensor[0]]
                if args.tokenizer.sp_model is not None:
                    generated_sentence = args.tokenizer.sp_model.decode(tokens)
                else:
                    generated_sentence = "".join(args.tokenizer.convert_ids_to_tokens(tokens))
                print(line)
                print(generated_sentence)
                outf.write(generated_sentence + "\n\n")
