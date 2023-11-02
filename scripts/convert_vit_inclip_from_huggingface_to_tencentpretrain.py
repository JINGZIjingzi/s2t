import argparse
import collections
import torch


def convert_vit_transformer_encoder_from_huggingface_to_tencentpretrain(input_model, output_model, layers_num):
    for i in range(layers_num):
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.q_proj.weight"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.q_proj.bias"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.k_proj.weight"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.k_proj.bias"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.v_proj.weight"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.v_proj.bias"]
        output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.out_proj.weight"]
        output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".self_attn.out_proj.bias"]
        output_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".layer_norm1.weight"]
        output_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".layer_norm1.bias"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".mlp.fc1.weight"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".mlp.fc1.bias"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".mlp.fc2.weight"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".mlp.fc2.bias"]
        output_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".layer_norm2.weight"]
        output_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = \
            input_model["vision_model.encoder.layers." + str(i) + ".layer_norm2.bias"]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
    parser.add_argument("--layers_num", type=int, default=12, help=".")


    args = parser.parse_args()

    input_model = torch.load(args.input_model_path, map_location="cpu")

    output_model = collections.OrderedDict()

    output_model["embedding.patch.cls_emb"] = input_model["vision_model.embeddings.class_embedding"].unsqueeze(0).unsqueeze(0)
    output_model["embedding.patch.projection.weight"] = input_model["vision_model.embeddings.patch_embedding.weight"]
    # output_model["embedding.patch.projection.bias"] = input_model["vision_model.embeddings.patch_embedding.bias"]
    output_model["embedding.pos.embedding.weight"] = input_model["vision_model.embeddings.position_embedding.weight"].squeeze(0)
    output_model["embedding.layer_norm.gamma"] = input_model["vision_model.pre_layrnorm.weight"]
    output_model["embedding.layer_norm.beta"] = input_model["vision_model.pre_layrnorm.bias"]

    convert_vit_transformer_encoder_from_huggingface_to_tencentpretrain(input_model, output_model, args.layers_num)

    output_model["encoder.layer_norm.gamma"] = input_model["vision_model.post_layernorm.weight"]
    output_model["encoder.layer_norm.beta"] = input_model["vision_model.post_layernorm.bias"]
    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
