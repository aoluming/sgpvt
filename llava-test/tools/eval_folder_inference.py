"""
对文件夹内全部图像批量调用 LLaVA 推理（用于评估/遗忘实验后的定性检查）。
运行前请设置环境变量 CUDA_VISIBLE_DEVICES，勿在脚本内写死 GPU 编号。
"""
import argparse
import os
import re
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import requests
from io import BytesIO


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def get_image_files_from_directory(directory, extensions=None):
    if extensions is None:
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
    image_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ]
    return sorted(image_files)


def eval_model(args):
    log_file = open(args.output_file, "w", encoding="utf-8") if args.output_file else None

    def log(message):
        print(message)
        if log_file:
            log_file.write(message + "\n")

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if hasattr(model, "peft_config"):
        print(f"[INFO] LoRA config found: {model.peft_config}")
    else:
        print("[INFO] No LoRA config in model (full finetune or merged weights).")

    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            vision_tower.load_model()

        vision_hidden_size = vision_tower.hidden_size if vision_tower is not None else None
        mm_projector = (
            model.get_model().mm_projector
            if hasattr(model.get_model(), "mm_projector")
            else None
        )

        if vision_hidden_size is not None and mm_projector is not None:
            if isinstance(mm_projector, torch.nn.Sequential):
                projector_input_dim = mm_projector[0].in_features
            elif isinstance(mm_projector, torch.nn.Linear):
                projector_input_dim = mm_projector.in_features
            else:
                projector_input_dim = None

            if projector_input_dim is not None and projector_input_dim != vision_hidden_size:
                print(
                    "[WARNING] mm_projector 输入维度与 vision_tower 输出不一致，"
                    "请使用与训练一致的 vision tower / checkpoint；详见脚本内注释逻辑。"
                )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            f"[WARNING] auto conv_mode={conv_mode}, --conv-mode={args.conv_mode}, using {args.conv_mode}"
        )
    else:
        args.conv_mode = conv_mode

    prompt = args.prompt if args.prompt else "What is the name of this person?"
    log(f"Prompt: {prompt}")

    image_files = get_image_files_from_directory(args.image_file)
    if not image_files:
        log(f"No images found in directory: {args.image_file}")
        if log_file:
            log_file.close()
        return

    device = model.device
    for image_path in image_files:
        log(f"Processing image: {image_path}")
        try:
            image = load_image(image_path)
        except Exception as e:
            log(f"Failed to load image {image_path}: {e}")
            continue

        conv = conv_templates[args.conv_mode].copy()
        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        images_tensor = process_images([image], image_processor, model.config).to(
            device, dtype=torch.float16
        )

        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                images=images_tensor,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        log(f"Generated_text for {os.path.basename(image_path)}: {generated_text}\n")

    if log_file:
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA folder batch inference")
    parser.add_argument("--model-path", type=str, required=True, help="模型 checkpoint 目录")
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="LoRA 合并用基座路径（与训练时一致）",
    )
    parser.add_argument(
        "--image-file",
        type=str,
        required=True,
        help="仅支持目录：遍历其中图片",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="将 stdout 同步写入该文件",
    )
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--prompt", type=str, default=None)

    _args = parser.parse_args()
    if not os.path.isdir(_args.image_file):
        raise ValueError(f"--image-file 必须是目录: {_args.image_file}")

    eval_model(_args)
