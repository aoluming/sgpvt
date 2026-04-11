import argparse
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from peft import PeftModel


def merge_lora(args):
    # 首先加载基础模型
    model_name = get_model_name_from_path(args.model_base)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_base, None, model_name, device_map='cpu')
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, args.model_path)
    
    # 合并LoRA权重
    model = model.merge_and_unload()
    
    # 保存合并后的模型
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    print(f"LoRA weights merged and saved to: {args.save_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
