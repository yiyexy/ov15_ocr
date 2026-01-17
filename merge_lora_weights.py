"""
合并 LoRA 权重到基础模型
生成完整的可独立使用的模型
"""

import torch
import sys
import os
import argparse

# 解析命令行参数（需要先解析以获取模型路径）
def parse_args():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基础模型")
    parser.add_argument(
        "--base_model", 
        default="./checkpoints/LLaVA-OneVision-1.5-4B-Instruct",
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_path", 
        default="./output/llava_ov15_4b_ocr_lora/final",
        help="LoRA adapter 权重路径"
    )
    parser.add_argument(
        "--output", 
        default="./output/llava_ov15_4b_ocr_merged",
        help="合并后模型保存路径"
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true",
        help="是否上传到 HuggingFace Hub"
    )
    return parser.parse_args()

args = parse_args()

# 添加模型目录到路径
sys.path.insert(0, args.base_model)

from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText
from peft import PeftModel

# 注册自定义模型
from configuration_llavaonevision1_5 import Llavaonevision1_5Config
from modeling_llavaonevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration

AutoConfig.register("llava_onevision1_5", Llavaonevision1_5Config)
AutoModelForImageTextToText.register(Llavaonevision1_5Config, LLaVAOneVision1_5_ForConditionalGeneration)


def merge_lora_weights(base_model_path: str, lora_path: str, output_path: str, push_to_hub: bool = False):
    """
    合并 LoRA 权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA adapter 权重路径
        output_path: 合并后模型保存路径
        push_to_hub: 是否上传到 HuggingFace Hub
    """
    print("=" * 60)
    print("LoRA 权重合并工具")
    print("=" * 60)
    
    # 1. 加载基础模型
    print(f"\n[1/4] 加载基础模型: {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 在 CPU 上合并以节省 GPU 显存
        trust_remote_code=True
    )
    print(f"   基础模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 加载 LoRA 权重
    print(f"\n[2/4] 加载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    
    # 统计 LoRA 参数
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
    print(f"   LoRA 参数量: {lora_params:,}")
    
    # 3. 合并权重
    print(f"\n[3/4] 合并 LoRA 权重到基础模型...")
    model = model.merge_and_unload()
    print(f"   合并后模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 保存合并后的模型
    print(f"\n[4/4] 保存合并后的模型: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(output_path, safe_serialization=True)
    print(f"   模型已保存")
    
    # 保存 processor
    processor.save_pretrained(output_path)
    print(f"   Processor 已保存")
    
    # 复制自定义模型文件（如果需要）
    custom_files = [
        "configuration_llavaonevision1_5.py",
        "modeling_llavaonevision1_5.py",
    ]
    for f in custom_files:
        src = os.path.join(base_model_path, f)
        dst = os.path.join(output_path, f)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            print(f"   复制: {f}")
    
    # 检查输出文件
    print(f"\n{'=' * 60}")
    print("合并完成！输出文件:")
    print("=" * 60)
    total_size = 0
    for f in sorted(os.listdir(output_path)):
        fpath = os.path.join(output_path, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            print(f"   {f}: {size / 1024 / 1024:.2f} MB")
    print(f"\n   总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")
    
    if push_to_hub:
        print("\n上传到 HuggingFace Hub...")
        model.push_to_hub(output_path.split("/")[-1])
        processor.push_to_hub(output_path.split("/")[-1])
        print("上传完成！")
    
    return output_path


def main():
    # 使用全局 args 变量（已在文件开头解析）
    
    merge_lora_weights(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output,
        push_to_hub=args.push_to_hub
    )


if __name__ == "__main__":
    main()
