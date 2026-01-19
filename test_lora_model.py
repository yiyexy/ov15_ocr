"""
测试 LLaVA-OneVision-1.5-4B + LoRA 模型
在发票验证集上评估效果
"""

import torch
import sys
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

# 解析命令行参数（需要先解析以获取模型路径）
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="./checkpoints/LLaVA-OneVision-1.5-4B-Instruct")
    parser.add_argument("--lora_path", default="./output/llava_ov15_4b_ocr_lora/final")
    parser.add_argument("--no_lora", action="store_true", help="不加载 LoRA 权重，直接使用原始模型测试")
    parser.add_argument("--val_data", default="./dataset/invoice_vqa_val.json")
    parser.add_argument("--image_root", default="./")
    parser.add_argument("--num_samples", type=int, default=20, help="测试样本数，None 表示全部")
    parser.add_argument("--output", default="./eval_results.json")
    return parser.parse_args()

args = parse_args()

from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

# 完全依赖 trust_remote_code=True 自动加载自定义模型，无需手动注册


def load_model(base_model_path, lora_path, use_lora=True):
    """加载基础模型和 LoRA 权重"""
    print(f"Loading base model from {base_model_path}...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 使用 AutoModelForCausalLM 配合 trust_remote_code=True 自动加载自定义模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if use_lora and lora_path:
        print(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        print("Using base model without LoRA...")
    
    model.eval()
    
    return model, processor


def inference(model, processor, image_path, question, max_new_tokens=256):
    """单张图片推理"""
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    
    # 限制图片大小
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    # 构建对话
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # 应用 chat template
    text_input = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 处理输入
    inputs = processor(
        images=image,
        text=text_input,
        return_tensors="pt"
    )
    
    # 移到 GPU
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    # 解码
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def evaluate_on_val_set(model, processor, val_data_path, image_root, num_samples=None):
    """在验证集上评估"""
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    
    if num_samples:
        val_data = val_data[:num_samples]
    
    results = []
    correct = 0
    total = 0
    
    for sample in tqdm(val_data, desc="Evaluating"):
        image_path = os.path.join(image_root, sample["images"][0])
        question = sample["messages"][0]["content"].replace("<image>\n", "").replace("<image>", "")
        gt_answer = sample["messages"][1]["content"]
        
        try:
            pred_answer = inference(model, processor, image_path, question)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            pred_answer = ""
        
        # 简单匹配评估
        is_correct = gt_answer.strip() in pred_answer or pred_answer.strip() in gt_answer
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "image": sample["images"][0],
            "question": question,
            "gt": gt_answer,
            "pred": pred_answer,
            "match": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    return results, accuracy


def main():
    # 使用全局 args 变量（已在文件开头解析）
    
    print("=" * 50)
    if args.no_lora:
        print("LLaVA-OneVision-1.5-4B 原始模型测试")
    else:
        print("LLaVA-OneVision-1.5-4B + LoRA 模型测试")
    print("=" * 50)
    
    # 加载模型
    model, processor = load_model(args.base_model, args.lora_path, use_lora=not args.no_lora)
    
    # 评估
    print(f"\nEvaluating on {args.num_samples if args.num_samples else 'all'} samples...")
    results, accuracy = evaluate_on_val_set(
        model, processor, 
        args.val_data, 
        args.image_root,
        args.num_samples
    )
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"\n{'='*50}")
    print(f"评估结果")
    print(f"{'='*50}")
    print(f"测试样本数: {len(results)}")
    print(f"匹配准确率: {accuracy*100:.2f}%")
    print(f"结果保存至: {args.output}")
    
    # 显示一些样例
    print(f"\n{'='*50}")
    print("样例预测结果")
    print(f"{'='*50}")
    for i, r in enumerate(results[:5]):
        print(f"\n--- 样例 {i+1} ---")
        print(f"图片: {r['image']}")
        print(f"问题: {r['question']}")
        print(f"GT: {r['gt'][:80]}{'...' if len(r['gt']) > 80 else ''}")
        print(f"预测: {r['pred'][:80]}{'...' if len(r['pred']) > 80 else ''}")
        print(f"匹配: {'✓' if r['match'] else '✗'}")


if __name__ == "__main__":
    main()
