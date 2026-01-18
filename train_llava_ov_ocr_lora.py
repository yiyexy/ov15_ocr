"""
LLaVA-OneVision-1.5-4B OCR 微调训练脚本
使用 LoRA 进行参数高效微调
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
import argparse

# 解析命令行参数（需要先解析以获取模型路径）
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./checkpoints/LLaVA-OneVision-1.5-4B-Instruct")
    parser.add_argument("--data_path", default="./dataset/invoice_vqa_dataset.json")
    parser.add_argument("--output_dir", default="./output/llava_ov15_4b_ocr_lora")
    parser.add_argument("--max_samples", type=int, default=None, help="最大训练样本数，用于调试")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度，降低可节省显存")
    parser.add_argument("--max_image_size", type=int, default=384, help="图像最大尺寸，降低可大幅节省显存")
    parser.add_argument("--use_4bit", action="store_true", help="使用 4-bit 量化加载模型")
    parser.add_argument("--use_8bit", action="store_true", help="使用 8-bit 量化加载模型")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="启用梯度检查点")
    return parser.parse_args()

args = parse_args()

from transformers import AutoProcessor, AutoConfig, AutoModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import json
from PIL import Image
from typing import Dict, List, Any
import numpy as np
import random

# 不需要手动注册，trust_remote_code=True 会自动加载自定义模型类

# 设置随机种子
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 提高 PIL 像素限制，避免 DecompressionBomb 警告
Image.MAX_IMAGE_PIXELS = None


class OCRInstructDataset(Dataset):
    """
    OCR 指令微调数据集
    数据格式：
    {
        "id": "identity_0",
        "images": ["path/to/image.png"],
        "messages": [
            {"role": "user", "content": "<image>"},
            {"role": "assistant", "content": "OCR结果..."}
        ]
    }
    """

    def __init__(self, data_path: str, processor, max_length: int = 1024, max_samples: int = None, max_image_size: int = 384):
        self.processor = processor
        self.max_length = max_length
        self.max_image_size = max_image_size
        self.samples = self._load_samples(data_path, max_samples)

    def _load_samples(self, data_path: str, max_samples: int = None) -> List[Dict]:
        """加载数据"""
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} items from JSON")
        
        # 如果设置了 max_samples，先截断再检查（节省时间）
        if max_samples:
            data = data[:max_samples]
            print(f"Using first {max_samples} samples")
        
        # 过滤无效样本（跳过文件存在性检查以加快速度）
        valid_samples = []
        skipped = 0
        for item in data:
            if not item.get("images") or not item.get("messages"):
                skipped += 1
                continue
            valid_samples.append(item)
        
        if skipped > 0:
            print(f"Skipped {skipped} invalid samples (missing images/messages)")
        
        print(f"Total valid samples: {len(valid_samples)}")
        
        if max_samples:
            valid_samples = valid_samples[:max_samples]
        
        print(f"Loaded {len(valid_samples)} valid samples")
        return valid_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 1. 加载图像并限制尺寸（避免产生过多 image tokens）
        image_path = sample["images"][0]
        try:
            image = Image.open(image_path).convert("RGB")
            # 限制最大尺寸，减少 image tokens 数量
            max_size = self.max_image_size
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个空白图像
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        
        # 2. 解析 messages
        messages = sample["messages"]
        user_content = ""
        assistant_content = ""
        
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]
        
        # 3. 构建对话格式（包含完整的输入和输出）
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_content.replace("<image>", "").strip() or "请识别图片中的文字内容。"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_content[:1500]}  # 截断过长的输出
                ]
            }
        ]
        
        # 4. 应用 chat template（完整对话）
        text_input = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 5. 处理输入
        encoding = self.processor(
            images=image,
            text=text_input,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,  # 不截断，保持 image tokens 完整
            return_tensors="pt"
        )
        
        # 检查长度，如果仍然超长则跳过（返回一个简单样本）
        if encoding["input_ids"].shape[1] > self.max_length:
            # 使用更小的图片重试
            image = image.resize((384, 384), Image.LANCZOS)
            encoding = self.processor(
                images=image,
                text=text_input,
                padding="max_length",
                max_length=self.max_length,
                truncation=False,
                return_tensors="pt"
            )
        
        # 6. 构造 labels - 只对 assistant 回复部分计算 loss
        labels = encoding["input_ids"].clone()
        input_ids = encoding["input_ids"].squeeze(0)
        
        # 找到 assistant 回复开始的位置
        # 先获取只有 user 部分的 text（用于定位 assistant 开始位置）
        user_only_conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_content.replace("<image>", "").strip() or "请识别图片中的文字内容。"}
                ]
            }
        ]
        user_only_text = self.processor.apply_chat_template(
            user_only_conversation,
            tokenize=False,
            add_generation_prompt=True  # 添加 assistant 开始标记
        )
        
        # 获取 user 部分的 token 数量（近似）
        user_encoding = self.processor(
            images=image,
            text=user_only_text,
            return_tensors="pt"
        )
        user_len = user_encoding["input_ids"].shape[1]
        
        # 将 user 部分（包括 image tokens）的 labels 设为 -100
        labels[0, :user_len] = -100
        
        # 将 padding 设为 -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # 7. 构造返回字典
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        
        # 添加图像相关字段
        if "pixel_values" in encoding:
            item["pixel_values"] = encoding["pixel_values"].squeeze(0)
        
        if "image_sizes" in encoding:
            item["image_sizes"] = encoding["image_sizes"]
            
        if "image_grid_thw" in encoding:
            item["image_grid_thw"] = encoding["image_grid_thw"]

        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """自定义 Collate 函数"""
    collated_batch = {}
    
    # 特殊处理的字段
    special_keys = {"image_sizes", "image_grid_thw"}
    
    for key in batch[0].keys():
        if key in special_keys:
            # 这些字段需要 concat 而不是 stack
            values = [item[key] for item in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                collated_batch[key] = torch.cat(values, dim=0)
            else:
                collated_batch[key] = values
        elif isinstance(batch[0][key], torch.Tensor):
            try:
                collated_batch[key] = torch.stack([item[key] for item in batch])
            except RuntimeError:
                # 如果无法 stack（形状不一致），使用 padding
                max_len = max(item[key].shape[0] for item in batch)
                padded = []
                for item in batch:
                    tensor = item[key]
                    if tensor.shape[0] < max_len:
                        pad_size = max_len - tensor.shape[0]
                        if tensor.dim() == 1:
                            tensor = torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype)])
                        else:
                            tensor = torch.cat([tensor, torch.zeros(pad_size, *tensor.shape[1:], dtype=tensor.dtype)])
                    padded.append(tensor)
                collated_batch[key] = torch.stack(padded)
        else:
            collated_batch[key] = [item[key] for item in batch]
    
    return collated_batch


def main():
    # 使用全局 args 变量（已在文件开头解析）
    
    print("=" * 50)
    print("LLaVA-OneVision-1.5-4B OCR LoRA 微调")
    print("=" * 50)
    
    # 1. 加载模型和处理器
    print(f"\nLoading model from {args.model_path}...")
    
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # 配置量化选项
    model_kwargs = {
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        print("Using 4-bit quantization...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.use_8bit:
        from transformers import BitsAndBytesConfig
        print("Using 8-bit quantization...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    # 使用 AutoModel 加载自定义模型（trust_remote_code=True 会自动识别）
    model = AutoModel.from_pretrained(
        args.model_path,
        **model_kwargs
    )
    
    # 2. 配置 LoRA
    print("\nConfiguring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        ]
    )
    
    # 启用梯度检查点（用计算换显存）
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    # 确保模型参数可训练
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 3. 准备数据集
    print("\nPreparing dataset...")
    train_dataset = OCRInstructDataset(
        data_path=args.data_path,
        processor=processor,
        max_length=args.max_length,
        max_samples=args.max_samples,
        max_image_size=args.max_image_size
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # VLM 必须用 batch_size=1
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=4,  # 增加数据加载并行度
        dataloader_pin_memory=True,  # 加速 GPU 数据传输
        dataloader_prefetch_factor=2,  # 预取数据
        remove_unused_columns=False,
        report_to="tensorboard",
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
    )
    
    # 5. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    # 6. 开始训练
    print("\nStarting training...")
    trainer.train()
    
    # 7. 保存模型
    print("\nSaving model...")
    trainer.save_model(f"{args.output_dir}/final")
    processor.save_pretrained(f"{args.output_dir}/final")
    
    print(f"\nTraining complete! Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
