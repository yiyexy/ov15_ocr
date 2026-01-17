# LLaVA-OneVision-1.5 OCR LoRA 微调项目

基于 LLaVA-OneVision-1.5-4B 的 OCR 识别 LoRA 微调。

## 环境配置

```bash
conda create -n ov15_ocr python=3.10
conda activate ov15_ocr
pip install -r requirements.txt
```

## 数据集图片下载路径
```
https://aistudio.baidu.com/datasetdetail/125158
```

## 预训练模型路径
```
https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct
```

## 项目结构

```
ov15_ocr/
├── train_llava_ov_ocr_lora.py    # LoRA 训练脚本
├── test_lora_model.py            # 模型测试脚本
├── merge_lora_weights.py         # LoRA 权重合并脚本
├── create_invoice_vqa_dataset.py # 训练集生成
├── create_invoice_vqa_val.py     # 验证集生成
├── requirements.txt              # 依赖库
└── dataset/                      # 数据集
    ├── invoice_vqa_dataset.json  # 训练集 (709 samples)
    └── invoice_vqa_val.json      # 验证集 (598 samples)
```

## 使用方法

### 1. 训练 LoRA

```bash
# 默认配置（启用 gradient checkpointing）
CUDA_VISIBLE_DEVICES=0 python train_llava_ov_ocr_lora.py \
    --model_path /path/to/LLaVA-OneVision-1.5-4B-Instruct \
    --data_path dataset/invoice_vqa_dataset.json \
    --output_dir output/llava_ov15_4b_ocr_lora \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

#### 显存优化模式

```bash
# 4-bit 量化 + 小图像（约 6-8GB 显存）
CUDA_VISIBLE_DEVICES=0 python train_llava_ov_ocr_lora.py \
    --model_path /path/to/LLaVA-OneVision-1.5-4B-Instruct \
    --use_4bit \
    --max_image_size 256 \
    --max_length 512

# 8-bit 量化（约 10-12GB 显存）
CUDA_VISIBLE_DEVICES=0 python train_llava_ov_ocr_lora.py \
    --model_path /path/to/LLaVA-OneVision-1.5-4B-Instruct \
    --use_8bit
```

#### 完整参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_length` | 1024 | 最大序列长度，降低可节省显存 |
| `--max_image_size` | 384 | 图像最大尺寸，降低可大幅节省显存 |
| `--use_4bit` | False | 使用 4-bit 量化加载模型 |
| `--use_8bit` | False | 使用 8-bit 量化加载模型 |
| `--gradient_checkpointing` | True | 启用梯度检查点（用计算换显存）|

### 2. 测试模型

```bash
python test_lora_model.py \
    --base_model /path/to/LLaVA-OneVision-1.5-4B-Instruct \
    --lora_path output/llava_ov15_4b_ocr_lora/final \
    --num_samples 20
```

### 3. 合并权重

```bash
python merge_lora_weights.py \
    --base_model /path/to/LLaVA-OneVision-1.5-4B-Instruct \
    --lora_path output/llava_ov15_4b_ocr_lora/final \
    --output output/llava_ov15_4b_ocr_merged
```

## 训练配置

| 参数 | 默认值 |
|------|--------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-5 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Epochs | 3 |

## License

MIT
