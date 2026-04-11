# Vision Transformer (ViT) 模型规模说明

## ViT 模型规模分类

ViT模型通常**不是按参数量（B）命名**，而是按照模型规模命名。以下是常见的ViT模型版本：

### 1. 标准ViT规模

| 模型规模 | 参数量 | Hidden Size | Layers | Heads | Patch Size | HuggingFace模型示例 |
|---------|--------|-------------|--------|-------|------------|-------------------|
| **ViT-Base (ViT-B)** | ~86M (0.086B) | 768 | 12 | 12 | 16/32 | `google/vit-base-patch16-224` |
| **ViT-Large (ViT-L)** | ~307M (0.3B) | 1024 | 24 | 16 | 16/32 | `google/vit-large-patch16-224` |
| **ViT-Huge (ViT-H)** | ~632M (0.6B) | 1280 | 32 | 16 | 14 | `google/vit-huge-patch14-224-in21k` |

### 2. 详细规格

#### ViT-Base
- **参数量**: ~86M (0.086B)
- **Hidden Size**: 768
- **Transformer Layers**: 12
- **Attention Heads**: 12
- **MLP Size**: 3072 (768 × 4)
- **常见模型**:
  - `google/vit-base-patch16-224`
  - `google/vit-base-patch16-224-in21k`
  - `google/vit-base-patch32-224`

#### ViT-Large
- **参数量**: ~307M (0.3B)
- **Hidden Size**: 1024
- **Transformer Layers**: 24
- **Attention Heads**: 16
- **MLP Size**: 4096 (1024 × 4)
- **常见模型**:
  - `google/vit-large-patch16-224`
  - `google/vit-large-patch16-224-in21k`
  - `google/vit-large-patch32-224`

#### ViT-Huge
- **参数量**: ~632M (0.6B)
- **Hidden Size**: 1280
- **Transformer Layers**: 32
- **Attention Heads**: 16
- **MLP Size**: 5120 (1280 × 4)
- **常见模型**:
  - `google/vit-huge-patch14-224-in21k`

### 3. 其他ViT变体

#### ViT-Small
- **参数量**: ~22M (0.022B)
- **Hidden Size**: 384
- **Transformer Layers**: 12
- **Attention Heads**: 6
- **常见模型**: `google/vit-small-patch16-224`

#### ViT-Tiny
- **参数量**: ~5.5M (0.0055B)
- **Hidden Size**: 192
- **Transformer Layers**: 12
- **Attention Heads**: 3

## 在LLaVA中使用ViT

### 当前状态
LLaVA目前**只支持CLIP视觉编码器**，不支持标准ViT。CLIP使用的是CLIP ViT，与标准ViT略有不同。

### CLIP ViT规模
CLIP ViT也有不同规模：

| CLIP模型 | ViT规模 | 参数量 | Hidden Size |
|---------|---------|--------|-------------|
| CLIP ViT-B/32 | Base | ~86M | 768 |
| CLIP ViT-B/16 | Base | ~86M | 768 |
| CLIP ViT-L/14 | Large | ~307M | 1024 |
| CLIP ViT-L/14@336px | Large | ~307M | 1024 |

### 如果要使用标准ViT
需要：
1. 实现`ViTVisionTower`类（参考我之前创建的`swin_encoder.py`）
2. 修改`builder.py`支持ViT
3. 重新训练`mm_projector`（因为`hidden_size`不同）

## 参数量对比

| 模型 | 参数量 | 换算 |
|------|--------|------|
| ViT-Tiny | 5.5M | 0.0055B |
| ViT-Small | 22M | 0.022B |
| ViT-Base | 86M | **0.086B** |
| ViT-Large | 307M | **0.3B** |
| ViT-Huge | 632M | **0.6B** |

**注意**: ViT的参数量远小于语言模型（如LLaMA-7B），因为：
- ViT主要用于视觉特征提取
- 参数量集中在Transformer层
- 没有语言模型的词嵌入层（vocab很大）

## 总结

- ViT**不按B（Billion）命名**，而是按Base/Large/Huge命名
- 最大的是ViT-Huge，约**0.6B参数**
- LLaVA目前只支持CLIP ViT，不支持标准ViT
- 如需使用标准ViT，需要自行实现VisionTower类

