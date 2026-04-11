# LLaVA 视觉Backbone扩展指南

## 当前状态分析

### 1. 现有实现
LLaVA目前**仅支持CLIP视觉编码器**，具体实现位于：
- `llava/model/multimodal_encoder/clip_encoder.py` - CLIPVisionTower类
- `llava/model/multimodal_encoder/builder.py` - build_vision_tower函数

### 2. 架构限制
从代码分析可以看出：
- **硬编码CLIP**：`builder.py`中只检查CLIP相关的路径和标识符
- **接口依赖**：`CLIPVisionTower`直接使用`CLIPVisionModel`和`CLIPImageProcessor`
- **特征提取**：依赖CLIP的`hidden_states`结构
- **投影层适配**：`mm_projector`需要适配不同backbone的`hidden_size`

### 3. 关键接口要求
任何新的视觉backbone需要实现以下接口：

```python
class VisionTowerBase(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        # 初始化逻辑
        pass
    
    def load_model(self, device_map=None):
        # 加载预训练模型
        pass
    
    def forward(self, images):
        # 返回: [batch_size, num_patches, hidden_size]
        pass
    
    @property
    def hidden_size(self):
        # 返回特征维度
        pass
    
    @property
    def num_patches(self):
        # 返回patch数量
        pass
    
    @property
    def image_processor(self):
        # 返回图像预处理器
        pass
```

## 扩展方案

### 方案1：添加Swin Transformer支持

创建新文件：`llava/model/multimodal_encoder/swin_encoder.py`

```python
import torch
import torch.nn as nn
from transformers import SwinModel, SwinImageProcessor, SwinConfig

class SwinVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SwinConfig.from_pretrained(self.vision_tower_name)
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded')
            return
        
        self.image_processor = SwinImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SwinModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
    
    def feature_select(self, image_forward_outs):
        # Swin的输出结构可能不同，需要适配
        if hasattr(image_forward_outs, 'hidden_states'):
            image_features = image_forward_outs.hidden_states[self.select_layer]
        else:
            # Swin可能直接返回last_hidden_state
            image_features = image_forward_outs.last_hidden_state
        
        if self.select_feature == 'patch':
            # Swin没有CLS token，直接使用所有patch
            image_features = image_features
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        return image_features
    
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        return image_features
    
    @property
    def dtype(self):
        return self.vision_tower.dtype
    
    @property
    def device(self):
        return self.vision_tower.device
    
    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only
    
    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    @property
    def num_patches_per_side(self):
        # Swin的patch计算方式可能不同
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        return image_size // patch_size
    
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
```

### 方案2：添加通用ViT支持

创建新文件：`llava/model/multimodal_encoder/vit_encoder.py`

```python
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor, ViTConfig

class ViTVisionTower(nn.Module):
    """支持标准ViT模型（非CLIP ViT）"""
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = ViTConfig.from_pretrained(self.vision_tower_name)
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        
        self.image_processor = ViTImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = ViTModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]  # 移除CLS token
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        return image_features
    
    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        return image_features
    
    @property
    def dtype(self):
        return self.vision_tower.dtype
    
    @property
    def device(self):
        return self.vision_tower.device
    
    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only
    
    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    @property
    def num_patches_per_side(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        return image_size // patch_size
    
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
```

### 方案3：修改builder.py支持多种backbone

修改 `llava/model/multimodal_encoder/builder.py`:

```python
import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .swin_encoder import SwinVisionTower
from .vit_encoder import ViTVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', 
                          getattr(vision_tower_cfg, 'vision_tower', None))
    
    # 检测backbone类型
    vision_tower_type = getattr(vision_tower_cfg, 'vision_tower_type', None)
    
    # 如果没有指定类型，尝试自动检测
    if vision_tower_type is None:
        vision_tower_lower = vision_tower.lower()
        if 'swin' in vision_tower_lower:
            vision_tower_type = 'swin'
        elif 'vit' in vision_tower_lower and 'clip' not in vision_tower_lower:
            vision_tower_type = 'vit'
        elif 'clip' in vision_tower_lower or vision_tower.startswith('openai') or vision_tower.startswith('laion'):
            vision_tower_type = 'clip'
        else:
            vision_tower_type = 'clip'  # 默认使用CLIP
    
    # 根据类型构建对应的vision tower
    if vision_tower_type == 'swin':
        return SwinVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower_type == 'vit':
        return ViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower_type == 'clip':
        is_absolute_path_exists = os.path.exists(vision_tower)
        use_s2 = getattr(vision_tower_cfg, 's2', False)
        if is_absolute_path_exists or vision_tower.startswith("openai") or \
           vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            if use_s2:
                return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {vision_tower} (type: {vision_tower_type})')
```

## 使用示例

### 训练脚本修改

在 `finetune_lora_copy.sh` 中添加backbone类型参数：

```bash
deepspeed --master_port=29503 --include="localhost:6" llava/train/train_copy.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /data2/dmz/llava_test/LLaVA-main/llava-v1.5-7b \
    --vision_tower microsoft/swin-base-patch4-window7-224 \
    --vision_tower_type swin \  # 新增参数
    --version v1 \
    ...
```

或者在 `ModelArguments` 中添加：

```python
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_tower: Optional[str] = field(default=None)
    vision_tower_type: Optional[str] = field(default=None, 
        metadata={"help": "Vision backbone type: 'clip', 'swin', 'vit'"})
    ...
```

## 交叉编码器泛化验证

### 问题分析
当前LLaVA缺乏交叉编码器泛化验证，主要问题：

1. **投影层适配**：不同backbone的`hidden_size`不同，需要重新训练`mm_projector`
2. **特征分布差异**：不同编码器的特征分布可能差异很大
3. **patch数量差异**：不同编码器产生的patch数量不同

### 验证方案

创建验证脚本：`scripts/validate_cross_encoder.py`

```python
"""
交叉编码器泛化验证脚本
测试在不同视觉backbone下的模型性能
"""
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

def validate_cross_encoder(model_path, vision_towers, test_data):
    """
    验证模型在不同视觉backbone下的性能
    
    Args:
        model_path: 训练好的模型路径
        vision_towers: 视觉backbone列表，如 ['clip', 'swin', 'vit']
        test_data: 测试数据
    """
    results = {}
    
    for vision_tower_name in vision_towers:
        print(f"\n测试视觉backbone: {vision_tower_name}")
        
        # 加载模型（需要重新初始化投影层）
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, 
            model_base=None,
            model_name=model_name,
            vision_tower=vision_tower_name  # 需要修改load_pretrained_model支持
        )
        
        # 评估性能
        accuracy = evaluate_model(model, tokenizer, image_processor, test_data)
        results[vision_tower_name] = accuracy
        
        print(f"{vision_tower_name} 准确率: {accuracy:.2f}%")
    
    # 对比分析
    print("\n交叉编码器泛化分析:")
    print("=" * 50)
    for backbone, acc in results.items():
        print(f"{backbone}: {acc:.2f}%")
    
    return results
```

## 注意事项

### 1. 投影层重新训练
切换backbone时，`mm_projector`需要重新训练，因为：
- `hidden_size`可能不同
- 特征分布不同

### 2. 特征对齐
不同backbone的特征可能需要对齐：
```python
# 在forward中添加特征对齐层
if self.needs_alignment:
    image_features = self.alignment_layer(image_features)
```

### 3. 性能影响
- **Swin**: 通常patch数量更多，计算量更大
- **ViT**: 与CLIP ViT类似，但预训练数据不同
- **CLIP**: 当前最优，因为与文本对齐

## 总结

1. **当前限制**：LLaVA仅支持CLIP视觉编码器
2. **扩展可行**：可以通过实现新的VisionTower类来支持其他backbone
3. **需要适配**：投影层、特征提取、图像预处理都需要适配
4. **验证缺失**：需要建立交叉编码器泛化验证流程

建议优先实现Swin Transformer支持，因为它在视觉任务上表现优异，且与ViT架构相似，适配相对容易。

