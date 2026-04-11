# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
import logging
import torch
from mmengine.logging import MMLogger
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F  # 新增：导入torch.nn.functional
from xtuner.registry import BUILDER
from .utils import (
    find_all_linear_names,
    get_peft_model_state_dict,
    guess_load_checkpoint,
    make_inputs_require_grad,
)
def print_ds_id_status(model, model_name="Model"):
    """
    打印模型所有参数的 ds_id 状态（若存在）
    Args:
        model: 待检查的模型（如学生模型 self.model、教师模型 self.teacher_model）
        model_name: 模型名称（用于日志区分）
    """
    print_log(f"\n=== {model_name} Parameter ds_id Status ===", logger="current")
    original_model = model.module if hasattr(model, "module") else model

    param_count = 0
    ds_id_count = 0
    for name, param in original_model.named_parameters():
        has_ds_id = hasattr(param, "ds_id")
        param_count += 1
        if has_ds_id:
            ds_id_count += 1

    print_log(
        f"=== {model_name} Summary ==="
        f"Total params: {param_count} | Params with ds_id: {ds_id_count}",
        logger="current"
    )
    if model_name == "Teacher Model" and ds_id_count > 0:
        print_log(
            "WARNING: Teacher model parameters have ds_id! This will cause ds_id conflict!",
            logger="current",
            level=logging.ERROR
        )

class InternVL_V1_5(BaseModel):
    def __init__(
        self,
        model_path,
        freeze_llm=False,
        freeze_visual_encoder=False,
        llm_lora=None,
        visual_encoder_lora=None,
        quantization_vit=False,
        quantization_llm=False,
        pretrained_pth=None,
        # -------------------------- 新增：双任务训练参数 --------------------------
        teacher_model_path=None,  # 教师模型路径（预训练InternVL）
        ga_weight=1.0,            # 梯度上升任务权重
        kl_weight=1.2,            # KL蒸馏任务权重
        target_entity="entity",  # 屏蔽的目标实体（用于梯度上升）；请按数据替换
        deepspeed_zero_config=None,  # DualTaskTrainer 等场景使用；None 时用仓库内 deepspeed_zero3.json
        num_image_tokens=None,    # 每个图像的token数量（None表示自动计算）
        top_k_ratio=0.15,          # 选择top-k%的图像token进行扰动
        enable_token_analysis=False,  # 是否启用详细的token分析（调试用）
        # -------------------------- 新增：扰动控制参数 --------------------------
        perturbation_period=50,     # 余弦函数周期（步数）
        perturbation_start_similarity=0.7,  # 起始相似度
        perturbation_max_similarity=0.95,   # 最大相似度
        perturbation_alpha=1.0,     # 进度速率控制器
        # -------------------------- 新增：多次采样参数 --------------------------
        num_samples=3,             # 每次step的采样次数（获得多个扰动embedding）
        kl_aggregation_strategy='mean',  # KL损失聚合策略
    ):
        print_log("Start to load InternVL_V1_5 model.", logger="current")
        super().__init__()
        # -------------------------- 原有初始化逻辑（不变） --------------------------
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert llm_lora is not None

        # 加载主模型（学生模型）
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = "flash_attention_2"
        else:
            config.llm_config._attn_implementation = "flash_attention_2"
    
        # 量化配置（原有逻辑不变）
        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ["mlp1"]
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append("vision_model")
            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append("language_model")
            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            quantization_clazz = quantization_config.pop("type")
            quantization = quantization_clazz(**quantization_config)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            config=config,
            trust_remote_code=True,
        )

        # 加载Tokenizer（原有逻辑不变）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # 新增：保存tokenizer供后续使用
        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id

        # 冻结部分网络（原有逻辑不变）
        if self.freeze_llm:
            print("此时冻结语言模型")
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            print("此时冻结视觉编码器")
            self.model.vision_model.requires_grad_(False)
        # if not self.freeze_visual_encoder:
        #     print("Force enabling requires_grad for visual encoder")
        #     for name, param in self.model.vision_model.named_parameters():
        #         param.requires_grad = True
        #         print(f"Param: {name}, requires_grad: {param.requires_grad}")
        # 梯度 checkpoint（原有逻辑不变）
        if hasattr(self.model.language_model, "enable_input_require_grads"):
            self.model.language_model.enable_input_require_grads()
        else:
            self.model.language_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
        self.gradient_checkpointing_enable()

        # LoRA配置（原有逻辑不变）
        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        # 加载预训练权重（原有逻辑不变）
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Load pretrained weight from {pretrained_pth}")

        # -------------------------- 新增：双任务训练初始化 --------------------------
        self.ga_weight = ga_weight  # 梯度上升任务权重
        self.kl_weight = kl_weight  # KL蒸馏任务权重
        self.target_entity = target_entity  # 需屏蔽的目标实体（用于梯度上升的标签处理）
        self.ignore_index = -100  # InternVL默认忽略标签（与LLaVA的IGNORE_INDEX一致）
        # 动态计算图像Token数量
        if num_image_tokens is None:
            # 从模型配置中获取图像尺寸和patch大小
            if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'config'):
                vision_config = self.model.vision_model.config
                image_size = getattr(vision_config, 'image_size', 448)
                patch_size = getattr(vision_config, 'patch_size', 14)
                self.num_image_tokens = (image_size // patch_size) ** 2
                print_log(f"Auto-calculated num_image_tokens: {self.num_image_tokens} (image_size={image_size}, patch_size={patch_size})", logger="current")
            else:
                # 如果无法获取配置，使用默认值
                self.num_image_tokens = 1024  # 448x448的默认值
                print_log(f"Using default num_image_tokens: {self.num_image_tokens}", logger="current")
        else:
            self.num_image_tokens = num_image_tokens
            print_log(f"Using specified num_image_tokens: {self.num_image_tokens}", logger="current")
        self.top_k_ratio = top_k_ratio  # 选择top-k%的图像token进行扰动
        self.enable_token_analysis = enable_token_analysis  # 是否启用详细的token分析
        # -------------------------- 新增：扰动控制参数初始化 --------------------------
        self.perturbation_period = perturbation_period  # 余弦函数周期
        self.perturbation_start_similarity = perturbation_start_similarity  # 起始相似度
        self.perturbation_max_similarity = perturbation_max_similarity  # 最大相似度
        self.perturbation_alpha = perturbation_alpha  # 进度速率控制器
        # -------------------------- 新增：多次采样参数初始化 --------------------------
        self.num_samples = num_samples  # 每次step的采样次数
        self.kl_aggregation_strategy = kl_aggregation_strategy  # KL损失聚合策略

        # 1. 初始化教师模型（预训练InternVL，冻结用于蒸馏）
        if teacher_model_path is not None:
            print_log(f"Loading teacher model from {teacher_model_path}", logger="current")
            
            # 关键：使用独立的DeepSpeed引擎加载教师模型，避免ds_id冲突
            self.teacher_model_path = teacher_model_path
            self.teacher_quantization = quantization
            self.deepspeed_zero_config = deepspeed_zero_config
            
            # 加载教师模型配置
            teacher_config = AutoConfig.from_pretrained(teacher_model_path, trust_remote_code=True)
            if teacher_config.llm_config.model_type == "internlm2":
                teacher_config.llm_config.attn_implementation = "flash_attention_2"
            else:
                teacher_config.llm_config._attn_implementation = "flash_attention_2"
            
            # 加载教师模型（量化配置与学生模型一致）
            teacher_model = AutoModel.from_pretrained(
                teacher_model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization,
                config=teacher_config,
                trust_remote_code=True,
            )
            teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
            teacher_model.img_context_token_id = teacher_tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            
            # 关键：为教师模型参数添加唯一标识，避免DeepSpeed识别为相同参数
            # 通过修改参数名称来确保每个参数都有唯一的ds_id
            for name, param in teacher_model.named_parameters():
                # 为教师模型参数添加唯一前缀
                param._teacher_param = True
                param._original_name = name
            
            self.teacher_model = teacher_model
            self.teacher_tokenizer = teacher_tokenizer
            
            # 冻结教师模型
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            # 移动到CPU，避免GPU内存冲突
            self.teacher_model = self.teacher_model.cpu()
            
            print_log("Teacher model loaded with unique parameter identifiers to avoid ds_id conflicts", logger="current")
        else:
            print_log("No teacher model provided, running in gradient ascent only mode", logger="current")
            self.teacher_model = None
            self.teacher_tokenizer = None

        # 1. 打印学生模型（self.model）的 ds_id（应有 ds_id，正常）
        print_ds_id_status(self.model, model_name="Student Model (self.model)")
        # 2. 打印教师模型（self.teacher_model）的 ds_id（应无 ds_id，因为使用Stage 0）
        if self.teacher_model is not None:
            print_ds_id_status(self.teacher_model, model_name="Teacher Model (self.teacher_model)")
        else:
            print_log("No teacher model to check ds_id status", logger="current")
        self._count = 0
        print_log(self, logger="current")
        print_log("InternVL_V1_5 construction (with dual-task) is complete", logger="current")


    

    def mask_answer_tokens(self, labels):
        """
        屏蔽答案中目标实体及之后的所有token（适配InternVL的标签格式）
        Args:
            labels: (B, N) 原始标签（含IGNORE_INDEX的文本-图像对齐标签）
        Returns:
            masked_labels: (B, N) 屏蔽后的标签（目标实体及之后设为IGNORE_INDEX）
        """
        masked_labels = labels.clone()
        batch_size, seq_len = masked_labels.shape

        # 目标实体按长度降序排列（优先匹配长实体，避免短实体误触发）
        target_entities = [
            f" {self.target_entity}", self.target_entity,  # 带空格/不带空格（适配实际文本）
            self.target_entity.lower(), f" {self.target_entity.lower()}"  # 小写形式（兼容大小写）
        ]

        for i in range(batch_size):
            label_ids = masked_labels[i].tolist()
            entity_found = False

            for entity in target_entities:
                if entity_found:
                    break
                try:
                    # 编码实体为token id（与InternVL的Tokenizer一致）
                    entity_token_ids = self.tokenizer.encode(entity, add_special_tokens=False)
                    entity_len = len(entity_token_ids)
                    if entity_len == 0:
                        continue

                    # 搜索范围：跳过前导的IGNORE_INDEX（仅在有效标签中搜索）
                    valid_start = 0
                    while valid_start < seq_len and label_ids[valid_start] == self.ignore_index:
                        valid_start += 1
                    max_search_pos = valid_start + (seq_len - valid_start) - entity_len
                    if max_search_pos < valid_start:
                        continue

                    # 匹配实体并屏蔽后续token
                    for k in range(valid_start, max_search_pos + 1):
                        if label_ids[k:k+entity_len] == entity_token_ids:
                            masked_labels[i, k:] = self.ignore_index  # 实体及之后设为忽略
                            entity_found = True
                            print(f"[INFO] Sample {i}: Masked entity '{entity}' at position {k}")
                            break
                except Exception as e:
                    print(f"[WARN] Failed to process entity '{entity}': {str(e)}")
                    continue

            # 验证屏蔽效果（可选）
            valid_labels = (masked_labels[i] != self.ignore_index).sum().item()
            print(f"[DEBUG] Sample {i}: Valid labels after masking: {valid_labels}")
        print("此时屏蔽labels完成")
        assert masked_labels.shape == labels.shape, "Masked labels shape mismatch with original!"
        print("此时屏蔽完成")
        return masked_labels

    def compute_attention_scores(self, model, input_embeds, attention_mask, image_token_positions):
        """
        计算文本token对图像token的注意力分数（参考LLaVA实现）
        Args:
            model: 语言模型
            input_embeds: (B, N, C) 输入embedding
            attention_mask: (B, N) 注意力掩码
            image_token_positions: dict 图像token位置信息
        Returns:
            attention_scores: 注意力分数张量
        """
        with torch.no_grad():
            # 获取语言模型的注意力层
            if hasattr(model, 'module'):
                original_model = model.module
            else:
                original_model = model
                
            # 通过语言模型获取注意力权重
            outputs = original_model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
            
            # 获取所有层的注意力权重
            attentions = outputs.attentions
            if isinstance(attentions, tuple):
                attentions = list(attentions)
            
            # 堆叠成张量 (num_layers, B, num_heads, N, N)
            attentions_tensor = torch.stack(attentions, dim=0)
            
            print(f"[DEBUG] 注意力分析开始:")
            print(f"  - 层数: {attentions_tensor.shape[0]}")
            print(f"  - 批次大小: {attentions_tensor.shape[1]}")
            print(f"  - 头数: {attentions_tensor.shape[2]}")
            print(f"  - 序列长度: {attentions_tensor.shape[3]}")
            
            return attentions_tensor

    def _compute_high_attention_image_tokens(self, attentions, image_token_positions, input_ids, top_k_ratio=0.3):
        """
        基于注意力分数计算高关注的图像Token（适配InternVL格式）
        Args:
            attentions: (num_layers, B, num_heads, N, N) 注意力权重
            image_token_positions: dict 图像token位置信息
            input_ids: (B, N) 输入token ids
            top_k_ratio: 选择比例
        Returns:
            high_attention_indices: dict 高关注token的绝对位置
        """
        num_layers, batch_size, num_heads, seq_len, _ = attentions.shape
        high_attention_indices = {}
        
        print(f"\n[DEBUG] 注意力分析开始:")
        print(f"  - 层数: {num_layers}, 批次大小: {batch_size}, 头数: {num_heads}, 序列长度: {seq_len}")
        print(f"  - 图像Token位置信息: {len(image_token_positions)}个样本")
        
        for batch_idx, positions in image_token_positions.items():
            print(f"\n[DEBUG] 分析样本{batch_idx}:")
            print(f"  - 图像Token位置数量: {len(positions)}")
            
            # 计算文本Token范围
            if positions:
                img_start = positions[0]['start']
                img_end = positions[0]['end']
                text_ranges = []
                
                # 图像之前的文本
                if img_start > 0:
                    text_ranges.append((0, img_start))
                    print(f"  - 图像前文本Token范围: [0:{img_start}] (共{img_start}个)")
                
                # 图像之后的文本（这是关键，用于计算注意力）
                if img_end < seq_len:
                    text_ranges.append((img_end, seq_len))
                    print(f"  - 图像后文本Token范围: [{img_end}:{seq_len}] (共{seq_len-img_end}个)")
                else:
                    print(f"  - 图像在序列末尾，无图像后文本")
                
                if not text_ranges:
                    text_ranges = [(0, img_start)]
                    print(f"  - 默认文本Token范围: [0:{img_start}] (共{img_start}个)")
            else:
                text_ranges = [(0, seq_len)]
                print(f"  - 全文本Token范围: [0:{seq_len}] (共{seq_len}个)")
            
            # 对每个图像token区域进行分析
            for pos_info in positions:
                img_start = pos_info['start']
                img_end = pos_info['end']
                num_image_tokens = img_end - img_start
                
                print(f"  - 图像Token范围: [{img_start}:{img_end}] (共{num_image_tokens}个)")
                
                # 提取当前样本的所有层Attention权重
                batch_attentions = attentions[:, batch_idx, :, :, :]
                
                # 计算每个图像Token的关注度
                token_attention_scores = torch.zeros(num_image_tokens, device=batch_attentions.device)
                
                # 使用加权层平均，后期层权重更高
                layer_weights = torch.linspace(0.1, 2.0, num_layers, device=batch_attentions.device)
                layer_weights = F.softmax(layer_weights, dim=0)
                print(f"  - 层权重分布: {layer_weights.tolist()}")
                
                for layer_idx in range(num_layers):
                    layer_attn = batch_attentions[layer_idx]
                    
                    # 修正策略：只使用图像后的文本token计算注意力
                    # 因为InternVL使用掩码注意力，只有图像后的文本才能看到图像token
                    post_image_text_range = None
                    for text_start, text_end in text_ranges:
                        if text_end > text_start and text_start >= img_end:  # 图像后的文本
                            post_image_text_range = (text_start, text_end)
                            break
                    
                    if post_image_text_range:
                        text_start, text_end = post_image_text_range
                        # 图像后文本token对图像token的注意力
                        text2img_attn = layer_attn[:, text_start:text_end, img_start:img_end]
                        text_tokens = text_end - text_start
                        
                        if text_tokens > 0:
                            # 对多头、多文本Token取平均
                            text_attn_per_token = text2img_attn.mean(dim=[0, 1])  # [num_image_tokens]
                            print(f"    * 图像后文本范围[{text_start}:{text_end}]对图像注意力形状: {text2img_attn.shape}")
                        else:
                            text_attn_per_token = torch.zeros(img_end - img_start, device=layer_attn.device)
                            print(f"    * 图像后文本范围为空，创建零张量: {text_attn_per_token.shape}")
                    else:
                        # 如果没有图像后的文本，使用随机注意力（作为fallback）
                        text_attn_per_token = torch.randn(img_end - img_start, device=layer_attn.device) * 0.01
                        print(f"    * 未找到图像后文本范围，使用随机注意力: {text_attn_per_token.shape}")
                    
                    # 加权累加
                    weight = layer_weights[layer_idx]
                    token_attention_scores += weight * text_attn_per_token
                
                # 使用L1归一化保持相对比例
                if token_attention_scores.sum() > 0:
                    token_attention_scores = F.normalize(token_attention_scores, p=1, dim=0)
                else:
                    # 如果所有注意力都是0，使用随机分布
                    token_attention_scores = torch.randn_like(token_attention_scores)
                    token_attention_scores = F.softmax(token_attention_scores, dim=0)
                    print(f"  - 所有注意力为0，使用随机分布")
                
                # 筛选Top K高关注Token
                top_k = max(1, int(num_image_tokens * top_k_ratio))
                top_values, top_rel_indices = torch.topk(token_attention_scores, k=top_k)
                top_abs_indices = [img_start + rel_idx for rel_idx in top_rel_indices.tolist()]
                
                print(f"  - Top-{top_k} 高关注图像Token:")
                for i, (abs_idx, rel_idx) in enumerate(zip(top_abs_indices, top_rel_indices.tolist())):
                    score = token_attention_scores[rel_idx].item()
                    print(f"    * Token {abs_idx} (相对位置{rel_idx}): {score:.6f}")
                
                # 保存结果
                if batch_idx not in high_attention_indices:
                    high_attention_indices[batch_idx] = []
                high_attention_indices[batch_idx].extend(top_abs_indices)
        
        print(f"\n[DEBUG] 注意力分析完成，共找到{len(high_attention_indices)}个样本的高关注Token")
        return high_attention_indices

    def _fallback_random_selection(self, image_token_positions, top_k_ratio=0.3):
        """
        当注意力计算失败时的fallback策略：随机选择图像token
        Args:
            image_token_positions: dict 图像token位置信息
            top_k_ratio: 选择比例
        Returns:
            high_attention_indices: dict 选中的token位置
        """
        high_attention_indices = {}
        
        print(f"\n[FALLBACK] 使用随机选择策略:")
        print(f"  - 选择比例: {top_k_ratio}")
        
        for batch_idx, positions in image_token_positions.items():
            print(f"\n[FALLBACK] 样本{batch_idx}:")
            
            for pos_info in positions:
                img_start = pos_info['start']
                img_end = pos_info['end']
                num_image_tokens = img_end - img_start
                
                # 随机选择top-k%的token
                top_k = max(1, int(num_image_tokens * top_k_ratio))
                selected_indices = torch.randperm(num_image_tokens)[:top_k]
                top_abs_indices = [img_start + idx.item() for idx in selected_indices]
                
                print(f"  - 图像Token范围: [{img_start}:{img_end}] (共{num_image_tokens}个)")
                print(f"  - 随机选择{top_k}个token: {top_abs_indices}")
                
                if batch_idx not in high_attention_indices:
                    high_attention_indices[batch_idx] = []
                high_attention_indices[batch_idx].extend(top_abs_indices)
        
        print(f"[FALLBACK] 随机选择完成，共找到{len(high_attention_indices)}个样本的选中Token")
        return high_attention_indices

    def _perturb_high_attention_embeds(self, image_embeds, high_attention_indices, image_token_positions, target_similarity=0.7, sample_idx=0):
        """
        基于高关注图像Token的Embedding进行扰动（支持多次采样）
        Args:
            image_embeds: (B, num_image_tokens, embed_dim) 图像embedding
            high_attention_indices: dict 高关注token的绝对位置
            image_token_positions: dict 图像token位置信息
            target_similarity: 目标相似度
            sample_idx: 采样索引（用于生成不同的随机噪声）
        Returns:
            perturbed_embeds: 扰动后的图像embedding
        """
        perturbed_image_embeds = image_embeds.clone()
        batch_size, num_image_tokens_total, embed_dim = image_embeds.shape
        
        print(f"\n[DEBUG] 开始针对性扰动 (Token级别) - 第{sample_idx+1}次采样:")
        print(f"  - 图像Embedding形状: {image_embeds.shape}")
        print(f"  - 需要扰动的样本数: {len(high_attention_indices)}")
        print(f"  - 采样索引: {sample_idx}")
        
        # 计算目标相似度（使用改进的余弦函数公式）
        import math
        current_step = getattr(self, '_count', 0)
        
        # 原公式参数映射
        s_min = self.perturbation_start_similarity  # 相似度目标下限
        T = self.perturbation_period  # 周期长度
        alpha = 1.0  # 进度速率控制器
        epsilon = 1e-2  # 小常数，确保不触及边界
        
        # 原公式实现
        # 基础偏移项：(s_min + 1) / 2
        base_offset = (s_min + 1) / 2
        
        # 周期性波动项计算
        # 1. 计算余弦函数的相位：(2π * current_step * alpha) / T
        phase = (2 * math.pi * current_step * alpha) / T
        
        # 2. 计算波动幅度因子：(1 - s_min) / 2 * (1 - 2ε)
        amplitude_factor = (1 - s_min) / 2 * (1 - 2 * epsilon)
        
        # 3. 计算完整的周期性波动项
        fluctuation = amplitude_factor * math.cos(phase)-0.05
        
        # 最终目标相似度：基础偏移项 + 周期性波动项
        target_similarity = base_offset + fluctuation
        
        print(f"  - 当前步数: {current_step}")
        print(f"  - 扰动周期: {T}, 余弦相位: {phase:.4f}")
        print(f"  - 相似度下限: {s_min:.4f}, 基础偏移: {base_offset:.4f}")
        print(f"  - 波动幅度: {amplitude_factor:.4f}, 最终目标相似度: {target_similarity:.4f}")
        
        for batch_idx, top_abs_indices in high_attention_indices.items():
            if batch_idx not in image_token_positions:
                continue
                
            print(f"\n[DEBUG] 扰动样本{batch_idx}:")
            print(f"  - 高关注Token绝对位置: {top_abs_indices}")
            
            # 找到对应的图像token位置信息
            positions = image_token_positions[batch_idx]
            
            # 对每个高关注token进行扰动
            for abs_idx in top_abs_indices:
                # 找到这个token属于哪个图像区域
                for pos_info in positions:
                    if pos_info['start'] <= abs_idx < pos_info['end']:
                        rel_idx = abs_idx - pos_info['start']
                        image_idx = pos_info['image_idx']
                        
                        # 提取对应的embedding
                        if image_idx < image_embeds.shape[0]:
                            token_embed = image_embeds[batch_idx, rel_idx:rel_idx+1]
                            
                            # 每个token都直接使用目标相似度
                            token_target_similarity = target_similarity
                            
                            # 应用扰动
                            token_embed_normalized = F.normalize(token_embed, p=2, dim=-1)
                            
                            # 生成垂直随机向量（每次采样使用不同的随机种子）
                            # 为每次采样设置不同的随机种子，确保生成不同的噪声
                            torch.manual_seed(torch.randint(0, 10000, (1,)).item() + sample_idx * 1000)
                            noise = torch.randn_like(token_embed_normalized)
                            noise = noise - torch.sum(noise * token_embed_normalized, dim=-1, keepdim=True) * token_embed_normalized
                            noise_normalized = F.normalize(noise, p=2, dim=-1)
                            
                            # 计算旋转角度
                            theta = torch.acos(torch.tensor(token_target_similarity, device=image_embeds.device))
                            
                            # 旋转扰动
                            perturbed_token_embed = (
                                token_embed_normalized * torch.cos(theta) + 
                                noise_normalized * torch.sin(theta)
                            )
                            
                            # 将扰动后的embedding放回原位置
                            perturbed_image_embeds[batch_idx, rel_idx:rel_idx+1] = perturbed_token_embed
                            
                            # 验证单个token的相似度
                            try:
                                similarity_tensor = F.cosine_similarity(token_embed, perturbed_token_embed, dim=-1)
                                if similarity_tensor.numel() > 0:
                                    actual_similarity = similarity_tensor.item()
                                else:
                                    actual_similarity = 0.0
                                print(f"    Token位置{abs_idx} (相对位置{rel_idx}): 目标相似度={token_target_similarity:.4f}, 实际相似度={actual_similarity:.4f}")
                            except Exception as e:
                                print(f"    Token位置{abs_idx} (相对位置{rel_idx}): 相似度计算失败: {e}")
                                actual_similarity = 0.0
                        break
        
        # 计算整体扰动效果
        overall_similarity = F.cosine_similarity(image_embeds, perturbed_image_embeds, dim=-1).mean()
        print(f"\n[DEBUG] Token级别扰动效果:")
        print(f"  - 整体相似度: {overall_similarity.item():.4f}")
        print(f"  - 扰动Token数量: {sum(len(indices) for indices in high_attention_indices.values())}")
        
        return perturbed_image_embeds

    def _batch_perturb_high_attention_embeds(self, image_embeds, high_attention_indices, image_token_positions, num_samples=3):
        """
        批量生成多个采样的扰动embedding（并行计算）
        Args:
            image_embeds: (B, num_image_tokens, embed_dim) 图像embedding
            high_attention_indices: dict 高关注token的绝对位置
            image_token_positions: dict 图像token位置信息
            num_samples: 采样次数
        Returns:
            batch_perturbed_embeds: (B * num_samples, num_image_tokens, embed_dim) 批量扰动embedding
        """
        batch_size, num_image_tokens_total, embed_dim = image_embeds.shape
        
        print(f"\n[DEBUG] 开始批量扰动生成:")
        print(f"  - 原始图像Embedding形状: {image_embeds.shape}")
        print(f"  - 采样次数: {num_samples}")
        print(f"  - 目标批量形状: [{batch_size * num_samples}, {num_image_tokens_total}, {embed_dim}]")
        
        # 计算目标相似度（使用改进的余弦函数公式）
        import math
        current_step = getattr(self, '_count', 0)
        
        # 原公式参数映射
        s_min = self.perturbation_start_similarity
        T = self.perturbation_period
        alpha = 1.0
        epsilon = 1e-2
        
        # 计算目标相似度
        base_offset = (s_min + 1) / 2
        phase = (2 * math.pi * current_step * alpha) / T
        amplitude_factor = (1 - s_min) / 2 * (1 - 2 * epsilon)
        fluctuation = amplitude_factor * math.cos(phase) - 0.05
        target_similarity = base_offset + fluctuation
        
        print(f"  - 目标相似度: {target_similarity:.4f}")
        
        # 批量生成扰动embedding
        # 方法：将原始embedding复制num_samples次，然后批量扰动
        batch_image_embeds = image_embeds.unsqueeze(0).repeat(num_samples, 1, 1, 1)  # [num_samples, B, num_tokens, embed_dim]
        batch_image_embeds = batch_image_embeds.view(-1, num_image_tokens_total, embed_dim)  # [num_samples * B, num_tokens, embed_dim]
        
        print(f"  - 批量复制后形状: {batch_image_embeds.shape}")
        
        # 对每个样本进行扰动
        for sample_idx in range(num_samples):
            start_idx = sample_idx * batch_size
            end_idx = (sample_idx + 1) * batch_size
            
            print(f"  - 处理第{sample_idx+1}次采样: 索引[{start_idx}:{end_idx}]")
            
            # 为这次采样设置随机种子
            torch.manual_seed(torch.randint(0, 10000, (1,)).item() + sample_idx * 1000)
            
            # 对当前采样的所有batch进行扰动
            for batch_idx, top_abs_indices in high_attention_indices.items():
                if batch_idx not in image_token_positions:
                    continue
                
                # 计算在批量张量中的实际索引
                actual_batch_idx = start_idx + batch_idx
                positions = image_token_positions[batch_idx]
                
                # 对每个高关注token进行扰动
                for abs_idx in top_abs_indices:
                    for pos_info in positions:
                        if pos_info['start'] <= abs_idx < pos_info['end']:
                            rel_idx = abs_idx - pos_info['start']
                            
                            # 提取对应的embedding
                            token_embed = batch_image_embeds[actual_batch_idx, rel_idx:rel_idx+1]
                            
                            # 应用扰动
                            token_embed_normalized = F.normalize(token_embed, p=2, dim=-1)
                            
                            # 生成垂直随机向量
                            noise = torch.randn_like(token_embed_normalized)
                            noise = noise - torch.sum(noise * token_embed_normalized, dim=-1, keepdim=True) * token_embed_normalized
                            noise_normalized = F.normalize(noise, p=2, dim=-1)
                            
                            # 计算旋转角度
                            theta = torch.acos(torch.tensor(target_similarity, device=image_embeds.device))
                            
                            # 旋转扰动
                            perturbed_token_embed = (
                                token_embed_normalized * torch.cos(theta) + 
                                noise_normalized * torch.sin(theta)
                            )
                            
                            # 将扰动后的embedding放回原位置
                            batch_image_embeds[actual_batch_idx, rel_idx:rel_idx+1] = perturbed_token_embed
                            break
        
        print(f"  - 批量扰动完成，最终形状: {batch_image_embeds.shape}")
        
        # 验证扰动效果
        original_flat = image_embeds.view(-1, embed_dim)
        perturbed_flat = batch_image_embeds.view(-1, embed_dim)
        
        # 计算每个样本的相似度
        similarities = []
        for i in range(num_samples):
            start_idx = i * batch_size * num_image_tokens_total
            end_idx = (i + 1) * batch_size * num_image_tokens_total
            sample_similarity = F.cosine_similarity(
                original_flat, 
                perturbed_flat[start_idx:end_idx], 
                dim=-1
            ).mean()
            similarities.append(sample_similarity.item())
            print(f"  - 第{i+1}次采样平均相似度: {sample_similarity.item():.4f}")
        
        return batch_image_embeds

    def _parse_lora_config(self, lora_config):
        if (
            isinstance(lora_config, dict)
            or isinstance(lora_config, Config)
            or isinstance(lora_config, ConfigDict)
        ):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.language_model = prepare_model_for_kbit_training(
            self.model.language_model, use_activation_checkpointing
        )
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.language_model)
            lora_config.target_modules = modules
        self.model.language_model = get_peft_model(
            self.model.language_model, lora_config
        )

    def _prepare_visual_encoder_for_lora(self, lora_config):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.vision_model)
            lora_config.target_modules = modules
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.language_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.vision_model, state_dict=state_dict
                )
            )
        elif not self.freeze_visual_encoder:
            to_return.update(
                {k: v for k, v in state_dict.items() if "model.vision_model." in k}
            )
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.language_model, state_dict=state_dict
                )
            )
        elif not self.freeze_llm:
            to_return.update(
                {k: v for k, v in state_dict.items() if "model.language_model." in k}
            )
        # Step 3. Projector
        to_return.update({k: v for k, v in state_dict.items() if "model.mlp1." in k})
        return to_return

    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode="loss"):
        """
        双任务训练的前向传播：
        1. 梯度上升任务：对屏蔽目标实体后的标签计算交叉熵损失，取负号
        2. KL蒸馏任务：对齐学生模型与教师模型的输出分布
        """
        if mode != "loss":
            raise NotImplementedError("Only 'loss' mode is supported for dual-task training.")

        # 1. 预处理输入数据（原有逻辑不变）
        pixel_values = data["pixel_values"]
        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values]
            concat_images = torch.cat(
                [image.to(self.model.vision_model.dtype) for image in pixel_values],
                dim=0,
            )
        else:
            raise NotImplementedError()
        data["pixel_values"] = concat_images  # 更新为拼接后的图像
        # data["pixel_values"].requires_grad_(True)
        print("步骤1完成")
        # 2. 梯度上升任务：屏蔽标签中的目标实体
        original_labels = data["labels"]
        masked_labels = self.mask_answer_tokens(original_labels)  # 调用步骤2的屏蔽方法
        print("此时计算ga loss")
        # 3. 计算梯度上升损失（负交叉熵损失）
        ga_outputs = self._llm_forward(
            model=self.model,
            pixel_values=data["pixel_values"],
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            position_ids=data["position_ids"],
            image_flags=self._get_image_flags(data["pixel_values"]),
            labels=original_labels,
            use_cache=False,
        )
        ga_base_loss = ga_outputs.loss
        ga_loss = -ga_base_loss  # 取负号实现梯度上升
        print(f"[DEBUG] Gradient Ascent Loss: {ga_loss.item():.6f} (base loss: {ga_base_loss.item():.6f})")
        print("计算ga loss完成")
        # 4. KL蒸馏任务：基于注意力的图像token扰动并计算KL损失
        # 4.1 提取图像特征
        vit_embeds = self.model.extract_feature(data["pixel_values"])
        image_flags = self._get_image_flags(data["pixel_values"])
        vit_embeds = vit_embeds[image_flags == 1]
        
        # 4.2 生成图像token位置信息
        image_token_positions = self._get_image_token_indices(data["input_ids"], data["attention_mask"])
        
        # 4.3 计算注意力分数
        # 首先需要构建完整的input_embeds来获取注意力
        input_embeds = self.model.language_model.get_input_embeddings()(data["input_ids"])
        # 将图像特征插入到对应位置
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = data["input_ids"].reshape(B * N)
        selected = input_ids == self.model.img_context_token_id
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)
        
        print(f"[DEBUG] Computing attention scores for {B} batches, {N} tokens per batch")
        print(f"[DEBUG] Image token positions: {len(image_token_positions)} samples")
        
        # 计算注意力分数
        attention_scores = self.compute_attention_scores(
            self.model, input_embeds, data["attention_mask"], image_token_positions
        )
        
        # 4.4 选择top-k%的图像token
        # 如果注意力计算失败，使用简化的随机选择策略
        try:
            high_attention_indices = self._compute_high_attention_image_tokens(
                attention_scores, image_token_positions, data["input_ids"], top_k_ratio=self.top_k_ratio
            )
        except Exception as e:
            print(f"[WARNING] 注意力计算失败: {e}")
            print("[INFO] 使用简化的随机选择策略")
            high_attention_indices = self._fallback_random_selection(image_token_positions, top_k_ratio=self.top_k_ratio)
        
        # 4.5 多次采样：对选中的图像token进行多次扰动
        print(f"[DEBUG] 开始多次采样，采样次数: {self.num_samples}")
        print(f"[DEBUG] 原始vit_embeds形状: {vit_embeds.shape}")
        print(f"[DEBUG] 图像token位置信息: {image_token_positions}")
        
        # 重新获取图像token的embedding
        # 通过语言模型的embedding层获取图像token的embedding
        input_embeds = self.model.language_model.get_input_embeddings()(data["input_ids"])
        batch_size, seq_len, embed_dim = input_embeds.shape
        
        # 提取图像token的embedding
        image_embeds_list = []
        for batch_idx, positions in image_token_positions.items():
            if positions:
                img_start = positions[0]['start']
                img_end = positions[0]['end']
                # 提取该样本的图像token embedding
                sample_image_embeds = input_embeds[batch_idx, img_start:img_end]  # [768, embed_dim]
                image_embeds_list.append(sample_image_embeds)
            else:
                # 如果没有图像token，创建零embedding
                sample_image_embeds = torch.zeros(
                    self.num_image_tokens, embed_dim, device=input_embeds.device
                )
                image_embeds_list.append(sample_image_embeds)
        
        # 堆叠成 [batch_size, 768, embed_dim]
        image_embeds_reshaped = torch.stack(image_embeds_list, dim=0)
        print(f"[DEBUG] 重新获取的图像embedding形状: {image_embeds_reshaped.shape}")
        
        # 批量多次采样：一次性生成所有采样的扰动embedding
        print(f"[DEBUG] 开始批量多次采样，采样次数: {self.num_samples}")
        
        # 批量生成所有采样的扰动embedding
        all_perturbed_embeds_batch = self._batch_perturb_high_attention_embeds(
            image_embeds_reshaped, high_attention_indices, image_token_positions, 
            num_samples=self.num_samples
        )
        
        print(f"[DEBUG] 批量扰动embedding形状: {all_perturbed_embeds_batch.shape}")
        print(f"[DEBUG] 预期形状: [batch_size * num_samples, num_image_tokens, embed_dim]")
        
        # 批量计算KL损失（仅当有教师模型时）
        if self.teacher_model is not None and self.kl_weight > 0:
            final_kl_loss = self._batch_compute_kl_loss(
                data, masked_labels, all_perturbed_embeds_batch, 
                num_samples=self.num_samples
            )
            print(f"[DEBUG] 批量计算KL Loss: {final_kl_loss.item():.6f}")
        else:
            final_kl_loss = torch.tensor(0.0, device=ga_loss.device, requires_grad=True)
            print(f"[DEBUG] KL Loss: 0.0 (teacher model disabled or kl_weight=0)")

        # 5. 组合双任务损失（加权求和）
        total_loss = self.ga_weight * ga_loss + self.kl_weight * final_kl_loss
        print(f"[DEBUG] Total Loss: {total_loss.item():.6f} (GA weight: {self.ga_weight}, KL weight: {self.kl_weight})")

        # 6. 返回损失字典（适配MMEngine训练流程）
        loss_dict = {
            "total_loss": total_loss,
            "ga_loss": ga_loss,
            "kl_loss": final_kl_loss,
            "ga_base_loss": ga_base_loss,
            "num_samples": self.num_samples,  # 添加采样次数信息
            "batch_computation": True,  # 标记使用了批量计算
        }
        return loss_dict

    def _llm_forward(
        self,
        model,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        perturbed_image_embeds: Optional[torch.FloatTensor] = None,  # 新增：扰动后的图像Embedding
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if hasattr(model, 'module') and isinstance(model.module, torch.nn.Module):
            original_model = model.module  # DeepSpeed包装：原始模型在model.module中
        else:
            original_model = model  # 非DeepSpeed包装：直接用model

        # 2. 从原始模型的config中获取use_return_dict
        return_dict = return_dict if return_dict is not None else original_model.config.use_return_dict
        # 确保所有输入张量都在正确的设备上
        device = next(original_model.parameters()).device
        image_flags = image_flags.squeeze(-1)
        logger = MMLogger.get_current_instance()
        # 1. 文本Embedding（原有逻辑不变）

        # 生成文本Embedding
        input_embeds = original_model.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = original_model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        # assert C == self.llm_hidden_size, f"Text embedding dim {C} != LLM hidden size {self.llm_hidden_size}"
        input_ids = input_ids.reshape(B * N)
        selected = input_ids == original_model.img_context_token_id
        # print("此时获取文本Embedding完成")
        # 获取图像Embedding
        if perturbed_image_embeds is not None:
            logger.debug("Using perturbed image embeddings")
            vit_embeds = perturbed_image_embeds
        else:
            logger.debug("Using original image embeddings")

        if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, "
                f"images per sample: {vit_batch_size / B}, "
                f"dynamic token length: {N}"
            )
        self._count += 1
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape="
                f"{input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        # # 语言模型前向传播
        # print(f"[DEBUG] Before language_model call:")
        # print(f"[DEBUG] input_embeds device: {input_embeds.device}")
        # print(f"[DEBUG] attention_mask device: {attention_mask.device if attention_mask is not None else 'None'}")
        # print(f"[DEBUG] position_ids device: {position_ids.device if position_ids is not None else 'None'}")
        # print(f"[DEBUG] position_ids shape: {position_ids.shape if position_ids is not None else 'None'}")
        # print(f"[DEBUG] position_ids dtype: {position_ids.dtype if position_ids is not None else 'None'}")
        
        # 强制确保语言模型在正确设备上（解决旋转位置编码缓存问题）
        try:
            # 确保模型在正确设备上
            original_model.language_model = original_model.language_model.to(device)
            # print(f"[DEBUG] Language model moved to device: {device}")
            
            # 清除旋转位置编码缓存，强制重新计算到正确设备
            if hasattr(original_model.language_model, 'model') and hasattr(original_model.language_model.model, 'layers'):
                for i, layer in enumerate(original_model.language_model.model.layers):
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                        # 清除可能存在的CPU缓存
                        if hasattr(layer.self_attn.rotary_emb, '_cos_cached'):
                            layer.self_attn.rotary_emb._cos_cached = None
                        if hasattr(layer.self_attn.rotary_emb, '_sin_cached'):
                            layer.self_attn.rotary_emb._sin_cached = None
                        # print(f"[DEBUG] Cleared rotary embedding cache for layer {i}")
                        
            # 额外检查：确保所有模型参数都在正确设备上
            for name, param in original_model.language_model.named_parameters():
                if param.device != device:
                    print(f"[DEBUG] Moving parameter {name} from {param.device} to {device}")
                    param.data = param.data.to(device)
                    
        except Exception as e:
            print(f"[DEBUG] Warning: Could not ensure model device consistency: {e}")
        
        outputs = original_model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        # 计算交叉熵损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _compute_kl_loss(self, data, masked_labels, perturbed_embeds):
        """计算KL蒸馏损失"""
        logger = MMLogger.get_current_instance()
        
        # 学生模型输出
        student_outputs = self._llm_forward(
            model=self.model,
            pixel_values=data["pixel_values"],
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            position_ids=data["position_ids"],
            image_flags=self._get_image_flags(data["pixel_values"]),
            labels=masked_labels,
            perturbed_image_embeds=perturbed_embeds,
            use_cache=False,
        )
        student_logits = student_outputs.logits

        # 教师模型输出（冻结梯度）
        with torch.no_grad():
            # 教师模型在CPU上，需要临时移动到GPU进行计算
            teacher_device = next(self.model.parameters()).device  # 使用学生模型的设备
            
            # 临时移动教师模型到GPU
            self.teacher_model = self.teacher_model.to(teacher_device)
            
            try:
                teacher_inputs = {
                    "model": self.teacher_model,
                    "pixel_values": data["pixel_values"].to(teacher_device),
                    "input_ids": data["input_ids"].to(teacher_device),
                    "attention_mask": data["attention_mask"].to(teacher_device),
                    "position_ids": data["position_ids"].to(teacher_device),
                    "image_flags": self._get_image_flags(data["pixel_values"]).to(teacher_device),
                    "labels": masked_labels.to(teacher_device),
                    "perturbed_image_embeds": perturbed_embeds.to(teacher_device),
                    "use_cache": False,
                }
                teacher_outputs = self._llm_forward(**teacher_inputs)
                teacher_logits = teacher_outputs.logits
            finally:
                # 计算完成后立即移回CPU，释放GPU内存
                self.teacher_model = self.teacher_model.cpu()
                torch.cuda.empty_cache()  # 清理GPU缓存

        # 计算KL散度
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl_loss = -(teacher_probs * torch.log(student_probs + 1e-12)).sum(dim=-1).mean()

        # 验证概率分布
        avg_student_prob = student_probs.mean().item()
        avg_teacher_prob = teacher_probs.mean().item()
        logger.debug(
            f"KL Loss Details: Avg Student Prob={avg_student_prob:.6f} | "
            f"Avg Teacher Prob={avg_teacher_prob:.6f}"
        )

        return kl_loss

    def _aggregate_kl_losses(self, kl_losses):
        """
        聚合多次采样的KL损失
        Args:
            kl_losses: list of tensors 多次采样的KL损失列表
        Returns:
            aggregated_loss: tensor 聚合后的KL损失
        """
        if not kl_losses:
            return torch.tensor(0.0, requires_grad=True)
        
        kl_losses_tensor = torch.stack(kl_losses)
        
        if self.kl_aggregation_strategy == 'mean':
            # 简单平均
            aggregated_loss = kl_losses_tensor.mean()
            print(f"[DEBUG] KL损失聚合策略: 平均 (mean) = {aggregated_loss.item():.6f}")
            
        elif self.kl_aggregation_strategy == 'weighted':
            # 加权平均（后期采样权重更高）
            weights = torch.softmax(torch.arange(len(kl_losses), dtype=torch.float, device=kl_losses_tensor.device), dim=0)
            aggregated_loss = (kl_losses_tensor * weights).sum()
            print(f"[DEBUG] KL损失聚合策略: 加权平均 (weighted) = {aggregated_loss.item():.6f}")
            
        elif self.kl_aggregation_strategy == 'max':
            # 取最大值
            aggregated_loss = kl_losses_tensor.max()
            print(f"[DEBUG] KL损失聚合策略: 最大值 (max) = {aggregated_loss.item():.6f}")
            
        elif self.kl_aggregation_strategy == 'min':
            # 取最小值
            aggregated_loss = kl_losses_tensor.min()
            print(f"[DEBUG] KL损失聚合策略: 最小值 (min) = {aggregated_loss.item():.6f}")
            
        else:
            # 默认使用平均
            aggregated_loss = kl_losses_tensor.mean()
            print(f"[DEBUG] KL损失聚合策略: 默认平均 = {aggregated_loss.item():.6f}")
        
        # 打印每次采样的详细损失
        print(f"[DEBUG] 各次采样KL损失详情:")
        for i, loss in enumerate(kl_losses):
            print(f"  - 第{i+1}次采样: {loss.item():.6f}")
        
        return aggregated_loss

    def _batch_compute_kl_loss(self, data, masked_labels, batch_perturbed_embeds, num_samples=3):
        """
        批量计算多次采样的KL损失（并行计算）
        Args:
            data: 输入数据
            masked_labels: 屏蔽后的标签
            batch_perturbed_embeds: (B * num_samples, num_image_tokens, embed_dim) 批量扰动embedding
            num_samples: 采样次数
        Returns:
            aggregated_kl_loss: 聚合后的KL损失
        """
        logger = MMLogger.get_current_instance()
        batch_size = masked_labels.shape[0]
        
        print(f"\n[DEBUG] 开始批量KL损失计算:")
        print(f"  - 批量扰动embedding形状: {batch_perturbed_embeds.shape}")
        print(f"  - 原始batch大小: {batch_size}")
        print(f"  - 采样次数: {num_samples}")
        
        # 扩展输入数据以匹配批量扰动embedding
        # 需要将原始数据复制num_samples次
        expanded_data = self._expand_data_for_batch_computation(data, num_samples)
        expanded_labels = masked_labels.repeat(num_samples, 1)  # [B * num_samples, seq_len]
        
        print(f"  - 扩展后标签形状: {expanded_labels.shape}")
        
        # 批量计算学生模型输出
        student_outputs = self._llm_forward(
            model=self.model,
            pixel_values=expanded_data["pixel_values"],
            input_ids=expanded_data["input_ids"],
            attention_mask=expanded_data["attention_mask"],
            position_ids=expanded_data["position_ids"],
            image_flags=self._get_image_flags(expanded_data["pixel_values"]),
            labels=expanded_labels,
            perturbed_image_embeds=batch_perturbed_embeds.view(-1, batch_perturbed_embeds.shape[-1]),  # 展平
            use_cache=False,
        )
        student_logits = student_outputs.logits  # [B * num_samples, seq_len, vocab_size]
        
        # 批量计算教师模型输出
        with torch.no_grad():
            teacher_device = next(self.model.parameters()).device
            self.teacher_model = self.teacher_model.to(teacher_device)
            
            try:
                teacher_inputs = {
                    "model": self.teacher_model,
                    "pixel_values": expanded_data["pixel_values"].to(teacher_device),
                    "input_ids": expanded_data["input_ids"].to(teacher_device),
                    "attention_mask": expanded_data["attention_mask"].to(teacher_device),
                    "position_ids": expanded_data["position_ids"].to(teacher_device),
                    "image_flags": self._get_image_flags(expanded_data["pixel_values"]).to(teacher_device),
                    "labels": expanded_labels.to(teacher_device),
                    "perturbed_image_embeds": batch_perturbed_embeds.view(-1, batch_perturbed_embeds.shape[-1]).to(teacher_device),
                    "use_cache": False,
                }
                teacher_outputs = self._llm_forward(**teacher_inputs)
                teacher_logits = teacher_outputs.logits  # [B * num_samples, seq_len, vocab_size]
            finally:
                self.teacher_model = self.teacher_model.cpu()
                torch.cuda.empty_cache()
        
        # 计算批量KL散度
        student_probs = F.softmax(student_logits, dim=-1)  # [B * num_samples, seq_len, vocab_size]
        teacher_probs = F.softmax(teacher_logits, dim=-1)  # [B * num_samples, seq_len, vocab_size]
        
        # 计算每个样本的KL损失
        kl_losses_per_sample = -(teacher_probs * torch.log(student_probs + 1e-12)).sum(dim=-1).mean(dim=1)  # [B * num_samples]
        
        print(f"  - 每个样本KL损失形状: {kl_losses_per_sample.shape}")
        
        # 按采样分组并聚合
        kl_losses_grouped = kl_losses_per_sample.view(num_samples, batch_size)  # [num_samples, B]
        kl_losses_per_sample_avg = kl_losses_grouped.mean(dim=1)  # [num_samples] 每个采样的平均KL损失
        
        print(f"[DEBUG] 各次采样KL损失详情:")
        for i, loss in enumerate(kl_losses_per_sample_avg):
            print(f"  - 第{i+1}次采样平均KL损失: {loss.item():.6f}")
        
        # 聚合多次采样的KL损失
        if self.kl_aggregation_strategy == 'mean':
            aggregated_loss = kl_losses_per_sample_avg.mean()
            print(f"[DEBUG] KL损失聚合策略: 平均 (mean) = {aggregated_loss.item():.6f}")
        elif self.kl_aggregation_strategy == 'weighted':
            weights = torch.softmax(torch.arange(num_samples, dtype=torch.float, device=kl_losses_per_sample_avg.device), dim=0)
            aggregated_loss = (kl_losses_per_sample_avg * weights).sum()
            print(f"[DEBUG] KL损失聚合策略: 加权平均 (weighted) = {aggregated_loss.item():.6f}")
        elif self.kl_aggregation_strategy == 'max':
            aggregated_loss = kl_losses_per_sample_avg.max()
            print(f"[DEBUG] KL损失聚合策略: 最大值 (max) = {aggregated_loss.item():.6f}")
        elif self.kl_aggregation_strategy == 'min':
            aggregated_loss = kl_losses_per_sample_avg.min()
            print(f"[DEBUG] KL损失聚合策略: 最小值 (min) = {aggregated_loss.item():.6f}")
        else:
            aggregated_loss = kl_losses_per_sample_avg.mean()
            print(f"[DEBUG] KL损失聚合策略: 默认平均 = {aggregated_loss.item():.6f}")
        
        return aggregated_loss

    def _expand_data_for_batch_computation(self, data, num_samples):
        """
        扩展输入数据以支持批量计算
        Args:
            data: 原始输入数据
            num_samples: 采样次数
        Returns:
            expanded_data: 扩展后的数据
        """
        expanded_data = {}
        
        # 扩展pixel_values
        if "pixel_values" in data:
            pixel_values = data["pixel_values"]
            if isinstance(pixel_values, torch.Tensor):
                # 如果是tensor，直接repeat
                expanded_data["pixel_values"] = pixel_values.repeat(num_samples, 1, 1, 1)
            else:
                # 如果是list，扩展list
                expanded_data["pixel_values"] = pixel_values * num_samples
        
        # 扩展其他tensor数据
        for key in ["input_ids", "attention_mask", "position_ids"]:
            if key in data:
                expanded_data[key] = data[key].repeat(num_samples, 1)
        
        return expanded_data

    def _get_image_flags(self, pixel_values):
        """
        辅助函数：生成图像标记（image_flags），标记有效图像（非全零）
        Args:
            pixel_values: (B*N, C, H, W) 拼接后的图像输入
        Returns:
            image_flags: (B*N,) 1表示有效图像，0表示无效图像
        """
        if type(pixel_values) is list:
            pixel_values = torch.cat([x.to(self.model.vision_model.dtype) for x in pixel_values], dim=0)
        # 全零像素视为无效图像
        image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
        return image_flags.long().to(pixel_values.device)

    def _analyze_tokens_detailed(self, input_ids, attention_mask):
        """
        详细分析tokenizer处理后的token，帮助定位图像和文本token位置
        Args:
            input_ids: (B, N) 输入token ids
            attention_mask: (B, N) 注意力掩码
        """
        batch_size, seq_len = input_ids.shape
        
        print(f"\n{'='*80}")
        print(f"[TOKEN ANALYSIS] 详细Token分析开始")
        print(f"{'='*80}")
        print(f"  - 输入序列形状: {input_ids.shape}")
        print(f"  - 注意力掩码形状: {attention_mask.shape}")
        
        # 获取特殊token的id
        special_tokens = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
        }
        
        # 尝试获取图像相关token
        try:
            img_context_token_id = self.model.img_context_token_id
            special_tokens['img_context_token_id'] = img_context_token_id
        except:
            img_context_token_id = None
            special_tokens['img_context_token_id'] = None
        
        print(f"\n[SPECIAL TOKENS] 特殊Token ID:")
        for name, token_id in special_tokens.items():
            print(f"  - {name}: {token_id}")
        
        for batch_idx in range(batch_size):
            print(f"\n{'-'*60}")
            print(f"[BATCH {batch_idx}] 样本{batch_idx}详细分析")
            print(f"{'-'*60}")
            
            seq = input_ids[batch_idx].tolist()
            attn_mask = attention_mask[batch_idx].tolist()
            
            print(f"  - 序列长度: {len(seq)}")
            print(f"  - 有效长度: {sum(attn_mask)}")
            
            # 分析特殊token位置
            print(f"\n[SPECIAL TOKEN POSITIONS] 特殊Token位置:")
            for name, token_id in special_tokens.items():
                if token_id is not None:
                    positions = [i for i, tid in enumerate(seq) if tid == token_id]
                    if positions:
                        print(f"  - {name} ({token_id}): 位置 {positions}")
                    else:
                        print(f"  - {name} ({token_id}): 未找到")
            
            # 分段分析token
            print(f"\n[TOKEN SEGMENTS] Token分段分析:")
            
            # 找到图像上下文token位置
            if img_context_token_id is not None:
                img_context_positions = [i for i, tid in enumerate(seq) if tid == img_context_token_id]
                print(f"  - 图像上下文token位置: {img_context_positions}")
                
                if img_context_positions:
                    # 分析每个图像区域
                    for img_idx, img_pos in enumerate(img_context_positions):
                        print(f"\n  [IMAGE {img_idx}] 图像{img_idx}分析:")
                        print(f"    - 图像上下文token位置: {img_pos}")
                        
                        # 分析图像上下文token前后的内容
                        start_analyze = max(0, img_pos - 5)
                        end_analyze = min(seq_len, img_pos + 10)
                        
                        print(f"    - 上下文区域[{start_analyze}:{end_analyze}]:")
                        for i in range(start_analyze, end_analyze):
                            token_id = seq[i]
                            try:
                                decoded = self.tokenizer.decode([token_id])
                                # 清理显示
                                decoded = decoded.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                if len(decoded) > 20:
                                    decoded = decoded[:20] + '...'
                                print(f"      [{i:3d}] {token_id:5d} -> '{decoded}'")
                            except:
                                print(f"      [{i:3d}] {token_id:5d} -> <decode_error>")
                        
                        # 分析图像token区域
                        img_start = img_pos + 1
                        img_end = min(img_start + self.num_image_tokens, seq_len)
                        actual_img_tokens = img_end - img_start
                        
                        print(f"    - 图像token区域[{img_start}:{img_end}] (共{actual_img_tokens}个):")
                        
                        # 显示图像token的前几个和最后几个
                        if actual_img_tokens > 0:
                            # 前5个图像token
                            print(f"      - 前5个图像token:")
                            for i in range(img_start, min(img_start + 5, img_end)):
                                token_id = seq[i]
                                print(f"        [{i:3d}] {token_id:5d}")
                            
                            if actual_img_tokens > 10:
                                print(f"      - ... (中间{actual_img_tokens-10}个token)")
                                # 最后5个图像token
                                print(f"      - 最后5个图像token:")
                                for i in range(max(img_start, img_end - 5), img_end):
                                    token_id = seq[i]
                                    print(f"        [{i:3d}] {token_id:5d}")
                            elif actual_img_tokens > 5:
                                # 显示剩余的token
                                print(f"      - 剩余图像token:")
                                for i in range(img_start + 5, img_end):
                                    token_id = seq[i]
                                    print(f"        [{i:3d}] {token_id:5d}")
                        
                        # 分析图像后的文本
                        if img_end < seq_len:
                            text_start = img_end
                            text_end = min(text_start + 10, seq_len)
                            print(f"    - 图像后文本区域[{text_start}:{text_end}]:")
                            for i in range(text_start, text_end):
                                token_id = seq[i]
                                try:
                                    decoded = self.tokenizer.decode([token_id])
                                    decoded = decoded.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                    if len(decoded) > 30:
                                        decoded = decoded[:30] + '...'
                                    print(f"      [{i:3d}] {token_id:5d} -> '{decoded}'")
                                except:
                                    print(f"      [{i:3d}] {token_id:5d} -> <decode_error>")
                else:
                    print(f"  - 未找到图像上下文token")
            else:
                print(f"  - 无法获取图像上下文token ID")
            
            # 分析文本区域
            print(f"\n[TEXT ANALYSIS] 文本区域分析:")
            
            # 找到所有非图像token的区域
            text_regions = []
            current_start = 0
            
            if img_context_token_id is not None:
                img_positions = [i for i, tid in enumerate(seq) if tid == img_context_token_id]
                img_ranges = [(pos, pos + 1 + self.num_image_tokens) for pos in img_positions]
                
                for img_start, img_end in img_ranges:
                    if current_start < img_start:
                        text_regions.append((current_start, img_start))
                    current_start = img_end
                
                if current_start < seq_len:
                    text_regions.append((current_start, seq_len))
            else:
                text_regions = [(0, seq_len)]
            
            for region_idx, (start, end) in enumerate(text_regions):
                if end - start > 0:
                    print(f"  - 文本区域{region_idx}[{start}:{end}] (共{end-start}个token):")
                    
                    # 显示前10个和后10个文本token
                    display_tokens = min(10, end - start)
                    for i in range(start, start + display_tokens):
                        token_id = seq[i]
                        try:
                            decoded = self.tokenizer.decode([token_id])
                            decoded = decoded.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                            if len(decoded) > 40:
                                decoded = decoded[:40] + '...'
                            print(f"    [{i:3d}] {token_id:5d} -> '{decoded}'")
                        except:
                            print(f"    [{i:3d}] {token_id:5d} -> <decode_error>")
                    
                    if end - start > 20:
                        print(f"    ... (中间{end-start-20}个token)")
                        for i in range(max(start, end - 10), end):
                            token_id = seq[i]
                            try:
                                decoded = self.tokenizer.decode([token_id])
                                decoded = decoded.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                if len(decoded) > 40:
                                    decoded = decoded[:40] + '...'
                                print(f"    [{i:3d}] {token_id:5d} -> '{decoded}'")
                            except:
                                print(f"    [{i:3d}] {token_id:5d} -> <decode_error>")
        
        print(f"\n{'='*80}")
        print(f"[TOKEN ANALYSIS] Token分析完成")
        print(f"{'='*80}")

    def _get_image_token_indices(self, input_ids, attention_mask):
        """
        智能检测图像Token范围，兼容InternVL格式（参考LLaVA实现）
        Args:
            input_ids: (B, N) 输入token ids
            attention_mask: (B, N) 注意力掩码
        Returns:
            image_token_positions: dict 图像token位置信息
        """
        # 根据配置决定是否进行详细的token分析
        if self.enable_token_analysis:
            self._analyze_tokens_detailed(input_ids, attention_mask)
        
        image_token_positions = {}
        batch_size, seq_len = input_ids.shape
        
        # 获取图像上下文token的id
        img_context_token_id = self.model.img_context_token_id
        
        print(f"\n[DEBUG] 开始图像Token定位:")
        print(f"  - 输入序列形状: {input_ids.shape}")
        print(f"  - 图像上下文token id: {img_context_token_id}")
        
        for batch_idx in range(batch_size):
            seq = input_ids[batch_idx].tolist()
            print(f"\n[DEBUG] 样本{batch_idx}的input_ids:")
            print(f"  - 序列长度: {len(seq)}")
            
            # 找到所有图像上下文token的位置
            image_positions = (input_ids[batch_idx] == img_context_token_id).nonzero(as_tuple=True)[0]
            
            if len(image_positions) == 0:
                print(f"[WARNING] 样本{batch_idx}: 未找到图像上下文token")
                continue
            
            print(f"  - 找到{len(image_positions)}个图像上下文token位置: {image_positions.tolist()}")
            
            # 检查图像token是否连续
            if len(image_positions) > 0:
                # 找到第一个图像token位置
                first_pos = image_positions[0].item()
                
                # 检查从第一个位置开始的连续图像token数量
                consecutive_count = 0
                for i in range(first_pos, seq_len):
                    if input_ids[batch_idx, i] == img_context_token_id:
                        consecutive_count += 1
                    else:
                        break
                
                print(f"  - 从位置{first_pos}开始的连续图像token数量: {consecutive_count}")
                
                if consecutive_count > 100:  # 合理的图像token数量
                    start_pos = first_pos
                    end_pos = first_pos + consecutive_count
                    actual_token_num = consecutive_count
                    
                    print(f"[DEBUG] 样本{batch_idx}: 图像Token范围[{start_pos}:{end_pos}] (共{actual_token_num}个token)")
                else:
                    print(f"[WARNING] 样本{batch_idx}: 连续图像token数量过少({consecutive_count})，跳过")
                    continue
            else:
                print(f"[WARNING] 样本{batch_idx}: 未找到图像token")
                continue
            
            if batch_idx not in image_token_positions:
                image_token_positions[batch_idx] = []
            
            image_token_positions[batch_idx].append({
                'image_idx': 0,  # 只有一个图像
                'start': start_pos,
                'end': end_pos,
                'num_tokens': actual_token_num
            })
        
        print(f"[DEBUG] 总共找到{len(image_token_positions)}个样本的图像Token位置")
        
        # 验证文本token识别
        self._verify_text_token_detection(input_ids, image_token_positions)
        
        return image_token_positions

    def _verify_text_token_detection(self, input_ids, image_token_positions):
        """
        验证文本token识别是否正确
        Args:
            input_ids: (B, N) 输入token ids
            image_token_positions: dict 图像token位置信息
        """
        print(f"\n[DEBUG] 验证文本Token识别:")
        
        for batch_idx, positions in image_token_positions.items():
            if not positions:
                continue
                
            img_start = positions[0]['start']
            img_end = positions[0]['end']
            seq_len = input_ids.shape[1]
            
            print(f"\n[DEBUG] 样本{batch_idx}文本Token验证:")
            print(f"  - 图像Token范围: [{img_start}:{img_end}] (共{img_end-img_start}个)")
            print(f"  - 序列总长度: {seq_len}")
            
            # 检查图像前的文本
            if img_start > 0:
                pre_text = input_ids[batch_idx, :img_start]
                try:
                    pre_text_decoded = self.tokenizer.decode(pre_text)
                    print(f"  - 图像前文本长度: {img_start}个token")
                    print(f"  - 图像前文本内容: {pre_text_decoded[:100]}...")
                except:
                    print(f"  - 图像前文本解码失败")
            
            # 检查图像后的文本
            if img_end < seq_len:
                post_text = input_ids[batch_idx, img_end:]
                try:
                    post_text_decoded = self.tokenizer.decode(post_text)
                    print(f"  - 图像后文本长度: {seq_len-img_end}个token")
                    print(f"  - 图像后文本内容: {post_text_decoded[:200]}...")
                    
                    # 检查是否包含预期的文本内容
                    if "Can you identify" in post_text_decoded:
                        print(f"  - ✓ 找到用户问题文本")
                    if "assistant" in post_text_decoded:
                        print(f"  - ✓ 找到助手回复文本")
                    te = getattr(self, "target_entity", "") or ""
                    if te and te in post_text_decoded:
                        print(f"  - ✓ 找到目标实体 '{te}'")
                        
                except:
                    print(f"  - 图像后文本解码失败")
            else:
                print(f"  - 图像在序列末尾，无图像后文本")
