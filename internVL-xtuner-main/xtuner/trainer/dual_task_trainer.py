# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from pathlib import Path
from typing import List, Optional
import json
from mmengine.logging import print_log


def _default_deepspeed_zero3_config() -> str:
    """仓库内置的 ZeRO-3 配置路径（随源码分发，避免写死本机绝对路径）。"""
    return str(
        Path(__file__).resolve().parent.parent
        / "configs"
        / "deepspeed"
        / "deepspeed_zero3.json"
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class DualTaskTrainer(Trainer):
    """双任务训练器：梯度上升 + KL蒸馏"""
    
    def __init__(self, dual_task_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 双任务训练参数
        self.dual_task_args = dual_task_args or {}
        self.teacher_model_path = self.dual_task_args.get('teacher_model_path')
        self.ga_weight = self.dual_task_args.get('ga_weight', 1.0)
        self.kl_weight = self.dual_task_args.get('kl_weight', 1.2)
        self.target_entity = self.dual_task_args.get('target_entity', 'Joe Biden')
        self.deepspeed_zero_config = self.dual_task_args.get('deepspeed_zero_config')
        if self.deepspeed_zero_config is None:
            self.deepspeed_zero_config = _default_deepspeed_zero3_config()
        
        # 初始化教师模型
        if self.teacher_model_path:
            self._initialize_teacher_model()
    
    def _initialize_teacher_model(self):
        """初始化教师模型（参考LLaVA成功模式）"""
        print_log(f"Loading teacher model from {self.teacher_model_path}", logger="current")
        
        # 加载教师模型
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        teacher_config = AutoConfig.from_pretrained(self.teacher_model_path, trust_remote_code=True)
        if teacher_config.llm_config.model_type == "internlm2":
            teacher_config.llm_config.attn_implementation = "flash_attention_2"
        else:
            teacher_config.llm_config._attn_implementation = "flash_attention_2"
        
        teacher_model = AutoModel.from_pretrained(
            self.teacher_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=None,
            config=teacher_config,
            trust_remote_code=True,
        )
        
        # 教师模型Tokenizer
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_path, trust_remote_code=True)
        teacher_model.img_context_token_id = self.teacher_tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        
        # 教师模型设为评估模式并冻结权重
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # 使用DeepSpeed初始化教师模型
        print_log("Initializing teacher model with DeepSpeed", logger="current")
        self.teacher_model, _, _, _ = deepspeed.initialize(
            model=teacher_model,
            model_parameters=teacher_model.parameters(),
            config_params=json.load(open(self.deepspeed_zero_config, "r"))
        )
        self.teacher_model.eval()
        
        print_log("Teacher model initialized with DeepSpeed successfully", logger="current")
    
    def mask_answer_tokens(self, labels):
        """屏蔽答案中目标实体及之后的所有token"""
        masked_labels = labels.clone()
        batch_size, seq_len = masked_labels.shape
        
        # 目标实体按长度降序排列
        target_entities = [
            f" {self.target_entity}", self.target_entity,
            self.target_entity.lower(), f" {self.target_entity.lower()}"
        ]
        
        for i in range(batch_size):
            label_ids = masked_labels[i].tolist()
            entity_found = False
            
            for entity in target_entities:
                if entity_found:
                    break
                try:
                    entity_token_ids = self.tokenizer.encode(entity, add_special_tokens=False)
                    entity_len = len(entity_token_ids)
                    if entity_len == 0:
                        continue
                    
                    max_search_pos = len(label_ids) - entity_len
                    if max_search_pos < 0:
                        continue
                    
                    for k in range(max_search_pos + 1):
                        if label_ids[k] == -100:  # IGNORE_INDEX
                            continue
                        if label_ids[k:k+entity_len] == entity_token_ids:
                            masked_labels[i, k:] = -100
                            entity_found = True
                            print(f"[INFO] Sample {i}: Masked entity '{entity}' at position {k}")
                            break
                except Exception as e:
                    print(f"Failed to process entity '{entity}': {str(e)}")
                    continue
        
        return masked_labels
    
    def perturb_image_embeds(self, vit_embeds):
        """扰动图像Embedding"""
        original_vit_embeds = vit_embeds.clone()
        target_similarity = 0.7
        
        # 分离"方向"和"幅值"
        original_magnitude = torch.norm(original_vit_embeds, p=2, dim=-1, keepdim=True)
        original_dir = original_vit_embeds / (original_magnitude + 1e-12)
        
        # 旋转扰动
        noise = torch.randn_like(original_dir)
        noise = noise - torch.sum(noise * original_dir, dim=-1, keepdim=True) * original_dir
        noise_dir = F.normalize(noise, p=2, dim=-1)
        theta = torch.acos(torch.tensor(target_similarity, device=original_vit_embeds.device))
        rotated_dir = original_dir * torch.cos(theta) + noise_dir * torch.sin(theta)
        
        # 恢复原始幅值
        perturbed_embeds = rotated_dir * original_magnitude
        noise_std = 0.05
        gaussian_noise = torch.randn_like(perturbed_embeds) * noise_std
        perturbed_embeds = perturbed_embeds + gaussian_noise
        
        # 验证相似度
        actual_similarity = F.cosine_similarity(original_vit_embeds, perturbed_embeds, dim=-1).mean()
        print(f"Image embedding similarity: Actual={actual_similarity:.4f}, Target={target_similarity}")
        return perturbed_embeds
    
    def _compute_kl_loss(self, model, inputs):
        """计算KL蒸馏损失"""
        # 获取学生模型输出
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # 获取教师模型输出
        with torch.no_grad():
            teacher_inputs = {
                'input_ids': inputs['input_ids'].to(self.teacher_model.device),
                'attention_mask': inputs['attention_mask'].to(self.teacher_model.device),
                'pixel_values': inputs['pixel_values'].to(self.teacher_model.device),
                'position_ids': inputs.get('position_ids'),
                'image_flags': inputs.get('image_flags'),
            }
            if teacher_inputs['position_ids'] is not None:
                teacher_inputs['position_ids'] = teacher_inputs['position_ids'].to(self.teacher_model.device)
            if teacher_inputs['image_flags'] is not None:
                teacher_inputs['image_flags'] = teacher_inputs['image_flags'].to(self.teacher_model.device)
            
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
        
        # 计算KL散度
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl_loss = -(teacher_probs * torch.log(student_probs + 1e-12)).sum(dim=-1).mean()
        
        return kl_loss
    
    def training_step(self, model, inputs):
        """双任务训练步骤"""
        model.train()
        
        # 准备输入数据
        inputs = self._prepare_inputs(inputs)
        
        # 计算基础损失
        base_loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        # 任务1：梯度上升
        ga_loss = -base_loss
        
        # 任务2：KL蒸馏
        masked_labels = self.mask_answer_tokens(inputs['labels'])
        inputs['labels'] = masked_labels
        
        # 扰动图像Embedding
        if 'pixel_values' in inputs:
            vit_embeds = model.extract_feature(inputs['pixel_values'])
            perturbed_embeds = self.perturb_image_embeds(vit_embeds)
            # 这里需要将扰动后的embedding传递给模型
            # 具体实现取决于InternVL的forward方法
        
        kl_loss = self._compute_kl_loss(model, inputs)
        
        # 组合损失
        total_loss = self.ga_weight * ga_loss + self.kl_weight * kl_loss
        
        # 反向传播
        self.accelerator.backward(total_loss)
        
        return total_loss





