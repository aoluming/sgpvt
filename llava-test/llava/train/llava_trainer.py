import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from llava.constants import IGNORE_INDEX
from torch.utils.data import Sampler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from transformers import AutoProcessor
from llava.model import *
from torch.cuda.amp import autocast
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
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


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in
                    get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: Optional[List[int]] = None,
            generator=None,
            group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                          generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                 generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(self, second_model_path, forgetting_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.second_model = second_model_path
        self.forgetting_args = forgetting_args

        deepspeed_config = getattr(forgetting_args, 'deepspeed_config', None) or getattr(kwargs.get('args', None), 'deepspeed', None)
        self.second_model, _, _, _ = deepspeed.initialize(model=self.second_model,
                                                          model_parameters=self.second_model.parameters(),
                                                          config=deepspeed_config)
        self.second_model.eval()

        # 添加缓存机制，避免重复计算
        self._retention_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # 参数监控
        self._monitor_step_count = 0
        self._monitor_interval = 5

        # 添加时间统计功能
        import time
        self._step_times = []
        self._total_training_time = 0.0
        self._training_start_time = None
        self._current_step_start_time = None
    # def compute_kl_loss(self, model, batch, device):
    #     normal_outputs = model(
    #         batch["input_ids"].to(device),
    #         attention_mask=batch["attention_mask"].int().to(device),
    #         labels=batch["labels"].to(device),
    #     )

    #     # self.second_model.to(device)
    #     with torch.no_grad():
    #         pretrained_outputs = self.second_model(
    #             batch["input_ids"].to(device),
    #             attention_mask=batch["attention_mask"].int().to(device),
    #             labels=batch["labels"].to(device),
    #         )

    #     prob_p = F.softmax(pretrained_outputs.logits, dim=-1)
    #     prob_q = F.softmax(normal_outputs.logits, dim=-1)

    #     kl_loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    #     return kl_loss, normal_outputs.loss



    def mask_answer_tokens(self, labels):
        """屏蔽答案部分中目标实体及之后的所有token（修复版）"""
        masked_labels = labels.clone()
        if self.tokenizer is None:
            raise ValueError("请在初始化时传入tokenizer，否则无法执行实体匹配")

        target_entities: List[str] = []
        path = getattr(self.forgetting_args, "forget_entities_file", None) if self.forgetting_args else None
        if path:
            if not os.path.isfile(path):
                logger.warning("forget_entities_file 不存在，跳过实体屏蔽: %s", path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, list):
                    raise ValueError("forget_entities_file 必须是 JSON 数组")
                target_entities = [str(x) for x in loaded]
        if not target_entities:
            return masked_labels

        batch_size, seq_len = masked_labels.shape
        for i in range(batch_size):
            label_ids = masked_labels[i].tolist()
            entity_found = False
            
            for entity in target_entities:
                if entity_found:
                    break
                try:
                    # 生成带空格实体的token ids（与实际解码结果一致）
                    entity_token_ids = self.tokenizer.encode(entity, add_special_tokens=False)
                    entity_len = len(entity_token_ids)
                    if entity_len == 0:
                        continue
                    
                    max_search_pos = len(label_ids) - entity_len
                    if max_search_pos < 0:
                        continue
                    
                    # 搜索并屏蔽实体
                    for k in range(max_search_pos + 1):
                        if label_ids[k] == IGNORE_INDEX:
                            continue
                        if label_ids[k:k+entity_len] == entity_token_ids:
                            masked_labels[i, k:] = IGNORE_INDEX
                            entity_found = True
                            print(f"[INFO] 样本{i}屏蔽敏感实体：{entity}（位置k={k}）")  # 新增：验证屏蔽效果
                            break
                except Exception as e:
                    print(f"处理实体「{entity}」出错：{str(e)}")
                    continue
          
            valid_labels = (masked_labels != IGNORE_INDEX).sum().item()
            print(f"[DEBUG] 样本{i}屏蔽后有效标签数: {valid_labels}")
        
        assert masked_labels.shape == labels.shape, "mask后长度异常"
        
        return masked_labels
    def _safe_convert_to_numpy(self, tensor):
        """
        安全地将tensor转换为numpy数组，处理BFloat16等特殊数据类型
        """
        if tensor.dtype == torch.bfloat16:
            return tensor.cpu().float().numpy()
        else:
            return tensor.cpu().numpy()

    def _compute_high_attention_image_tokens_new(self, attentions, image_token_positions, input_ids, top_k_ratio):
        """
        基于新的位置信息计算高关注的图像Token - 修复版
        """
        num_layers, batch_size, num_heads, seq_len, _ = attentions.shape
        high_attention_indices = {}
        
        print(f"\n[DEBUG] 新Attention分析开始:")
        print(f"  - 层数: {num_layers}, 批次大小: {batch_size}, 头数: {num_heads}, 序列长度: {seq_len}")
        print(f"  - 图像Token位置信息: {len(image_token_positions)}个样本")
        
        for batch_idx, positions in image_token_positions.items():
            print(f"\n[DEBUG] 分析样本{batch_idx}:")
            print(f"  - 图像Token位置数量: {len(positions)}")
            
            # 计算文本Token范围（基于实际的图像token位置）
            if positions:
                # 图像token在位置[35:611]，所以文本token在[0:35]和[611:seq_len]
                img_start = positions[0]['start']
                img_end = positions[0]['end']
                text_ranges = []
                
                # 图像之前的文本
                if img_start > 0:
                    text_ranges.append((0, img_start))
                    print(f"  - 图像前文本Token范围: [0:{img_start}] (共{img_start}个)")
                
                # 图像之后的文本
                if img_end < seq_len:
                    text_ranges.append((img_end, seq_len))
                    print(f"  - 图像后文本Token范围: [{img_end}:{seq_len}] (共{seq_len-img_end}个)")
                
                # 如果没有找到文本token，使用默认范围
                if not text_ranges:
                    text_ranges = [(0, img_start)]
                    print(f"  - 默认文本Token范围: [0:{img_start}] (共{img_start}个)")
            else:
                # 没有图像token，整个序列都是文本
                text_ranges = [(0, seq_len)]
                print(f"  - 全文本Token范围: [0:{seq_len}] (共{seq_len}个)")
            
            print(f"  - 文本Token范围数量: {len(text_ranges)}")
            
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
                    
                    # print(f"  - 层{layer_idx}注意力矩阵形状: {layer_attn.shape}")
                    print(f"  - 文本范围数量: {len(text_ranges)}")
                    print(f"  - 图像范围: [{img_start}:{img_end}]")
                    
                    # 检查索引是否超出范围
                    max_text_end = max([text_end for text_start, text_end in text_ranges]) if text_ranges else 0
                    if max_text_end > layer_attn.shape[1] or img_end > layer_attn.shape[2]:
                        print(f"  - 警告：索引超出范围！注意力矩阵形状: {layer_attn.shape}")
                        continue
                    
                    # 关键修复：正确提取文本Token对图像Token的注意力权重
                    # 注意力矩阵的形状是 (num_heads, seq_len, seq_len)
                    # 其中 attn_weights[head, query_pos, key_pos] 表示第head个头中query_pos对key_pos的注意力
                    
                    # 只使用第二段文本范围（图像后的文本）进行注意力分析
                    second_text_range = None
                    for text_start, text_end in text_ranges:
                        if text_end > text_start and text_start >= img_end:  # 图像后的文本
                            second_text_range = (text_start, text_end)
                            break
                    
                    if second_text_range:
                        text_start, text_end = second_text_range
                        # 第二段文本token对图像token的注意力
                        text2img_attn = layer_attn[:, text_start:text_end, img_start:img_end]
                        total_text_tokens = text_end - text_start
                        # print(f"    * 使用第二段文本范围[{text_start}:{text_end}]对图像注意力形状: {text2img_attn.shape}")
                        # print(f"  - text2img_attn统计: 最大值={text2img_attn.max().item():.6f}, 最小值={text2img_attn.min().item():.6f}, 平均值={text2img_attn.mean().item():.6f}")
                    else:
                        # 如果没有第二段文本，创建零张量
                        text2img_attn = torch.zeros(layer_attn.shape[0], 0, img_end - img_start, device=layer_attn.device)
                        total_text_tokens = 0
                        print(f"    * 未找到第二段文本范围，创建零张量: {text2img_attn.shape}")
                    
                    # # 简化调试信息：只检查第二段文本对图像的注意力
                    # print(f"  - 第二段文本注意力分析:")
                    # print(f"    * 图像范围: [{img_start}:{img_end}] (共{img_end - img_start}个token)")
                    # print(f"    * 第二段文本token数量: {total_text_tokens}")
                    
                    if second_text_range and total_text_tokens > 0:
                        text_start, text_end = second_text_range
                        # print(f"    * 第二段文本范围: [{text_start}:{text_end}]")
                        # 检查前3个第二段文本token对图像token的注意力
                        for i in range(min(3, text_end - text_start)):
                            token_idx = text_start + i
                            attn_values = layer_attn[0, token_idx, img_start:img_start+5].tolist()
                            # print(f"    * 第二段文本token {token_idx} 对图像token的注意力: {attn_values}")
                    else:
                        print(f"    * 无第二段文本，跳过注意力检查")
                    
                    # 计算第二段文本对图像token的注意力分数
                    if text2img_attn.shape[1] > 0:  # 确保有第二段文本token
                        # 对多头、多文本Token取平均，得到每个图像token被第二段文本关注的程度
                        text_attn_per_token = text2img_attn.mean(dim=[0, 1])  # [576] - 每个图像token被第二段文本关注的平均程度
                        # print(f"  - 第二段文本对图像token注意力形状: {text_attn_per_token.shape}")
                        # print(f"  - 第二段文本注意力前5个值: {text_attn_per_token[:5].tolist()}")
                        
                        # 直接使用第二段文本的注意力分数
                        avg_attn_per_token = text_attn_per_token
                        # print(f"  - 使用第二段文本注意力进行token筛选")
                    else:
                        # 如果没有第二段文本，创建零张量
                        avg_attn_per_token = torch.zeros(img_end - img_start, device=layer_attn.device)
                        print(f"  - 无第二段文本，创建零张量: {avg_attn_per_token.shape}")
                    
                    # 加权累加
                    weight = layer_weights[layer_idx]
                    token_attention_scores += weight * avg_attn_per_token
                
                # 使用L1归一化保持相对比例
                token_attention_scores = F.normalize(token_attention_scores, p=1, dim=0)
                
                # 打印第二段文本注意力分数统计
                # print(f"  - 第二段文本注意力分数统计:")
                # print(f"    * 最大值: {token_attention_scores.max().item():.6f}")
                # print(f"    * 最小值: {token_attention_scores.min().item():.6f}")
                # print(f"    * 平均值: {token_attention_scores.mean().item():.6f}")
                # print(f"    * 标准差: {token_attention_scores.std().item():.6f}")
                
                # 筛选Top K高关注Token（基于第二段文本注意力）
                top_k = max(1, int(num_image_tokens * top_k_ratio))
                top_values, top_rel_indices = torch.topk(token_attention_scores, k=top_k)
                top_abs_indices = [img_start + rel_idx for rel_idx in top_rel_indices.tolist()]
                
                # 打印Top K的注意力分数
                # print(f"  - Top-{top_k} 高关注图像Token（基于第二段文本注意力）:")
                for i, (abs_idx, rel_idx) in enumerate(zip(top_abs_indices, top_rel_indices.tolist())):
                    score = token_attention_scores[rel_idx].item()
                    # print(f"    * Token {abs_idx} (相对位置{rel_idx}): {score:.6f}")
                
                # 保存结果
                if batch_idx not in high_attention_indices:
                    high_attention_indices[batch_idx] = []
                high_attention_indices[batch_idx].extend(top_abs_indices)
        
        # print(f"\n[DEBUG] 新Attention分析完成，共找到{len(high_attention_indices)}个样本的高关注Token")
        return high_attention_indices

    
    def _perturb_token_spherical_rotation(self, token_embed, target_similarity, device):
        """
        球面旋转扰动方法
        """
        # 应用扰动
        token_embed_normalized = F.normalize(token_embed, p=2, dim=-1)
        
        # 生成垂直随机向量
        noise = torch.randn_like(token_embed_normalized)
        noise = noise - torch.sum(noise * token_embed_normalized, dim=-1, keepdim=True) * token_embed_normalized
        noise_normalized = F.normalize(noise, p=2, dim=-1)
        
        # 计算旋转角度
        theta = torch.acos(torch.tensor(target_similarity, device=device))
        
        # 旋转扰动
        perturbed_token_embed = (
            token_embed_normalized * torch.cos(theta) + 
            noise_normalized * torch.sin(theta)
        )
        
        # 补充高斯噪声 (辅助扰动)
        noise_std = getattr(self.forgetting_args, 'gaussian_noise_std', 0.01)
        if noise_std > 0:
            gaussian_noise = torch.randn_like(perturbed_token_embed) * noise_std
            perturbed_token_embed += gaussian_noise
        
        return perturbed_token_embed
    
    def _perturb_token_noise_injection(self, token_embed, device):
        """
        噪声注入扰动方法
        """
        noise_scale = getattr(self.forgetting_args, 'noise_injection_scale', 0.1)
        noise_type = getattr(self.forgetting_args, 'noise_injection_type', 'gaussian')
        
        if noise_type == 'gaussian':
            # 高斯噪声
            noise = torch.randn_like(token_embed) * noise_scale
        elif noise_type == 'uniform':
            # 均匀噪声
            noise = (torch.rand_like(token_embed) * 2 - 1) * noise_scale  # [-noise_scale, noise_scale]
        else:
            raise ValueError(f"未知的噪声类型: {noise_type}")
        
        # 直接添加噪声
        perturbed_token_embed = token_embed + noise
        
        return perturbed_token_embed
    
    def _perturb_token_feature_recombination(self, token_embed, batch_idx, rel_idx, image_embeds, device):
        """
        特征重组扰动方法
        """
        recombination_alpha = getattr(self.forgetting_args, 'recombination_alpha', 0.5)
        recombination_strategy = getattr(self.forgetting_args, 'recombination_strategy', 'random')
        
        # image_embeds的形状是[batch_size, num_image_tokens, embed_dim]
        # rel_idx是相对于图像embeddings的索引
        num_tokens = image_embeds.shape[1]
        
        # 检查rel_idx是否在有效范围内
        if not (0 <= rel_idx < num_tokens):
            return token_embed
        
        # 获取该图像区域的所有embeddings
        region_embeds = image_embeds[batch_idx, :, :]  # [num_image_tokens, embed_dim]
        
        if recombination_strategy == 'random':
            # 随机选择一个其他token的embedding
            other_indices = [i for i in range(num_tokens) if i != rel_idx]
            if len(other_indices) > 0:
                random_idx = torch.randint(0, len(other_indices), (1,), device=device).item()
                other_embed = region_embeds[other_indices[random_idx]:other_indices[random_idx]+1]
            else:
                # 如果没有其他token，使用当前token本身
                other_embed = token_embed
                
        elif recombination_strategy == 'neighbor':
            # 选择邻近的token（优先选择相邻的）
            neighbor_indices = []
            if rel_idx > 0:
                neighbor_indices.append(rel_idx - 1)
            if rel_idx < num_tokens - 1:
                neighbor_indices.append(rel_idx + 1)
            
            if len(neighbor_indices) > 0:
                neighbor_idx = neighbor_indices[torch.randint(0, len(neighbor_indices), (1,), device=device).item()]
                other_embed = region_embeds[neighbor_idx:neighbor_idx+1]
            else:
                other_embed = token_embed
                
        elif recombination_strategy == 'average':
            # 使用所有其他token的平均embedding
            other_indices = [i for i in range(num_tokens) if i != rel_idx]
            if len(other_indices) > 0:
                other_embeds = region_embeds[other_indices]
                other_embed = other_embeds.mean(dim=0, keepdim=True)
            else:
                other_embed = token_embed
        else:
            raise ValueError(f"未知的特征重组策略: {recombination_strategy}")
        
        # 线性插值混合
        perturbed_token_embed = recombination_alpha * token_embed + (1 - recombination_alpha) * other_embed
        
        return perturbed_token_embed
    
    def _perturb_high_attention_embeds_new(self, image_embeds, high_attention_indices, image_token_positions):
        """
        基于新的位置信息对高关注图像Token的Embedding进行扰动 - 支持多种扰动方法
        方法包括：
        1. spherical_rotation: 球面旋转（原方法）
        2. noise_injection: 噪声注入
        3. feature_recombination: 特征重组
        """
        perturbed_image_embeds = image_embeds.clone()
        batch_size, num_image_tokens_total, embed_dim = image_embeds.shape
        
        # 获取扰动方法
        perturbation_method = getattr(self.forgetting_args, 'perturbation_method', 'spherical_rotation')
        
        print(f"\n[DEBUG] 开始新针对性扰动 (Token级别, 方法: {perturbation_method}):")
        print(f"  - 图像Embedding形状: {image_embeds.shape}")
        print(f"  - 需要扰动的样本数: {len(high_attention_indices)}")
        
        # 获取当前训练步骤
        current_step = getattr(self.state, 'global_step', 0)
        device = image_embeds.device
        
        # 计算目标相似度（仅用于球面旋转方法）
        target_similarity = None
        if perturbation_method == 'spherical_rotation':
            # 原公式参数映射
            s_min = 0.6  # 相似度目标下限
            T = getattr(self.forgetting_args, 'perturbation_max_steps', 50)  # 周期长度
            alpha = getattr(self.forgetting_args, 'alpha', 1.0)  # 进度速率控制器
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
            fluctuation = amplitude_factor * math.cos(phase)
            
            # 最终目标相似度：基础偏移项 + 周期性波动项
            target_similarity = base_offset + fluctuation
            print(f"原始进度: {current_step:.2f}, 目标相似度: {target_similarity:.4f}")
            
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
                            
                            # 根据扰动方法选择不同的处理方式
                            if perturbation_method == 'spherical_rotation':
                                perturbed_token_embed = self._perturb_token_spherical_rotation(
                                    token_embed, target_similarity, device
                                )
                            elif perturbation_method == 'noise_injection':
                                perturbed_token_embed = self._perturb_token_noise_injection(
                                    token_embed, device
                                )
                            elif perturbation_method == 'feature_recombination':
                                perturbed_token_embed = self._perturb_token_feature_recombination(
                                    token_embed, batch_idx, rel_idx, image_embeds, device
                                )
                            else:
                                raise ValueError(f"未知的扰动方法: {perturbation_method}")

                            # 将扰动后的embedding放回原位置
                            perturbed_image_embeds[batch_idx, rel_idx:rel_idx+1] = perturbed_token_embed
                            
                        break
        
        # 计算整体扰动效果
        overall_similarity = F.cosine_similarity(image_embeds, perturbed_image_embeds, dim=-1).mean()
        print(f"\n[DEBUG] Token级别扰动效果 ({perturbation_method}):")
        print(f"  - 整体相似度: {overall_similarity.item():.4f}")
        print(f"  - 扰动Token数量: {sum(len(indices) for indices in high_attention_indices.values())}")
        
        return perturbed_image_embeds

    def _compute_retention_loss(self, model, inputs):
        """KL散度损失用于保留任务 - 基于图像Embedding扰动和输出分布蒸馏，支持多次采样"""
        num_samples = getattr(self.forgetting_args, 'num_samples_per_step', 1)
        if self.is_world_process_zero():
            print(f"[DEBUG] Computing retention loss with image embedding perturbation, num_samples: {num_samples}")
        
        # 1. 初始化模型和设备
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.second_model.module if hasattr(self.second_model, 'module') else self.second_model
        device = next(student_model.parameters()).device  # 统一使用学生模型的设备
        perturbed_image_embeds = None
        image_embeds = None  # 提前初始化，避免未定义错误

        # 2. 确保教师模型配置与学生模型一致
        if hasattr(teacher_model, 'base_model'):
            # 对于PeftModelForCausalLM，修改底层模型的配置
            teacher_model.base_model.config.mm_use_im_start_end = True
            teacher_model.base_model.config.mm_use_im_patch_token = True
            print(f"[DEBUG] 已设置教师模型配置: mm_use_im_start_end=True, mm_use_im_patch_token=True")
        elif hasattr(teacher_model, 'config'):
            # 对于普通模型，直接修改配置
            teacher_model.config.mm_use_im_start_end = True
            teacher_model.config.mm_use_im_patch_token = True
            print(f"[DEBUG] 已设置教师模型配置: mm_use_im_start_end=True, mm_use_im_patch_token=True")
        
        # 3. 定位图像Token范围（前提：先获取输入序列）
        input_ids = inputs['input_ids'].to(device)
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            raise ValueError("LLaVATrainer未初始化Tokenizer，请在__init__中传入tokenizer参数")
        
        print(f"\n[DEBUG] 开始图像Token定位:")
        print(f"  - 输入序列形状: {input_ids.shape}")
        print(f"  - 设备: {device}")
        print(f"  - inputs keys: {list(inputs.keys())}")
        print(f"  - 是否有images: {'images' in inputs}")
        if 'images' in inputs:
            print(f"  - images shape: {inputs['images'].shape if inputs['images'] is not None else 'None'}")
        
        image_token_ranges = self._get_image_token_indices(input_ids, self.tokenizer)

        # 3. 生成原始图像Embedding（关键：先于模型调用和扰动）
        if 'images' in inputs and inputs['images'] is not None and hasattr(student_model, 'get_model'):
            vision_tower = student_model.get_model().get_vision_tower()
            if vision_tower is not None:
                # 确保图像在正确设备上
                images = inputs['images'].to(device) if isinstance(inputs['images'], torch.Tensor) else inputs['images']
                image_embeds = vision_tower(images)  # [batch_size, num_image_tokens, embed_dim]
            else:
                print("[WARNING] 未找到Vision Tower，无法生成图像Embedding")
        else:
            print("[INFO] 当前batch无图像输入，使用纯文本逻辑")

        if vision_tower is not None:
            images = inputs['images'].to(device)
            image_embeds = vision_tower(images)  # [batch_size, 576, 1024]
            
            # 校验：image_embeds的长度是否与预期一致（v1版本需为576）
            expected_embed_len = 576
            if image_embeds.shape[1] != expected_embed_len:
                raise ValueError(
                    f"图像Embedding长度异常！预期{expected_embed_len}（CLIP ViT-L），"
                    f"实际{image_embeds.shape[1]}，请检查Vision Tower配置或图像预处理逻辑"
                )

        # 4. 准备蒸馏任务的文本（屏蔽答案）
        masked_labels = self.mask_answer_tokens(inputs['labels'].to(device))

        # 5. 处理无图像样本的情况（提前返回，避免后续无效计算）
        if not image_token_ranges or image_embeds is None:
            print("[WARNING] 无有效图像Token或Embedding，跳过针对性扰动")
            perturbed_image_embeds = image_embeds  # 直接使用原始Embedding（可能为None）
        else:
            # 6. 使用新的attention分析机制
            # 存储图像token位置信息
            self.image_token_positions = {}
            
            def attention_analysis_callback(batch_idx, image_idx, image_token_start, image_token_end, image_features):
                """回调函数，记录图像token的位置信息"""
                print(f"[DEBUG] 注意力分析回调被调用:")
                print(f"  - batch_idx: {batch_idx}")
                print(f"  - image_idx: {image_idx}")
                print(f"  - image_token_start: {image_token_start}")
                print(f"  - image_token_end: {image_token_end}")
                print(f"  - image_features形状: {image_features.shape}")
                
                if batch_idx not in self.image_token_positions:
                    self.image_token_positions[batch_idx] = []
                self.image_token_positions[batch_idx].append({
                    'image_idx': image_idx,
                    'start': image_token_start,
                    'end': image_token_end,
                    'features': image_features
                })
                
                # 记录文本token范围，用于后续注意力分析
                attention_analysis_callback.text_token_range = (0, image_token_start)
            
            # 首次调用学生模型获取Attention权重（使用新的attention分析机制）
            with torch.no_grad():  # 仅推理，不计算梯度
                # 保存原始配置
                original_output_attentions = student_model.config.output_attentions
                # 临时开启注意力输出
                student_model.config.output_attentions = True
                
                student_attention_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=inputs.get('attention_mask').to(device) if inputs.get('attention_mask') is not None else None,
                    images=inputs['images'] if 'images' in inputs else None,
                    image_embeds=image_embeds,  # 使用原始Embedding获取Attention
                    analyze_attention=True,  # 启用attention分析
                    attention_analysis_callback=attention_analysis_callback,  # 传入回调函数
                    return_dict=True
                )
                # 恢复原始配置
                student_model.config.output_attentions = original_output_attentions
            student_attentions = student_attention_outputs.attentions

            # 7. 筛选高关注图像Token（使用新的位置信息）
            if isinstance(student_attentions, tuple):
                # 将元组转换为列表，再堆叠成张量
                student_attentions_list = list(student_attentions)
                student_attentions_tensor = torch.stack(student_attentions_list, dim=0).to(device)
            elif isinstance(student_attentions, list):
                student_attentions_tensor = torch.stack(student_attentions, dim=0).to(device)
            else:
                # 如果是单个张量（较少见），直接迁移设备
                student_attentions_tensor = student_attentions.to(device)

            top_k_ratio = getattr(self.forgetting_args, 'top_k_ratio', 0.2)
            
            high_attention_indices = self._compute_high_attention_image_tokens_new(
                student_attentions_tensor,
                self.image_token_positions,
                input_ids,
                top_k_ratio
            )

            # 8. 针对性扰动高关注Token的Embedding
            if high_attention_indices:
                perturbed_image_embeds = self._perturb_high_attention_embeds_new(
                    image_embeds,
                    high_attention_indices,
                    self.image_token_positions
                )
            else:
                print("[WARNING] 未筛选出高关注图像Token，使用原始Embedding")
                perturbed_image_embeds = image_embeds

        # 9. 用扰动后的Embedding调用学生模型，获取输出分布
        print(f"[DEBUG] 调用学生模型前:")
        print(f"  - input_ids中-200的数量: {(input_ids == -200).sum().item()}")
        print(f"  - perturbed_image_embeds形状: {perturbed_image_embeds.shape if perturbed_image_embeds is not None else 'None'}")
        
        # 学生模型也使用prepare_inputs_labels_for_multimodal确保一致性
        print(f"[DEBUG] 学生模型配置:")
        print(f"  - mm_use_im_start_end: {getattr(student_model.config, 'mm_use_im_start_end', 'Not set')}")
        print(f"  - mm_use_im_patch_token: {getattr(student_model.config, 'mm_use_im_patch_token', 'Not set')}")
        
        student_input_ids, student_position_ids, student_attention_mask, student_past_key_values, student_new_input_embeds, student_new_labels = student_model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=inputs.get('position_ids') if inputs.get('position_ids') is not None else None,
            attention_mask=inputs.get('attention_mask') if inputs.get('attention_mask') is not None else None,
            past_key_values=None,
            labels=masked_labels,
            images=inputs['images'] if ('images' in inputs and inputs['images'] is not None) else None,
            image_embeds=perturbed_image_embeds if perturbed_image_embeds is not None else None,
            analyze_attention=True,
            attention_analysis_callback=attention_analysis_callback
        )
        
        student_outputs = student_model(
            input_ids=student_input_ids,
            position_ids=student_position_ids,
            attention_mask=student_attention_mask,
            past_key_values=student_past_key_values,
            inputs_embeds=student_new_input_embeds,
            labels=student_new_labels,
            return_dict=True
        )
        
        print(f"[DEBUG] 学生模型调用后:")
        print(f"  - 输出logits形状: {student_outputs.logits.shape}")
        print(f"  - 输出hidden_states形状: {student_outputs.hidden_states[0].shape if hasattr(student_outputs, 'hidden_states') and student_outputs.hidden_states else 'None'}")
        print(f"  - 处理后的序列长度: {student_new_input_embeds.shape[1]}")
        student_logits = student_outputs.logits

        # 10. 教师模型使用相同的扰动Embedding，获取输出分布
        with torch.no_grad():
            # 确保教师模型使用与学生模型相同的图像处理逻辑
            # 通过prepare_inputs_labels_for_multimodal统一处理
            try:
                # 检查teacher_model的实际类型
                print(f"[DEBUG] teacher_model类型: {type(teacher_model)}")
                print(f"[DEBUG] teacher_model是否有prepare_inputs_labels_for_multimodal: {hasattr(teacher_model, 'prepare_inputs_labels_for_multimodal')}")
                
                # 对于PeftModelForCausalLM，需要访问底层的LlavaLlamaForCausalLM对象
                if hasattr(teacher_model, 'base_model') and hasattr(teacher_model.base_model, 'prepare_inputs_labels_for_multimodal'):
                    # 通过base_model访问（PeftModelForCausalLM的情况）
                    print(f"[DEBUG] 通过base_model访问prepare_inputs_labels_for_multimodal")
                    print(f"[DEBUG] 教师模型配置:")
                    print(f"  - mm_use_im_start_end: {getattr(teacher_model.base_model.config, 'mm_use_im_start_end', 'Not set')}")
                    print(f"  - mm_use_im_patch_token: {getattr(teacher_model.base_model.config, 'mm_use_im_patch_token', 'Not set')}")
                    
                    teacher_input_ids, teacher_position_ids, teacher_attention_mask, teacher_past_key_values, teacher_new_input_embeds, teacher_new_labels = teacher_model.base_model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids.to(teacher_model.device),
                        position_ids=inputs.get('position_ids').to(teacher_model.device) if inputs.get('position_ids') is not None else None,
                        attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                        past_key_values=None,
                        labels=masked_labels.to(teacher_model.device),
                        images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                        image_embeds=perturbed_image_embeds.to(teacher_model.device) if perturbed_image_embeds is not None else None
                    )
                elif hasattr(teacher_model, 'prepare_inputs_labels_for_multimodal'):
                    # 直接调用
                    print(f"[DEBUG] 直接调用prepare_inputs_labels_for_multimodal")
                    teacher_input_ids, teacher_position_ids, teacher_attention_mask, teacher_past_key_values, teacher_new_input_embeds, teacher_new_labels = teacher_model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids.to(teacher_model.device),
                        position_ids=inputs.get('position_ids').to(teacher_model.device) if inputs.get('position_ids') is not None else None,
                        attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                        past_key_values=None,
                        labels=masked_labels.to(teacher_model.device),
                        images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                        image_embeds=perturbed_image_embeds.to(teacher_model.device) if perturbed_image_embeds is not None else None
                    )
                elif hasattr(teacher_model, 'module') and hasattr(teacher_model.module, 'prepare_inputs_labels_for_multimodal'):
                    # 通过module访问
                    print(f"[DEBUG] 通过module访问prepare_inputs_labels_for_multimodal")
                    teacher_input_ids, teacher_position_ids, teacher_attention_mask, teacher_past_key_values, teacher_new_input_embeds, teacher_new_labels = teacher_model.module.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids.to(teacher_model.device),
                        position_ids=inputs.get('position_ids').to(teacher_model.device) if inputs.get('position_ids') is not None else None,
                        attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                        past_key_values=None,
                        labels=masked_labels.to(teacher_model.device),
                        images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                        image_embeds=perturbed_image_embeds.to(teacher_model.device) if perturbed_image_embeds is not None else None
                    )
                elif hasattr(teacher_model, 'get_model') and hasattr(teacher_model.get_model(), 'prepare_inputs_labels_for_multimodal'):
                    # 通过get_model访问
                    print(f"[DEBUG] 通过get_model访问prepare_inputs_labels_for_multimodal")
                    teacher_input_ids, teacher_position_ids, teacher_attention_mask, teacher_past_key_values, teacher_new_input_embeds, teacher_new_labels = teacher_model.get_model().prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids.to(teacher_model.device),
                        position_ids=inputs.get('position_ids').to(teacher_model.device) if inputs.get('position_ids') is not None else None,
                        attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                        past_key_values=None,
                        labels=masked_labels.to(teacher_model.device),
                        images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                        image_embeds=perturbed_image_embeds.to(teacher_model.device) if perturbed_image_embeds is not None else None
                    )
                else:
                    raise AttributeError("无法找到prepare_inputs_labels_for_multimodal方法")
                
                # 使用处理后的输入调用教师模型
                teacher_outputs = teacher_model(
                    input_ids=teacher_input_ids,
                    position_ids=teacher_position_ids,
                    attention_mask=teacher_attention_mask,
                    past_key_values=teacher_past_key_values,
                    inputs_embeds=teacher_new_input_embeds,
                    labels=teacher_new_labels,
                    return_dict=True
                )
            except AttributeError as e:
                print(f"[WARNING] 教师模型无法调用prepare_inputs_labels_for_multimodal: {e}")
                print("[INFO] 回退到原始调用方式")
                # 回退到原始调用方式
                teacher_outputs = teacher_model(
                    input_ids=input_ids.to(teacher_model.device),
                    attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                    labels=masked_labels.to(teacher_model.device),
                    images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                    image_embeds=perturbed_image_embeds.to(teacher_model.device) if perturbed_image_embeds is not None else None,
                    return_dict=True
                )
            
            teacher_logits = teacher_outputs.logits
            print(f"[DEBUG] 教师模型调用后:")
            print(f"  - 输出logits形状: {teacher_outputs.logits.shape}")
            print(f"  - 处理后的序列长度: {teacher_new_input_embeds.shape[1]}")

        # 11. 多次采样处理
        all_student_logits = []
        all_teacher_logits = []
        
        # 第一次采样（已经计算过）
        all_student_logits.append(student_logits)
        all_teacher_logits.append(teacher_logits)
        
        # 进行额外的采样
        for sample_idx in range(1, num_samples):
            if self.is_world_process_zero():
                print(f"[DEBUG] 进行第{sample_idx + 1}次采样")
            
            # 重新生成扰动后的图像embedding（每次采样都不同）
            if image_embeds is not None and high_attention_indices:
                # 重新计算高关注token（可能会有轻微变化）
                perturbed_image_embeds_sample = self._perturb_high_attention_embeds_new(
                    image_embeds,
                    high_attention_indices,
                    self.image_token_positions
                )
            else:
                perturbed_image_embeds_sample = perturbed_image_embeds
            
            # 学生模型推理
            student_input_ids, student_position_ids, student_attention_mask, student_past_key_values, student_new_input_embeds, student_new_labels = student_model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=inputs.get('position_ids') if inputs.get('position_ids') is not None else None,
                attention_mask=inputs.get('attention_mask') if inputs.get('attention_mask') is not None else None,
                past_key_values=None,
                labels=masked_labels,
                images=inputs['images'] if ('images' in inputs and inputs['images'] is not None) else None,
                image_embeds=perturbed_image_embeds_sample if perturbed_image_embeds_sample is not None else None,
                analyze_attention=True,
                attention_analysis_callback=attention_analysis_callback
            )
            
            student_outputs_sample = student_model(
                input_ids=student_input_ids,
                position_ids=student_position_ids,
                attention_mask=student_attention_mask,
                past_key_values=student_past_key_values,
                inputs_embeds=student_new_input_embeds,
                labels=student_new_labels,
                return_dict=True
            )
            
            # 教师模型推理
            with torch.no_grad():
                try:
                    if hasattr(teacher_model, 'base_model') and hasattr(teacher_model.base_model, 'prepare_inputs_labels_for_multimodal'):
                        teacher_input_ids, teacher_position_ids, teacher_attention_mask, teacher_past_key_values, teacher_new_input_embeds, teacher_new_labels = teacher_model.base_model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids.to(teacher_model.device),
                            position_ids=inputs.get('position_ids').to(teacher_model.device) if inputs.get('position_ids') is not None else None,
                            attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                            past_key_values=None,
                            labels=masked_labels.to(teacher_model.device),
                            images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                            image_embeds=perturbed_image_embeds_sample.to(teacher_model.device) if perturbed_image_embeds_sample is not None else None
                        )
                        teacher_outputs_sample = teacher_model(
                            input_ids=teacher_input_ids,
                            position_ids=teacher_position_ids,
                            attention_mask=teacher_attention_mask,
                            past_key_values=teacher_past_key_values,
                            inputs_embeds=teacher_new_input_embeds,
                            labels=teacher_new_labels,
                            return_dict=True
                        )
                    elif hasattr(teacher_model, 'prepare_inputs_labels_for_multimodal'):
                        teacher_input_ids, teacher_position_ids, teacher_attention_mask, teacher_past_key_values, teacher_new_input_embeds, teacher_new_labels = teacher_model.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids.to(teacher_model.device),
                            position_ids=inputs.get('position_ids').to(teacher_model.device) if inputs.get('position_ids') is not None else None,
                            attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                            past_key_values=None,
                            labels=masked_labels.to(teacher_model.device),
                            images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                            image_embeds=perturbed_image_embeds_sample.to(teacher_model.device) if perturbed_image_embeds_sample is not None else None
                        )
                        teacher_outputs_sample = teacher_model(
                            input_ids=teacher_input_ids,
                            position_ids=teacher_position_ids,
                            attention_mask=teacher_attention_mask,
                            past_key_values=teacher_past_key_values,
                            inputs_embeds=teacher_new_input_embeds,
                            labels=teacher_new_labels,
                            return_dict=True
                        )
                    else:
                        teacher_outputs_sample = teacher_model(
                            input_ids=input_ids.to(teacher_model.device),
                            attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                            labels=masked_labels.to(teacher_model.device),
                            images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                            image_embeds=perturbed_image_embeds_sample.to(teacher_model.device) if perturbed_image_embeds_sample is not None else None,
                            return_dict=True
                        )
                except AttributeError:
                    teacher_outputs_sample = teacher_model(
                        input_ids=input_ids.to(teacher_model.device),
                        attention_mask=inputs.get('attention_mask').to(teacher_model.device) if inputs.get('attention_mask') is not None else None,
                        labels=masked_labels.to(teacher_model.device),
                        images=inputs['images'].to(teacher_model.device) if ('images' in inputs and inputs['images'] is not None) else None,
                        image_embeds=perturbed_image_embeds_sample.to(teacher_model.device) if perturbed_image_embeds_sample is not None else None,
                        return_dict=True
                    )
                
                teacher_logits_sample = teacher_outputs_sample.logits
            
            all_student_logits.append(student_outputs_sample.logits)
            all_teacher_logits.append(teacher_logits_sample)
        
        # 12. 合并多次采样的结果并计算KL散度损失
        # 将所有采样的logits堆叠成一个大的batch
        combined_student_logits = torch.cat(all_student_logits, dim=0)  # [num_samples * batch_size, seq_len, vocab_size]
        combined_teacher_logits = torch.cat(all_teacher_logits, dim=0)  # [num_samples * batch_size, seq_len, vocab_size]
        
        # 计算KL散度损失
        student_probs = F.softmax(combined_student_logits.to(device), dim=-1)
        teacher_probs = F.softmax(combined_teacher_logits.to(device), dim=-1)
        kl_loss = -(teacher_probs * torch.log(student_probs + 1e-12)).sum(-1).mean()

        if self.is_world_process_zero():
            print(f"[DEBUG] KL loss (with {num_samples} samples): {kl_loss.item():.6f}")
            print(f"[DEBUG] Combined batch size: {combined_student_logits.shape[0]} (original: {input_ids.shape[0]}, samples: {num_samples})")

        return kl_loss

    def _get_image_token_indices(self, input_ids, tokenizer):
        """
        智能检测图像Token范围，兼容不同的LLaVA版本
        """
        image_token_ranges = []
        batch_size, seq_len = input_ids.shape
        
        # 检查是否使用im_start_end模式
        use_im_start_end = getattr(self.args, 'mm_use_im_start_end', False)
        print(f"[DEBUG] use_im_start_end: {use_im_start_end}")
        
        for batch_idx in range(batch_size):
            seq = input_ids[batch_idx].tolist()
            # print(f"\n[DEBUG] 样本{batch_idx}的input_ids:")
            # print(f"  - 序列长度: {len(seq)}")
            # print(f"  - 完整序列: {seq}")
            
            # 解码前20个token看看内容
            try:
                decoded_tokens = [tokenizer.decode([token_id]) for token_id in seq[:20]]
                print(f"  - 前20个token解码: {decoded_tokens}")
            except Exception as e:
                print(f"  - 解码失败: {e}")
            
            if use_im_start_end:
                # 模式1：使用<im_start>和<im_end>标记
                im_start_id = tokenizer.convert_tokens_to_ids("<im_start>")
                im_end_id = tokenizer.convert_tokens_to_ids("<im_end>")
                
                if im_start_id == tokenizer.unk_token_id or im_end_id == tokenizer.unk_token_id:
                    print(f"[WARNING] 样本{batch_idx}: 未找到<im_start>或<im_end>特殊Token")
                    continue
                
                try:
                    start_idx = seq.index(im_start_id) + 1
                    end_idx = seq.index(im_end_id)
                    if start_idx >= end_idx:
                        print(f"[WARNING] 样本{batch_idx}: <im_start>位置({start_idx}) >= <im_end>位置({end_idx})")
                        continue
                    
                    # 关键修复：如果图像token数量不足，说明数据预处理有问题
                    actual_token_num = end_idx - start_idx
                    if actual_token_num < 100:  # 正常情况下应该有几百个token
                        print(f"[WARNING] 样本{batch_idx}: 图像Token数量过少({actual_token_num})，可能数据预处理有问题")
                        continue
                    
                    image_token_ranges.append((batch_idx, start_idx, end_idx))
                    print(f"[DEBUG] 样本{batch_idx}: 图像Token范围[{start_idx}:{end_idx}] (共{actual_token_num}个token)")
                except ValueError as e:
                    print(f"[WARNING] 样本{batch_idx}: 无有效<im_start>/<im_end>标记: {e}")
                    continue
            else:
                # 模式2：使用单个<image> token 或 <im_start><im_patch>...<im_end>格式
                # 首先尝试找<im_start>和<im_end>（即使use_im_start_end=False，数据可能仍使用这种格式）
                im_start_id = tokenizer.convert_tokens_to_ids("<im_start>")
                im_end_id = tokenizer.convert_tokens_to_ids("<im_end>")
                
                if im_start_id != tokenizer.unk_token_id and im_end_id != tokenizer.unk_token_id:
                    try:
                        start_idx = seq.index(im_start_id) + 1
                        end_idx = seq.index(im_end_id)
                        if start_idx < end_idx:
                            actual_token_num = end_idx - start_idx
                            # 打印图像token的详细信息
                            image_tokens = seq[start_idx:end_idx]
                            print(f"[DEBUG] 样本{batch_idx}: 检测到<im_start>/<im_end>格式，图像Token范围[{start_idx}:{end_idx}] (共{actual_token_num}个token)")
                            print(f"  - 图像Token内容: {image_tokens[:10]}...{image_tokens[-5:] if len(image_tokens) > 10 else image_tokens}")
                            print(f"  - 图像Token类型: {set(image_tokens)}")
                            image_token_ranges.append((batch_idx, start_idx, end_idx))
                            continue
                    except ValueError:
                        pass
                
                # 如果没找到<im_start>/<im_end>，尝试找IMAGE_TOKEN_INDEX (-200)
                # 这是<image> token被替换后的实际token id
                image_token_positions = [i for i, token_id in enumerate(seq) if token_id == -200]  # IMAGE_TOKEN_INDEX
                
                if image_token_positions:
                    for img_pos in image_token_positions:
                        # 智能检测图像特征数量
                        # 方法1：从vision tower获取实际特征数量
                        if hasattr(self, 'model') and hasattr(self.model, 'get_model'):
                            try:
                                vision_tower = self.model.get_model().get_vision_tower()
                                if vision_tower is not None and hasattr(vision_tower, 'num_patches'):
                                    num_patches = vision_tower.num_patches
                                else:
                                    num_patches = 576  # 默认值
                            except:
                                num_patches = 576  # 默认值
                        else:
                            num_patches = 576  # 默认值
                        
                        start_idx = img_pos
                        end_idx = min(img_pos + num_patches, seq_len)
                        image_token_ranges.append((batch_idx, start_idx, end_idx))
                        print(f"[DEBUG] 样本{batch_idx}: 图像Token范围[{start_idx}:{end_idx}] (共{end_idx-start_idx}个token, 预期{num_patches}个)")
                    continue
                
                # 如果没找到IMAGE_TOKEN_INDEX，尝试找<image> token（作为备选）
                image_token_id = tokenizer.convert_tokens_to_ids("<image>")
                if image_token_id != tokenizer.unk_token_id:
                    image_positions = [i for i, token_id in enumerate(seq) if token_id == image_token_id]
                    
                    if image_positions:
                        for img_pos in image_positions:
                            num_patches = 576
                            start_idx = img_pos
                            end_idx = min(img_pos + num_patches, seq_len)
                            image_token_ranges.append((batch_idx, start_idx, end_idx))
                            print(f"[DEBUG] 样本{batch_idx}: 图像Token范围[{start_idx}:{end_idx}] (共{end_idx-start_idx}个token)")
                        continue
                
                print(f"[WARNING] 样本{batch_idx}: 未找到任何图像Token标记")
        
        print(f"[DEBUG] 总共找到{len(image_token_ranges)}个图像Token范围")
        return image_token_ranges
    
    def _monitor_parameters(self, model, model_name, step):
        """监控模型参数状态"""
        try:
            # 获取实际模型（处理DeepSpeed包装）
            actual_model = model.module if hasattr(model, 'module') else model
            
            # 统计参数
            total_params = 0
            trainable_params = 0
            frozen_params = 0
            
            trainable_names = []
            frozen_names = []
            
            for name, param in actual_model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    trainable_names.append(name)
                else:
                    frozen_params += param.numel()
                    frozen_names.append(name)
            
            # 计算百分比
            trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0
            
            # 按模块分类
            module_stats = {}
            for name in trainable_names + frozen_names:
                module = name.split('.')[0] if '.' in name else name
                if module not in module_stats:
                    module_stats[module] = {'trainable': 0, 'frozen': 0, 'total': 0}
                
                param = dict(actual_model.named_parameters())[name]
                module_stats[module]['total'] += param.numel()
                if param.requires_grad:
                    module_stats[module]['trainable'] += param.numel()
                else:
                    module_stats[module]['frozen'] += param.numel()
            
            # 打印监控信息
            print(f"  📊 {model_name} 参数统计 (Step {step}):")
            print(f"    总参数: {total_params:,}")
            print(f"    可训练: {trainable_params:,} ({trainable_pct:.2f}%)")
            print(f"    冻结: {frozen_params:,}")
            
            # 打印关键模块状态
            key_modules = ['lora', 'mm_projector', 'vision_tower', 'model', 'base_model']
            print(f"  🔍 关键模块状态:")
            for module in key_modules:
                if module in module_stats:
                    stats = module_stats[module]
                    trainable_pct = (stats['trainable'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    status = "🟢 可训练" if trainable_pct > 0 else "🔴 冻结"
                    print(f"    {module}: {status} ({trainable_pct:.1f}% - {stats['trainable']:,}/{stats['total']:,})")
            
            # 打印LoRA相关参数
            lora_params = [name for name in trainable_names if 'lora' in name.lower()]
            if lora_params:
                print(f"  🎯 LoRA参数 ({len(lora_params)}个):")
                for name in lora_params[:5]:  # 只显示前5个
                    print(f"    - {name}")
                if len(lora_params) > 5:
                    print(f"    ... 还有{len(lora_params)-5}个LoRA参数")
            
            # 打印投影层参数
            projector_params = [name for name in trainable_names if 'mm_projector' in name.lower()]
            if projector_params:
                print(f"  🖼️ 投影层参数 ({len(projector_params)}个):")
                for name in projector_params:
                    print(f"    - {name}")
            else:
                print(f"  🖼️ 投影层参数: 无 (可能被冻结)")
                
        except Exception as e:
            print(f"  ❌ 参数监控出错: {e}")

   
    def training_step(self, model, inputs):
        """
        两任务微调的training_step方法，支持多次采样：
        1. 梯度上升任务：标准交叉熵损失取负号
        2. 模型蒸馏对齐：使用KL散度损失（支持多次采样）
        """
        import time
        
        # 记录步骤开始时间
        if self._current_step_start_time is None:
            # 第一次调用，记录训练开始时间
            if self._training_start_time is None:
                self._training_start_time = time.time()
        step_start_time = time.time()
        
        model.train()
        device = next(model.parameters()).device
        # 准备输入数据
        inputs = self._prepare_inputs(inputs)
        
        # 参数监控
        self._monitor_step_count += 1
        if self._monitor_step_count % self._monitor_interval == 0:
            print(f"\n[Step {self._monitor_step_count}] 参数状态监控:")
            self._monitor_parameters(model, "StudentModel", self._monitor_step_count)
        
        # 获取采样次数
        num_samples = getattr(self.forgetting_args, 'num_samples_per_step', 1)
        
        # 使用HuggingFace的内置compute_loss方法确保维度匹配
        base_loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        print(f"[DEBUG] base_loss值: {base_loss.item()}, 是否有梯度: {base_loss.requires_grad}")
        
        # 任务1：梯度上升 - 使用base_loss的负值
        # ga_loss 不受采样次数影响，保持原始尺度
        ga_loss = -base_loss  # 取负号实现梯度上升
        
        # 任务2：模型蒸馏对齐 - 使用KL散度损失（支持多次采样）
        kl_loss = self._compute_retention_loss(model, inputs)
        
        # 组合损失
        ga_weight = getattr(self.forgetting_args, 'ga_weight', 0.3)
        kl_weight = getattr(self.forgetting_args, 'kl_weight', 0.7)
        
        print(f"[DEBUG] ga_weight: {ga_weight}, kl_weight: {kl_weight}, num_samples: {num_samples}")
        print(f"[DEBUG] ga_loss: {ga_loss.item():.6f}, kl_loss: {kl_loss.item():.6f}")
        total_loss = ga_weight * ga_loss + kl_weight * kl_loss
        
        # 反向传播
        self.accelerator.backward(total_loss)
        
        # 多GPU场景下平均损失
        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()
        
        # 记录步骤结束时间并计算耗时
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        self._step_times.append(step_duration)
        self._current_step_start_time = step_end_time
        
        # 每10步输出一次时间统计
        if len(self._step_times) % 10 == 0 and self.is_world_process_zero():
            avg_step_time = sum(self._step_times[-10:]) / 10
            total_time_so_far = sum(self._step_times)
            print(f"\n[时间统计] Step {len(self._step_times)}: "
                  f"当前步骤耗时={step_duration:.4f}秒, "
                  f"最近10步平均耗时={avg_step_time:.4f}秒, "
                  f"累计训练时间={total_time_so_far:.2f}秒 ({total_time_so_far/60:.2f}分钟)")
        
        return total_loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            default_lr = getattr(self.args, 'learning_rate', 1e-5)
            if default_lr == 0:
                print(f"[WARNING] 检测到学习率为0，设置为默认值1e-5")
                default_lr = 2e-4
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
    
    def _print_training_time_summary(self):
        """打印训练时间统计摘要"""
        if not self._step_times:
            return
        
        import time
        import os
        import json
        
        total_steps = len(self._step_times)
        total_time = sum(self._step_times)
        avg_step_time = total_time / total_steps
        min_step_time = min(self._step_times)
        max_step_time = max(self._step_times)
        
        # 计算总训练时间（从开始到结束）
        if self._training_start_time is not None:
            elapsed_time = time.time() - self._training_start_time
        else:
            elapsed_time = total_time
        
        print("\n" + "="*80)
        print("训练时间统计摘要 (COPY方法 - 带注意力分析和扰动)")
        print("="*80)
        print(f"总训练步数: {total_steps}")
        print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟, {total_time/3600:.2f}小时)")
        print(f"从开始到结束的耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟, {elapsed_time/3600:.2f}小时)")
        print(f"平均每步耗时: {avg_step_time:.4f}秒")
        print(f"最快步骤耗时: {min_step_time:.4f}秒")
        print(f"最慢步骤耗时: {max_step_time:.4f}秒")
        print(f"每步耗时标准差: {(sum((t - avg_step_time)**2 for t in self._step_times) / total_steps)**0.5:.4f}秒")
        print("="*80)
        
        # 保存统计信息到文件
        if self.is_world_process_zero():
            output_dir = getattr(self.args, 'output_dir', None)
            if output_dir is not None:
                stats_file = os.path.join(output_dir, "training_time_stats.json")
                stats = {
                    "method": "COPY",
                    "total_steps": total_steps,
                    "total_training_time_seconds": total_time,
                    "elapsed_time_seconds": elapsed_time,
                    "avg_step_time_seconds": avg_step_time,
                    "min_step_time_seconds": min_step_time,
                    "max_step_time_seconds": max_step_time,
                    "step_times": self._step_times[:100]  # 只保存前100步的详细时间，避免文件过大
                }
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"时间统计信息已保存到: {stats_file}")