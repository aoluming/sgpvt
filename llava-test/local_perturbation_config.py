"""
局部扰动配置文件示例
用于配置图像特定区域的扰动参数
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class LocalPerturbationConfig:
    """局部扰动配置类（示例；与主训练 CLI 解耦，按需自行接入 LLaVATrainer）"""

    # 目标实体列表 - 这些实体区域将被重点扰动
    target_entities: List[str] = field(
        default_factory=lambda: ["Elon Musk", "person", "face", "human"]
    )
    
    # 扰动强度
    perturbation_std: float = 0.1
    
    # 分割方法选择
    segmentation_method: str = "heuristic"  # "sam", "clip", "face_detection", "heuristic"
    
    # SAM模型配置
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    
    # 人脸检测配置
    face_detection_confidence: float = 0.5
    
    # CLIP模型配置
    clip_model_name: str = "ViT-B/32"
    
    # 掩码生成参数
    mask_center_weight: float = 1.0  # 中心区域权重
    mask_edge_weight: float = 0.3    # 边缘区域权重
    mask_threshold: float = 0.3      # 实体检测阈值

# 预定义配置示例
ELON_MUSK_CONFIG = LocalPerturbationConfig(
    target_entities=["Elon Musk", "person", "face", "human"],
    perturbation_std=0.15,
    segmentation_method="clip",  # 使用CLIP进行文本引导分割
    mask_center_weight=1.0,
    mask_edge_weight=0.2
)

PERSON_CONFIG = LocalPerturbationConfig(
    target_entities=["person", "human", "face"],
    perturbation_std=0.12,
    segmentation_method="sam",  # 使用SAM进行精确分割
    mask_center_weight=0.8,
    mask_edge_weight=0.4
)

HEURISTIC_CONFIG = LocalPerturbationConfig(
    target_entities=["person"],
    perturbation_std=0.1,
    segmentation_method="heuristic",  # 使用启发式规则
    mask_center_weight=1.0,
    mask_edge_weight=0.5
)

# 使用示例
if __name__ == "__main__":
    # 在训练脚本中使用
    config = ELON_MUSK_CONFIG
    
    print("局部扰动配置:")
    print(f"目标实体: {config.target_entities}")
    print(f"扰动强度: {config.perturbation_std}")
    print(f"分割方法: {config.segmentation_method}")
    print(f"中心权重: {config.mask_center_weight}")
    print(f"边缘权重: {config.mask_edge_weight}")
