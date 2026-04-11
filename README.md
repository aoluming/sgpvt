# SGPVT: Self-Generated Proximal Visual Tokens for Mitigating Proximal Collateral Damage in MLLM Unlearning

**ACL 2026 Main Conference**

This repository contains the official implementation of SGPVT, a novel approach for machine unlearning in Multimodal Large Language Models (MLLMs) that effectively mitigates proximal collateral damage through self-generated proximal visual tokens.

## 📋 Abstract

Machine unlearning in MLLMs faces the challenge of **proximal collateral damage** - where removing specific knowledge inadvertently degrades the model's performance on related but distinct concepts. Our proposed SGPVT method addresses this by generating proximal visual tokens that serve as semantic anchors during the unlearning process, preserving the model's capabilities on related concepts while effectively removing target knowledge.

## ✨ Key Features

- **Self-Generated Proximal Tokens**: Automatically generates visual tokens that represent concepts semantically proximal to the target forgetting concept
- **Collateral Damage Mitigation**: Significantly reduces performance degradation on related concepts during unlearning
- **Dual-Model Training Framework**: Implements student/teacher architecture with gradient ascent and KL retention mechanisms
- **Advanced Perturbation Strategies**: Supports spherical rotation, noise injection, and feature recombination methods
- **Local Image Perturbation**: Targeted perturbation of specific image regions for precise unlearning

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SGPVT Framework                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐         ┌─────────────┐                   │
│  │   Student   │         │   Teacher   │                   │
│  │   Model     │         │   Model     │                   │
│  │  (Learning) │         │  (Frozen)   │                   │
│  └──────┬──────┘         └──────┬──────┘                   │
│         │                       │                            │
│         │    ┌──────────────────┼──────────────────┐        │
│         │    │                                      │        │
│         │    │    ┌─────────────────────────┐      │        │
│         │    │    │  Proximal Visual Token  │      │        │
│         │    │    │       Generator         │      │        │
│         │    │    └─────────────────────────┘      │        │
│         │    │                                      │        │
│         │    │    ┌─────────────────────────┐      │        │
│         │    └────│   Perturbation Engine   │──────┘        │
│         │         └─────────────────────────┘              │
│         │                                                   │
│         │    ┌─────────────────────────────────┐          │
│         └─────│    Composite Loss Function     │          │
│               │  • Gradient Ascent (GA)         │          │
│               │  • KL Divergence (Retention)    │          │
│               │  • Proximal Token Preservation  │          │
│               └─────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
sgpvt/
├── llava-test/              # Main implementation
│   ├── llava/
│   │   ├── train/
│   │   │   ├── train.py                 # Training entry point
│   │   │   └── llava_trainer.py         # Custom trainer with SGPVT
│   ├── scripts/
│   │   ├── v1_5/finetune_lora.sh       # Training script
│   │   └── zero3.json                   # DeepSpeed config
│   ├── tools/
│   │   └── eval_folder_inference.py    # Evaluation tools
│   ├── clip_image_similarity.py        # CLIP similarity analysis
│   ├── local_perturbation_config.py    # Local perturbation configs
│   ├── README.md                        # Detailed documentation
│   └── README_local_perturbation.md     # Local perturbation guide
└── internVL-xtuner-main/    # Additional implementations
```

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA-capable GPU (recommend ≥ 24GB VRAM)
- DeepSpeed

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sgpvt.git
cd sgpvt/llava-test

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Training

```bash
# Edit configuration in scripts/v1_5/finetune_lora.sh
bash scripts/v1_5/finetune_lora.sh
```

### Evaluation

```bash
python tools/eval_folder_inference.py \
  --model-path /path/to/checkpoint \
  --model-base /path/to/base/model \
  --image-file /path/to/test/images \
  --output-file results.txt \
  --prompt "Your evaluation prompt here"
```

## 🎯 Core Components

### 1. Proximal Visual Token Generation
Automatically generates visual tokens that represent concepts semantically close to the target forgetting concept, serving as anchors to preserve related knowledge.

### 2. Dual-Model Training Framework
- **Student Model**: Learns to forget target concepts
- **Teacher Model**: Frozen reference model for KL divergence retention
- **Gradient Ascent**: Actively suppresses target knowledge
- **KL Retention**: Preserves general capabilities

### 3. Advanced Perturbation Methods
- **Spherical Rotation**: Rotates embeddings in high-dimensional space
- **Noise Injection**: Adds controlled Gaussian noise
- **Feature Recombination**: Mixes features from different semantic regions

### 4. Local Image Perturbation
Targeted perturbation of specific image regions using:
- SAM (Segment Anything Model)
- CLIP-guided segmentation
- Face detection
- Heuristic methods

## 📊 Experimental Results

Our method demonstrates significant improvements in mitigating proximal collateral damage:

| Method | Target Removal | Related Concept Preservation | Overall Performance |
|--------|---------------|------------------------------|---------------------|
| Standard Unlearning | ✓ | ✗ | ✗ |
| SGPVT (Ours) | ✓ | ✓ | ✓ |

*Detailed results available in the paper*

## 🔧 Configuration

### Key Training Parameters

```python
# Forgetting Arguments
ga_weight = 1.0              # Gradient ascent weight
kl_weight = 0.5              # KL retention weight
perturbation_method = "spherical_rotation"
top_k_ratio = 0.1           # Top-K token perturbation ratio
perturbation_max_steps = 10
alpha = 0.5                  # Perturbation intensity
```

### Local Perturbation Configuration

```python
from local_perturbation_config import LocalPerturbationConfig

config = LocalPerturbationConfig(
    target_entities=["Elon Musk"],
    perturbation_std=0.2,
    segmentation_method="clip",
    mask_center_weight=1.0,
    mask_edge_weight=0.1
)
```

## 📖 Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{sgpvt2026,
  title={SGPVT: Self-Generated Proximal Visual Tokens for Mitigating Proximal Collateral Damage in MLLM Unlearning},
  author={Jiaqi Li, Zhijing Zhang, Jiahui Geng, Sheng Bi, Chuanyi Zhang, Fan Liu, Guilin Qi},
  booktitle={Proceedings of the 64rd Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```

## 🙏 Acknowledgments

- **[LLaVA](https://github.com/haotian-liu/LLaVA)**: Base multimodal framework
- **[InternVL](https://github.com/OpenGVLab/InternVL)**: Additional implementation support
- **DeepSpeed Team**: For the efficient training framework

## 📄 License

This project is released under the Apache 2.0 license. Please see the LICENSE file for more details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and feedback, please contact [your email here].

---

**Note**: This is the official implementation for ACL 2026. The code is provided for research purposes. Please ensure proper citation when using this work.