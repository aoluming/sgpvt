# LLaVA Multimodal Unlearning / Perturbation Training Extensions

This repository extends [LLaVA](https://github.com/haotian-liu/LLaVA) with **dual-model (student/teacher) training**, **image embedding perturbation**, and **gradient ascent + KL retention** mechanisms for controllable unlearning or robustness experiments in multimodal large models. Designed for researchers familiar with LLaVA data formats and training workflows.

## Feature Overview

| Module | Description |
|------|------|
| **Training** | DeepSpeed ZeRO + LoRA, implements perturbation and composite loss in `LLaVATrainer` |
| **Forget Entity Masking** | Configure entity substrings to mask in answer regions via JSON (`forget_entities_file`) |
| **Evaluation Scripts** | `tools/eval_folder_inference.py`: Batch inference for images in folders |
| **CLIP Similarity** | `clip_image_similarity.py`: Embedding similarity statistics between main image and multiple folder images |
| **Local Perturbation Examples** | `local_perturbation_config.py` + [README_local_perturbation.md](README_local_perturbation.md) |

## Requirements

- Python ≥ 3.8, PyTorch (recommend 2.x) and corresponding CUDA
- Refer to LLaVA official documentation for optional dependencies like `flash-attn`

### Installation

```bash
cd "/path/to/this/repo"   # Recommend using paths without spaces to avoid shell/toolchain issues
pip install -r requirements.txt
pip install -e .
```

Add the project root directory to `PYTHONPATH` (example provided in training scripts):

```bash
export PYTHONPATH="/path/to/this/repo:$PYTHONPATH"
```

## Core Training Code

Training entry point and customization logic are concentrated in the following files:

| File | Purpose |
|------|------|
| [scripts/v1_5/finetune_lora.sh](scripts/v1_5/finetune_lora.sh) | DeepSpeed launch commands, LoRA/data paths, **unlearning and perturbation hyperparameters** |
| [llava/train/train.py](llava/train/train.py) | Model and data loading, `ForgettingArguments`, construct `LLaVATrainer` |
| [llava/train/llava_trainer.py](llava/train/llava_trainer.py) | `LLaVATrainer`: `mask_answer_tokens`, image embedding perturbation, GA/KL, etc. |

### Quick Start Training

1. Edit `PROJECT_DIR`, `MODEL_PATH`, `DATA_PATH`, `IMAGE_FOLDER`, `VISION_TOWER`, `OUTPUT_DIR` at the top of `scripts/v1_5/finetune_lora.sh`.
2. Prepare conversation JSON data consistent with LLaVA format (including `image` field and `conversations`).
3. Prepare **forget entity list** JSON (array of strings, substrings matching tokenizer-decoded text), refer to [configs/forget_entities_example.json](configs/forget_entities_example.json).
4. Execute:

```bash
bash scripts/v1_5/finetune_lora.sh
```

If entity-based answer token masking is not needed, remove the `--forget_entities_file ...` line from the shell (in this case `mask_answer_tokens` will not perform entity masking).

### Main `ForgettingArguments` Parameters (CLI)

The following parameters are exemplified in `finetune_lora.sh`, definitions found in `ForgettingArguments` in `llava/train/train.py`.

| Parameter | Meaning |
|------|------|
| `ga_weight` / `kl_weight` | Gradient ascent term and KL (relative to frozen teacher) term weights |
| `perturbation_method` | `spherical_rotation` / `noise_injection` / `feature_recombination` |
| `top_k_ratio` | Ratio of high-attention image tokens to perturb |
| `perturbation_max_steps` / `alpha` | Perturbation period and progress control |
| `gaussian_noise_std` | Additional Gaussian noise standard deviation |
| `noise_injection_scale` / `noise_injection_type` | Noise injection intensity and type |
| `recombination_alpha` / `recombination_strategy` | Feature recombination mixing ratio and strategy |
| `num_samples_per_step` | Number of times to sample embeddings per step |
| `forget_entities_file` | Points to JSON array file for `mask_answer_tokens` |

DeepSpeed configuration found in [scripts/zero3.json](scripts/zero3.json). Modify `--include` and ports in `finetune_lora.sh` according to your cluster GPU count.

## Evaluation: Folder Batch Inference

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/eval_folder_inference.py \
  --model-path /path/to/your/lora_checkpoint \
  --model-base /path/to/llava-v1.5-7b \
  --image-file /path/to/image_folder \
  --output-file /path/to/log.txt \
  --prompt "What is the name of this person?"
```

## Tools: CLIP Image Similarity

Using Hugging Face format CLIP, calculate cosine similarity between **one main image** and all images in **multiple folders**, output rankings (default writes to `similarity_results.txt`).

```bash
python clip_image_similarity.py \
  --clip_path openai/clip-vit-large-patch14-336 \
  --main_image /path/to/main.jpg \
  --comparison_folders /path/to/folder_a /path/to/folder_b \
  --output similarity_results.txt
```

## Local Perturbation Configuration (Example)

`local_perturbation_config.py` provides `LocalPerturbationConfig` and several presets, detailed in [README_local_perturbation.md](README_local_perturbation.md). **Default training pipeline follows `ForgettingArguments`**; to integrate this configuration into `LLaVATrainer`, you need to extend parameter passing yourself.

## Directory Structure (Excerpt)

```
.
├── configs/
│   └── forget_entities_example.json
├── llava/
│   └── train/
│       ├── train.py
│       └── llava_trainer.py
├── scripts/
│   ├── v1_5/finetune_lora.sh
│   └── zero3.json
├── tools/
│   └── eval_folder_inference.py
├── clip_image_similarity.py
├── local_perturbation_config.py
├── requirements.txt
└── setup.py
```

## Open Source and Citation

- This repository is based on LLaVA, please follow both the original project license and citation requirements.
- If releasing code, please add `LICENSE` yourself and replace example paths, remove sensitive data.
- When using this extension for research, recommend citing both LLaVA and related foundation model papers.

## Acknowledgments

Thanks to Haotian Liu and other LLaVA authors for their open-source work.