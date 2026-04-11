# LLaVA Local Image Perturbation Feature Usage Guide

> **Note**: The main training entry point (`llava/train/train.py`) loads `ForgettingArguments` via CLI. The `LocalPerturbationConfig` described in this document is an independent example, describing the design of local masking and perturbation hyperparameters; to integrate with `LLaVATrainer`, you need to extend parameter passing and reading logic yourself.

## Overview

This feature allows you to apply local perturbation to specific regions of images (such as Elon Musk's portrait area), rather than global perturbation to the entire image. This is very useful for entity-specific forgetting training.

## Feature Capabilities

### 1. Multiple Segmentation Methods
- **SAM (Segment Anything Model)**: Most precise segmentation, requires pretrained weights
- **CLIP-guided Segmentation**: Intelligent segmentation based on text prompts
- **Face Detection**: Segmentation specifically for face regions
- **Heuristic Rules**: Simple segmentation based on image geometry

### 2. Flexible Configuration
- Customizable target entity list
- Adjustable perturbation intensity
- Configurable mask weights (center region vs edge region)

## Installing Dependencies

### Basic Dependencies
```bash
pip install opencv-python pillow numpy
```

### SAM Model (Optional)
```bash
pip install segment-anything
# Download pretrained weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### CLIP Model (Optional)
```bash
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

## Usage

### 1. Basic Configuration

In the training script, pass the configuration to the Trainer:

```python
from local_perturbation_config import ELON_MUSK_CONFIG

# Create training parameters
training_args = TrainingArguments(
    # ... other parameters ...
)

# Add configuration to forgetting_args
training_args.forgetting_args = ELON_MUSK_CONFIG

# Create Trainer
trainer = LLaVATrainer(
    second_model_path=second_model,
    forgetting_args=training_args.forgetting_args,
    # ... other parameters ...
)
```

### 2. Custom Configuration

```python
from local_perturbation_config import LocalPerturbationConfig

# Create custom configuration
custom_config = LocalPerturbationConfig(
    target_entities=["Elon Musk", "Tesla CEO"],
    perturbation_std=0.2,
    segmentation_method="clip",
    mask_center_weight=1.0,
    mask_edge_weight=0.1
)

training_args.forgetting_args = custom_config
```

### 3. Runtime Configuration

You can also modify configuration dynamically at runtime:

```python
# Modify target entities
trainer.forgetting_args.target_entities = ["Elon Musk", "person"]

# Modify perturbation intensity
trainer.forgetting_args.perturbation_std = 0.15

# Modify segmentation method
trainer.forgetting_args.segmentation_method = "sam"
```

## Configuration Parameters Explained

### target_entities
- **Type**: List[str]
- **Description**: List of target entities to be heavily perturbed
- **Example**: `["Elon Musk", "person", "face"]`

### perturbation_std
- **Type**: float
- **Description**: Standard deviation of perturbation, larger values mean stronger perturbation
- **Recommended Range**: 0.05 - 0.3

### segmentation_method
- **Type**: str
- **Options**: "sam", "clip", "face_detection", "heuristic"
- **Description**: Choose image segmentation method

### mask_center_weight / mask_edge_weight
- **Type**: float
- **Description**: Control the perturbation intensity ratio between center and edge regions

## Workflow

1. **Image Input**: Receive input image batch
2. **Segmentation Processing**: Create local masks based on selected method
3. **Mask Application**: Apply masks to image embeddings
4. **Local Perturbation**: Add noise only to masked regions
5. **Model Inference**: Use perturbed embeddings for forward propagation

## Performance Optimization Tips

### 1. Segmentation Method Selection
- **Fast Training**: Use "heuristic" method
- **Precise Segmentation**: Use "sam" method
- **Text-guided**: Use "clip" method

### 2. Batch Processing Optimization
- For images of the same size, masks can be reused
- Consider using caching mechanisms to avoid repeated computation

### 3. Memory Management
- SAM model is large, pay attention to GPU memory usage
- Can reduce batch size to save memory

## Troubleshooting

### Common Issues

1. **SAM Model Loading Failure**
   - Check pretrained weight file path
   - Ensure sufficient GPU memory

2. **CLIP Import Error**
   - Check if CLIP is installed correctly
   - Try reinstalling CLIP

3. **Mask Generation Failure**
   - System will automatically fall back to heuristic method
   - Check image format and size

### Debug Mode

Enable debug information:

```python
# Set in training script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example Scenarios

### Scenario 1: Elon Musk Forgetting Training
```python
elon_config = LocalPerturbationConfig(
    target_entities=["Elon Musk", "Tesla CEO", "SpaceX CEO"],
    perturbation_std=0.2,
    segmentation_method="clip"
)
```

### Scenario 2: Generic Person Forgetting
```python
person_config = LocalPerturbationConfig(
    target_entities=["person", "human", "face"],
    perturbation_std=0.15,
    segmentation_method="sam"
)
```

### Scenario 3: Fast Prototyping
```python
quick_config = LocalPerturbationConfig(
    target_entities=["person"],
    perturbation_std=0.1,
    segmentation_method="heuristic"
)
```

## Important Notes

1. **Computational Overhead**: SAM and CLIP methods will increase training time
2. **Memory Usage**: Advanced segmentation methods require more GPU memory
3. **Model Compatibility**: Ensure all dependency model versions are compatible
4. **Data Privacy**: Pay attention to privacy protection when processing sensitive images

## Technical Support

If you encounter issues, please check:
1. Dependency package version compatibility
2. Sufficient GPU memory
3. Pretrained weights downloaded correctly
4. Image format supported