# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoTokenizer
from xtuner.model import InternVL_V1_5
from xtuner.dataset import InternVL_V1_5_Dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
import os

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
_ROOT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_ROOT, "..", "..", "..", "..")  # xtuner/configs/internvl/v2 -> 仓库根
_ROOT = os.path.normpath(_ROOT)
# Model
path = os.environ.get("INTERNVL_MODEL_PATH", os.path.join(_ROOT, "InternVL2-8B"))

# Data
data_path = os.environ.get(
    "INTERNVL_ANN_PATH", os.path.join(_ROOT, "examples", "data", "sample_annotations.json")
)
image_folder = os.environ.get(
    "INTERNVL_IMAGE_DIR", os.path.join(_ROOT, "examples", "data", "images")
)
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 4096  # 从8192减少到4096，显著降低显存使用

# Scheduler & Optimizer
batch_size = 2  # per_device (从8减少到2)
accumulative_counts = 4  # 从2增加到8，保持总的有效批次大小
dataloader_num_workers = 4
max_epochs = 5  # 从1增加到3，确保学习率调度器有足够的步数
optim_type = AdamW
# official 1024 -> 4e-5
lr = 6e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.1

# Save
save_steps = 1000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=InternVL_V1_5,
    model_path=path,
    freeze_llm=True,
    freeze_visual_encoder=False,
    # comment the following lines if you don't want to use Lora in llm
    llm_lora=dict(
        type=LoraConfig,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=None,
        task_type="CAUSAL_LM",
    ),
    # visual_encoder_lora=dict(
    #     type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05,
    #     target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']),
    visual_encoder_lora=dict(
        type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05,
        target_modules=['mlp.fc1', 'mlp.fc2']),
    quantization_llm=False,  # 4bit量化降低显存占用
    quantization_vit=False,
    # # -------------------------- 双任务参数 --------------------------
    teacher_model_path=os.environ.get("INTERNVL_TEACHER_PATH", path),
    ga_weight=1.0,
    kl_weight=1.2,
    target_entity=os.environ.get("INTERNVL_TARGET_ENTITY", "entity"),
    deepspeed_zero_config=os.path.join(
        _ROOT, "xtuner", "configs", "deepspeed", "deepspeed_zero3.json"
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=InternVL_V1_5_Dataset,
    model_path=path,
    data_paths=data_path,
    image_folders=image_folder,
    template=prompt_template,
    max_length=max_length,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=llava_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=default_collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
)

custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

