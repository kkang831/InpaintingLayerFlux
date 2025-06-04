#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import load_file

import accelerate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    # free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available, load_image, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from pipeline_ours_AAA import FluxControlOursPipeline

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")

logger = get_logger(__name__)

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def log_validation(flux_transformer, args, accelerator, weight_dtype, step, is_final_validation=False, save_model_dir=None, save_image_dir=None):
    logger.info("Running validation... ")

    if not is_final_validation:
        flux_transformer = accelerator.unwrap_model(flux_transformer)
        pipeline = FluxControlOursPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=flux_transformer,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
        )
        initial_channels = transformer.config.in_channels
        pipeline = FluxControlOursPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )
        pipeline.load_lora_weights(save_model_dir)
        assert (
            pipeline.transformer.config.in_channels == initial_channels * 4
        ), f"{pipeline.transformer.config.in_channels=}"

    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # validation_root = '/data1/kkang/desobav2_full_res'
    validation_root = '/home/kkang831/dataset/desobav2_full_res'
    validation_txt  = os.path.join(validation_root, 'TestLabel.txt') 

    validation_example_num = 20
    with open(validation_txt, 'r') as f:
        validation_example = f.readlines()
    validation_example = [x.strip() for x in validation_example]
    validation_example = validation_example[:validation_example_num]
    
    validation_images = [os.path.join(validation_root, 'shadowfree_imgs', x) for x in validation_example]
    validation_masks  = [os.path.join(validation_root, 'object_masks', x) for x in validation_example]
    validation_prompts = ['"Realistic image with natural shadow"'] * validation_example_num
    
        
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type, weight_dtype)
        
    validation_index = 0

    for validation_prompt, validation_image, validation_mask in zip(validation_prompts, validation_images, validation_masks):
        validation_index += 1
        validation_image = load_image(validation_image)
        validation_mask = load_image(validation_mask)
        
        # maybe need to inference on 1024 to get a good image
        # validation_image = Image.resize(validation_image, (args.resolution, args.resolution), Image.LANCZOS)
        # validation_mask = Image.resize(validation_mask, (args.resolution, args.resolution), Image.BILINEAR)
        
        validation_image = validation_image.resize((args.resolution, args.resolution))
        validation_mask = validation_mask.resize((args.resolution, args.resolution))
        
        validation_image2 = validation_image.copy()
        validation_image2_np = np.array(validation_image2)
        validation_mask_np = np.array(validation_mask)
        validation_image2_np[validation_mask_np < 0.9] = 255
        validation_image2 = Image.fromarray(validation_image2_np)
        
        image_logs = []
        for _ in range(args.num_validation_images):
            validation_images1 = []
            validation_images2 = []
            validation_images1.append(validation_image)
            validation_images2.append(validation_image2)
            with autocast_ctx:
                image = pipeline(
                    prompt=validation_prompt,
                    control_image=validation_image,
                    control_mask=validation_mask,
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    max_sequence_length=512,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            image = image.resize((args.resolution, args.resolution))
            validation_images1.append(image)
            
            with autocast_ctx:
                image = pipeline(
                    prompt=validation_prompt,
                    control_image=validation_image2,
                    control_mask=validation_mask,
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    max_sequence_length=512,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            image = image.resize((args.resolution, args.resolution))
            validation_images2.append(image)

            image_logs.append(
                {"validation_images1": validation_images1, "validation_images2": validation_images2}
            )
        
        for index, example in enumerate(image_logs):
            for k, v in example.items():
                images = []
                for i in range(len(v)):
                    image = v[i] # PIL image
                    image = np.array(image)
                    images.append(image)
                image = np.concatenate(images, axis=1)
                image = Image.fromarray(image)
                image.save(os.path.join(save_image_dir, f"{step}_{validation_index}_{index}_{i}_{k}.png"))

    del pipeline
    # free_memory()
    return



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Control LoRA training script.")
    
    parser.add_argument("--save_root", type=str, default="training_results")
    # parser.add_argument("--save_image_root", type=str, default="/Bean/log/kkang/EditingLayerSynthesis/train_flux_network")
    # parser.add_argument("--save_model_root", type=str, default="/Bean/log/kkang/EditingLayerSynthesis/train_flux_network")
    parser.add_argument("--save_image_root", type=str, default="/home/kkang831/log/EditingLayerSynthesis/train_flux_network")
    parser.add_argument("--save_model_root", type=str, default="/home/kkang831/log/EditingLayerSynthesis/train_flux_network")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained LoRA.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--use_lora_bias", action="store_true", help="If training the bias of lora_B layers.")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )
    parser.add_argument(
        "--gaussian_init_lora",
        action="store_true",
        help="If using the Gaussian init strategy. When False, we follow the original LoRA init strategy.",
    )
    parser.add_argument("--train_norm_layers", action="store_true", help="Whether to train the norm scales.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument("--log_dataset_samples", action="store_true", help="Whether to log somple dataset samples.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="flux_train_control_lora",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--jsonl_for_train",
        type=str,
        default=None,
        help="Path to the jsonl file containing the training data.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the guidance scale used for transformer.",
    )

    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoders to CPU when they are not used.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class TrainRemovalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 resolution,
                 preprocess_fn=None):
        self.root = data_root
        self.maskroot = os.path.join(self.root, 'object_masks')
        self.image_with_shadow_root = os.path.join(self.root, 'shadow_imgs')
        self.image_without_shadow_root = os.path.join(self.root, 'shadowfree_imgs')
        
        mask_name_txt = os.path.join(self.root, 'TrainLabel.txt')
        with open(mask_name_txt, 'r') as f:
            mask_name = f.readlines()
        self.mask_name = [x.strip() for x in mask_name]
        self._length = len(self.mask_name)

        # Store transformation function
        self.resolution = resolution
        self.preprocess_fn = preprocess_fn
        # print(self.mask_path[1])
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        mask = Image.open(os.path.join(self.maskroot, self.mask_name[index])).convert('L') # load the binary mask
        image_with_shadow = Image.open(os.path.join(self.image_with_shadow_root, self.mask_name[index])).convert('RGB') # load the target image
        image_without_shadow = Image.open(os.path.join(self.image_without_shadow_root, self.mask_name[index])).convert('RGB') # load the target image
        
        sample = {
            "masks": mask,
            "image_with_shadow": image_with_shadow,
            "image_without_shadow": image_without_shadow
        }
        
        # Apply the transformation if available
        if self.preprocess_fn:
            sample = self.preprocess_fn(sample)  # apply the preprocess_fn
        
        return sample


def get_train_dataset(args, accelerator):
    dataset = TrainRemovalDataset(args.train_data_dir, resolution=args.resolution, preprocess_fn=None)

    with accelerator.main_process_first():
        # train_dataset = dataset.shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            dataset = dataset.select(range(args.max_train_samples))
    return dataset


def prepare_train_dataset(dataset, accelerator):
    image_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    
    mask_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        image_with_shadow = examples["image_with_shadow"].convert("RGB") if not isinstance(examples["image_with_shadow"], str) else Image.open(examples["image_with_shadow"]).convert("RGB")
        image_with_shadow = image_transforms(image_with_shadow) 

        masks = examples["masks"].convert("L") if not isinstance(examples["masks"], str) else Image.open(examples["masks"]).convert("L")
        masks = mask_transforms(masks) 
        
        image_without_shadow = examples["image_without_shadow"].convert("RGB") if not isinstance(examples["image_without_shadow"], str) else Image.open(examples["image_without_shadow"]).convert("RGB")
        image_without_shadow = image_transforms(image_without_shadow)
        
        image_foreground = image_with_shadow.clone()
        image_foreground[(masks < 0.5).repeat(3, 1, 1)] = -1
        
        examples["image_with_shadow"] = image_with_shadow
        examples["image_without_shadow"] = image_without_shadow
        # examples["masked_images"] = masked_images
        examples["image_foreground"] = image_foreground
        examples["masks"] = masks
        return examples

    # Update dataset with the transformation function
    dataset.preprocess_fn = preprocess_train
    return dataset


def collate_fn(examples):
    image_with_shadow = torch.stack([example["image_with_shadow"] for example in examples])
    image_with_shadow = image_with_shadow.to(memory_format=torch.contiguous_format).float()
    image_without_shadow = torch.stack([example["image_without_shadow"] for example in examples])
    image_without_shadow = image_without_shadow.to(memory_format=torch.contiguous_format).float()
    # masked_images = torch.stack([example["masked_images"] for example in examples])
    # masked_images = masked_images.to(memory_format=torch.contiguous_format).float()
    image_foreground = torch.stack([example["image_foreground"] for example in examples])
    image_foreground = image_foreground.to(memory_format=torch.contiguous_format).float()
    masks = torch.stack([example["masks"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()
    # save_dir = '/home/yinzijin/BrushNet-main/examples/brushnet/image_temp'
    # os.makedirs(save_dir, exist_ok=True)  

    # for i, example in enumerate(examples):
    #     pixel_values = example['pixel_values']
    #     conditioning_pixel_values = example['conditioning_pixel_values']
    #     mask = example['masks']
    #     save_image(pixel_values, os.path.join(save_dir, f'pixel_values_{i}.png'), normalize=True)
    #     save_image(conditioning_pixel_values, os.path.join(save_dir, f'conditioning_pixel_values_{i}.png'), normalize=True)
    #     save_image(mask, os.path.join(save_dir, f'mask_{i}.png'), normalize=False)
    
    # assert False
    
    return {"image_with_shadow": image_with_shadow, 
            "image_without_shadow": image_without_shadow, 
            # "masked_images": masked_images, 
            "image_foreground": image_foreground,
            "masks": masks}


def main(args):
    
    #---------------
    # Define save directory & accelerator
    from datetime import datetime
    now = datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')

    save_root = args.save_root
    if args.save_image_root is not None:
        save_image_root = args.save_image_root
    else:
        save_image_root = save_root
    if args.save_model_root is not None:
        save_model_root = args.save_model_root
    else:
        save_model_root = save_root

    exp_name = f'ShadowGen'
    if args.exp_name is not None:
        exp_name = exp_name + '_' + args.exp_name
    
    save_image_dir = f'{save_image_root}/{exp_name}_{__file__[-6:-3]}_{now}'
    save_model_dir = f'{save_model_root}/{exp_name}_{__file__[-6:-3]}_{now}/model'
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

    import shutil
    shutil.copy(__file__, os.path.join(save_image_dir, os.path.basename(__file__)))
    shutil.copy(__file__, os.path.join(save_model_dir, os.path.basename(__file__)))
    
    import json
    with open(os.path.join(save_image_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(save_model_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    project_dir = save_model_dir
    logging_dir = save_image_dir

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load models. We will load the text encoders later in a pipeline to compute
    # embeddings.
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    logger.info("All models loaded successfully")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)

    # cast down and move to the CPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # let's not move the VAE to the GPU yet.
    vae.to(dtype=torch.float32)  # keep the VAE in float32.
    flux_transformer.to(dtype=weight_dtype, device=accelerator.device)

    # enable image inputs
    with torch.no_grad():
        initial_input_channels = flux_transformer.config.in_channels
        new_linear = torch.nn.Linear(
            flux_transformer.x_embedder.in_features*4,
            flux_transformer.x_embedder.out_features,
            bias=flux_transformer.x_embedder.bias is not None,
            dtype=flux_transformer.dtype,
            device=flux_transformer.device,
        )
        new_linear.weight.zero_()
        new_linear.weight[:, :initial_input_channels].copy_(flux_transformer.x_embedder.weight)
        if flux_transformer.x_embedder.bias is not None:
            new_linear.bias.copy_(flux_transformer.x_embedder.bias)
        flux_transformer.x_embedder = new_linear

    # assert torch.all(flux_transformer.x_embedder.weight[:, initial_input_channels:].data == 0)
    flux_transformer.register_to_config(in_channels=initial_input_channels*4, out_channels=initial_input_channels)

    if args.train_norm_layers:
        for name, param in flux_transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad = True

    if args.lora_layers is not None:
        if args.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
            # add the input layer to the mix.
            if "x_embedder" not in target_modules:
                target_modules.append("x_embedder")
        elif args.lora_layers == "all-linear":
            target_modules = set()
            for name, module in flux_transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
    else:
        target_modules = [
            "x_embedder",
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        # lora_bias=args.use_lora_bias,
    )
    flux_transformer.add_adapter(transformer_lora_config)

    if args.pretrained_lora_path:
        logger.info(f"Loading from pretrained LoRA checkpoint {args.pretrained_lora_path}")
        
        lora_state_dict = load_file(args.pretrained_lora_path)
        transformer_lora_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and "lora" in k
        }
        incomatible_keys = set_peft_model_state_dict(
            flux_transformer, transformer_lora_state_dict, adapter_name="default"
        )
        if incomatible_keys is not None:
            # check
            unexpected_keys = getattr(incomatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, save_model_dir):
            if accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        model = unwrap_model(model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                        if args.train_norm_layers:
                            transformer_norm_layers_to_save = {
                                f"transformer.{name}": param
                                for name, param in model.named_parameters()
                                if any(k in name for k in NORM_LAYER_PREFIXES)
                            }
                            transformer_lora_layers_to_save = {
                                **transformer_lora_layers_to_save,
                                **transformer_norm_layers_to_save,
                            }
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                FluxControlOursPipeline.save_lora_weights(
                    save_model_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            transformer_ = None

            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(unwrap_model(flux_transformer))):
                        transformer_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
            else:
                transformer_ = FluxTransformer2DModel.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                ).to(accelerator.device, weight_dtype)

                # Handle input dimension doubling before adding adapter
                with torch.no_grad():
                    initial_input_channels = transformer_.config.in_channels # 64
                    new_linear = torch.nn.Linear(
                        transformer_.x_embedder.in_features*4,
                        transformer_.x_embedder.out_features,
                        bias=transformer_.x_embedder.bias is not None,
                        dtype=transformer_.dtype,
                        device=transformer_.device,
                    )
                    new_linear.weight.zero_()
                    new_linear.weight[:, :initial_input_channels].copy_(flux_transformer.x_embedder.weight)
                    if transformer_.x_embedder.bias is not None:
                        new_linear.bias.copy_(transformer_.x_embedder.bias)
                    transformer_.x_embedder = new_linear
                    transformer_.register_to_config(in_channels=initial_input_channels*4)

                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = FluxControlOursPipeline.lora_state_dict(input_dir)
            transformer_lora_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and "lora" in k
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_lora_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
            if args.train_norm_layers:
                transformer_norm_state_dict = {
                    k: v
                    for k, v in lora_state_dict.items()
                    if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
                }
                transformer_._transformer_norm_layers = FluxControlOursPipeline._load_norm_into_transformer(
                    transformer_norm_state_dict,
                    transformer=transformer_,
                    discard_original_layers=False,
                )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if args.mixed_precision == "fp16":
                models = [transformer_]
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [flux_transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimization parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))
    optimizer = optimizer_class(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare dataset and dataloader.
    train_dataset = get_train_dataset(args, accelerator)
    train_dataset = prepare_train_dataset(train_dataset, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Create a pipeline for text encoding. We will move this pipeline to GPU/CPU as needed.
    text_encoding_pipeline = FluxControlOursPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(save_model_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(save_model_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
        

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        flux_transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                # Convert images to latent space
                # vae encode
                image_with_shadow_latents = encode_images(batch["image_with_shadow"], vae.to(accelerator.device), weight_dtype) 
                image_without_shadow_latents = encode_images(batch["image_without_shadow"], vae.to(accelerator.device), weight_dtype) 
                image_foreground_latents = encode_images(batch["image_foreground"], vae.to(accelerator.device), weight_dtype) # 1,16,128,128
                # pixel_masked_image_latents = encode_images(
                #     batch["masked_images"], vae.to(accelerator.device), weight_dtype
                # )

                # masks = batch["masks"][:, 0, :, :]
                # masks = masks.view(
                #     B, 
                #     2*(int(H) // (vae_scale_factor * 2)), 
                #     vae_scale_factor, 
                #     2*(int(W) // (vae_scale_factor * 2)), 
                #     vae_scale_factor
                # )
                # masks = masks.permute(0, 2, 4, 1, 3)
                # masks = masks.reshape(
                #     masks.shape[0], 
                #     vae_scale_factor * vae_scale_factor, 
                #     2*(int(H) // (vae_scale_factor * 2)), 
                #     2*(int(W) // (vae_scale_factor * 2))
                # ).to(weight_dtype)
                
                latent_gt = image_with_shadow_latents
                latent_condition_1 = image_without_shadow_latents
                latent_condition_2 = image_foreground_latents

                masks = torch.nn.functional.interpolate(
                    batch["masks"], size=(batch["masks"].shape[2] // vae_scale_factor , batch["masks"].shape[3] // vae_scale_factor)
                    ).to(weight_dtype)
                masks = masks.repeat(1, latent_gt.shape[1], 1, 1)
                if args.offload:
                    # offload vae to CPU.
                    vae.cpu()

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                bsz = latent_gt.shape[0]
                noise = torch.randn_like(latent_gt, device=accelerator.device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latent_gt.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=latent_gt.ndim, dtype=latent_gt.dtype)
                noisy_model_input = (1.0 - sigmas) * latent_gt + sigmas * noise
                # Concatenate across channels.
                # Question: Should we concatenate before adding noise?
                concatenated_noisy_model_input = torch.cat([noisy_model_input, 
                                                            latent_condition_1,
                                                            latent_condition_2,
                                                            masks], dim=1)
                # print(concatenated_noisy_model_input.shape)
                # pack the latents.
                packed_noisy_model_input = FluxControlOursPipeline._pack_latents( # 1,4096,256
                    concatenated_noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=concatenated_noisy_model_input.shape[1],
                    height=concatenated_noisy_model_input.shape[2],
                    width=concatenated_noisy_model_input.shape[3],
                )
                # print(packed_noisy_model_input.shape)
                # latent image ids for RoPE.
                latent_image_ids = FluxControlOursPipeline._prepare_latent_image_ids(
                    bsz,
                    concatenated_noisy_model_input.shape[2] // 2,
                    concatenated_noisy_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # handle guidance
                if unwrap_model(flux_transformer).config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=noisy_model_input.device,
                        dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                # text encoding.
                captions = "Realistic image with natural shadow"
                text_encoding_pipeline = text_encoding_pipeline.to("cuda")
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                        captions, prompt_2=None
                    )
                # this could be optimized by not having to do any text encoding and just
                # doing zeros on specified shapes for `prompt_embeds` and `pooled_prompt_embeds`
                if args.proportion_empty_prompts and random.random() < args.proportion_empty_prompts:
                    prompt_embeds.zero_()
                    pooled_prompt_embeds.zero_()
                if args.offload:
                    text_encoding_pipeline = text_encoding_pipeline.to("cpu")

                # Predict.
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxControlOursPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow-matching loss
                target = noise - latent_gt
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = flux_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(save_model_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(save_model_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(save_model_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 1:
                        log_validation(
                            flux_transformer=flux_transformer,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                            save_model_dir=save_model_dir,
                            save_image_dir=save_image_dir,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        flux_transformer = unwrap_model(flux_transformer)
        if args.upcast_before_saving:
            flux_transformer.to(torch.float32)
        transformer_lora_layers = get_peft_model_state_dict(flux_transformer)
        if args.train_norm_layers:
            transformer_norm_layers = {
                f"transformer.{name}": param
                for name, param in flux_transformer.named_parameters()
                if any(k in name for k in NORM_LAYER_PREFIXES)
            }
            transformer_lora_layers = {**transformer_lora_layers, **transformer_norm_layers}
        FluxControlOursPipeline.save_lora_weights(
            save_directory=save_model_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        del flux_transformer
        del text_encoding_pipeline
        del vae
        # free_memory()

        # Run a final round of validation.
        image_logs = None
        log_validation(
            flux_transformer=None,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            step=global_step,
            is_final_validation=True,
            save_model_dir=save_model_dir,
            save_image_dir=save_image_dir,
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('DONE')