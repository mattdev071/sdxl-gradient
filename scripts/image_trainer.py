#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import os
import subprocess
import sys

import toml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path

def create_config(task_id, model, model_type, expected_repo_name):
    """Create the diffusion config file"""
    config_template_path = train_paths.get_image_training_config_template_path(model_type)

    with open(config_template_path, "r") as file:
        config = toml.load(file)

    # Update config
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = train_paths.get_image_training_images_dir(task_id)
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    if model_type.lower() == "sdxl":
        config["resolution"] = "1024,1024"  # SDXL native
        config["enable_bucket"] = True
        config["train_batch_size"] = 1
        config["min_bucket_reso"] = 256
        config["max_bucket_reso"] = 2048
        config["max_train_steps"] = 1
        config["mixed_precision"] = "bf16"
        config["num_batches_per_epoch"] = 43
        config["max_token_length"] = 75
        config["multires_noise_discount"] = 0.3
        config["max_grad_norm"] = 1.0
        config["full_fp16"] = False
        config["ip_noise_gamma_random_strength"] = False
        config["gradient_accumulation_steps"] = 2
        config["epoch"] = 10
        config["loss_type"] = "l2"
        config["noise_offset_random_strength"] = False
        config["lr_scheduler"] = "cosine"
        config["full_bf16"] = True
        config["min_snr_gamma"] = None
        config["lowram"] = False
        config["scale_weight_norms"] = 5
        config["unet_lr"] = 0.0001
        config["caption_dropout_every_n_epochs"] = 0
        config["cache_latents"] = True
        config["lr_warmup_steps"] = 0
        config["text_encoder_lr"] = 0.0001
        config["clip_skip"] = 1
        config["prior_loss_weight"] = 1
        config["optimizer_type"] = "AdamW"
        config["optimizer"] = "AdamW"
        config["steps"] = 430
        config["gradient_checkpointing"] = True
        config["caption_tag_dropout_rate"] = 0.0
        config["network_dim"] = 32
        config["ip_noise_gamma"] = None
        config["noise_offset"] = None
        config["adaptive_noise_scale"] = None
        config["num_epochs"] = 10
        config["debiased_estimation"] = False
        config["huber_schedule"] = "snr"
        config["caption_dropout_rate"] = 0.0
        config["network_args"] = ["conv_dim=8", "conv_alpha=8", "dropout=0.1"]
        config["huber_c"] = 0.1
        config["multires_noise_iterations"] = None
        config["network_dropout"] = None
        config["learning_rate"] = 1e-5
        config["shuffle_caption"] = True
        config["weighted_captions"] = False
        config["keep_tokens"] = 1
        config["zero_terminal_snr"] = False
        config["network_alpha"] = 32

    elif model_type.lower() == "flux":
        config["max_resolution"] = "1024,1024"
        config["resolution"] = "1024,1024"

        # Bucketing — maintain flexibility but limit max resolution
        config["enable_bucket"] = True
        config["min_bucket_reso"] = 256  # Slightly lower min to match reduced res
        config["max_bucket_reso"] = 1024  # Reduce max to save VRAM
        config["bucket_reso_steps"] = 64
        config["bucket_no_upscale"] = False
        config["loss_type"] = "l2"
        # Training params — increase batch size to leverage full 80GB VRAM
        config["train_batch_size"] = 1  # Increased from 1 to maximize VRAM usage; test stability
        config["epoch"] = 25  # Increased epochs for more thorough training on small dataset
        config["num_batches_per_epoch"] = 10
        config["max_train_steps"] = 3000  # Increased to allow more training steps
        config["logit_mean"] = 0.0
        config["gradient_accumulation_steps"] = 1  # Reduced to balance with higher batch size (effective batch size = 16)
        config["gradient_checkpointing"] = True  # Enable to trade compute time for VRAM savings

        # Learning rate & scheduler — lower LR for finer adjustments and lower loss
        config["learning_rate"] = 1e-04  # Reduced from 1e-5 to prevent overfitting on small dataset
        config["unet_lr"] = 1e-04
        config["text_encoder_lr"] = 5e-06
        config["t5xxl_lr"] = 5e-03
        config["lr_scheduler"] = "constant"
        config["lr_scheduler_num_cycles"] = 3  # Increased cycles for better annealing
        config["lr_warmup_steps"] = 0  # Slightly increased for gradual start

        # Optimizer — unchanged, but ensure stability with lower LR
        config["optimizer_type"] = "AdamW"
        config["optimizer"] = "AdamW"
        config["optimizer_args"] = ["weight_decay=0.01","betas=(0.9, 0.999)","eps=1e-8"]

        config["max_grad_norm"] = 1.0

        # LoRA settings — higher rank for more capacity on small dataset
        config["network_dim"] = 128  # Increased from 64 to capture more nuances
        config["network_alpha"] = 128  # Adjusted proportionally
        config["network_dropout"] = 0.1  # Slight increase to combat overfitting
        config["network_train_unet_only"] = False  # Train more components for better adaptation
        config["network_args"] = ["train_double_block_indices=all", "train_single_block_indices=all", "train_t5xxl=True","dropout=null"]

        # Precision — bf16 is good, but enable more efficiencies
        config["mixed_precision"] = "bf16"
        config["full_bf16"] = True
        config["highvram"] = True  # Enable to fully utilize 80GB VRAM
        config["lowram"] = False
        config["clip_skip"] = 1
        config["sdpa"] = True

        # Cache — use disk caching to offload from VRAM
        config["cache_latents"] = True
        config["cache_latents_to_disk"] = True
        config["flux1_cache_text_encoder_outputs"] = True
        config["flux1_cache_text_encoder_outputs_to_disk"] = True

        # Train text encoders — enable to improve text-image alignment
        config["clip_skip"] = 1
        config["huber_schedule"] = "snr"
        # Blocks — increase swap depth for efficiency

        # Loss/noise (unchanged, as these don't heavily impact VRAM)
        config["loss_type"] = "l2"
        config["model_prediction_type"] = "raw"
        config["discrete_flow_shift"] = 3
        config["timestep_sampling"] = "sigmoid"
        config["guidance_scale"] = 1
        config["ip_noise_gamma_random_strength"] = False
        config["min_snr_gamma"] = "None"  # Keep as None for stable training
        config["noise_offset"] = "None"
        config["multires_noise_discount"] = 0.3
        config["multires_noise_iterations"] = 6

        config["v2"] = False
        config["logit_std"] = 1.0
        config["guidance_scale"] = 1.0
        # Captions (unchanged)
        config["caption_extension"] = ".txt"
        config["shuffle_caption"] = False
        config["weighted_captions"] = False
        config["keep_tokens"] = 0
        config["caption_dropout_rate"] = 0
        config["caption_dropout_every_n_epochs"] = 0

        # Misc (unchanged)
        config["seed"] = 0
        config["max_data_loader_n_workers"] = 8
        config["persistent_data_loader_workers"] = True
        config["xformers"] = "xformers"


    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Save config to file
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    training_command = [
        "accelerate", "launch",
        "--dynamo_backend", "no",
        "--dynamo_mode", "default",
        "--mixed_precision", "bf16",
        "--num_processes", "1",
        "--num_machines", "1",
        "--num_cpu_threads_per_process", "2",
        f"/app/sd-scripts/{model_type}_train_network.py",
        "--config_file", config_path
    ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    # Create config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model_type,
        args.expected_repo_name,
    )

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
