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
        config["min_bucket_reso"] = 256  # Slightly lower min to match reduced res
        config["mixed_precision"] = "bf16"
        config["num_batches_per_epoch"] = 43
        config["multires_noise_discount"] = 0.3
        config["ip_noise_gamma_random_strength"] = False
        config["epoch"] = 10
        config["loss_type"] = "l2"
        config["noise_offset_random_strength"] = False
        config["full_bf16"] = True
        config["lowram"] = False
        config["caption_dropout_every_n_epochs"] = 0
        config["lr_warmup_steps"] = 0
        config["steps"] = 430
        config["caption_tag_dropout_rate"] = 0.0
        config["ip_noise_gamma"] = None
        config["noise_offset"] = None
        config["adaptive_noise_scale"] = None
        config["num_epochs"] = 10
        config["debiased_estimation"] = False
        config["huber_schedule"] = "snr"
        config["caption_dropout_rate"] = 0.0
        config["network_args"] = ["conv_dim=8", "conv_alpha=8", "dropout=0.1"]
        config["multires_noise_iterations"] = None
        config["network_dropout"] = None
        config["shuffle_caption"] = True
        config["weighted_captions"] = False
        config["keep_tokens"] = 1
        config["zero_terminal_snr"] = False

        config["no_metadata"] = True
        config["async_upload"] = True
        config["bucket_no_upscale"] = True
        config["bucket_reso_steps"] = 64
        config["cache_latents"] = True
        config["cache_latents_to_disk"] = True
        config["caption_extension"] = ".txt"
        config["clip_skip"] = 1
        config["dynamo_backend"] = "no"
        config["gradient_accumulation_steps"] = 4
        config["gradient_checkpointing"] = False
        config["huber_c"] = 0.05
        config["huggingface_path_in_repo"] = "checkpoint"
        config["huggingface_repo_id"] = ""
        config["huggingface_repo_type"] = "model"
        config["huggingface_repo_visibility"] = "public"
        config["huggingface_token"] = ""
        config["learning_rate"] = 0.0003
        config["lr_scheduler"] = "cosine"
        config["lr_scheduler_args"] = []
        config["lr_scheduler_num_cycles"] = 3
        config["lr_scheduler_power"] = 1
        config["max_bucket_reso"] = 1024
        config["max_data_loader_n_workers"] = 0
        config["max_grad_norm"] = 1
        config["max_timestep"] = 1000
        config["max_token_length"] = 128
        config["max_train_steps"] = 800
        config["min_snr_gamma"] = 5
        config["network_alpha"] = 16
        config["network_dim"] = 32
        config["no_half_vae"] = True
        config["noise_offset_type"] = "Original"
        config["optimizer_args"] = ["weight_decay=0.01", "betas=(0.9,0.99)"]
        config["optimizer_type"] = "Lion"
        config["prior_loss_weight"] = 1
        config["sample_sampler"] = "euler_a"
        config["save_every_n_epochs"] = 10
        config["save_precision"] = "bf16"
        config["scale_weight_norms"] = 5
        config["text_encoder_lr"] = 0.000003
        config["train_batch_size"] = 2
        config["unet_lr"] = 0.00003
        config["xformers"] = True

    elif model_type.lower() == "flux":
        config["resolution"] = "1024,1024"
        config["gradient_checkpointing"] = True
        config["learning_rate"] = 1e-5
        config["lr_warmup_steps"] = 0 
        config["max_grad_norm"] = 1.0

        # Precision — bf16 is good, but enable more efficiencies
        config["mixed_precision"] = "bf16"
        config["full_bf16"] = True
        config["highvram"] = True  # Enable to fully utilize 80GB VRAM
        config["clip_skip"] = 1
    
        # Loss/noise (unchanged, as these don't heavily impact VRAM)
        config["loss_type"] = "l2"
        config["model_prediction_type"] = "raw"
        config["timestep_sampling"] = "sigmoid"
        config["ip_noise_gamma_random_strength"] = False
        config["min_snr_gamma"] = None
        config["noise_offset"] = None
        config["multires_noise_discount"] = 0.3
        config["multires_noise_iterations"] = 6

        # Captions (unchanged)
        config["caption_extension"] = ".txt"
        config["shuffle_caption"] = False
        config["weighted_captions"] = False
        config["keep_tokens"] = 0
        config["caption_dropout_rate"] = 0
        config["caption_dropout_every_n_epochs"] = 0


        config["no_metadata"] = True
        config["apply_t5_attn_mask"] = True
        config["bucket_no_upscale"] = True
        config["bucket_reso_steps"] = 64
        config["cache_latents"] = True
        config["cache_latents_to_disk"] = True
        config["discrete_flow_shift"] = 3.1582
        config["gradient_accumulation_steps"] = 2
        config["guidance_scale"] = 80.0
        config["huber_c"] = 0.1
        config["huber_scale"] = 1
        config["lr_scheduler"] = "cosine"
        config["lr_scheduler_num_cycles"] = 1
        config["lr_scheduler_power"] = 1
        config["max_bucket_reso"] = 2048
        config["max_data_loader_n_workers"] = 0
        config["max_timestep"] = 1000
        config["max_train_steps"] = 240
        config["mem_eff_save"] = True
        config["min_bucket_reso"] = 256
        config["network_alpha"] = 128
        config["network_args"] = [
            "train_double_block_indices=all",
            "train_single_block_indices=all",
            "train_t5xxl=True",
        ]
        config["network_dim"] = 128
        config["noise_offset_type"] = "Original"
        config["optimizer_args"] = ["weight_decay=0.01", "betas=(0.9,0.99)"]
        config["optimizer_type"] = "AdamW8bit"
        config["prior_loss_weight"] = 1
        config["sample_sampler"] = "euler_a"
        config["save_every_n_epochs"] = 10
        config["save_precision"] = "float"
        config["t5xxl_max_token_length"] = 512
        config["text_encoder_lr"] = 0.000003
        config["train_batch_size"] = 4
        config["unet_lr"] = 0.00003
        config["vae_batch_size"] = 4
        config["xformers"] = True
    
    
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
