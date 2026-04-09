"""Train LoRA adapter for H&E → TIL overlay generation.

Adapted from train_lora.py for the TIL overlay task.
Fine-tunes PixCell-1024 cross-attention layers to generate
TIL overlay images conditioned on UNI2-h embeddings.

Usage:
  accelerate launch --num_processes 1 \
    virtual_staining/train_til_lora.py \
    --root_dir /path/to/lora_training/ \
    --task blend \
    --train_batch_size 2 \
    --num_epochs 10 \
    --gradient_accumulation_steps 2
"""

import os
import subprocess
import torch
import einops

from peft import LoraConfig
from pixcell_transformer_2d_lora import PixCellTransformer2DModelLoRA
from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from til_dataset import TILOverlayDataset
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from tqdm.auto import tqdm

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for H&E → TIL overlay")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to extracted training data")
    parser.add_argument("--task", type=str, choices=["blend", "classmap", "probmap", "gtt"], default="gtt")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default=None,
                        help="Set to 'bf16' for faster training on supported hardware")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./training_output")
    parser.add_argument("--uncond_prob", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=250,
                        help="Save checkpoint every N steps (0 = epoch-end only)")
    parser.add_argument("--eval_every", type=int, default=250,
                        help="Run eval every N steps (0 = disabled)")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Path to eval patches dir (required if eval_every > 0)")
    parser.add_argument("--eval_output_dir", type=str, default="./eval_grids")
    parser.add_argument("--smoke_test_step", type=int, default=5,
                        help="Run eval at this step to verify plumbing (0 = skip)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load UNI2-h
    print("Loading UNI2-h encoder...")
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval()

    # Load VAE and scheduler
    print("Loading SD3.5 VAE and scheduler...")
    sd3_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    scheduler = DPMSolverMultistepScheduler.from_pretrained("StonyBrook-CVLab/PixCell-1024", subfolder="scheduler")

    # Create transformer with LoRA
    print("Loading PixCell-1024 transformer...")
    config = {
        "_class_name": "PixCellTransformer2DModel",
        "_diffusers_version": "0.32.2",
        "_name_or_path": "pixart_1024/transformer",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 72,
        "attention_type": "default",
        "caption_channels": 1536,
        "caption_num_tokens": 16,
        "cross_attention_dim": 1152,
        "dropout": 0.0,
        "in_channels": 16,
        "interpolation_scale": 2,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "norm_num_groups": 32,
        "norm_type": "ada_norm_single",
        "num_attention_heads": 16,
        "num_embeds_ada_norm": 1000,
        "num_layers": 28,
        "out_channels": 32,
        "patch_size": 2,
        "sample_size": 128,
        "upcast_attention": False,
        "use_additional_conditions": False,
    }
    lora_transformer = PixCellTransformer2DModelLoRA(**config)

    # Load pretrained weights
    ckpt_path = hf_hub_download(
        repo_id="StonyBrook-CVLab/PixCell-1024",
        filename="transformer/diffusion_pytorch_model.safetensors",
        local_dir="downloads/",
    )
    lora_transformer.load_state_dict(load_file(ckpt_path), strict=False)

    # Add LoRA to cross-attention layers
    target_modules = [
        "attn2.add_k_proj",
        "attn2.add_q_proj",
        "attn2.add_v_proj",
        "attn2.to_add_out",
        "attn2.to_k",
        "attn2.to_out.0",
        "attn2.to_q",
        "attn2.to_v",
    ]
    rank = 16  # Higher rank for larger H&E → gtt transformation
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    lora_transformer.add_adapter(transformer_lora_config)

    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            lora_transformer.load_state_dict(ckpt["model_state_dict"], strict=False)
            start_epoch = ckpt.get("epoch", 0)
            start_step = ckpt.get("step", 0)
            print(f"  Resuming from epoch {start_epoch}, step {start_step}")
        else:
            lora_transformer.load_state_dict(ckpt, strict=False)
            print("  Loaded model weights (no epoch/step info)")

    # Training config
    vae_scale = sd3_vae.config.scaling_factor
    vae_shift = getattr(sd3_vae.config, "shift_factor", 0)

    # Dataset
    print(f"Loading TIL overlay dataset (task={args.task})...")
    dataset = TILOverlayDataset(
        root_dir=args.root_dir,
        split="train",
        task=args.task,
    )
    train_dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4
    )

    # Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
        project_dir=os.path.join(args.output_dir, "logs"),
        kwargs_handlers=[ddp_kwargs],
    )

    vae = sd3_vae
    noise_scheduler = scheduler

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.requires_grad_(False)

    # Trainable: only LoRA params
    lora_parameters = list(filter(lambda p: p.requires_grad, lora_transformer.parameters()))
    print(f"Trainable LoRA params: {sum(p.numel() for p in lora_parameters):,d}")

    optimizer = torch.optim.AdamW([
        {"params": lora_parameters, "lr": args.learning_rate},
    ])

    # Sanity check
    assert not lora_transformer.transformer_blocks[0].attn2.to_q.base_layer.weight.requires_grad
    assert lora_transformer.transformer_blocks[0].attn2.to_q.lora_A.default.weight.requires_grad

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=None,
        num_cycles=1,
        power=0,
    )

    lora_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_transformer, optimizer, train_dataloader, lr_scheduler
    )
    vae = accelerator.prepare_model(vae, evaluation_mode=True)
    uni_model = accelerator.prepare_model(uni_model, evaluation_mode=True)

    # Restore optimizer state after accelerator.prepare
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"  Restored optimizer state")

    global_step = 0
    for epoch in range(start_epoch, args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps already completed when resuming
            if epoch == start_epoch and step < start_step:
                if step % 100 == 0:
                    progress_bar.update(min(100, start_step - step))
                continue
            with accelerator.accumulate(lora_transformer):
                he, target = batch
                bs = target.shape[0]

                # Extract UNI features from H&E INPUT (not target!)
                # This ensures conditioning matches inference (both use H&E)
                uni_patches = einops.rearrange(he, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4)
                uni_input = uni_transform(uni_patches)
                with torch.inference_mode():
                    uni_emb_target = uni_model(uni_input.to(lora_transformer.device))
                uni_emb_target = uni_emb_target.unsqueeze(0).reshape(bs, 16, -1)

                # Encode TARGET images via VAE
                target_norm = (2 * (target - 0.5)).to(dtype=vae.dtype)
                target_latents = vae.encode(target_norm.to(lora_transformer.device)).latent_dist.sample()
                target_latents = (target_latents - vae_shift) * vae_scale

                # Add noise
                t = torch.randint(0, 1000, (bs,), device="cpu", dtype=torch.int64)
                atbar = noise_scheduler.alphas_cumprod[t].view(bs, 1, 1, 1).to(lora_transformer.device)
                epsilon = torch.randn_like(target_latents)
                noisy_latents = torch.sqrt(atbar) * target_latents + torch.sqrt(1 - atbar) * epsilon

                current_timestep = t.clone().to(lora_transformer.device)

                # Random unconditional dropout (classifier-free guidance)
                if args.uncond_prob > 0:
                    uncond = lora_transformer.caption_projection.uncond_embedding.clone().tile(
                        uni_emb_target.shape[0], 1, 1
                    )
                    mask = (torch.rand((bs, 1, 1), device=lora_transformer.device) < args.uncond_prob).float()
                    uni_emb_target = (
                        (1 - mask) * uni_emb_target.to(lora_transformer.device) + mask * uncond
                    )

                # Predict noise
                epsilon_pred = lora_transformer(
                    noisy_latents,
                    encoder_hidden_states=uni_emb_target.to(lora_transformer.device),
                    timestep=current_timestep,
                    return_dict=False,
                )[0]

                # Denoising loss
                loss = ((epsilon_pred[:, :16, :, :] - epsilon) ** 2).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Determine if we should save/eval
                should_save = args.save_every > 0 and global_step % args.save_every == 0
                is_smoke_test = args.smoke_test_step > 0 and global_step == args.smoke_test_step
                should_eval = (
                    args.eval_every > 0 and args.eval_dir and
                    (global_step % args.eval_every == 0 or is_smoke_test)
                )

                if should_save or is_smoke_test:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        lora_unwrapped = accelerator.unwrap_model(lora_transformer)
                        save_path = os.path.join(
                            args.save_dir,
                            f"til_{args.task}_lora_e{epoch}_s{global_step}.pth",
                        )
                        torch.save({
                            "model_state_dict": lora_unwrapped.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "step": step + 1,
                            "global_step": global_step,
                        }, save_path)
                        label = "SMOKE TEST checkpoint" if is_smoke_test else "Checkpoint"
                        print(f"\n  {label} saved: {save_path}", flush=True)

                        # Run eval inline (pauses training ~2-3 min for 17 patches)
                        if should_eval:
                            eval_script = os.path.join(
                                os.path.dirname(__file__), "eval_til_lora.py"
                            )
                            eval_cmd = [
                                "python", eval_script,
                                "--checkpoint", save_path,
                                "--eval_dir", args.eval_dir,
                                "--output_dir", args.eval_output_dir,
                                "--task", args.task,
                            ]
                            print(f"  Running eval...", flush=True)
                            result = subprocess.run(eval_cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                for line in result.stdout.strip().split("\n")[-5:]:
                                    print(f"    {line}", flush=True)
                                # Run reference comparison
                                step_label = f"e{epoch}_s{global_step}"
                                grid_path = os.path.join(
                                    args.eval_output_dir,
                                    f"eval_til_{args.task}_lora_{step_label}.jpg"
                                )
                                ref_script = os.path.join(
                                    os.path.dirname(__file__), "compare_with_reference.py"
                                )
                                if os.path.exists(grid_path) and os.path.exists(ref_script):
                                    ref_cmd = [
                                        "python", ref_script,
                                        "--eval_grid", grid_path,
                                        "--manifest", os.path.join(args.eval_dir, "manifest.csv"),
                                        "--output_dir", os.path.join(args.eval_output_dir, "reference"),
                                        "--step_label", step_label,
                                    ]
                                    print(f"  Running reference comparison...", flush=True)
                                    ref_result = subprocess.run(ref_cmd, capture_output=True, text=True)
                                    if ref_result.returncode == 0:
                                        for line in ref_result.stdout.strip().split("\n")[-8:]:
                                            print(f"    {line}", flush=True)
                                    else:
                                        print(f"  Reference comparison failed", flush=True)
                            else:
                                print(f"  Eval FAILED (rc={result.returncode})", flush=True)
                                for line in result.stderr.strip().split("\n")[-5:]:
                                    print(f"    {line}", flush=True)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        # Save checkpoint each epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            lora_unwrapped = accelerator.unwrap_model(lora_transformer)
            save_path = os.path.join(args.save_dir, f"til_{args.task}_lora_{epoch+1}.pth")
            torch.save({
                "model_state_dict": lora_unwrapped.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "step": 0,
                "global_step": global_step,
            }, save_path)
            print(f"  Epoch {epoch+1} saved: {save_path}", flush=True)

    accelerator.end_training()
    print("Training complete!")


if __name__ == "__main__":
    main()
