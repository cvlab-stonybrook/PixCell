"""Evaluate a TIL LoRA checkpoint by generating overlay images for eval patches.

Loads a checkpoint, runs diffusion inference on eval patches, and produces
a comparison grid: H&E | Ground Truth | Generated per row.

Usage:
  python virtual_staining/eval_til_lora.py \
    --checkpoint ~/checkpoints/til_blend_lora_e0_s500.pth \
    --eval_dir ~/data/eval_patches \
    --output_dir ~/outputs/eval_grids \
    --task blend
"""

import os
import csv
import torch
import einops
import numpy as np
from PIL import Image
from pathlib import Path

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from peft import LoraConfig
from pixcell_transformer_2d_lora import PixCellTransformer2DModelLoRA
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms

from tqdm.auto import tqdm
import argparse


TRANSFORMER_CONFIG = {
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

LORA_TARGET_MODULES = [
    "attn2.add_k_proj", "attn2.add_q_proj", "attn2.add_v_proj",
    "attn2.to_add_out", "attn2.to_k", "attn2.to_out.0",
    "attn2.to_q", "attn2.to_v",
]


def load_models(checkpoint_path: str, device: torch.device):
    """Load UNI2-h, VAE, scheduler, and LoRA transformer."""
    # UNI2-h
    print("Loading UNI2-h encoder...")
    timm_kwargs = {
        "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
        "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667 * 2,
        "num_classes": 0, "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU,
        "reg_tokens": 8, "dynamic_img_size": True,
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval().to(device)

    # VAE + scheduler
    print("Loading VAE and scheduler...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", subfolder="vae"
    ).to(device)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "StonyBrook-CVLab/PixCell-1024", subfolder="scheduler"
    )

    # Transformer + LoRA
    print("Loading PixCell-1024 + LoRA...")
    transformer = PixCellTransformer2DModelLoRA(**TRANSFORMER_CONFIG)

    # Load base weights
    ckpt_path = hf_hub_download(
        repo_id="StonyBrook-CVLab/PixCell-1024",
        filename="transformer/diffusion_pytorch_model.safetensors",
        local_dir="downloads/",
    )
    transformer.load_state_dict(load_file(ckpt_path), strict=False)

    # Add LoRA adapter
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0,
        init_lora_weights="gaussian", target_modules=LORA_TARGET_MODULES,
    )
    transformer.add_adapter(lora_config)

    # Load LoRA checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    transformer.load_state_dict(state_dict, strict=False)
    transformer.eval().to(device)

    step_info = ""
    if isinstance(ckpt, dict):
        step_info = f"epoch={ckpt.get('epoch', '?')}, step={ckpt.get('global_step', '?')}"

    return uni_model, uni_transform, vae, scheduler, transformer, step_info


@torch.no_grad()
def generate_image(
    he_tensor: torch.Tensor,
    gt_overlay_tensor: torch.Tensor,
    uni_model, uni_transform, vae, scheduler, transformer,
    device: torch.device,
    num_inference_steps: int = 20,
    guidance_scale: float = 1.2,
) -> np.ndarray:
    """Generate an overlay image conditioned on H&E UNI embeddings."""
    bs = 1

    # Extract UNI embeddings from H&E INPUT (not ground truth!)
    uni_patches = einops.rearrange(
        he_tensor.unsqueeze(0),
        "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4
    )
    uni_input = uni_transform(uni_patches)
    uni_emb = uni_model(uni_input.to(device))
    uni_emb = uni_emb.unsqueeze(0).reshape(bs, 16, -1)

    # Diffusion sampling
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    uncond = transformer.caption_projection.uncond_embedding.clone().tile(bs, 1, 1).to(device)

    xt = torch.randn((1, 16, 128, 128), device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)

    for tt in scheduler.timesteps:
        current_timestep = torch.tensor([tt]).to(device)
        with torch.autocast("cuda"):
            epsilon = transformer(
                xt, encoder_hidden_states=uni_emb,
                timestep=current_timestep, return_dict=False,
            )[0][:, :16, :, :]

        if guidance_scale > 1.0:
            with torch.autocast("cuda"):
                epsilon_uncond = transformer(
                    xt, encoder_hidden_states=uncond,
                    timestep=current_timestep, return_dict=False,
                )[0][:, :16, :, :]
            epsilon = epsilon_uncond + guidance_scale * (epsilon - epsilon_uncond)

        xt = scheduler.step(epsilon, tt, xt, return_dict=False)[0]

    # Decode
    gen_image = vae.decode((xt / vae_scale) + vae_shift, return_dict=False)[0]
    gen_image = (0.5 * (gen_image + 1)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (gen_image * 255).astype(np.uint8)


def make_comparison_grid(
    results: list[dict], output_path: Path, step_info: str = "",
):
    """Create a comparison grid image: H&E | Ground Truth | Generated per row."""
    n = len(results)
    patch_size = results[0]["he"].shape[0]
    col_width = patch_size
    label_height = 30
    grid_width = col_width * 3
    grid_height = n * patch_size + label_height

    grid = np.full((grid_height, grid_width, 3), 255, dtype=np.uint8)

    for i, r in enumerate(results):
        y = i * patch_size + label_height
        grid[y:y + patch_size, 0:col_width] = r["he"]
        grid[y:y + patch_size, col_width:2 * col_width] = r["gt"]
        grid[y:y + patch_size, 2 * col_width:3 * col_width] = r["gen"]

    img = Image.fromarray(grid)

    # Add labels using simple text (no PIL font dependency)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((col_width // 3, 5), "H&E", fill=(0, 0, 0))
    draw.text((col_width + col_width // 3, 5), "Ground Truth", fill=(0, 0, 0))
    draw.text((2 * col_width + col_width // 4, 5), f"Generated ({step_info})", fill=(0, 0, 0))

    img.save(output_path, quality=95)
    print(f"Saved comparison grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TIL LoRA checkpoint")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--eval_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--task", choices=["blend", "classmap", "probmap", "gtt"], default="gtt")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    uni_model, uni_transform, vae, scheduler, transformer, step_info = load_models(
        args.checkpoint, device
    )

    # Read manifest
    manifest_path = args.eval_dir / "manifest.csv"
    with open(manifest_path) as f:
        patches = list(csv.DictReader(f))

    print(f"\nRunning inference on {len(patches)} eval patches...")

    to_tensor = transforms.ToTensor()
    results = []

    for p in tqdm(patches, desc="Evaluating"):
        patch_id = p["patch_id"]

        he_path = args.eval_dir / "he" / f"{patch_id}.jpg"
        if args.task in ("probmap", "gtt"):
            gt_path = args.eval_dir / args.task / f"{patch_id}.png"
        elif args.task == "blend":
            gt_path = args.eval_dir / "overlay" / f"{patch_id}.jpg"
        else:
            gt_path = args.eval_dir / "classmap" / f"{patch_id}.jpg"

        if not he_path.exists() or not gt_path.exists():
            print(f"  Skipping {patch_id}: missing files")
            continue

        he_img = np.array(Image.open(he_path).convert("RGB"))
        gt_img = np.array(Image.open(gt_path).convert("RGB"))

        he_tensor = to_tensor(Image.open(he_path).convert("RGB"))
        gt_tensor = to_tensor(Image.open(gt_path).convert("RGB"))

        gen_img = generate_image(
            he_tensor, gt_tensor,
            uni_model, uni_transform, vae, scheduler, transformer,
            device, args.num_inference_steps, args.guidance_scale,
        )

        # Compute simple metrics
        mse = np.mean((gen_img.astype(float) - gt_img.astype(float)) ** 2)
        print(f"  {patch_id}: MSE={mse:.1f}")

        results.append({
            "patch_id": patch_id,
            "he": he_img,
            "gt": gt_img,
            "gen": gen_img,
            "mse": mse,
        })

    # Extract step number for filename
    ckpt_name = Path(args.checkpoint).stem
    grid_path = args.output_dir / f"eval_{ckpt_name}.jpg"
    make_comparison_grid(results, grid_path, step_info)

    # Print summary
    mses = [r["mse"] for r in results]
    print(f"\nMean MSE: {np.mean(mses):.1f} (std: {np.std(mses):.1f})")
    print(f"Grid saved to: {grid_path}")


if __name__ == "__main__":
    main()
