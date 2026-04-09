"""Run PixCell LoRA inference on a whole slide (or region) and produce overlays.

Generates a PixCell overlay for every tissue window, stitches into a
low-res overview, thresholds colors to classify tumor/TIL, and optionally
compares against WSInfer patch-based results.

Usage:
  # Full slide
  python virtual_staining/infer_whole_slide.py \
    --checkpoint ~/checkpoints/til_blend_lora_e0_s1000.pth \
    --slide ~/data/slides/TCGA-B6-A0RH-....svs \
    --output_dir ~/outputs/wsi_inference/A0RH_s1000 \
    --batch_size 4

  # Region only (fast test)
  python virtual_staining/infer_whole_slide.py \
    --checkpoint ~/checkpoints/til_blend_lora_e0_s1000.pth \
    --slide ~/data/slides/TCGA-B6-A0RH-....svs \
    --output_dir ~/outputs/wsi_inference/A0RH_s1000_region \
    --region 40000,30000,50000,40000 \
    --batch_size 4

  # With WSInfer comparison
  python virtual_staining/infer_whole_slide.py \
    --checkpoint ... --slide ... --output_dir ... \
    --tumor_csv ~/data/results/model-outputs-csv/TCGA-B6-A0RH-....csv \
    --lymph_csv ~/data/results-lymphocytes/model-outputs-csv/TCGA-B6-A0RH-....csv
"""

import os
import csv
import math
import argparse
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
    """Load all models for inference."""
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

    print("Loading VAE and scheduler...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", subfolder="vae"
    ).to(device)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "StonyBrook-CVLab/PixCell-1024", subfolder="scheduler"
    )

    print("Loading PixCell-1024 + LoRA...")
    transformer = PixCellTransformer2DModelLoRA(**TRANSFORMER_CONFIG)
    ckpt_path = hf_hub_download(
        repo_id="StonyBrook-CVLab/PixCell-1024",
        filename="transformer/diffusion_pytorch_model.safetensors",
        local_dir="downloads/",
    )
    transformer.load_state_dict(load_file(ckpt_path), strict=False)

    lora_config = LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0,
        init_lora_weights="gaussian", target_modules=LORA_TARGET_MODULES,
    )
    transformer.add_adapter(lora_config)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    transformer.load_state_dict(state_dict, strict=False)
    transformer.eval().to(device)

    return uni_model, uni_transform, vae, scheduler, transformer


@torch.no_grad()
def generate_batch(
    he_batch: torch.Tensor,
    uni_model, uni_transform, vae, scheduler, transformer,
    device: torch.device,
    num_inference_steps: int = 10,
    guidance_scale: float = 1.2,
) -> np.ndarray:
    """Generate overlay images for a batch of H&E patches.

    Args:
        he_batch: (B, 3, 1024, 1024) tensor, values in [0, 1]

    Returns:
        (B, 1024, 1024, 3) uint8 numpy array
    """
    bs = he_batch.shape[0]

    # Extract UNI embeddings from H&E (condition on input, not target)
    # Rearrange each 1024x1024 into 16 256x256 patches
    uni_patches = einops.rearrange(
        he_batch, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4
    )
    uni_input = uni_transform(uni_patches)
    uni_emb = uni_model(uni_input.to(device))
    uni_emb = uni_emb.unsqueeze(0).reshape(bs, 16, -1)

    # Diffusion sampling — process one sample at a time because
    # PixCell transformer's timestep embedding doesn't support batch>1
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    uncond_single = transformer.caption_projection.uncond_embedding.clone().unsqueeze(0).to(device)

    all_gen = []
    for i in range(bs):
        emb_i = uni_emb[i:i+1]
        xt = torch.randn((1, 16, 128, 128), device=device)
        scheduler.set_timesteps(num_inference_steps, device=device)

        for tt in scheduler.timesteps:
            current_timestep = torch.tensor([tt]).to(device)
            with torch.autocast("cuda"):
                epsilon = transformer(
                    xt, encoder_hidden_states=emb_i,
                    timestep=current_timestep, return_dict=False,
                )[0][:, :16, :, :]

            if guidance_scale > 1.0:
                with torch.autocast("cuda"):
                    epsilon_uncond = transformer(
                        xt, encoder_hidden_states=uncond_single,
                        timestep=current_timestep, return_dict=False,
                    )[0][:, :16, :, :]
                epsilon = epsilon_uncond + guidance_scale * (epsilon - epsilon_uncond)

            xt = scheduler.step(epsilon, tt, xt, return_dict=False)[0]

        gen_img = vae.decode((xt / vae_scale) + vae_shift, return_dict=False)[0]
        gen_img = (0.5 * (gen_img + 1)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        all_gen.append((gen_img * 255).astype(np.uint8))

    return np.stack(all_gen)


def threshold_overlay(
    overlay: np.ndarray,
    tumor_yellow_thresh: float = 0.3,
    til_red_thresh: float = 0.3,
) -> np.ndarray:
    """Threshold a generated overlay image into tumor/TIL/other classification.

    Uses color space analysis:
    - Yellow-dominant pixels → tumor
    - Red-dominant pixels → TIL
    - Otherwise → other tissue

    Returns class map: 0=background, 1=other tissue, 2=tumor, 3=TIL
    """
    r = overlay[:, :, 0].astype(float) / 255
    g = overlay[:, :, 1].astype(float) / 255
    b = overlay[:, :, 2].astype(float) / 255

    # Yellow = high R, high G, low B
    yellowness = (r + g) / 2 - b
    # Red = high R, low G, low B
    redness = r - (g + b) / 2

    # Background detection (very light or very dark)
    brightness = (r + g + b) / 3
    is_tissue = (brightness > 0.15) & (brightness < 0.95)

    class_map = np.zeros(overlay.shape[:2], dtype=np.uint8)
    class_map[is_tissue] = 1  # other tissue
    class_map[is_tissue & (yellowness > tumor_yellow_thresh)] = 2  # tumor
    class_map[is_tissue & (redness > til_red_thresh)] = 3  # TIL

    return class_map


def stitch_overview(
    patches: list[dict],
    x_min: int, y_min: int,
    patch_size: int = 1024,
    downsample: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch patches into low-res overview images.

    Returns (he_overview, overlay_overview, classmap_overview) as uint8 arrays.
    """
    if not patches:
        empty = np.zeros((1, 1, 3), dtype=np.uint8)
        return empty, empty, empty

    xs = [p["wx"] for p in patches]
    ys = [p["wy"] for p in patches]
    x_max = max(xs) + patch_size
    y_max = max(ys) + patch_size

    ow = (x_max - x_min) // downsample
    oh = (y_max - y_min) // downsample
    small_ps = patch_size // downsample

    he_canvas = np.full((oh, ow, 3), 255, dtype=np.uint8)
    overlay_canvas = np.full((oh, ow, 3), 255, dtype=np.uint8)
    class_canvas = np.zeros((oh, ow), dtype=np.uint8)

    for p in patches:
        px = (p["wx"] - x_min) // downsample
        py = (p["wy"] - y_min) // downsample

        if px < 0 or py < 0 or px + small_ps > ow or py + small_ps > oh:
            continue

        # Downsample patch
        he_small = np.array(Image.fromarray(p["he"]).resize((small_ps, small_ps), Image.LANCZOS))
        ov_small = np.array(Image.fromarray(p["overlay"]).resize((small_ps, small_ps), Image.LANCZOS))

        he_canvas[py:py + small_ps, px:px + small_ps] = he_small
        overlay_canvas[py:py + small_ps, px:px + small_ps] = ov_small

        # Downsample classmap (nearest neighbor to preserve categories)
        cm_small = np.array(Image.fromarray(p["classmap"]).resize(
            (small_ps, small_ps), Image.NEAREST
        ))
        class_canvas[py:py + small_ps, px:px + small_ps] = cm_small

    return he_canvas, overlay_canvas, class_canvas


def load_wsinfer_classmap(
    tumor_csv: str, lymph_csv: str,
    x_min: int, y_min: int, x_max: int, y_max: int,
    downsample: int = 8,
    tumor_thresh: float = 0.5,
    lymph_thresh: float = 0.5,
) -> np.ndarray:
    """Build a WSInfer-based classification map for comparison."""
    # Read CSVs
    tumor_lookup = {}
    tumor_ps = 0
    with open(tumor_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c.startswith("prob_") and c != "prob_Other"]
        tumor_col = cols[0]
    with open(tumor_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y = int(float(row["minx"])), int(float(row["miny"]))
            w = int(float(row["width"]))
            tumor_ps = w
            tumor_lookup[(x, y)] = float(row[tumor_col])

    lymph_lookup = {}
    lymph_ps = 0
    with open(lymph_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = [c for c in reader.fieldnames if c.startswith("prob_") and c != "prob_Other"]
        lymph_col = cols[0]
    with open(lymph_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y = int(float(row["minx"])), int(float(row["miny"]))
            w = int(float(row["width"]))
            lymph_ps = w
            lymph_lookup[(x, y)] = float(row[lymph_col])

    ow = (x_max - x_min) // downsample
    oh = (y_max - y_min) // downsample

    canvas = np.zeros((oh, ow), dtype=np.uint8)

    # Paint tumor patches
    small_tps = max(tumor_ps // downsample, 1)
    for (tx, ty), prob in tumor_lookup.items():
        if tx < x_min or ty < y_min or tx >= x_max or ty >= y_max:
            continue
        px = (tx - x_min) // downsample
        py = (ty - y_min) // downsample
        if prob >= tumor_thresh:
            canvas[py:py + small_tps, px:px + small_tps] = 2  # tumor

    # Paint lymph patches (override where lymph is high)
    small_lps = max(lymph_ps // downsample, 1)
    for (lx, ly), prob in lymph_lookup.items():
        if lx < x_min or ly < y_min or lx >= x_max or ly >= y_max:
            continue
        px = (lx - x_min) // downsample
        py = (ly - y_min) // downsample
        if prob >= lymph_thresh:
            canvas[py:py + small_lps, px:px + small_lps] = 3  # TIL

    # Mark tissue (anything with a tumor prediction, even if below threshold)
    for (tx, ty) in tumor_lookup:
        if tx < x_min or ty < y_min or tx >= x_max or ty >= y_max:
            continue
        px = (tx - x_min) // downsample
        py = (ty - y_min) // downsample
        if canvas[py:py + small_tps, px:px + small_tps].max() == 0:
            canvas[py:py + small_tps, px:px + small_tps] = 1  # other tissue

    return canvas


def render_classmap_rgb(class_map: np.ndarray) -> np.ndarray:
    """Render a class map as RGB image."""
    h, w = class_map.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    rgb[class_map == 1] = [200, 200, 200]  # other tissue
    rgb[class_map == 2] = [255, 255, 0]    # tumor
    rgb[class_map == 3] = [255, 0, 0]      # TIL
    return rgb


def main():
    import tiffslide

    parser = argparse.ArgumentParser(description="Whole-slide PixCell LoRA inference")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--slide", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.2)
    parser.add_argument("--downsample", type=int, default=8,
                        help="Downsample factor for overview images")
    parser.add_argument("--region", type=str, default=None,
                        help="x_min,y_min,x_max,y_max — process only this region")
    parser.add_argument("--tumor_csv", type=str, default=None,
                        help="WSInfer tumor CSV for comparison")
    parser.add_argument("--lymph_csv", type=str, default=None,
                        help="WSInfer lymphocyte CSV for comparison")
    parser.add_argument("--skip_patches", action="store_true",
                        help="Don't save individual patch images (saves disk)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    uni_model, uni_transform, vae, scheduler, transformer = load_models(
        args.checkpoint, device
    )

    # Open slide
    print(f"Opening slide: {args.slide}")
    slide = tiffslide.TiffSlide(args.slide)
    slide_w, slide_h = slide.dimensions
    print(f"  Dimensions: {slide_w} x {slide_h}")

    # Determine region
    if args.region:
        x_min, y_min, x_max, y_max = [int(v) for v in args.region.split(",")]
    else:
        x_min, y_min = 0, 0
        x_max, y_max = slide_w, slide_h

    # Enumerate tissue windows
    # If we have tumor CSV, use it to identify tissue regions
    tissue_windows = []
    if args.tumor_csv:
        with open(args.tumor_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tx = int(float(row["minx"]))
                ty = int(float(row["miny"]))
                if x_min <= tx < x_max and y_min <= ty < y_max:
                    # Map to 1024x1024 grid
                    wx = (tx // args.patch_size) * args.patch_size
                    wy = (ty // args.patch_size) * args.patch_size
                    tissue_windows.append((wx, wy))
        tissue_windows = sorted(set(tissue_windows))
    else:
        # Grid all windows in region
        for wy in range(y_min, y_max - args.patch_size, args.patch_size):
            for wx in range(x_min, x_max - args.patch_size, args.patch_size):
                tissue_windows.append((wx, wy))

    print(f"  Region: ({x_min},{y_min}) to ({x_max},{y_max})")
    print(f"  Windows to process: {len(tissue_windows)}")
    est_time = len(tissue_windows) * 3 / args.batch_size / 60
    print(f"  Estimated time: ~{est_time:.0f} minutes")

    to_tensor = transforms.ToTensor()

    # Process in batches
    all_results = []
    patch_dir = args.output_dir / "patches"
    if not args.skip_patches:
        patch_dir.mkdir(exist_ok=True)

    for batch_start in tqdm(range(0, len(tissue_windows), args.batch_size),
                            desc="Generating overlays"):
        batch_coords = tissue_windows[batch_start:batch_start + args.batch_size]
        he_tensors = []
        he_arrays = []
        valid_coords = []

        for wx, wy in batch_coords:
            try:
                region = slide.read_region((wx, wy), 0,
                                           (args.patch_size, args.patch_size))
                he_arr = np.array(region.convert("RGB"))
            except Exception:
                continue

            # Skip mostly white
            if np.mean(np.mean(he_arr, axis=2) > 220) > 0.5:
                continue

            he_tensors.append(to_tensor(Image.fromarray(he_arr)))
            he_arrays.append(he_arr)
            valid_coords.append((wx, wy))

        if not he_tensors:
            continue

        he_batch = torch.stack(he_tensors)
        gen_batch = generate_batch(
            he_batch, uni_model, uni_transform, vae, scheduler, transformer,
            device, args.num_inference_steps, args.guidance_scale,
        )

        for i, (wx, wy) in enumerate(valid_coords):
            he_arr = he_arrays[i]
            gen_arr = gen_batch[i]
            class_map = threshold_overlay(gen_arr)

            result = {
                "wx": wx, "wy": wy,
                "he": he_arr,
                "overlay": gen_arr,
                "classmap": class_map,
            }
            all_results.append(result)

            if not args.skip_patches:
                Image.fromarray(gen_arr).save(
                    patch_dir / f"overlay_{wx}_{wy}.jpg", quality=90
                )

    slide.close()

    print(f"\nProcessed {len(all_results)} windows")

    # Stitch overview images
    print("Stitching overview images...")
    he_overview, overlay_overview, class_overview = stitch_overview(
        all_results, x_min, y_min, args.patch_size, args.downsample,
    )

    # Render class overview as RGB
    classmap_rgb = render_classmap_rgb(class_overview)

    Image.fromarray(he_overview).save(args.output_dir / "overview_he.jpg", quality=90)
    Image.fromarray(overlay_overview).save(args.output_dir / "overview_pixcell.jpg", quality=90)
    Image.fromarray(classmap_rgb).save(args.output_dir / "overview_pixcell_classmap.jpg", quality=90)
    print(f"  Saved overview images ({he_overview.shape[1]}x{he_overview.shape[0]})")

    # WSInfer comparison
    if args.tumor_csv and args.lymph_csv:
        print("Building WSInfer comparison map...")
        wsinfer_map = load_wsinfer_classmap(
            args.tumor_csv, args.lymph_csv,
            x_min, y_min, x_max, y_max,
            args.downsample,
        )
        wsinfer_rgb = render_classmap_rgb(wsinfer_map)
        Image.fromarray(wsinfer_rgb).save(
            args.output_dir / "overview_wsinfer_classmap.jpg", quality=90
        )

        # Side-by-side comparison image
        h = max(classmap_rgb.shape[0], wsinfer_rgb.shape[0])
        w1, w2 = classmap_rgb.shape[1], wsinfer_rgb.shape[1]
        gap = 20
        comparison = np.full((h, w1 + gap + w2, 3), 255, dtype=np.uint8)
        comparison[:classmap_rgb.shape[0], :w1] = classmap_rgb
        comparison[:wsinfer_rgb.shape[0], w1 + gap:] = wsinfer_rgb
        Image.fromarray(comparison).save(
            args.output_dir / "comparison_pixcell_vs_wsinfer.jpg", quality=90
        )
        print("  Saved comparison: PixCell (left) vs WSInfer (right)")

        # Compute agreement stats — align sizes
        min_h = min(class_overview.shape[0], wsinfer_map.shape[0])
        min_w = min(class_overview.shape[1], wsinfer_map.shape[1])
        class_overview_aligned = class_overview[:min_h, :min_w]
        wsinfer_map_aligned = wsinfer_map[:min_h, :min_w]
        both_tissue = (class_overview_aligned > 0) & (wsinfer_map_aligned > 0)
        if both_tissue.sum() > 0:
            agree = (class_overview_aligned[both_tissue] == wsinfer_map_aligned[both_tissue]).mean()
            print(f"  Pixel agreement (where both have tissue): {agree:.1%}")

            # Per-class stats
            for cls, name in [(2, "tumor"), (3, "TIL")]:
                pixcell_cls = class_overview_aligned == cls
                wsinfer_cls = wsinfer_map_aligned == cls
                intersection = (pixcell_cls & wsinfer_cls).sum()
                union = (pixcell_cls | wsinfer_cls).sum()
                iou = intersection / max(union, 1)
                print(f"  {name} IoU: {iou:.3f} (PixCell: {pixcell_cls.sum():,d}px, WSInfer: {wsinfer_cls.sum():,d}px)")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
