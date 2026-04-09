"""Pre-extract UNI2-h embeddings for all training patches.

Caches embeddings to disk so Flow MLP and LoRA training don't need to
run UNI2-h on every batch. This is the expensive step — run once.

Usage:
  python virtual_staining/extract_uni_embeddings.py \
    --root_dir /path/to/lora_training/ \
    --task blend \
    --output_dir /path/to/lora_training/embeddings/ \
    --device mps \
    --batch_size 16
"""

import argparse
import os
import torch
import einops
import numpy as np
from pathlib import Path
from tqdm import tqdm

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from til_dataset import TILOverlayDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Pre-extract UNI2-h embeddings")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["blend", "classmap"], default="blend")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for UNI2-h (images, not embedding batches)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load UNI2-h
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
    uni_model.eval()
    uni_model.to(device)

    for split in ["train", "val"]:
        print(f"\nProcessing {split} split...")
        dataset = TILOverlayDataset(root_dir=args.root_dir, split=split, task=args.task)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        all_he_embs = []
        all_target_embs = []

        for he, target in tqdm(loader, desc=f"Extracting {split}"):
            bs = he.shape[0]

            # Split 1024×1024 into 4×4 grid of 256×256
            he_patches = einops.rearrange(he, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4)
            target_patches = einops.rearrange(target, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4)

            he_input = uni_transform(he_patches)
            target_input = uni_transform(target_patches)

            with torch.inference_mode():
                all_input = torch.cat([he_input, target_input], dim=0).to(device)
                all_emb = uni_model(all_input)
                he_emb, target_emb = torch.chunk(all_emb, 2, dim=0)

            # Reshape: (bs*16, 1536) → (bs, 16, 1536)
            he_emb = he_emb.cpu().reshape(bs, 16, -1)
            target_emb = target_emb.cpu().reshape(bs, 16, -1)

            all_he_embs.append(he_emb)
            all_target_embs.append(target_emb)

        he_embs = torch.cat(all_he_embs, dim=0)
        target_embs = torch.cat(all_target_embs, dim=0)

        print(f"  H&E embeddings: {he_embs.shape}")
        print(f"  Target embeddings: {target_embs.shape}")

        torch.save(he_embs, os.path.join(args.output_dir, f"{split}_he_embs.pt"))
        torch.save(target_embs, os.path.join(args.output_dir, f"{split}_target_embs.pt"))
        print(f"  Saved to {args.output_dir}/{split}_*.pt")

    print("\nDone! Embeddings cached for fast MLP training.")


if __name__ == "__main__":
    main()
