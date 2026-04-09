"""Train Flow MLP for H&E → TIL overlay embedding translation.

Adapted from train_flow_mlp.py for the TIL overlay task.
Maps UNI2-h embeddings of H&E patches to embeddings of TIL overlay patches
using rectified flow matching.

Usage:
  python virtual_staining/train_til_flow_mlp.py \
    --root_dir /path/to/lora_training/ \
    --task blend \
    --device mps \
    --train_batch_size 4 \
    --num_epochs 100 \
    --save_every 25
"""

import os
import numpy as np
import torch
import einops

from resmlp import SimpleMLP
from til_dataset import TILOverlayDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train flow MLP for H&E → TIL overlay")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to extracted training data (e.g., data/lora_training/)")
    parser.add_argument("--task", type=str, choices=["blend", "classmap"], default="blend")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

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
    uni_model.to(device)

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

    # Flow MLP
    uni_mlp = SimpleMLP(
        in_channels=1536,
        time_embed_dim=1024,
        model_channels=1024,
        bottleneck_channels=1024,
        out_channels=1536,
        num_res_blocks=6,
    ).to(device)

    print(f"MLP params: {sum(p.numel() for p in uni_mlp.parameters()):,d}")
    opt = torch.optim.AdamW(uni_mlp.parameters(), lr=args.learning_rate)

    # Train
    uni_mlp.train()
    losses = []
    for e in range(args.num_epochs):
        print(f"Epoch [{e+1}/{args.num_epochs}]")
        bar = tqdm(train_dataloader)
        for batch in bar:
            he, target = batch
            bs = he.shape[0]

            # Extract UNI features from H&E and target (overlay)
            uni_patches_he = einops.rearrange(he, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4)
            uni_input_he = uni_transform(uni_patches_he)

            uni_patches_target = einops.rearrange(target, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4)
            uni_input_target = uni_transform(uni_patches_target)

            with torch.inference_mode():
                uni_emb = uni_model(
                    torch.cat((uni_input_he, uni_input_target), dim=0).to(device)
                )
                uni_emb_he, uni_emb_target = torch.chunk(uni_emb, chunks=2, dim=0)
            uni_emb_he = uni_emb_he.unsqueeze(0).reshape(bs, 16, -1)
            uni_emb_target = uni_emb_target.unsqueeze(0).reshape(bs, 16, -1)

            # Flatten for MLP
            uni_emb_he = uni_emb_he.reshape(-1, 1536)
            uni_emb_target = uni_emb_target.reshape(-1, 1536)

            # Rectified flow matching: t=1 is target, t=0 is source
            batch_size = uni_emb_he.shape[0]
            t = torch.rand((batch_size,), device=device).view(-1, 1)
            xt = t * uni_emb_target + (1 - t) * uni_emb_he
            flow_target = uni_emb_target - uni_emb_he

            pred = uni_mlp(xt, 999 * t.view(-1))
            loss = ((pred - flow_target) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            if len(losses) > 100:
                losses = losses[-100:]
            bar.set_postfix({"Loss": np.mean(losses)})

        if (e + 1) % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"til_{args.task}_mlp_{e+1}.pth")
            torch.save(uni_mlp.state_dict(), save_path)
            print(f"  Saved: {save_path}")

    # Final save
    save_path = os.path.join(args.save_dir, f"til_{args.task}_mlp_final.pth")
    torch.save(uni_mlp.state_dict(), save_path)
    print(f"Final model saved: {save_path}")


if __name__ == "__main__":
    main()
