"""Train Flow MLP on pre-extracted UNI2-h embeddings (fast — no image loading).

Requires running extract_uni_embeddings.py first to cache embeddings.

Usage:
  python virtual_staining/train_til_flow_mlp_cached.py \
    --emb_dir /path/to/lora_training/embeddings/ \
    --device mps \
    --train_batch_size 256 \
    --num_epochs 100 \
    --save_dir ./checkpoints
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from resmlp import SimpleMLP


def main():
    parser = argparse.ArgumentParser(description="Train Flow MLP on cached embeddings")
    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load cached embeddings
    print("Loading cached embeddings...")
    he_embs = torch.load(os.path.join(args.emb_dir, "train_he_embs.pt"), weights_only=True)
    target_embs = torch.load(os.path.join(args.emb_dir, "train_target_embs.pt"), weights_only=True)
    print(f"  H&E: {he_embs.shape}, Target: {target_embs.shape}")

    # Flatten: (N, 16, 1536) → (N*16, 1536)
    he_flat = he_embs.reshape(-1, 1536)
    target_flat = target_embs.reshape(-1, 1536)
    print(f"  Flattened: {he_flat.shape[0]} embedding pairs")

    dataset = TensorDataset(he_flat, target_flat)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

    # Flow MLP
    mlp = SimpleMLP(
        in_channels=1536,
        time_embed_dim=1024,
        model_channels=1024,
        bottleneck_channels=1024,
        out_channels=1536,
        num_res_blocks=6,
    ).to(device)

    print(f"MLP params: {sum(p.numel() for p in mlp.parameters()):,d}")
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.learning_rate)

    mlp.train()
    losses = []
    for e in range(args.num_epochs):
        bar = tqdm(loader, desc=f"Epoch {e+1}/{args.num_epochs}")
        for he_batch, target_batch in bar:
            he_batch = he_batch.to(device)
            target_batch = target_batch.to(device)

            # Rectified flow: t=1 target, t=0 source
            batch_size = he_batch.shape[0]
            t = torch.rand((batch_size,), device=device).view(-1, 1)
            xt = t * target_batch + (1 - t) * he_batch
            flow_target = target_batch - he_batch

            pred = mlp(xt, 999 * t.view(-1))
            loss = ((pred - flow_target) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            if len(losses) > 100:
                losses = losses[-100:]
            bar.set_postfix({"Loss": f"{np.mean(losses):.4f}"})

        if (e + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"til_blend_mlp_{e+1}.pth")
            torch.save(mlp.state_dict(), path)
            print(f"  Saved: {path}")

    path = os.path.join(args.save_dir, "til_blend_mlp_final.pth")
    torch.save(mlp.state_dict(), path)
    print(f"Final: {path}")


if __name__ == "__main__":
    main()
