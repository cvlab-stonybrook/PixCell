# Run with:
# python train_flow_mlp.py --dataset MIST --root_dir /path/to/MIST/
#  --split train --stain HER2 --device cuda:1 --train_batch_size 4 --num_epochs 1 --save_every 1

import os
import numpy as np
import torch
import einops

from resmlp import SimpleMLP
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train rectified flow between H&E and IHC UNI embeddings")
    parser.add_argument("--dataset", type=str, choices=['MIST', 'HER2Match'])
    parser.add_argument("--root_dir", type=str, default='/path/to/data/')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--stain", type=str, default='')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default='./downloads')

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load UNI
    timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval()
    uni_model.to(device)

    # Create dataloader
    if args.dataset == 'MIST':
        from mist_dataset import MISTDataset

        dataset = MISTDataset(
            root_dir=args.root_dir,
            split=args.split,
            stain=args.stain,
        )
        train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    elif args.dataset == 'HER2Match':
        from her2match_dataset import HER2MatchDataset

        dataset = HER2MatchDataset(
            root_dir=args.root_dir,
            split=args.split,
        )
        train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)

    # ResMLP
    uni_mlp = SimpleMLP(
        in_channels=1536,
        time_embed_dim=1024,
        model_channels=1024,
        bottleneck_channels=1024,
        out_channels=1536,
        num_res_blocks=6,
    ).to(device)

    print(f"MLP # params:, {sum([p.numel() for p in uni_mlp.parameters()]):,d}")
    opt = torch.optim.AdamW(uni_mlp.parameters(), lr=args.learning_rate)

    # Train
    uni_mlp.train()
    losses = []
    epochs = args.num_epochs
    for e in range(epochs):
        print(f"Epoch [{e+1}/{args.num_epochs}]")
        bar = tqdm(train_dataloader)
        for batch in bar:
            he, ihc = batch
            bs = he.shape[0]

            # Extract UNI features from H&E and IHC
            # Rearrange 1024x1024 images into 16 256x256 patches
            uni_patches_he = einops.rearrange(he, 'b c (d1 h) (d2 w) -> (b d1 d2) c h w', d1=4, d2=4)
            uni_input_he = uni_transform(uni_patches_he)

            uni_patches_ihc = einops.rearrange(ihc, 'b c (d1 h) (d2 w) -> (b d1 d2) c h w', d1=4, d2=4)
            uni_input_ihc = uni_transform(uni_patches_ihc)

            # Get embeddings
            with torch.inference_mode():
                uni_emb = uni_model(torch.cat((uni_input_he, uni_input_ihc), dim=0).to(device))
                uni_emb_he, uni_emb_ihc = torch.chunk(uni_emb, chunks=2, dim=0)
            uni_emb_he = uni_emb_he.unsqueeze(0).reshape(bs,16,-1)
            uni_emb_ihc = uni_emb_ihc.unsqueeze(0).reshape(bs,16,-1)

            # Flatten embeddings
            uni_emb_he = uni_emb_he.reshape(-1,1536)
            uni_emb_ihc = uni_emb_ihc.reshape(-1,1536)

            # Flow matching
            # 1: ihc, 0: he
            batch_size = uni_emb_he.shape[0]
            t = torch.rand((batch_size,), device=device).view(-1,1)
            xt = t*uni_emb_ihc + (1-t)*uni_emb_he
            target = uni_emb_ihc - uni_emb_he

            pred = uni_mlp(xt, 999*t.view(-1)) #, context=uni_emb_he)
            loss = ((pred - target)**2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            if len(losses) > 100:
                losses = losses[-100:]
            bar.set_postfix({'Loss': np.mean(losses)})

        if (e+1) % args.save_every == 0:
            torch.save(uni_mlp.state_dict(), os.path.join(args.save_dir, f"{args.dataset}_{args.stain}_mlp_{e+1}.pth"))


if __name__ == "__main__":
    main()
