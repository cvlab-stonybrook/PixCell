import os
import sys
from tqdm import tqdm
sys.path.append("/home/srikarym/code/PixArt-sigma")
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from diffusers.models import AutoencoderKL
import argparse
from torchvision import transforms as T
import numpy as np
from tools.utils import DatasetForFeatureExt, my_collate
import timm
from torchvision import transforms


ds_lengths = {
    "cptac": 2_196_291,
    "gtex": 7_870_674,
    "tcga_diagnostic": 9_180_591,
    "tcga_fresh_frozen": 3_680_187,
    "sbu_olympus": 6_683_850,
    "others": 1_143_288,
}


def checkpoint_update(checkpoint_path, start_idx):
    with open(checkpoint_path, 'w') as f:
        f.write(str(start_idx))

def extract_img_vae(args):
    device = torch.device("cuda:0")
    root = Path(args.root)

    ### load VAE ###
    vae_path = root / "pretrained_models/sd-3.5-vae"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()

    ### load UNI ###

    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
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

    uni_model = timm.create_model(pretrained=False, **timm_kwargs).to(device).eval()
    local_path = root / "pretrained_models/uni2/pytorch_model.bin"
    uni_model.load_state_dict(
        torch.load(local_path, map_location="cpu"),
        strict=True,)
    
    tf_uni = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset_name = args.dataset_name
    size = args.size


    tf_vae = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
    ### Read the checkpoint ###
    # create checkpoint directory if it does not exist
    if not os.path.exists(args.checkpoint_dir):
        Path(args.checkpoint_dir).mkdir(parents=True)
    
    checkpoint_path = os.path.join(
        args.checkpoint_dir, f'{dataset_name}_{args.start_idx}_{args.end_idx}.txt'
    )
    if os.path.exists(checkpoint_path):
        # read new start_idx from checkpoint
        with open(checkpoint_path, 'r') as f:
            start_idx = int(f.read())
    else:
        start_idx = args.start_idx
        
    checkpoint_update(checkpoint_path, start_idx)

    ### load dataset, only load the indices that are needed ###


    if args.end_idx == -1:
        # if end_idx is not provided, use the full dataset
        args.end_idx = ds_lengths[dataset_name]

    indices = np.arange(start_idx, args.end_idx)


    ds = DatasetForFeatureExt(root / "patches", dataset_name, size, tf_vae, tf_uni, indices)
    dl = DataLoader(
        ds, batch_size=args.batch_size, num_workers=2, collate_fn=my_collate
    )

    for i, (img_vae, img_uni, name_arr) in enumerate(tqdm(dl, desc="Processing Batches")):
        img_vae = img_vae.to(device)
        img_uni_grid = img_uni.to(device)

        feat_uni_arr = []

        with torch.no_grad():
            posterior = vae.encode(img_vae).latent_dist

            for img_uni in img_uni_grid:
                feat_uni = (
                    uni_model(img_uni)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                feat_uni_arr.append(feat_uni)
        z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy().squeeze()
        z = z.astype(np.float16)

        feat_uni_arr = np.stack(feat_uni_arr)

        ## save the features to disk

        for name_full, ar_vae, ar_uni in zip(name_arr, z, feat_uni_arr):

            name_full = name_full.replace(".jpeg", "")
            
            name_full = Path(name_full)
            output_dir = (root / "features") / name_full.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            name = name_full.stem

            np.save(output_dir / f"{name}_sd3_vae.npy", ar_vae)
            np.save(output_dir / f"{name}_uni.npy", ar_uni)


        if (i + 1) % 10 == 0:
            checkpoint_update(checkpoint_path, start_idx + (i+1)*args.batch_size)
            tqdm.write(
                f"Processed {i+1}/{len(dl)} batches [{start_idx + (i+1)*args.batch_size}/{start_idx + len(dl)*args.batch_size} samples], starting from {args.start_idx} to {args.end_idx}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/lus/grand/projects/GeomicVar/srikar/pan_cancer",
    )
    parser.add_argument("--checkpoint_dir", type=str, default='./feature_extraction_checkpoints')
    parser.add_argument("--dataset_name", type=str, required=True, choices=ds_lengths.keys())
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    args = parser.parse_args()

    extract_img_vae(args)
