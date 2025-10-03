import sys
sys.path.insert(0, "/home/srikarym/code/PixArt-sigma")
from uuid import uuid4
from PIL import Image
import numpy as np
import torch
from pathlib import Path
from diffusers.models import AutoencoderKL
from diffusion import DPMS
from diffusion.utils.misc import read_config
from diffusion.data.datasets.pan_cancer import PanCancerSingleType
from tqdm import tqdm
from torch.utils.data import DataLoader
import shutil
from tools.utils import build_model_new

def main(args):


    device = torch.device("cuda:0")
    batch_size = args.batch_size
    size = args.size


    root = Path(args.workdir)
    model, vae, config = build_model_new(root, device, args.checkpoint)

    size = config.image_size
    latent_size = size // 8

    data_kwargs = {**config.data, 'transform': None, 'subtype': args.subtype, 'data_source': args.data_source}
    ds = PanCancerSingleType(**data_kwargs)


    vae_scale = vae.config.scaling_factor
    vae_shift = vae.config.shift_factor
    ch = vae.config.latent_channels

    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)

    model_kwargs = dict(
        data_info={
            "img_hw": torch.tensor([size, size]),
            "aspect_ratio": torch.tensor(1),
            "mask_type": "null",
        },
        mask=None,
    )
    it = iter(dl)

    n_batches = args.n_images // batch_size

    out_dir = root / args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    for _ in tqdm(range(n_batches)):
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch = next(it)
            null_y = model.y_embedder.y_embedding[None].repeat(
                batch_size, 1, 1
            )[:, None]
            _,  y, _,_ = batch
            if len(y.shape) == 2:
                y = y.to(device).unsqueeze(1).unsqueeze(1)
            elif len(y.shape) == 3:
                y = y.to(device).unsqueeze(1)

            z = torch.randn(
                batch_size, ch, latent_size, latent_size, device=device
            )
            dpm_solver = DPMS(
                model.forward_with_dpmsolver,
                condition=y,
                uncondition=null_y,
                cfg_scale=args.guidance_strength,
                model_kwargs=model_kwargs,
            )
            samples = dpm_solver.sample(
                z,
                steps=args.sampling_steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            samples = (samples / vae_scale) + vae_shift
            samples_decoded = vae.decode(samples).sample
            samples_decoded = torch.clamp(samples_decoded, -1, 1)
            samples_decoded = (
                ((samples_decoded + 1) * 127.5).to(torch.uint8).cpu().numpy()
            )
        samples_decoded = np.transpose(samples_decoded, (0, 2, 3, 1))

        for img_ar in samples_decoded:
            img = Image.fromarray(img_ar)
            name = str(uuid4())[:8]
            img.save(out_dir / f"{name}.jpg")


    if args.delete_outdir:

        shutil.rmtree(out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/lus/grand/projects/GeomicVar/srikar/pan_cancer",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_images", type=int, default=10000)
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--guidance_strength", type=float, default=2)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_source", type=str, default="tcga_diagnostic_1024")
    parser.add_argument("--subtype", type=str, default="brca")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--fid_stats_root",
        type=str,
        default="/lus/grand/projects/GeomicVar/srikar/data/fid_stats/pan_cancer",
    )

    parser.add_argument("--delete_outdir", action="store_true")

    args = parser.parse_args()

    main(args)
