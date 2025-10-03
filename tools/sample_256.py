import sys
sys.path.insert(0, "/home/srikarym/code/PixArt-sigma")
from uuid import uuid4
from PIL import Image
import numpy as np
import torch
from pathlib import Path
# from diffusion import DPMS
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusion.data.datasets.pan_cancer import PanCancerSingleTypeLowRes, PanCancerDataLowRes
from tqdm import tqdm
from torch.utils.data import DataLoader
import shutil
from tools.utils import build_model_new

def main(args):


    device = torch.device("cuda:0")
    batch_size = args.batch_size

    root = Path(args.workdir)
    model, vae, config = build_model_new(root, device, args.checkpoint)
    size = config.image_size
    latent_size = size // 8

    
    if args.data_source is None:
        # Use random training set embeddings to generate images
        data_kwargs = {**config.data, 'transform': None, 'return_img':True,}
        ds = PanCancerDataLowRes(**data_kwargs)
    else:
        # Generate images and compute FID for a specific source and subtype
        data_kwargs = {**config.data, 'transform': None, 'subtype': args.subtype, 'data_source': args.data_source,}
        ds = PanCancerSingleTypeLowRes(**data_kwargs)

    print(f"length of dataset: {len(ds)}")
    vae_scale = vae.config.scaling_factor
    vae_shift = vae.config.shift_factor
    ch = vae.config.latent_channels

    dl = DataLoader(ds, batch_size=batch_size, num_workers=8, shuffle=True)

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

    (out_dir / "real").mkdir(exist_ok=True, parents=True)
    (out_dir / "syn").mkdir(exist_ok=True, parents=True)


    for _ in tqdm(range(n_batches)):
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch = next(it)
            null_y = model.y_embedder.y_embedding[None].repeat(
                batch_size, 1, 1
            )[:, None]
            _,  y, _,real_arr = batch
            if len(y.shape) == 2:
                y = y.to(device).unsqueeze(1).unsqueeze(1)
            elif len(y.shape) == 3:
                y = y.to(device).unsqueeze(1)

            z = torch.randn(
                batch_size, ch, latent_size, latent_size, device=device
            )

            if args.sampling_algo in ['iddpm', 'ddim']:
                z = torch.randn(batch_size, ch, latent_size, latent_size, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([y, null_y]),
                                    cfg_scale=args.guidance_strength, data_info={}, mask=None)
                
                diffusion = IDDPM(str(args.sampling_steps))

                loop_fn = diffusion.p_sample_loop if args.sampling_algo == 'iddpm' else diffusion.ddim_sample_loop
                samples = loop_fn(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)


            elif args.sampling_algo == 'dpm-solver':
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

            elif args.sampling_algo == 'sa-solver':
                model_kwargs = dict(data_info={}, mask=None)
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=args.sampling_steps,
                    batch_size=batch_size,
                    shape=(ch, latent_size, latent_size),
                    eta=1,
                    conditioning=y,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=args.guidance_strength,
                    model_kwargs=model_kwargs,
                )[0]

            samples = (samples / vae_scale) + vae_shift
            samples_decoded = vae.decode(samples).sample
            samples_decoded = torch.clamp(samples_decoded, -1, 1)
            samples_decoded = (
                ((samples_decoded + 1) * 127.5).to(torch.uint8).cpu().numpy()
            )
        samples_decoded = np.transpose(samples_decoded, (0, 2, 3, 1))

        for syn, real in zip(samples_decoded, real_arr.numpy()):
            name = str(uuid4())[:8]
            Image.fromarray(real).save(out_dir / f"real/{name}.jpg")
            Image.fromarray(syn).save(out_dir / f"syn/{name}.jpg")


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
    parser.add_argument("--workdir", type=str, required=True,)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last_ema.ckpt")
    parser.add_argument("--data_source", type=str) # tcga_diagnostic_1024
    parser.add_argument("--subtype", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--delete_outdir", action="store_true")
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver', 'ddim'])

    args = parser.parse_args()

    main(args)
