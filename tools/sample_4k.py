import sys
sys.path.insert(0, "/home/srikarym/code/PixArt-sigma")
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import argparse
from tqdm import tqdm
import torch
from diffusion import IDDPM
from PIL import Image
import numpy as np
from tools.utils import build_model_new
from einops import rearrange
import matplotlib
from torchvision import transforms as T
import timm
import time
import uuid

matplotlib.rcParams.update({"font.size": 14})


def get_conditioning(i, j, embeddings, embedding_stride=32):

    i_idx = i // embedding_stride
    j_idx = j // embedding_stride
    return embeddings[i_idx : i_idx + 4, j_idx : j_idx + 4]


def gaussian_kernel(size=64, mu=0, sigma=1):
    x = torch.linspace(-1, 1, size)
    x = torch.stack((x.tile(size, 1), x.tile(size, 1).T), dim=0)

    d = torch.linalg.norm(x - mu, dim=0)
    x = torch.exp(-(d**2) / sigma**2)
    x = x / x.max()
    return x




def sample_images(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    
    
    def vis_panorama_window_weighted(latent, sliding_window_size=32, sigma=0.8):
        f = 8
        lt_sz = 128
        out_img = torch.zeros(
            (latent.shape[0], 3, f * latent.shape[2], f * latent.shape[3])
        ).to(latent.device)
        avg_map = torch.zeros_like(out_img).to(latent.device)

        # Blending kernel that focuses at the center of each patch
        kernel = gaussian_kernel(size=f * lt_sz, sigma=sigma).to(device)

        for i in tqdm(range(0, latent.shape[2] - lt_sz + 1, sliding_window_size)):
            for j in range(0, latent.shape[3] - lt_sz + 1, sliding_window_size):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    latent_part = latent[:, :, i : i + lt_sz, j : j + lt_sz]
                    latent_part = (latent_part / vae_scale) + vae_shift
                    decoded = vae.decode(latent_part).sample
                    out_img[
                        :, :, i * f : (i + lt_sz) * f, j * f : (j + lt_sz) * f
                    ] += decoded * kernel.view(1, 1, 1024, 1024)
                    avg_map[
                        :, :, i * f : (i + lt_sz) * f, j * f : (j + lt_sz) * f
                    ] += torch.ones_like(decoded) * kernel.view(1, 1, 1024, 1024)

        out_img /= avg_map
        out_img = torch.clamp(out_img, -1, 1)
        out_img = ((out_img + 1) * 127.5).to(torch.uint8).cpu().numpy()
        return out_img.transpose([0, 2, 3, 1])


    workdir = Path(
        "/home/srikarym/grand_data/pan_cancer/checkpoints/pixart_20x_1024_sd3_vae_pos_embed/"
    )
    model, vae, config = build_model_new(workdir, device, "checkpoints/last_ema.ckpt")

    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)

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
    local_path = "/lus/grand/projects/GeomicVar/srikar/pan_cancer/pretrained_models/uni2/pytorch_model.bin"
    uni_model.load_state_dict(
        torch.load(local_path, map_location="cpu"),
        strict=True,
    )

    tf_uni = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image_list = np.load(
        "/home/srikarym/grand_data/data/patches/brca/single_4096/image_list.npy"
    )

    idx_choice = np.random.randint(0, len(image_list), args.num_samples)
    # image_list = image_list[: args.num_samples]

    # np.random.shuffle(image_list)

    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for idx in idx_choice:
        name = image_list[idx]

        start = time.perf_counter()

        if (dst / f"{idx}.jpg").exists():
            print(f"Image {idx} already exists, skipping...")
            continue
        print(f"Processing image {idx}: {name}")

        img_real = np.array(Image.open(name))
        img_grid_256 = rearrange(
            img_real, "(n1 h) (n2 w) c -> (n1 n2) h w c", h=256, w=256
        )
        uni_cond = []
        for i in range(0, len(img_grid_256), 16):
            img_arr = img_grid_256[i : i + 16]
            img_arr = torch.stack([tf_uni(Image.fromarray(im)) for im in img_arr])
            img_arr = img_arr.to(device)
            with torch.inference_mode():
                out = uni_model(img_arr).cpu()
            uni_cond.append(out)
        uni_cond = torch.stack(uni_cond)

        sliding_window_size = args.sliding_window_size

        total_steps = 1000
        t_stride = total_steps // args.num_timesteps

        w = args.guidance_scale

        if sliding_window_size not in [32, 64, 128]:
            raise ValueError("sliding_window_size must be one of [32, 64, 128]")

        f = 8
        lt_sz = 128
        panorama_resolution = (4 * 1024, 4 * 1024)
        xt = (
            torch.randn(
                (1, 16, panorama_resolution[0] // f, panorama_resolution[1] // f)
            )
            .float()
            .to(device)
        )

        sampler = IDDPM("")
        for tt in range(total_steps, 0, -t_stride):

            atbar = sampler.alphas_cumprod[tt - 1]
            atbar_prev = sampler.alphas_cumprod[max((tt - 1) - t_stride, 0)]

            # Denoise sliding window views
            x0_map = torch.zeros_like(xt)
            eps_map = torch.zeros_like(xt)
            avg_map = torch.zeros_like(xt)

            for j in range(0, xt.shape[2] - lt_sz + 1, sliding_window_size):
                for k in range(0, xt.shape[3] - lt_sz + 1, sliding_window_size):

                    x_slice = xt[:, :, j : j + lt_sz, k : k + lt_sz]

                    ref_uni = get_conditioning(j, k, uni_cond).to(device)
                    ref_uni = rearrange(ref_uni, "n1 n2 dim -> 1 (n1 n2) dim")
                    uncond = model.y_embedder.y_embedding[None]

                    current_timestep = torch.tensor([tt - 1] * ref_uni.shape[0]).to(
                        device
                    )

                    with torch.no_grad(), torch.cuda.amp.autocast():
                        eps_cond = model.forward(
                            x_slice,
                            timestep=current_timestep,
                            y=ref_uni,
                        )[:, :16, :, :]

                        eps_uncond = model.forward(
                            x_slice,
                            timestep=current_timestep,
                            y=uncond,
                        )[:, :16, :, :]

                        epsilon_combined = (w + 1) * eps_cond - w * eps_uncond
                        x0_pred = x_slice / np.sqrt(atbar) - epsilon_combined * np.sqrt(
                            (1 - atbar) / atbar
                        )

                    eps_map[:, :, j : j + lt_sz, k : k + lt_sz] += epsilon_combined
                    x0_map[:, :, j : j + lt_sz, k : k + lt_sz] += x0_pred
                    avg_map[:, :, j : j + lt_sz, k : k + lt_sz] += 1

            x0_pred = x0_map / avg_map

            epsilon = (xt.float() - np.sqrt(atbar) * x0_pred) / np.sqrt(1 - atbar)

            xt = np.sqrt(atbar_prev) * x0_pred + np.sqrt(1 - atbar_prev) * epsilon

        out = vis_panorama_window_weighted(xt, sliding_window_size=sliding_window_size)

        Image.fromarray(out[0]).save(dst / f"{idx}.jpg")

        end = time.perf_counter()
        print(
            f" stride={sliding_window_size}, ddim steps={args.num_timesteps}, time taken={end-start:.2f} seconds"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=20,
        help="Number of timesteps for sampling",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2,
        help="Guidance scale for sampling",
    )
    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=64,
        help="Sliding window size for sampling",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use for sampling",
    )

    args = parser.parse_args()
    sample_images(args)
