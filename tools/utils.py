from pathlib import Path
import h5py
from torch.utils.data import Dataset,  default_collate
from PIL import Image

from einops import rearrange
import numpy as np
import torch


from diffusion.model.builder import build_model
from diffusion.utils.misc import read_config

from diffusers.models import AutoencoderKL


def build_model_new(root, device, checkpoint):

    weight_dtype = torch.float16
    config = read_config(root / "config.py")
    latent_size = config.image_size // 8


    kv_compress_config = (
        config.kv_compress_config if config.kv_compress else None
    )
    model_kwargs = {
        "pe_interpolation": config.pe_interpolation,
        "config": config,
        "model_max_length": config.model_max_length,
        "qk_norm": config.qk_norm,
        "class_dropout_prob": 0,
        "kv_compress_config": kv_compress_config,
    }

    model = (
        build_model(
            config.model,
            input_size=latent_size,
            learn_sigma=True,
            pred_sigma=True,
            **model_kwargs,
        )
        .eval()
        .to(device)
        .to(weight_dtype)
    )

    state_dict = torch.load(root / checkpoint)
    model.load_state_dict(state_dict["state_dict"], strict=False)
    vae = (
        AutoencoderKL.from_pretrained(config.vae_pretrained)
        .to(device)
        .to(weight_dtype)
        .eval()
    )

    return model, vae, config



class DatasetForFeatureExt(Dataset):
    def __init__(self, root, dataset_name, size, tf_vae, tf_uni, indices=None):

        self.root = Path(root) 
        self.arr_name = f"{dataset_name}_{size}"
        self.h5 = None
        
        with h5py.File(root / "patch_names_all.hdf5", "r") as f:
            self.len = len(f[self.arr_name])

        self.tf_vae = tf_vae
        self.tf_uni = tf_uni

        if indices is not None:
            self.len = len(indices)
            self.indices = indices

    def __len__(self):
        return self.len



    def __getitem__(self, idx_orig):
        if self.h5 is None:
            self.h5 = h5py.File(self.root / "patch_names_all.hdf5", "r")

        if hasattr(self, "indices"):
            idx = self.indices[idx_orig]
        else:
            idx = idx_orig

        img_name_rel = self.h5[self.arr_name][idx].decode()
        img_name = self.root / img_name_rel

        try:

            img = Image.open(img_name)
            img_vae = self.tf_vae(img)
            
            # divide image into 256x256 patches for UNI
            
            img_uni_grid = rearrange(np.array(img), '(n1 h) (n2 w) c -> (n1 n2) h w c', h=256, w=256)
            img_uni_grid = np.stack([self.tf_uni(Image.fromarray(patch)) for patch in img_uni_grid])

            return img_vae, img_uni_grid, img_name_rel


        except Exception as e:
            
            print(f"Error opening {img_name_rel}, {e}")
            with open("invalid_files", "a") as f:
                f.write(img_name_rel + "\n")

from cleanfid.resize import build_resizer
from torchvision import transforms as T

class PatchDataset(Dataset):
    def __init__(self, root, resolution, img_list, resizer=None):
        self.root = root
        self.img_list = np.load(root / img_list)
        self.resolution = resolution
        self.resizer = build_resizer("clean") if resizer is None else resizer
        self.transforms = T.ToTensor()


    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.root / self.img_list[idx]))
        except Exception as e:
            print(f"Error opening {self.img_list[idx]}, {e}")
            return None

        if img.size != (self.resolution, self.resolution):
            img_list = rearrange(img, '(n1 h) (n2 w) c -> (n1 n2) h w c', h=self.resolution, w=self.resolution)

        else:
            img_list = [img]

        outputs = []
        for img_np in img_list:

            # fn_resize expects a np array and returns a np array
            img_resized = self.resizer(img_np)

            # ToTensor() converts to [0,1] only if input in uint8
            if img_resized.dtype == "uint8":
                img_t = self.transforms(np.array(img_resized))*255
            elif img_resized.dtype == "float32":
                img_t = self.transforms(img_resized)
            outputs.append(img_t)

        return np.array(outputs)

def my_collate(batch):
    items = [x for x in batch if x is not None]
    return default_collate(items)


class FlatPatchDataset(Dataset):
    def __init__(self, root, resolution, img_list, resizer=None):
        self.root = Path(root)
        self.resolution = resolution
        self.resizer = build_resizer("clean") if resizer is None else resizer
        self.img_paths = np.load(self.root / img_list)
        self.patch_indices = []
        self.transforms = T.ToTensor()

        # Precompute patch indices
        for path in self.img_paths:
            for i in range(0, 1024, resolution):
                for j in range(0, 1024, resolution):
                    self.patch_indices.append((path, i, j))

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        path, i, j = self.patch_indices[idx]

        try:
            img = Image.open(self.root / path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

        img = np.array(img)

        # Crop directly
        patch = img[i:i+self.resolution, j:j+self.resolution]

        if type(self.resizer) == T.Compose:
            patch = Image.fromarray(patch)

        # Resize if needed (usually not required if cropped correctly)
        patch_resized = self.resizer(patch)


        if patch_resized.dtype == "uint8":
            patch_resized = self.transforms(np.array(patch_resized))*255
        elif patch_resized.dtype == "float32":
            patch_resized = self.transforms(patch_resized)

        return patch_resized




import numpy as np
from pytorch_fid.inception import InceptionV3
from torchvision import transforms as T
import clip
import torch
from tqdm import tqdm

def calculate_activation_statistics(images, device, choice, batch_size=32,):
    
    if choice == 'inception':
    
        tf = T.Compose([
            T.ToTensor(),])

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device).eval()
        pred_arr = np.empty((len(images), 2048))

    else:
        
        model, _ = clip.load("ViT-B/32")
        model = model.to(device).eval()
        tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        pred_arr = np.empty((len(images), 512))
    
    start_idx = 0

    for i in tqdm(range(0, len(images), batch_size)):
        chunk = images[i : i + batch_size]
        inp = torch.stack([tf(img) for img in chunk]).to(device)

        with torch.no_grad():
            if choice == 'inception':
                pred = model(inp)[0].squeeze(3).squeeze(2)
            else:
                pred = model.encode_image(inp)

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred.cpu().numpy()

        start_idx = start_idx + pred.shape[0]

    mu = np.mean(pred_arr[:start_idx], axis=0)
    sigma = np.cov(pred_arr[:start_idx], rowvar=False)


    return mu, sigma

def compute_statistics_of_path(path):
    with np.load(path) as f:
        m, s = f["mu"][:], f["sigma"][:]
    return m, s
