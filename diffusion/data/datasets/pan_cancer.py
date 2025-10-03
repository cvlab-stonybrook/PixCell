from pathlib import Path
from PIL import Image
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from diffusers.utils.torch_utils import randn_tensor
from diffusion.data.builder import DATASETS
from einops import rearrange


class LowResCropMixin:
    def crop_features(self, vae_feat_1024, ssl_feat_1024, _img, idx):
        sub_idx = idx % (self.factor ** 2)

        # Crop VAE features
        vae_feat_1024 = rearrange(
            vae_feat_1024, 'c (n1 h) (n2 w) -> c (n1 n2) h w',
            n1=self.factor, n2=self.factor
        )

        # Crop SSL features
        hh = ww = int((vae_feat_1024.shape[0] // (self.factor ** 2)) ** 0.5)
        ssl_feat_1024 = rearrange(
            ssl_feat_1024, '(n1 hh n2 ww) c -> (n1 n2) (hh ww) c',
            n1=self.factor, n2=self.factor, hh=hh, ww=ww
        )

        vae_feat = vae_feat_1024[:, sub_idx]
        ssl_feat = ssl_feat_1024[sub_idx]

        if self.return_img:
            img = rearrange(
                _img, '(n1 h) (n2 w) c -> (n1 n2) h w c',
                n1=self.factor, n2=self.factor
            )
            return vae_feat, ssl_feat, 0, img[sub_idx]

        data_info = {
            "img_hw": torch.tensor([self.crop_size] * 2, dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
            "mask_type": "null",
            "idx": idx
        }

        return vae_feat, ssl_feat, 0, data_info


# Register the dataset class with the DATASETS registry
@DATASETS.register_module()
class PanCancerData(Dataset):
    def __init__(self, root, resolution, **kwargs):
        """
        Initialize the PanCancerData dataset.

        Args:
            root (str): Root directory of the dataset.
            resolution (int): Resolution of the images.
            **kwargs: Additional keyword arguments.
        """
        self.root = Path(root)
        self.load_vae_feat = True
        self.resolution = resolution
        self.vae_prefix = kwargs.get("vae_prefix", "sd3_vae")
        self.ssl_prefix = kwargs.get("ssl_prefix", "uni")
        self.return_img = kwargs.get("return_img", False)
        self.train_subset_keys = kwargs.get("train_subset_keys", None)

        patch_names_file = kwargs.get("patch_names_file", "patch_names_all.hdf5")

        self.image_list_h5 = self.root / f"patches/metadata/{patch_names_file}"
        self.features_dir = self.root / "features"

        # Load metadata from the HDF5 file
        self._load_metadata()

        self.dummy_data = False

    def _load_metadata(self):
        """Load dataset keys and lengths from the HDF5 file."""
        self.h5 = None
        lengths = {}
        with h5py.File(self.root / self.image_list_h5, "r") as h5:
            # Filter keys based on the resolution
            keys_all = list(h5.keys())
            if self.train_subset_keys is not None:
                keys_all = [key for key in keys_all if key in self.train_subset_keys]

            self.keys = [item for item in keys_all if f"_{self.resolution}" in item]

            # Store the length of each key
            for key in self.keys:
                lengths[key] = len(h5[key])

        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(list(lengths.values()))])

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.cumulative_lengths[-1]

    @staticmethod
    def _vae_feat_loader(path):
        """Load VAE features from a .npy file."""
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(
            mean.shape, generator=None, device=mean.device, dtype=mean.dtype
        )
        return mean + std * sample
    
    def load_from_image_name(self, img_name):
        # Load VAE and SSL features
        vae_path = img_name.replace(".jpeg", f"_{self.vae_prefix}.npy")
        ssl_path = img_name.replace(".jpeg", f"_{self.ssl_prefix}.npy")

        vae_feat = self._vae_feat_loader(self.features_dir / vae_path)
        ssl_feat = torch.from_numpy(np.load(self.features_dir / ssl_path))

        lt_sz = self.resolution // 8
        assert vae_feat.shape == (16, lt_sz, lt_sz)

        data_info = {
            "img_hw": torch.tensor([self.resolution] * 2, dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
            "mask_type": "null",
            "img_name": img_name
        }

        if self.return_img:
            img = Image.open(self.root / f"patches/{img_name}")
            img = np.array(img)
            return vae_feat, ssl_feat, 0, img

        return vae_feat, ssl_feat, 0, data_info

    def get_data(self, idx):
        """
        Get data for a given index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: VAE features, SSL features, _, and additional data info.
        """
        if self.h5 is None:
            self.h5 = h5py.File(self.root / self.image_list_h5, "r")
        key_idx = (
            np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        )
        key = self.keys[key_idx]
        idx = idx - self.cumulative_lengths[key_idx]
        img_name = self.h5[key][idx].decode("utf-8")

        if self.dummy_data:
            return img_name

        return self.load_from_image_name(img_name)
    
    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: VAE features, SSL features, _, and additional data info.
        """
        # return self.get_data(idx)
        retries = 10
        for _ in range(retries):
            try:
                return self.get_data(idx)
            except Exception as e:
                print(f"Failed to load data for index {idx}. Retrying...")
                idx = np.random.randint(len(self))

        raise RuntimeError(f"Failed to load data after {retries} attempts.")



@DATASETS.register_module()
class PanCancerDataLowRes(LowResCropMixin, PanCancerData):

    # if resolution is set to 256/512, take crops of 1024 images
    def __init__(self, root, resolution, **kwargs):

        super().__init__(root, 1024, **kwargs)
        self.crop_size = resolution

        # update the total number of samples
        self.factor = (1024 // resolution)
        self.cumulative_lengths[-1] *= (self.factor ** 2)


    def get_data(self, idx):
        
        vae_feat_1024, ssl_feat_1024, _, _img = super().get_data(idx // (self.factor ** 2))
        return self.crop_features(vae_feat_1024, ssl_feat_1024, _img, idx)


