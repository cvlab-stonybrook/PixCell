import os
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class HER2MatchDataset(Dataset):
    """
    HER2Match Dataset 
    https://zenodo.org/records/15797050
    Contains IHC images for HER2
    """
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split
        assert split in ['train', 'test', 'val'], "Split must be train/test/val"

        # Locate images
        self.he_paths = os.listdir(os.path.join(self.root_dir, f'HE_20x/{self.split}'))
        self.ihc_paths = os.listdir(os.path.join(self.root_dir, f'IHC_20x/{self.split}'))
        # Ignore hidden files
        self.he_paths = [p for p in self.he_paths if not p.startswith('.')]
        self.ihc_paths = [p for p in self.ihc_paths if not p.startswith('.')]
        assert len(self.he_paths) == len(self.ihc_paths)
        print("Found", len(self.he_paths), "images in split", f"{self.split}")

        # Transforms
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.he_paths) 

    def __getitem__(self, idx):
        # Load HE and IHC images
        he_path = os.path.join(self.root_dir, f'HE_20x/{self.split}', self.he_paths[idx])
        ihc_path = os.path.join(self.root_dir, f'IHC_20x/{self.split}', self.he_paths[idx])

        he_image = Image.open(he_path)
        ihc_image = Image.open(ihc_path)

        he_image = self.transforms(he_image)
        ihc_image = self.transforms(ihc_image)

        return he_image, ihc_image
