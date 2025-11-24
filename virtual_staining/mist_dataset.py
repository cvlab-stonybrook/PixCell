import os
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class MISTDataset(Dataset):
    """
    MIST Dataset 
    https://link.springer.com/chapter/10.1007/978-3-031-43987-2_61
    Contains IHC images for HER2, Ki67, ER, PR
    """
    def __init__(self, root_dir, split, stain):
        self.root_dir = root_dir
        self.split = split
        assert split in ['train', 'val'], "Split must be train/val"

        self.stain = stain
        assert stain in ['HER2', 'ER', 'PR', 'Ki67']

        # Locate images
        self.he_paths = os.listdir(os.path.join(self.root_dir, f'{self.stain}/TrainValAB/', self.split + "A"))
        self.ihc_paths = os.listdir(os.path.join(self.root_dir, f'{self.stain}/TrainValAB', self.split + "B"))
        assert len(self.he_paths) == len(self.ihc_paths)
        print("Found", len(self.he_paths), "images in split", f"{self.stain}/{self.split}")

        # Transforms resize the image to 1024x124
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((1024,1024), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        ])

    def __len__(self):
        return len(self.he_paths) 

    def __getitem__(self, idx):
        # Load HE and IHC images
        he_path = os.path.join(self.root_dir, f'{self.stain}/TrainValAB/', self.split + "A", self.he_paths[idx])
        ihc_path = os.path.join(self.root_dir, f'{self.stain}/TrainValAB/', self.split + "B", self.he_paths[idx])

        he_image = Image.open(he_path)
        ihc_image = Image.open(ihc_path)

        he_image = self.transforms(he_image)
        ihc_image = self.transforms(ihc_image)

        return he_image, ihc_image
