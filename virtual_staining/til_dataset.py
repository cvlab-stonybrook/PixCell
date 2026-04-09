"""TIL Overlay Dataset for PixCell LoRA training.

Paired H&E → target images at 1024×1024.
Same interface as MISTDataset / HER2MatchDataset.

Directory structure:
  root_dir/
    blend/
      trainA/   # H&E images (JPEG)
      trainB/   # blended overlay targets (JPEG)
    classmap/
      trainA/   # H&E images (JPEG)
      trainB/   # classification map targets (JPEG)
    probmap/
      trainA/   # H&E images (JPEG)
      trainB/   # probability map targets (PNG, lossless)
                # R=tumor_prob, G=TIL_prob, B=tissue_mask
"""

import os
import torchvision
from torch.utils.data import Dataset
from PIL import Image


class TILOverlayDataset(Dataset):
    """Paired H&E → TIL overlay dataset for PixCell LoRA fine-tuning.

    Args:
        root_dir: Path to extraction output (e.g., data/lora_training/)
        split: 'train' or 'val'
        task: 'blend', 'classmap', or 'probmap'
    """
    def __init__(self, root_dir, split, task="probmap"):
        self.root_dir = root_dir
        self.split = split
        self.task = task
        assert split in ["train", "val"], "Split must be train/val"
        assert task in ["blend", "classmap", "probmap", "gtt"], "Task must be blend/classmap/probmap/gtt"

        he_dir = os.path.join(root_dir, task, f"{split}A")
        target_dir = os.path.join(root_dir, task, f"{split}B")

        self.he_paths = sorted(os.listdir(he_dir))
        self.target_paths = sorted(os.listdir(target_dir))

        # For probmap/gtt, H&E is .jpg and target is .png — match by stem
        if task in ("probmap", "gtt"):
            he_stems = {os.path.splitext(f)[0]: f for f in self.he_paths}
            target_stems = {os.path.splitext(f)[0]: f for f in self.target_paths}
            common = sorted(he_stems.keys() & target_stems.keys())
            self.he_paths = [he_stems[s] for s in common]
            self.target_paths = [target_stems[s] for s in common]
        else:
            assert len(self.he_paths) == len(self.target_paths), (
                f"Mismatch: {len(self.he_paths)} H&E vs {len(self.target_paths)} targets"
            )

        print(f"TILOverlayDataset: {len(self.he_paths)} {task}/{split} pairs")

        self.he_dir = he_dir
        self.target_dir = target_dir

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                (1024, 1024),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
        ])

        # Probmap targets should use nearest-neighbor resize to preserve values
        self.target_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                (1024, 1024),
                interpolation=(
                    torchvision.transforms.InterpolationMode.NEAREST
                    if task in ("probmap", "gtt") else
                    torchvision.transforms.InterpolationMode.BICUBIC
                ),
            ),
        ])

    def __len__(self):
        return len(self.he_paths)

    def __getitem__(self, idx):
        he_path = os.path.join(self.he_dir, self.he_paths[idx])
        target_path = os.path.join(self.target_dir, self.target_paths[idx])

        he_image = Image.open(he_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        he_image = self.transforms(he_image)
        target_image = self.target_transforms(target_image)

        return he_image, target_image
