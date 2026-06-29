import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CorrosionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to the folder containing images and masks (e.g. .../test)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Find all jpg images
        self.image_paths = list(self.root_dir.rglob("*.jpg"))
        
        # Filter to make sure masks exist
        self.valid_data = []
        for img_path in self.image_paths:
            mask_name = img_path.stem + "_mask.png"
            mask_path = img_path.parent / mask_name
            if mask_path.exists():
                self.valid_data.append((img_path, mask_path))
                
        print(f"Found {len(self.valid_data)} valid image-mask pairs in {root_dir}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_data[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # L = grayscale
        
        # Resize for consistent tensor sizes
        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)
        
        # Convert to numpy array
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        # Binarize mask (0 or 1)
        mask = (mask > 0.5).astype(np.float32)
        
        # Convert to PyTorch tensors
        # image: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        # mask: (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.transform:
            # You can add albumentations here if needed
            pass
            
        return image, mask

if __name__ == "__main__":
    # Test the dataset
    ds = CorrosionDataset("../YoloV8Corrosion.v1i.png-mask-semantic/test")
    if len(ds) > 0:
        img, mask = ds[0]
        print("Image shape:", img.shape)
        print("Mask shape:", mask.shape)
