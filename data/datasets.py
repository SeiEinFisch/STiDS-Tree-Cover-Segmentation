import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, extensions=('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
        """
        Dataset for unlabeled images (for SimCLR pre-training)
        """
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_paths = []
        for file in os.listdir(image_dir):
            if file.lower().endswith(extensions):
                self.image_paths.append(os.path.join(image_dir, file))
        
        print(f"Found {len(self.image_paths)} images for pre-training")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            return self.transform(img)
        return img


class TreeSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256,256), augment=True, channels=(0,1,2)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.channels = channels
        
        self.img_resize = transforms.Resize(img_size)
        self.mask_resize = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)

        if augment:
            self.joint_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.joint_transform = None

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        
        self.data_pairs = []
        image_files = {f.split('.')[0]: f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))}
        mask_files = {f.split('.')[0]: f for f in os.listdir(mask_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))}
        
        for name in image_files:
            if name in mask_files:
                self.data_pairs.append((
                    os.path.join(image_dir, image_files[name]),
                    os.path.join(mask_dir, mask_files[name])
                ))
        
        print(f"Found {len(self.data_pairs)} image-mask pairs for segmentation")
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.data_pairs[idx]
        
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert('L')

        mask_np = np.array(mask)
        
        img = self.img_resize(img)
        mask = self.mask_resize(mask)
        
        if self.joint_transform:
            seed = np.random.randint(2147483647)
            
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.joint_transform(img)
            
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.joint_transform(mask)

        img_tensor = self.to_tensor(img)
        if img_tensor.shape[0] < (max(self.channels)+1):
            raise RuntimeError(f"Image has {img_tensor.shape[0]} channels, but tried to select channel {max(self.channels)+1}")
        
        img_tensor = img_tensor[list(self.channels), :, :]
        img_tensor = self.normalize(img_tensor)

        mask_tensor = self.to_tensor(mask)
        mask_tensor = (mask_tensor * 255).long().squeeze(0) 
        mask_tensor = (mask_tensor > 127).long()
        
        return img_tensor, mask_tensor