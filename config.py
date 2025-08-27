import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

seed = 42
random.seed(seed)

data_config = {
    'unlabeled_dir': '/kaggle/input/training-data/Training_data/aerial_unlabeled',  # For SimCLR pre-training
    'image_dir': '/kaggle/input/small-split-binary/small_split_binary/images',                # Labeled images for segmentation
    #'unlabeled_dir2': '/kaggle/input/gta-dataset/GTA-V/tiles',
    'mask_dir': '/kaggle/input/small-split-binary/small_split_binary/masks',                  # Corresponding segmentation masks
    'batch_size_pretrain': 32,
    'batch_size_seg': 16,
    'img_size': (256, 256),  
    'num_workers': 4 
}