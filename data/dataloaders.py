import torch
from torch.utils.data import DataLoader, ConcatDataset
from .datasets import UnlabeledImageDataset, TreeSegmentationDataset
from .transforms import SimCLRAugmentation


def create_dataloaders(image_dir=None, mask_dir=None, unlabeled_dir=None, unlabeled_dir2=None,
                      batch_size_pretrain=32, batch_size_seg=16,
                      img_size=(256, 256), num_workers=4):
    """
    Create data loaders for both pre-training and segmentation
    """
   
    pretrain_transform = SimCLRAugmentation(size=img_size[0])
    if unlabeled_dir2 is not None:
        pretrain_dataset1 = UnlabeledImageDataset(
            image_dir=unlabeled_dir,
            transform=pretrain_transform
        )
        pretrain_dataset2 = UnlabeledImageDataset(
            image_dir=unlabeled_dir2,
            transform=pretrain_transform
        )
        pretrain_dataset = ConcatDataset([pretrain_dataset1, pretrain_dataset2])
    else:
        pretrain_dataset = UnlabeledImageDataset(
            image_dir=unlabeled_dir,
            transform=pretrain_transform
        )
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size_pretrain,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
   
    seg_dataset = TreeSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_size=img_size,
        augment=True
    )
   
    train_size = int(0.8 * len(seg_dataset))
    val_size = len(seg_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    seg_train_dataset, seg_val_dataset = torch.utils.data.random_split(seg_dataset, [train_size, val_size], generator=generator)
   
    seg_train_loader = DataLoader(
        seg_train_dataset,
        batch_size=batch_size_seg,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    seg_val_loader = DataLoader(
        seg_val_dataset,
        batch_size=batch_size_seg,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
   
    return pretrain_loader, seg_train_loader, seg_val_loader