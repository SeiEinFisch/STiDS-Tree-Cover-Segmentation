import torch
import torch.nn as nn
from datetime import datetime
from pytorch_metric_learning.losses import NTXentLoss

from config import device, data_config
from data.dataloaders import create_dataloaders
from models.encoders import ResNet50Encoder
from models.simclr import SimCLRModel, ProjectionHead
from models.segmentation import TreeSegmentationModel, BaselineTreeSegmentationModel
from training.simclr_trainer import train_simclr
from training.segmentation_trainer import train_segmentation, evaluate_segmentation
from training.baseline_trainer import train_baseline_segmentation, evaluate_baseline_segmentation
from utils.visualization import plot_losses_comparison, visualize_segmentation_results_colored
from utils.io_utils import save_comparison_data


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training run timestamp: {timestamp}")
    
    print("Creating data loaders...")
    pretrain_loader, seg_train_loader, seg_val_loader = create_dataloaders(**data_config)

    print("=" * 60)
    print("BASELINE: ImageNet pretrained ResNet50 + U-Net (No SimCLR)")
    print("=" * 60)
    
    baseline_model = BaselineTreeSegmentationModel(num_classes=2).to(device)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-4, weight_decay=1e-5)
    baseline_criterion = nn.CrossEntropyLoss()
    
    baseline_train_losses, baseline_val_losses = train_baseline_segmentation(
        baseline_model, seg_train_loader, seg_val_loader,
        baseline_optimizer, baseline_criterion, device, epochs=15, patience=5
    )
    
    if seg_val_loader is not None:
        print("Final evaluation on validation set (Baseline)")
        baseline_val_loss, baseline_pixel_acc, baseline_iou_score, baseline_dice_score = evaluate_baseline_segmentation(
            baseline_model, seg_val_loader, device
        )

    print("\n" + "=" * 60)
    print("SIMCLR PRETRAINED MODEL")
    print("=" * 60)
    
    encoder = ResNet50Encoder(hidden_dim=512)
    projection_head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
    simclr_model = SimCLRModel(encoder, projection_head).to(device)
    
    print("Phase 1: Self-supervised pre-training with SimCLR")
    optimizer = torch.optim.Adam(simclr_model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = NTXentLoss(temperature=0.5)
    simclr_losses = train_simclr(simclr_model, pretrain_loader, optimizer, criterion, device, epochs=50)

    print("Phase 2: Fine-tuning for tree segmentation")
    seg_model = TreeSegmentationModel(simclr_model.encoder, num_classes=2).to(device)
    seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=1e-4, weight_decay=1e-5)
    seg_criterion = nn.CrossEntropyLoss()
    seg_train_losses, seg_val_losses = train_segmentation(
        seg_model, seg_train_loader, seg_val_loader,
        seg_optimizer, seg_criterion, device, epochs=50, patience=5
    )

    if seg_val_loader is not None:
        print("Final evaluation on validation set")
        final_val_loss, final_pixel_acc, final_iou_score, final_dice_score = evaluate_segmentation(
            seg_model, seg_val_loader, device
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\nBaseline Model Results:")
    print(f"  Final train loss: {baseline_train_losses[-1]:.4f}")
    if baseline_val_losses:
        print(f"  Final val loss: {baseline_val_losses[-1]:.4f}")
    if seg_val_loader is not None:
        print(f"  Final pixel accuracy: {baseline_pixel_acc:.4f}")
        print(f"  Final IoU score: {baseline_iou_score:.4f}")
        print(f"  Final Dice score: {baseline_dice_score:.4f}")
    
    print(f"\nSimCLR Pretrained Model Results:")
    print(f"  SimCLR final loss: {simclr_losses[-1]:.4f}")
    print(f"  Segmentation final train loss: {seg_train_losses[-1]:.4f}")
    if seg_val_losses:
        print(f"  Segmentation final val loss: {seg_val_losses[-1]:.4f}")
    if seg_val_loader is not None:
        print(f"  Final pixel accuracy: {final_pixel_acc:.4f}")
        print(f"  Final IoU score: {final_iou_score:.4f}")
        print(f"  Final Dice score: {final_dice_score:.4f}")

    if seg_val_loader is not None:
        iou_improvement = ((final_iou_score - baseline_iou_score) / baseline_iou_score) * 100
        dice_improvement = ((final_dice_score - baseline_dice_score) / baseline_dice_score) * 100
        acc_improvement = ((final_pixel_acc - baseline_pixel_acc) / baseline_pixel_acc) * 100
       
        print(f"\nImprovement with SimCLR pretraining:")
        print(f"  IoU improvement: {iou_improvement:+.2f}%")
        print(f"  Dice improvement: {dice_improvement:+.2f}%")
        print(f"  Pixel accuracy improvement: {acc_improvement:+.2f}%")

    plot_filename = plot_losses_comparison(
        simclr_losses, seg_train_losses, baseline_train_losses, timestamp,
        baseline_iou_score, baseline_dice_score, baseline_pixel_acc,
        final_iou_score, final_dice_score, final_pixel_acc
    )
    
    json_filename = save_comparison_data(
        simclr_losses, seg_train_losses, seg_val_losses, final_iou_score, final_dice_score,
        baseline_train_losses, baseline_val_losses, baseline_iou_score, baseline_dice_score,
        timestamp
    )
    
    baseline_model_path = f'baseline_tree_segmentation_model_{timestamp}.pth'
    simclr_model_path = f'simclr_pretrained_resnet50_{timestamp}.pth'
    seg_model_path = f'tree_segmentation_model_resnet50_{timestamp}.pth'
    torch.save(baseline_model.state_dict(), baseline_model_path)
    torch.save(simclr_model.state_dict(), simclr_model_path)
    torch.save(seg_model.state_dict(), seg_model_path)
    
    print(f"\nModels saved:")
    print(f"  Baseline: {baseline_model_path}")
    print(f"  SimCLR: {simclr_model_path}")
    print(f"  Segmentation: {seg_model_path}")

    visualize_segmentation_results_colored(seg_model, seg_val_loader, device, num_images=5)


if __name__ == "__main__":
    main()