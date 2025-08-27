import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import JaccardIndex


def plot_losses(simclr_losses, seg_losses, timestamp, save_dir='./'):
    """Plot and save loss curves for both training phases"""
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
   
    ax1.plot(simclr_losses, 'b-', linewidth=2, label='SimCLR Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('SimCLR Self-Supervised Pre-training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
   
    if seg_losses:
        ax2.plot(seg_losses, 'r-', linewidth=2, label='Segmentation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Tree Segmentation Fine-tuning Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
   
    plt.tight_layout()
   
    plot_filename = f'{save_dir}training_losses_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved as: {plot_filename}")
   
    return plot_filename


def plot_losses_comparison(simclr_losses, seg_train_losses, baseline_train_losses, timestamp, 
                          baseline_iou_score, baseline_dice_score, baseline_pixel_acc,
                          final_iou_score, final_dice_score, final_pixel_acc):
    """
    Modified plotting function to compare baseline vs SimCLR results
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
   
    axes[0].plot(simclr_losses, 'b-', label='SimCLR Loss')
    axes[0].set_title('SimCLR Pre-training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
   
    axes[1].plot(baseline_train_losses, 'r-', label='Baseline (ImageNet)')
    axes[1].plot(seg_train_losses, 'g-', label='SimCLR Pretrained')
    axes[1].set_title('Segmentation Training Loss Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
   
    metrics = ['IoU', 'Dice', 'Pixel Acc']
    baseline_vals = [baseline_iou_score, baseline_dice_score, baseline_pixel_acc]
    simclr_vals = [final_iou_score, final_dice_score, final_pixel_acc]
   
    x = range(len(metrics))
    width = 0.35
    axes[2].bar([i - width/2 for i in x], baseline_vals, width, label='Baseline', color='red', alpha=0.7)
    axes[2].bar([i + width/2 for i in x], simclr_vals, width, label='SimCLR', color='green', alpha=0.7)
    axes[2].set_title('Final Metrics Comparison')
    axes[2].set_xlabel('Metrics')
    axes[2].set_ylabel('Score')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
   
    plt.tight_layout()
    plot_filename = f'training_comparison_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
   
    return plot_filename


def visualize_segmentation_results_colored(model, val_dataloader, device, num_images=5):
    model.eval()
   
    fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    if num_images == 1:
        axes = axes.reshape(1, -1)
   
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 2].set_title('Predicted Mask', fontsize=14, fontweight='bold')
   
    accuracies = []
    dices = []
    ious = []
   
    with torch.no_grad():
        image_count = 0
        for batch_idx, (images, masks) in enumerate(val_dataloader):
            images, masks = images.to(device), masks.to(device)
           
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
           
            batch_size = images.size(0)
            for j in range(batch_size):
                if image_count >= num_images:
                    break
               
                image = images[j].cpu()
                true_mask = masks[j].cpu()
                pred_mask = predictions[j].cpu()
               
                correct_pixels = (true_mask == pred_mask).sum().item()
                total_pixels = true_mask.numel()
                accuracy = correct_pixels / total_pixels * 100
                accuracies.append(accuracy)
               
                dice_metric = DiceScore(num_classes=2, average='macro', include_background=True).to(device)
                iou_metric = JaccardIndex(num_classes=2, task='multiclass', average='macro').to(device)
               
                p = pred_mask.unsqueeze(0).to(device)
                t = true_mask.unsqueeze(0).to(device)
               
                dice_metric.update(p, t)
                iou_metric.update(p, t)
               
                dice_val = dice_metric.compute().item()
                iou_val = iou_metric.compute().item()
               
                dices.append(dice_val)
                ious.append(iou_val)
               
                i = image_count
               
                if image.shape[0] == 3:
                    image_np = image.permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_np = std * image_np + mean
                    image_np = np.clip(image_np, 0, 1)
                else:
                    image_np = image.squeeze().numpy()
               
                axes[i, 0].imshow(image_np if len(image_np.shape)==3 else image_np, cmap=None if len(image_np.shape)==3 else 'gray')
                axes[i, 0].axis('off')
                axes[i, 1].imshow(true_mask.numpy(), cmap='viridis', alpha=0.8)
                axes[i, 1].axis('off')
                axes[i, 2].imshow(pred_mask.numpy(), cmap='viridis', alpha=0.8)
                axes[i, 2].axis('off')
               
                axes[i, 0].set_ylabel(
                    f'Image {image_count + 1}\nAcc: {accuracy:.1f}%', fontsize=12, fontweight='bold'
                )
               
                image_count += 1
           
            if image_count >= num_images:
                break
   
    plt.tight_layout()
    plt.show()
   
    print("\nPer-Image Segmentation Metrics:")
    print(f"{'Image':<8}{'Pixel Acc (%)':<15}{'Dice Score':<12}{'IoU Score':<12}")
    print("-" * 47)
    for idx in range(num_images):
        print(f"{idx+1:<8}{accuracies[idx]:<15.2f}{dices[idx]:<12.4f}{ious[idx]:<12.4f}")
   
    print(f"\nAverage Pixel Accuracy: {np.mean(accuracies):.2f}%")
    print(f"Average Dice Score: {np.mean(dices):.4f}")
    print(f"Average IoU Score: {np.mean(ious):.4f}")