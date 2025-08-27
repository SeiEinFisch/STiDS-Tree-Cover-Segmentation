"""
Training and evaluation functions for baseline segmentation models.
"""

import torch
import torch.nn as nn
import copy
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import JaccardIndex


def train_baseline_segmentation(model, dataloader, val_dataloader, optimizer, criterion, device, epochs=50, patience=10, min_delta=0.001, restore_best_weights=True):
    """
    Training function for baseline segmentation model (ImageNet pretrained only)
    """
    model.train()
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Baseline Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        val_loss = 0
        if val_dataloader is not None:
            model.eval()
            val_loss = evaluate_baseline_loss_only(model, val_dataloader, criterion, device)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if restore_best_weights:
                    best_model_state = copy.deepcopy(model.state_dict())
                print(f'Baseline Epoch {epoch+1}/{epochs} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f} (Best)')
            else:
                epochs_without_improvement += 1
                print(f'Baseline Epoch {epoch+1}/{epochs} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f} (No improvement: {epochs_without_improvement}/{patience})')
            
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs. No improvement for {patience} epochs.')
                if restore_best_weights and best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print('Restored best model weights.')
                break
                
        else:
            print(f'Baseline Epoch {epoch+1}/{epochs} completed. Train Loss: {avg_train_loss:.4f}')
    
    return train_losses, val_losses


def evaluate_baseline_loss_only(model, dataloader, criterion, device):
    """Quick validation loss calculation for baseline model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def evaluate_baseline_segmentation(model, dataloader, device):
    """
    Evaluation function for baseline model
    """
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    criterion = nn.CrossEntropyLoss()
    dice_metric = DiceScore(num_classes=2, average='macro', include_background=True).to(device)
    iou_metric = JaccardIndex(num_classes=2, task='multiclass', average='macro').to(device)
    
    dice_metric.reset()
    iou_metric.reset()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            
            dice_metric.update(predictions, masks)
            iou_metric.update(predictions, masks)
            
    avg_loss = total_loss / len(dataloader)
    pixel_accuracy = correct_pixels / total_pixels
    
    avg_dice_score = dice_metric.compute().item()
    avg_iou_score = iou_metric.compute().item()
    
    print(f"Baseline Validation Loss: {avg_loss:.4f}")
    print(f"Baseline Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Baseline IoU: {avg_iou_score:.4f}")
    print(f"Baseline Dice Score: {avg_dice_score:.4f}")
    
    return avg_loss, pixel_accuracy, avg_iou_score, avg_dice_score
