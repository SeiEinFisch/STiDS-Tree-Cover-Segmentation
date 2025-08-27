import json


def save_loss_data(simclr_losses, seg_losses, val_losses, timestamp, final_iou_score, final_dice_score, save_dir='./'):
    """Save loss data as JSON file"""
   
    loss_data = {
        'timestamp': timestamp,
        'simclr_losses': simclr_losses,
        'segmentation_losses': seg_losses,
        'validation_losses': val_losses,
        'num_simclr_epochs': len(simclr_losses),
        'num_seg_epochs': len(seg_losses),
        'final_iou_score': final_iou_score,
        'final_dice_score': final_dice_score
    }
   
    json_filename = f'{save_dir}loss_data_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(loss_data, f, indent=2)
   
    print(f"Loss data saved as: {json_filename}")
    return json_filename


def save_comparison_data(simclr_losses, seg_train_losses, seg_val_losses, final_iou_score, final_dice_score,
                        baseline_train_losses, baseline_val_losses, baseline_iou_score, baseline_dice_score,
                        timestamp):
    """
    Save comparison data as JSON
    """
    data = {
        'timestamp': timestamp,
        'simclr_model': {
            'simclr_losses': simclr_losses,
            'seg_train_losses': seg_train_losses,
            'seg_val_losses': seg_val_losses,
            'final_iou_score': final_iou_score,
            'final_dice_score': final_dice_score
        },
        'baseline_model': {
            'train_losses': baseline_train_losses,
            'val_losses': baseline_val_losses,
            'final_iou_score': baseline_iou_score,
            'final_dice_score': baseline_dice_score
        },
        'improvement': {
            'iou_improvement_percent': ((final_iou_score - baseline_iou_score) / baseline_iou_score) * 100,
            'dice_improvement_percent': ((final_dice_score - baseline_dice_score) / baseline_dice_score) * 100
        }
    }
   
    json_filename = f'training_comparison_data_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)
   
    return json_filename