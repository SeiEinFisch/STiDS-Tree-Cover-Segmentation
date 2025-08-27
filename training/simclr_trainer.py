"""
SimCLR training functions for contrastive learning.
"""

import torch
import torch.nn.functional as F

def train_simclr(model, dataloader, optimizer, criterion, device, epochs=50, accumulation_steps=8):
    """
    Training function for simCLR model, applies Gradient accumulation to simulate having larger batches
    """
    model.train()
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Clear gradients outside the batch loop
        optimizer.zero_grad()

        for batch_idx, (x_i, x_j) in enumerate(dataloader):
            x_i, x_j = x_i.to(device), x_j.to(device)
            
            z_i = model(x_i)
            z_j = model(x_j)
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)
            
            embeddings = torch.cat([z_i, z_j], dim=0)  # [2B, D]
            labels = torch.arange(x_i.size(0), device=x_i.device)
            labels = torch.cat([labels, labels], dim=0)
            
            loss = criterion(embeddings, labels)
            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()

            # Gradient Accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps  # Multiply back for logging
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'SimCLR Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}')
        
        # If batches aren't divisible by accumulation_steps, step the optimizer once more
        if num_batches % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f'SimCLR Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}')
    
    return epoch_losses