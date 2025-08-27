"""
SimCLR model combining encoder and projection head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head that projects encoded feature vector into lower-dimensional space
    """
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class SimCLRModel(nn.Module):
    """
    SimCLR model consisting of encoder and projection head for contrastive learning
    """
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(self, x):
        h, _ = self.encoder(x)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)
