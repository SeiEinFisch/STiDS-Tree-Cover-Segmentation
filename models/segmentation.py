"""
Segmentation models using ResNet50 encoder with U-Net decoder and baseline
"""

import torch.nn as nn
from .encoders import ResNet50SegmentationEncoder, BaselineResNet50Encoder
from .decoders import ResNet50UNetDecoder


class TreeSegmentationModel(nn.Module):
    """
    Encoder-decoder model using ResNet50 for image segmentation
    """
    def __init__(self, pretrained_encoder, num_classes=2, freeze_early_layers=True):
        super().__init__()
        self.encoder = ResNet50SegmentationEncoder(pretrained_encoder)
        self.decoder = ResNet50UNetDecoder(num_classes)
        if freeze_early_layers:
            for param in self.encoder.resnet50_encoder.backbone[:6].parameters():
                param.requires_grad = False
               
    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class BaselineTreeSegmentationModel(nn.Module):
    """
    For Training without SimCLR
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = BaselineResNet50Encoder()
        self.decoder = ResNet50UNetDecoder(num_classes)
        
    def forward(self, x):
        features = self.encoder.get_layer_features(x)
        return self.decoder(features)