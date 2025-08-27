"""
ResNet50 Encoder for SimCLR and segmentation tasks.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Encoder(nn.Module):
    """
    ResNet50 Encoder for SimCLR decoder
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
       
        self.backbone = models.resnet50(pretrained=True)
        self.feature_dim = 2048
       
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
       
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.feature_dim, hidden_dim)
       
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        return self.fc(pooled), features
   
    def get_layer_features(self, x):
        features = []
       
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # batchnormalization
        x = self.backbone[2](x)  # relu
        features.append(x)       # 64 channels, H/2 x W/2
       
        x = self.backbone[3](x)  # maxpool
       
        x = self.backbone[4](x)  # layer1
        features.append(x)       # 256 channels, H/4 x W/4
       
        x = self.backbone[5](x)  # layer2
        features.append(x)       # 512 channels, H/8 x W/8
       
        x = self.backbone[6](x)  # layer3
        features.append(x)       # 1024 channels, H/16 x W/16
       
        x = self.backbone[7](x)  # layer4
        features.append(x)       # 2048 channels, H/32 x W/32
       
        return features


class BaselineResNet50Encoder(nn.Module):
    """
    Baseline ResNet50 encoder that uses ImageNet pretrained weights
    without any SimCLR pretraining on the same dataset
    """
    def __init__(self):
        super().__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
    def get_layer_features(self, x):
        features = []
        
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # batchnorm1
        x = self.backbone[2](x)  # relu
        features.append(x)       # 64 channels, H/2 x W/2
        
        x = self.backbone[3](x)  # maxpool
        
        x = self.backbone[4](x)  # layer1
        features.append(x)       # 256 channels, H/4 x W/4
        
        x = self.backbone[5](x)  # layer2
        features.append(x)       # 512 channels, H/8 x W/8
        
        x = self.backbone[6](x)  # layer3
        features.append(x)       # 1024 channels, H/16 x W/16
        
        x = self.backbone[7](x)  # layer4
        features.append(x)       # 2048 channels, H/32 x W/32
        
        return features


class ResNet50SegmentationEncoder(nn.Module):
    """
    Encoder wrapper for feature extraction using (pretrained) ResNet50
    """
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.resnet50_encoder = pretrained_encoder
       
    def forward(self, x):
        return self.resnet50_encoder.get_layer_features(x)
