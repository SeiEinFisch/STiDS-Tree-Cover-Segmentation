class BaselineResNet50Encoder(nn.Module):
    
    """
    Baseline ResNet50 encoder that uses ImageNet pretrained weights
    without any SimCLR pretraining on the same dataset
    """
import torch.nn as nn
import torchvision.models as models
from .segmentation_models import ResNet50UNetDecoder

class BaselineResNet50Encoder(nn.Module):
    """
    Baseline ResNet50 encoder that uses ImageNet pretrained weights
    without any SimCLR pretraining on the same dataset
    """
    def __init__(self):
        super().__init__()
        
        # Load ImageNet pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove the final classification layer and global average pooling
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
    def get_layer_features(self, x):
        """Extract features from different layers for U-Net skip connections"""
        features = []
        
        # Initial conv and pooling
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # batchnorm1
        x = self.backbone[2](x)  # relu
        features.append(x)       # 64 channels, H/2 x W/2
        
        x = self.backbone[3](x)  # maxpool
        
        # ResNet blocks
        x = self.backbone[4](x)  # layer1
        features.append(x)       # 256 channels, H/4 x W/4
        
        x = self.backbone[5](x)  # layer2
        features.append(x)       # 512 channels, H/8 x W/8
        
        x = self.backbone[6](x)  # layer3
        features.append(x)       # 1024 channels, H/16 x W/16
        
        x = self.backbone[7](x)  # layer4
        features.append(x)       # 2048 channels, H/32 x W/32
        
        return features


# For Training without simCLR
class BaselineTreeSegmentationModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = BaselineResNet50Encoder()
        self.decoder = ResNet50UNetDecoder(num_classes)  # Reuse decoder
        
    def forward(self, x):
        features = self.encoder.get_layer_features(x)
        return self.decoder(features)