"""
ResNet50 Decoder
"""
import torch
import torch.nn as nn


class ResNet50UNetDecoder(nn.Module):
    """
    U-Net style decoder adapted to ResNet50 encoder features. Upsamples feature maps from encoder while using skip connections.
    """
    def __init__(self, num_classes=2):
        super().__init__()
       
        self.channels = [64, 256, 512, 1024, 2048]
       
        self.up1 = nn.ConvTranspose2d(self.channels[4], self.channels[3], 2, stride=2)
        self.conv1 = self._make_conv_block(self.channels[3] * 2, self.channels[3])
       
        self.up2 = nn.ConvTranspose2d(self.channels[3], self.channels[2], 2, stride=2)
        self.conv2 = self._make_conv_block(self.channels[2] * 2, self.channels[2])
       
        self.up3 = nn.ConvTranspose2d(self.channels[2], self.channels[1], 2, stride=2)
        self.conv3 = self._make_conv_block(self.channels[1] * 2, self.channels[1])
       
        self.up4 = nn.ConvTranspose2d(self.channels[1], self.channels[0], 2, stride=2)
        self.conv4 = self._make_conv_block(self.channels[0] * 2, self.channels[0])
       
        self.up5 = nn.ConvTranspose2d(self.channels[0], 32, 2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, 1)
       
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
   
    def forward(self, features):
        x = features[4]
       
        x = self.up1(x)
        x = torch.cat([x, features[3]], dim=1)
        x = self.conv1(x)
       
        x = self.up2(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.conv2(x)
       
        x = self.up3(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.conv3(x)
       
        x = self.up4(x)
        x = torch.cat([x, features[0]], dim=1)
        x = self.conv4(x)
       
        x = self.up5(x)
        return self.final_conv(x)