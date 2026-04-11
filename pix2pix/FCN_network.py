import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        # ==================== Encoder (Convolutional Layers) ====================
        # Layer 1: 256x256 -> 128x128, 3 channels -> 8 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        
        # Layer 2: 128x128 -> 64x64, 8 channels -> 16 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        
        # Layer 3: 64x64 -> 32x32, 16 channels -> 32 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        
        # Layer 4: 32x32 -> 16x16, 32 channels -> 64 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        
        # ==================== Decoder (Deconvolutional Layers) ====================
        # Layer 1: 16x16 -> 32x32, 64 channels -> 32 channels
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        
        # Layer 2: 32x32 -> 64x64, 32 channels -> 16 channels
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        
        # Layer 3: 64x64 -> 128x128, 16 channels -> 8 channels
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        
        # Layer 4: 128x128 -> 256x256, 8 channels -> 3 channels (RGB output)
        # 注意：最后一层使用Tanh激活函数，将输出归一化到[-1, 1]范围
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]，与训练数据的归一化范围一致
        )

    def forward(self, x):
        # ==================== Encoder forward pass ====================
        # 输入: (B, 3, 256, 256)
        e1 = self.conv1(x)      # (B, 8, 128, 128)
        e2 = self.conv2(e1)     # (B, 16, 64, 64)
        e3 = self.conv3(e2)     # (B, 32, 32, 32)
        e4 = self.conv4(e3)     # (B, 64, 16, 16)
        
        # ==================== Decoder forward pass ====================
        d1 = self.deconv1(e4)   # (B, 32, 32, 32)
        d2 = self.deconv2(d1)   # (B, 16, 64, 64)
        d3 = self.deconv3(d2)   # (B, 8, 128, 128)
        output = self.deconv4(d3)  # (B, 3, 256, 256)
        
        return output
