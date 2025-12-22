# models/efficientvit_simple.py
import torch
import torch.nn as nn

class EfficientViTSimple(nn.Module):
    """极简EfficientViT-B0实现，仅依赖PyTorch"""
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        
        # Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )
        
        # Mobile Building Blocks (简化版)
        self.blocks = nn.Sequential(
            # Block 1: 112x112
            MBConvBlock(16, 16, kernel_size=3, stride=1, expansion=1, se_ratio=0.25),
            MBConvBlock(16, 16, kernel_size=3, stride=1, expansion=1, se_ratio=0.25),
            
            # Block 2: 56x56
            MBConvBlock(16, 24, kernel_size=3, stride=2, expansion=4, se_ratio=0.25),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expansion=4, se_ratio=0.25),
            
            # Block 3: 28x28
            MBConvBlock(24, 40, kernel_size=3, stride=2, expansion=4, se_ratio=0.25),
            MBConvBlock(40, 40, kernel_size=3, stride=1, expansion=4, se_ratio=0.25),
            
            # Block 4: 14x14 (使用Token Mixer)
            MBConvBlock(40, 80, kernel_size=3, stride=2, expansion=4, se_ratio=0.25, use_token_mixer=True),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion=4, se_ratio=0.25, use_token_mixer=True),
            
            # Block 5: 7x7
            MBConvBlock(80, 112, kernel_size=3, stride=2, expansion=6, se_ratio=0.25),
            MBConvBlock(112, 112, kernel_size=3, stride=1, expansion=6, se_ratio=0.25),
        )
        
        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = 112
        
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.uncertainty_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.stem(x)          # [B, 16, 112, 112]
        x = self.blocks(x)        # [B, 112, 7, 7]
        x = self.global_pool(x).flatten(1)  # [B, 112]
        
        position = self.position_head(x)
        uncertainty = torch.exp(self.uncertainty_head(x))
        
        return {
            'position': position,
            'uncertainty': uncertainty
        }

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck with SE + Token Mixer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion, se_ratio, use_token_mixer=False):
        super().__init__()
        self.stride = stride
        self.use_token_mixer = use_token_mixer
        mid_channels = int(in_channels * expansion)
        
        # Expansion
        if expansion > 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1),
                nn.BatchNorm2d(mid_channels),
                nn.Hardswish()
            )
        else:
            self.expand = nn.Identity()
        
        # Depthwise Conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, 
                     stride=stride, padding=kernel_size//2, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish()
        )
        
        # SE模块
        se_channels = max(1, int(mid_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, se_channels, kernel_size=1),
            nn.Hardswish(),
            nn.Conv2d(se_channels, mid_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Token Mixer (简化版: 池化+1x1卷积)
        if use_token_mixer:
            self.token_mixer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
                nn.Hardswish(),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        identity = x
        
        # Expand
        x = self.expand(x)
        
        # Depthwise
        x = self.depthwise(x)
        
        # SE
        se_weight = self.se(x)
        x = x * se_weight
        
        # Token Mixer
        if self.use_token_mixer:
            token_weight = self.token_mixer(x)
            x = x * token_weight
        
        # Project
        x = self.project(x)
        
        # Skip连接
        if identity.shape == x.shape:
            x = x + identity
        
        return x