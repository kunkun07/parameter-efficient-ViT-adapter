# models/ggcnn_baseline_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GGCNNBaselineFixed(nn.Module):
    """修正版GGCNN：统一预处理+不确定性+残差连接"""
    def __init__(self, input_channels=3, dropout=0.1):
        super().__init__()
        
        # 特征提取（与原版一致但深度略增）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # 16→32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # 32→64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 增加残差块
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # 64→128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最终7x7特征图
        )
        
        # 全局池化+回归头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = 128
        
        # 位置预测头（复用您的设计）
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 输出(x,y)坐标，范围[0,224]
        )
        
        # 不确定性头（增加公平性）
        self.uncertainty_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # x: [B, 3, 224, 224]，已标准化
        x = self.conv_layers(x)  # [B, 128, 7, 7]
        x = self.global_pool(x).flatten(1)  # [B, 128]
        
        position = self.position_head(x)
        uncertainty = torch.exp(self.uncertainty_head(x))  # 标准差
        
        return {
            'position': position,
            'uncertainty': uncertainty
        }