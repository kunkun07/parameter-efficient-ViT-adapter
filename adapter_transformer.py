import torch
import torch.nn as nn
from transformers import ViTModel

class LightweightAdapterReal(nn.Module):
    """RTX 5060优化的轻量级抓取模型"""
    def __init__(self, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        
        # 冻结ViT主干
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        for param in self.vit.parameters():
            param.requires_grad = False
            param.grad = None
        
        # Adapter层（可训练）
        hidden_dim = self.vit.config.hidden_size
        self.adapter_down = nn.Linear(hidden_dim, bottleneck_dim)
        self.adapter_up = nn.Linear(bottleneck_dim, hidden_dim)
        self.adapter_act = nn.GELU()
        self.adapter_dropout = nn.Dropout(dropout)
        
        # 任务头：位置回归
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # x,y坐标
        )
        
        # 不确定性估计（预测置信度）
        self.uncertainty_head = nn.Linear(hidden_dim, 2)  # x,y的标准差
    
    def forward(self, x):
        # ViT编码（冻结）
        with torch.no_grad():
            vit_output = self.vit(pixel_values=x)
            features = vit_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Adapter（可训练）
        adapted = self.adapter_act(self.adapter_up(self.adapter_down(features)))
        adapted = self.adapter_dropout(adapted)
        features = features + adapted
        
        # 预测
        position = self.position_head(features)
        uncertainty = torch.exp(self.uncertainty_head(features))  # 标准差
        
        return {
            'position': position,
            'uncertainty': uncertainty
        }
