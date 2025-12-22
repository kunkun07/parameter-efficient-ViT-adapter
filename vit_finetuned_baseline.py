# models/vit_finetuned_baseline.py
import torch
import torch.nn as nn
from transformers import ViTModel

class ViTFinetunedBaseline(nn.Module):
    """全参数微调的ViT（无冻结）"""
    def __init__(self, dropout=0.1):
        super().__init__()
        
        # ViT主干（全部可训练）
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # 任务头
        hidden_dim = self.vit.config.hidden_size
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # 不确定性头
        self.uncertainty_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        vit_output = self.vit(pixel_values=x)
        features = vit_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        position = self.position_head(features)
        uncertainty = torch.exp(self.uncertainty_head(features))
        
        return {
            'position': position,
            'uncertainty': uncertainty
        }