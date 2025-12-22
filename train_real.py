import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os

# 设置随机种子（确保可复现）
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True  

from models.adapter_transformer_real import LightweightAdapterReal
from dataset_cornell_pure import CornellPureDataset

def combined_loss(pred_mean, pred_std, target, alpha=0.5):
    """组合损失函数（NLL + L1，提升对异常值的鲁棒性）"""
    nll_loss = 0.5 * torch.log(pred_std**2) + 0.5 * ((target - pred_mean) ** 2) / pred_std**2
    l1_loss = torch.abs(target - pred_mean) / pred_std  # 标准化L1损失
    return (1 - alpha) * nll_loss.mean() + alpha * l1_loss.mean()

def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    total_position_loss = 0
    
    for batch in tqdm(dataloader, desc="训练中"):
        images = batch['image'].to(device)
        position_true = batch['position'].to(device)
        
        outputs = model(images)
        
        # 使用组合损失（核心优化：降低异常值影响）
        loss = combined_loss(
            outputs['position'], 
            outputs['uncertainty'], 
            position_true
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    position_errors = []
    uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            images = batch['image'].to(device)
            position_true = batch['position'].to(device)
            
            outputs = model(images)
            pred_pos = outputs['position']
            pred_std = outputs['uncertainty']
            
            # 计算误差
            error = torch.norm(pred_pos - position_true, dim=1)
            position_errors.extend(error.cpu().numpy())
            uncertainties.extend(pred_std.mean(dim=1).cpu().numpy())
    
    # 计算关键指标
    rmse = np.sqrt(np.mean(np.square(position_errors)))
    mean_error = np.mean(position_errors)
    median_error = np.median(position_errors)
    # 新增：不确定性与误差相关性（评估模型置信度是否可靠）
    uncertainty_corr = np.corrcoef(position_errors, uncertainties)[0, 1]
    
    return {
        'rmse': rmse,
        'mean_error': mean_error,
        'median_error': median_error,
        'uncertainty_corr': uncertainty_corr
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据（新增随机种子确保划分一致）
    train_dataset = CornellPureDataset(
        'data/cornell', 
        split='train', 
        augment=True, 
        random_seed=SEED
    )
    val_dataset = CornellPureDataset(
        'data/cornell', 
        split='val', 
        augment=False, 
        random_seed=SEED
    )
    
    # 优化数据加载（增大batch_size，使用pin_memory加速）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # 从8增至16（加速训练，需保证显存足够）
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 增大模型容量（核心优化：提升特征提取能力）
    model = LightweightAdapterReal(bottleneck_dim=128).to(device)  # 从64增至128
    
    # 统计参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%} 可训练)")
    
    # 优化优化器参数（调整学习率和权重衰减）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,  # 从1e-4提高到3e-4（加速前期收敛）
        weight_decay=0.005  # 降低权重衰减，减少过拟合抑制
    )
    
    # 改进学习率调度器（更灵敏的调整）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5, min_lr=1e-6
    )
    
    best_rmse = float('inf')
    patience = 10  # 从15减至10，避免过晚停止
    no_improve = 0
    
    print("\n开始训练...")
    print("="*60)
    
    for epoch in range(150):  # 增加训练轮次至150
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        
        # 打印详细指标（含不确定性相关性）
        print(f"第 {epoch+1:03d}/150轮 | 训练损失: {train_loss:.4f} | "
              f"RMSE: {metrics['rmse']:.2f} | 平均误差: {metrics['mean_error']:.2f} | "
              f"不确定度相关性: {metrics['uncertainty_corr']:.2f}")
        
        # 学习率调度
        scheduler.step(metrics['rmse'])
        
        # 早停与模型保存
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rmse': best_rmse,
                'metrics': metrics
            }
            torch.save(checkpoint, 'best_adapter_transformer_model.pth')
            print(f"✅ 保存最佳模型！RMSE: {best_rmse:.2f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⛔ 提前停止，最佳RMSE: {best_rmse:.2f}")
                break
    
    # 最终报告
    print("\n" + "="*60)
    print("训练完成！最佳性能:")
    checkpoint = torch.load('best_adapter_transformer_model.pth', weights_only=False)
    best_metrics = checkpoint['metrics']
    print(f"  验证集 RMSE: {best_metrics['rmse']:.2f}像素")
    print(f"  验证集 平均误差: {best_metrics['mean_error']:.2f}像素")
    print(f"  验证集 中位数误差: {best_metrics['median_error']:.2f}像素")
    print(f"  模型参数: {trainable_params:,}")
    print("="*60)

if __name__ == '__main__':
    main()