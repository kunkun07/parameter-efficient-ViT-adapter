import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from models.ggcnn_baseline import GGCNNBaselineFixed 
from dataset_cornell_pure import CornellPureDataset, IMAGE_MEAN, IMAGE_STD

# ==================== 核心修正1: 使用与主模型相同的组合损失 ====================
def ggcnn_loss(pred_mean, pred_std, target, alpha=0.5):
    """GGCNN也使用NLL+L1组合损失（公平对比）"""
    nll_loss = 0.5 * torch.log(pred_std**2) + 0.5 * ((target - pred_mean) ** 2) / pred_std**2
    l1_loss = torch.abs(target - pred_mean) / pred_std
    return (1 - alpha) * nll_loss.mean() + alpha * l1_loss.mean()

def train_epoch_ggcnn_fixed(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="GGCNN训练中"):
        # ==================== 核心修正2: 统一预处理（与Adapter一致） ====================
        images = batch['image'].to(device)  # 已归一化到[0,1]并标准化
        position_true = batch['position'].to(device)  # 范围[0,224]
        
        outputs = model(images)
        # 直接预测像素坐标，无需反归一化
        loss = ggcnn_loss(
            outputs['position'], 
            outputs['uncertainty'], 
            position_true, 
            alpha=0.5  # 与主模型一致
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_ggcnn_fixed(model, dataloader, device):
    model.eval()
    position_errors = []
    uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="GGCNN评估中"):
            images = batch['image'].to(device)
            position_true = batch['position'].to(device)
            
            outputs = model(images)
            pred_pos = outputs['position']
            pred_std = outputs['uncertainty']
            
            error = torch.norm(pred_pos - position_true, dim=1)
            position_errors.extend(error.cpu().numpy())
            uncertainties.extend(pred_std.mean(dim=1).cpu().numpy())
    
    rmse = np.sqrt(np.mean(np.square(position_errors)))
    mean_error = np.mean(position_errors)
    median_error = np.median(position_errors)
    # ==================== 核心修正3: 计算不确定性相关性 ====================
    uncertainty_corr = np.corrcoef(position_errors, uncertainties)[0, 1] if len(uncertainties) > 0 else 0
    
    return {
        'rmse': rmse,
        'mean_error': mean_error,
        'median_error': median_error,
        'uncertainty_corr': uncertainty_corr
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GGCNN使用设备: {device}")
    
    # ==================== 核心修正4: 完全一致的数据加载 ====================
    train_dataset = CornellPureDataset('data/cornell', split='train', augment=True)
    val_dataset = CornellPureDataset('data/cornell', split='val', augment=False)
    
    # 检查数据范围
    sample = train_dataset[0]
    print(f"标签范围检查 - min: {sample['position'].min()}, max: {sample['position'].max()}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    model = GGCNNBaselineFixed().to(device)
    
    # 统计参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GGCNN可训练参数: {trainable_params:,}")
    
    # ==================== 核心修正5: 优化器超参调整 ====================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,  # 降低学习率（原1e-3太大）
        weight_decay=0.01  # 增加权重衰减防过拟合
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=200, steps_per_epoch=len(train_loader)
)  # 使用OneCycle加速收敛
    
    
    best_rmse = float('inf')
    patience = 15  # 增加耐心
    
    print("\n开始GGCNN训练...")
    print("="*60)
    
    for epoch in range(150):  # 增加训练轮次
        train_loss = train_epoch_ggcnn_fixed(model, train_loader, optimizer, device)
        metrics = evaluate_ggcnn_fixed(model, val_loader, device)
        
        print(f"第 {epoch+1:03d}/150轮 | 训练损失: {train_loss:.4f} | "
              f"RMSE: {metrics['rmse']:.2f} | 中位数误差: {metrics['median_error']:.2f} | "
              f"不确定度相关性: {metrics['uncertainty_corr']:.2f}")
        
        scheduler.step(metrics['rmse'])
        
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rmse': best_rmse,
                'metrics': metrics
            }
            torch.save(checkpoint, 'best_ggcnn_baseline_model.pth')
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
    checkpoint = torch.load('best_ggcnn_baseline_model.pth', weights_only=False)
    best_metrics = checkpoint['metrics']
    print(f"  验证集 RMSE: {best_metrics['rmse']:.2f}像素")
    print(f"  验证集 平均误差: {best_metrics['mean_error']:.2f}像素")
    print(f"  验证集 中位数误差: {best_metrics['median_error']:.2f}像素")
    print(f"  模型参数: {trainable_params:,}")
    print("="*60)
    
if __name__ == '__main__':
    main()