import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random

from grasp_utils import decode_angle_vector, encode_angle_radians

# 新增：图像标准化参数（与预训练模型对齐）
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet均值
IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet标准差

class CornellPureDataset(Dataset):
    """优化版Cornell数据集加载器（提升标注准确性和数据多样性）"""
    def __init__(self, data_dir, split='train', augment=False, random_seed=42):
        self.data_dir = data_dir
        self.augment = augment
        self.random_seed = random_seed
        random.seed(random_seed)
        
        self.image_files = []
        self.grasp_centers = []  # 直接存储中心点（避免重复计算）
        self.grasp_angle_vectors = []
        self.valid_masks = []    # 存储物体掩码（用于验证中心点是否在物体上）
        
        # 加载并验证数据（核心优化：确保标注质量）
        self._load_and_validate_data()
        
        # 优化：随机划分训练/验证集（替换顺序划分，避免分布不均）
        train_indices, val_indices = train_test_split(
            range(len(self.image_files)),
            test_size=0.2,
            random_state=random_seed,
            shuffle=True
        )
        
        if split == 'train':
            self.indices = train_indices
        else:
            self.indices = val_indices
        
        print(f"{split}集: {len(self.indices)} 样本 (总有效样本: {len(self.image_files)})")

    @staticmethod
    def _reshape_rectangles(raw_rects):
        raw_rects = np.asarray(raw_rects, dtype=np.float32)

        if raw_rects.ndim == 1:
            if raw_rects.size == 8:
                return raw_rects.reshape(1, 4, 2)
            if raw_rects.size % 2 != 0:
                raise ValueError('Cornell标注维度异常，无法重组为矩形')
            raw_rects = raw_rects.reshape(-1, 2)

        if raw_rects.ndim == 2 and raw_rects.shape[1] == 8:
            return raw_rects.reshape(-1, 4, 2)

        if raw_rects.ndim == 2 and raw_rects.shape[1] == 2:
            if raw_rects.shape[0] % 4 != 0:
                raise ValueError('Cornell标注行数不是4的倍数，无法按抓取矩形解析')
            return raw_rects.reshape(-1, 4, 2)

        raise ValueError(f'不支持的Cornell标注格式: {raw_rects.shape}')

    @staticmethod
    def _extract_center_and_angle(rect_points):
        center = rect_points.mean(axis=0)

        edge_a = rect_points[1] - rect_points[0]
        edge_b = rect_points[2] - rect_points[3]
        grasp_direction = edge_a + edge_b
        if np.linalg.norm(grasp_direction) < 1e-6:
            grasp_direction = edge_a

        angle = np.arctan2(grasp_direction[1], grasp_direction[0])
        return center, encode_angle_radians(angle)

    @staticmethod
    def _angle_vector_to_direction(angle_vector):
        angle_radians = float(decode_angle_vector(angle_vector))
        return np.array([np.cos(angle_radians), np.sin(angle_radians)], dtype=np.float32)

    @staticmethod
    def _direction_to_angle_vector(direction):
        direction = np.asarray(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        direction = direction / norm
        angle = np.arctan2(direction[1], direction[0])
        return encode_angle_radians(angle)

    def _load_and_validate_data(self):
        """加载数据并验证标注（确保中心点在物体区域）"""
        for folder_idx in range(1, 11):
            folder = f'{self.data_dir}/{folder_idx:02d}'
            if not os.path.exists(folder):
                print(f"警告: 文件夹 {folder} 不存在，已跳过")
                continue
            
            # 只处理有深度图的样本（过滤无效图像）
            depth_files = [f for f in os.listdir(folder) if f.endswith('d.tiff')]
            for depth_file in depth_files:
                img_file = depth_file.replace('d.tiff', 'r.png')
                pos_file = depth_file.replace('d.tiff', 'cpos.txt')
                
                img_path = os.path.join(folder, img_file)
                pos_path = os.path.join(folder, pos_file)
                depth_path = os.path.join(folder, depth_file)
                
                # 检查文件完整性
                if not all(os.path.exists(p) for p in [img_path, pos_path, depth_path]):
                    continue
                
                try:
                    # 读取图像和深度图（用于验证物体区域）
                    image = cv2.imread(img_path)
                    depth = cv2.imread(depth_path, -1)  # 读取深度图（单通道）
                    if image is None or depth is None:
                        continue
                    h, w = image.shape[:2]
                    
                    # 生成物体掩码（深度图中非零区域视为物体）
                    mask = (depth > 0).astype(np.uint8)  # 物体区域为1，背景为0
                    
                    # 读取并处理所有标注（取均值提升鲁棒性）
                    rects = self._reshape_rectangles(np.loadtxt(pos_path, dtype=np.float32))
                    
                    valid_centers = []
                    valid_angle_vectors = []
                    for rect in rects:
                        if not np.isfinite(rect).all():
                            continue

                        center, angle_vector = self._extract_center_and_angle(rect)
                        center_x, center_y = center

                        if not np.isfinite(center_x) or not np.isfinite(center_y):
                            continue
                        
                        # 验证中心点是否在物体区域（核心优化：过滤背景标注）
                        cx_int, cy_int = int(round(center_x)), int(round(center_y))
                        if 0 <= cx_int < w and 0 <= cy_int < h and mask[cy_int, cx_int] == 1:
                            valid_centers.append([center_x, center_y])
                            valid_angle_vectors.append(angle_vector)
                    
                    # 只保留有有效标注的样本
                    if valid_centers and valid_angle_vectors:
                        final_center = np.mean(valid_centers, axis=0)  # 多标注取均值
                        final_angle = np.mean(valid_angle_vectors, axis=0)
                        if not np.isfinite(final_center).all() or not np.isfinite(final_angle).all():
                            continue
                        final_angle = final_angle / max(np.linalg.norm(final_angle), 1e-6)
                        self.image_files.append(img_path)
                        self.grasp_centers.append(final_center)
                        self.grasp_angle_vectors.append(final_angle.astype(np.float32))
                        self.valid_masks.append(mask)  # 保存掩码用于后续调试
                        
                except Exception as e:
                    print(f"处理文件 {pos_path} 出错: {str(e)}，已跳过")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.image_files[real_idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        center = self.grasp_centers[real_idx].copy()
        angle_vector = self.grasp_angle_vectors[real_idx].copy()
        
        # 增强数据增强策略（提升泛化能力）
        if self.augment:
            # 1. 随机水平翻转
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                center[0] = w - center[0]  # 同步翻转x坐标
                angle_vector[1] = -angle_vector[1]
            
            # 2. 随机旋转（-10~10度）
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            
            # 旋转中心点坐标
            center_hom = np.array([center[0], center[1], 1])
            rotated_center = M @ center_hom
            center[0], center[1] = rotated_center[0], rotated_center[1]

            direction = self._angle_vector_to_direction(angle_vector)
            rotated_direction = M[:, :2] @ direction
            angle_vector = self._direction_to_angle_vector(rotated_direction)
            
            # 3. 随机亮度/对比度调整
            if random.random() > 0.5:
                alpha = random.uniform(0.7, 1.3)  # 对比度
                beta = random.uniform(-30, 30)     # 亮度
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # 确保中心点在图像范围内
        center[0] = np.clip(center[0], 0, w-1)
        center[1] = np.clip(center[1], 0, h-1)
        
        # 缩放到224x224
        scale_x, scale_y = 224 / w, 224 / h
        scaled_center = [center[0] * scale_x, center[1] * scale_y]
        image = cv2.resize(image, (224, 224))
        
        # 图像标准化（核心优化：与预训练模型对齐）
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        for c in range(3):
            image[c] = (image[c] - IMAGE_MEAN[c]) / IMAGE_STD[c]
        
        return {
            'image': image,
            'position': torch.FloatTensor(scaled_center),
            'angle': torch.FloatTensor(angle_vector),
            'original_path': img_path  # 保留路径用于调试
        }

    def visualize_sample(self, idx):
        """可视化样本标注（验证中心点是否在物体上）"""
        sample = self.__getitem__(idx)
        img_path = sample['original_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 还原原始坐标
        scaled_center = sample['position'].numpy()
        orig_center = (int(scaled_center[0] * w/224), int(scaled_center[1] * h/224))
        
        # 绘制中心点和物体掩码
        mask = self.valid_masks[self.indices[idx]]
        mask_vis = np.zeros_like(image)
        mask_vis[mask == 1] = [0, 255, 0]  # 物体区域标为绿色半透明
        image = cv2.addWeighted(image, 0.7, mask_vis, 0.3, 0)
        cv2.circle(image, orig_center, 5, (0, 0, 255), -1)  # 中心点标为红色

        angle_radians = float(decode_angle_vector(sample['angle'].numpy()))
        line_length = max(20, min(h, w) // 8)
        dx = int(np.cos(angle_radians) * line_length)
        dy = int(np.sin(angle_radians) * line_length)
        cv2.line(
            image,
            (orig_center[0] - dx, orig_center[1] - dy),
            (orig_center[0] + dx, orig_center[1] + dy),
            (255, 0, 0),
            2,
        )
        
        cv2.imshow("Sample Visualization", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 测试
if __name__ == '__main__':
    dataset = CornellPureDataset('data/cornell', split='train', augment=False)
    print(f"数据集大小: {len(dataset)}")
    # 随机可视化3个样本，检查标注是否在物体上
    for i in random.sample(range(len(dataset)), 3):
        dataset.visualize_sample(i)