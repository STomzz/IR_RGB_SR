import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rgb_extractor import FourierFeatureExtractor
from model.ir_extractor import InfraredFeatureExtractor


class FeatureFusion(nn.Module):
    """特征融合类 - 融合红外和傅里叶特征"""
    def __init__(self, ir_extractor, fourier_extractor, base_channels=16):
        """
        参数:
          ir_extractor: 红外特征提取器实例
          fourier_extractor: 傅里叶特征提取器实例
          base_channels: 基础通道数
        """
        super().__init__()
        self.ir_extractor = ir_extractor#(torch.Size([1, 64, 256, 256]))
        self.fourier_extractor = fourier_extractor#torch.Size([1, 16, 256, 256])
        
        # 验证特征图尺寸匹配
        # assert ir_extractor.feat_H == fourier_extractor.H
        # assert ir_extractor.feat_W == fourier_extractor.W
        
        # 获取特征通道数
        ir_channels = ir_extractor.final_channels
        fourier_channels = fourier_extractor.final_channels
        
        # 融合层
        self.fusion_convs = nn.Sequential(
            # 初始融合 - 减少通道数
            nn.Conv2d(ir_channels + fourier_channels, base_channels*8, 1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*8),
            
            # 空间特征融合
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*8),
            
            # 通道特征融合
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*4),
            
            # 最终融合
            nn.Conv2d(base_channels*4, base_channels*4, 1),
            nn.ReLU(),
        )
        
        # 自适应输出层
        self.output_conv = nn.Conv2d(base_channels*4, base_channels*4, 1)
        
        # 输出归一化
        self.norm = nn.BatchNorm2d(base_channels*4)
        
    def forward(self, ir_image, visible_image):
        """
        前向传播
        参数:
          ir_image: 红外图像 [B, 1, H, W]
          visible_image: 可见光图像 [B, 3, H, W]
        """
        # 提取红外特征
        ir_features = self.ir_extractor(ir_image)['enhanced_features']
        
        # 提取傅里叶特征（使用可见光图像）
        fourier_features = self.fourier_extractor(visible_image)['filtered_spatial']
        
        # 特征融合 - 通道维度拼接
        fused_features = torch.cat([ir_features, fourier_features], dim=1)
        
        # 融合处理
        fused = self.fusion_convs(fused_features)
        
        # 输出处理
        output = self.output_conv(fused)
        output = self.norm(output)
        
        return {
            'ir_features': ir_features,
            'fourier_features': fourier_features,
            'fused_features': fused,
            'output': output
        }

class InfraredFourierFusionModel(nn.Module):
    """端到端红外与傅里叶特征融合模型"""
    def __init__(self, img_size=(256, 256), origianl_channels=16):
        super().__init__()
        # 创建特征提取器
        self.ir_extractor = InfraredFeatureExtractor(
            in_channels=1, 
            base_channels=origianl_channels, 
            img_size=img_size
        )
        self.fourier_extractor = FourierFeatureExtractor(
            in_channels=3,  # RGB图像输入
            out_channels=origianl_channels, 
            img_size=img_size
        )
        
        # 创建融合模块
        self.fusion_module = FeatureFusion(
            ir_extractor=self.ir_extractor,
            fourier_extractor=self.fourier_extractor,
            base_channels=origianl_channels
        )
        
        # 创建恢复头
        self.augmente_head = nn.Sequential(
            # 保持尺寸的卷积层（使用kernel_size=1进行通道压缩）
            nn.Conv2d(origianl_channels*4, origianl_channels*2, 1),  # 1x1卷积保持尺寸
            nn.ReLU(),
            nn.Conv2d(origianl_channels*2, origianl_channels, 1),   # 继续压缩通道
            nn.ReLU(),
            # 最终输出通道为1
            nn.Conv2d(origianl_channels, 1, 3, padding=1),  # 3x3卷积保持尺寸
            nn.Sigmoid()
        )
    
    def forward(self, ir_image, visible_image):
        # 提取和融合特征
        features = self.fusion_module(ir_image, visible_image)
        
        # 去噪测试
        output =  self.augmente_head(features['output'])
        
        
        # 输出
        return output
        


# ----------------------- 测试代码 -----------------------
def test_fusion_model():
    # 创建测试数据
    batch_size = 2
    img_size = 256
    
    # 红外图像 (1通道)
    ir_images = torch.rand(batch_size, 1, img_size, img_size)
    
    # 可见光图像 (3通道)
    visible_images = torch.rand(batch_size, 3, img_size, img_size)
    
    # 创建模型
    model = InfraredFourierFusionModel(
        img_size=(img_size, img_size),
        origianl_channels=16
    )
    
    print("模型结构:")
    print(model)
    
    # 前向传播
    outputs = model(ir_images, visible_images)
    
    # 打印输出尺寸
    print("\n输出尺寸:")
    print(f"红外特征: {outputs['ir_features'].shape}")
    print(f"傅里叶特征: {outputs['fourier_features'].shape}")
    print(f"融合特征: {outputs['fused_features'].shape}")
    print(f"最终输出: {outputs['output'].shape}")
    print(f"分割结果: {outputs['segmentation'].shape}")
    
    # 训练测试
    print("\n训练融合模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 模拟分割目标
    targets = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float()
    
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(ir_images, visible_images)
        loss = criterion(outputs['segmentation'], targets)
        loss.backward()
        optimizer.step()
        
        # 计算IOU
        preds = (outputs['segmentation'] > 0.5).float()
        intersection = (preds * targets).sum()
        union = (preds + targets).clamp(0, 1).sum()
        iou = intersection / (union + 1e-6)
        
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}, IoU: {iou.item():.4f}")

if __name__ == "__main__":
    test_fusion_model()