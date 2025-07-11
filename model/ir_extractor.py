import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import GaussianBlur

class InfraredFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, img_size=(256, 256)):
        """
        红外图像特征提取器
        特点:
          1. 使用卷积提取空间特征
          2. 通过Pooling-MLP-Softmax生成通道注意力权重
          3. 将注意力权重与可学习滤波器结合筛选特征
          4. 无分支设计，特征图尺寸固定
        
        参数:
          in_channels: 输入通道数 (红外图像通常为1)
          base_channels: 基础通道数
          img_size: 图像尺寸 (H, W)，用于初始化滤波器尺寸
        """
        super().__init__()
        self.img_size = img_size
        self.H, self.W = img_size
        
        # 计算卷积后的特征图尺寸
        self.feat_H = img_size[0]
        self.feat_W = img_size[1]
        
        # 1. 卷积特征提取层 - 固定输出尺寸
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels),
            
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*2),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*4)
        )
        self.final_channels = base_channels * 4
        
        # 2. 通道注意力机制 (Pooling-MLP-Softmax)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.final_channels, self.final_channels // 4),
            nn.ReLU(),
            nn.Linear(self.final_channels // 4, self.final_channels),
            nn.Softmax(dim=1)
        )
        
        # 3. 可学习滤波器 - 固定尺寸
        self.filter = nn.Parameter(torch.ones(1, self.final_channels, self.feat_H, self.feat_W))
        self.filter_bias = nn.Parameter(torch.zeros(1, self.final_channels, 1, 1))
        
        # 4. 特征增强层
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(self.final_channels, self.final_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.final_channels // 2, self.final_channels, 3, padding=1)
        )
        
        # 5. 输出归一化层
        self.norm = nn.BatchNorm2d(self.final_channels)

    def forward(self, x):
        
        # 1. 卷积特征提取 - 尺寸固定为 (H, W)
        features = self.conv_layers(x)
        batch_size, channels, H, W = features.shape
        
        # 2. 生成通道注意力权重
        pooled = self.global_pool(features).view(batch_size, -1)
        channel_weights = self.channel_mlp(pooled).view(batch_size, channels, 1, 1)
        
        # 3. 应用通道注意力权重
        weighted_features = features * channel_weights
        
        # 4. 应用固定尺寸的可学习滤波器
        filtered_features = weighted_features * self.filter + self.filter_bias
        
        # 5. 特征增强
        enhanced_features = self.enhance_conv(filtered_features) + filtered_features  # 残差连接
        
        # 6. 特征激活与归一化
        activated_features = F.relu(enhanced_features)
        normalized_features = self.norm(activated_features)
        
        return {
            'raw_features': features,
            'channel_weights': channel_weights,
            'weighted_features': weighted_features,
            'filtered_features': filtered_features,
            'enhanced_features': normalized_features,
            'learned_filter': self.filter
        }


def create_infrared_test_tensor(size=256, channels=1):
    """创建红外风格的测试张量"""
    # 确保尺寸能被4整除
    size = (size // 4) * 4
    # 创建基础网格
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # 创建热源点
    heat_sources = torch.zeros(size, size)
    for i in range(5):
        cx, cy = np.random.rand(), np.random.rand()
        intensity = 0.5 + np.random.rand() * 0.5
        std = 0.05 + np.random.rand() * 0.1
        heat_source = torch.exp(-((grid_x - cx)**2 + (grid_y - cy)**2) / (2 * std**2))
        heat_sources += intensity * heat_source
    
    # 添加热梯度
    heat_gradient = grid_x * 0.3 + grid_y * 0.2
    
    # 添加低频噪声
    noise = torch.randn(size, size) * 0.1 * torch.sin(2 * np.pi * 2 * grid_x) * torch.sin(2 * np.pi * 2 * grid_y)
    
    # 组合成红外图像
    infrared_img = heat_sources + heat_gradient + noise
    
    # 归一化并添加通道维度
    infrared_img = (infrared_img - infrared_img.min()) / (infrared_img.max() - infrared_img.min())
    return infrared_img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

def visualize_results(results, original_img,model):
    #解决中文标题乱码
    plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    """可视化特征提取结果"""
    # 获取结果
    raw_features = results['raw_features'][0].detach()
    channel_weights = results['channel_weights'][0].detach()
    weighted_features = results['weighted_features'][0].detach()
    filtered_features = results['filtered_features'][0].detach()
    enhanced_features = results['enhanced_features'][0].detach()
    learned_filter = results['learned_filter'][0].detach()
    
    # 可视化
    plt.figure(figsize=(20, 15))
    
    # 原始红外图像
    plt.subplot(3, 4, 1)
    plt.imshow(original_img[0, 0], cmap='hot')
    plt.title("原始红外图像")
    plt.colorbar()
    
    # 通道权重分布
    plt.subplot(3, 4, 2)
    weights = channel_weights.squeeze().numpy()
    plt.bar(range(len(weights)), weights)
    plt.title("通道注意力权重")
    plt.xlabel("通道索引")
    plt.ylabel("权重")
    plt.grid(True, alpha=0.3)
    
    # 可学习滤波器
    plt.subplot(3, 4, 3)
    filter_mean = learned_filter.mean(dim=0).detach()
    plt.imshow(filter_mean, cmap='viridis')
    plt.title("可学习滤波器 (均值)")
    plt.colorbar()
    
    # 滤波器权重分布
    plt.subplot(3, 4, 4)
    filter_values = filter_mean.flatten().numpy()
    plt.hist(filter_values, bins=20, alpha=0.7)
    plt.title("滤波器权重分布")
    plt.xlabel("权重值")
    plt.ylabel("频率")
    
    # 原始特征图
    plt.subplot(3, 4, 5)
    plt.imshow(raw_features.mean(dim=0), cmap='viridis')
    plt.title("原始特征图 (通道平均)")
    plt.colorbar()
    
    # 加权后特征图
    plt.subplot(3, 4, 6)
    plt.imshow(weighted_features.mean(dim=0), cmap='viridis')
    plt.title("加权后特征图 (通道平均)")
    plt.colorbar()
    
    # 滤波后特征图
    plt.subplot(3, 4, 7)
    plt.imshow(filtered_features.mean(dim=0), cmap='viridis')
    plt.title("滤波后特征图 (通道平均)")
    plt.colorbar()
    
    # 增强后特征图
    plt.subplot(3, 4, 8)
    plt.imshow(enhanced_features.mean(dim=0), cmap='viridis')
    plt.title("增强后特征图 (通道平均)")
    plt.colorbar()
    
    # 权重最高的原始特征通道
    top_channel = weights.argmax()
    plt.subplot(3, 4, 9)
    plt.imshow(raw_features[top_channel], cmap='viridis')
    plt.title(f"权重最高通道 ({top_channel}) - 原始")
    plt.colorbar()
    
    # 权重最高通道滤波后
    plt.subplot(3, 4, 10)
    plt.imshow(filtered_features[top_channel], cmap='viridis')
    plt.title(f"权重最高通道 ({top_channel}) - 滤波后")
    plt.colorbar()
    
    # 权重最高通道增强后
    plt.subplot(3, 4, 11)
    plt.imshow(enhanced_features[top_channel], cmap='viridis')
    plt.title(f"权重最高通道 ({top_channel}) - 增强后")
    plt.colorbar()
    
    
    plt.tight_layout()
    plt.show()

def test_infrared_extractor():
    # 创建红外测试张量 - 尺寸确保能被4整除
    size = 256
    infrared_tensor = create_infrared_test_tensor(size=size)
    print("红外图像张量形状:", infrared_tensor.shape)
    
    # 创建模型
    model = InfraredFeatureExtractor(
        in_channels=1, 
        base_channels=16, 
        img_size=(size, size)
    )
    print(f"特征图输出尺寸: {model.feat_H}x{model.feat_W}")
    
    # 前向传播
    results = model(infrared_tensor)
    print("特征图实际尺寸:", results['enhanced_features'].shape[2:])
    
    # 可视化结果
    visualize_results(results, infrared_tensor,model)
    
    # 训练测试
    print("\n训练红外特征提取器...")
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # # 简单的自编码目标
    # target = torch.randn_like(results['enhanced_features'])
    
    # for epoch in range(10):
    #     optimizer.zero_grad()
    #     results = model(infrared_tensor)
    #     loss = F.mse_loss(results['enhanced_features'], target)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
    
    # # 训练后可视化
    # print("\n训练后结果:")
    # results = model(infrared_tensor)
    # visualize_results(results, infrared_tensor,model)
    
    # # 打印训练后的参数
    # print(f"学习到的温度缩放因子: {model.temperature_scaler.item():.4f}")
    # print(f"滤波器均值: {model.filter.mean().item():.4f}, 标准差: {model.filter.std().item():.4f}")

# 运行测试
if __name__ == "__main__":
    test_infrared_extractor()