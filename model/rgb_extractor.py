import torch
import torch.nn as nn
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class LearnableFrequencyFilter(nn.Module):
    def __init__(self, in_channels, img_size, init_type='identity'):
        """
        可学习频域滤波器模块
        
        参数:
            in_channels: 输入通道数
            img_size: 图像尺寸 (H, W)
            init_type: 滤波器初始化类型 ('identity', 'lowpass', 'highpass')
        """
        super().__init__()
        self.img_size = img_size
        self.H, self.W = img_size
        
        # 创建可学习滤波器权重
        if init_type == 'identity':
            init_weight = torch.ones(1, 1, self.H, self.W)
        elif init_type == 'lowpass':
            init_weight = self.create_lowpass_filter()
        elif init_type == 'highpass':
            init_weight = self.create_highpass_filter()
        else:
            raise ValueError(f"未知初始化类型: {init_type}")
            
        self.filter_weight = nn.Parameter(init_weight.repeat(1, in_channels, 1, 1))
        
    def create_lowpass_filter(self, cutoff_ratio=0.3):
        """创建低通滤波器初始化"""
        y, x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
        center_x, center_y = self.W // 2, self.H // 2
        radius = cutoff_ratio * min(self.H, self.W)
        
        # 创建高斯低通滤波器
        dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        lowpass = torch.exp(-(dist**2) / (2 * (radius/3)**2))
        return lowpass.unsqueeze(0).unsqueeze(0)
    
    def create_highpass_filter(self, cutoff_ratio=0.3):
        """创建高通滤波器初始化"""
        lowpass = self.create_lowpass_filter(cutoff_ratio)
        return 1.0 - lowpass
    
    def forward(self, x_fft):
        """
        应用可学习频域滤波器
        
        参数:
            x_fft: 频域特征 [batch, channels, H, W] (复数)
        
        返回:
            滤波后的频域特征
        """
        # 确保滤波器权重与输入尺寸匹配
        if self.filter_weight.shape[-2:] != (self.H, self.W):
            raise ValueError(f"滤波器尺寸 {self.filter_weight.shape[-2:]} 与输入尺寸 {(self.H, self.W)} 不匹配")
        
        # 应用滤波器 (实值权重乘以复数特征)
        filtered_fft = self.filter_weight * x_fft
        return filtered_fft

class FourierFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size
        
        # 卷积特征提取层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )
        
        # 可学习频域滤波器
        self.freq_filter = LearnableFrequencyFilter(out_channels, img_size, init_type='highpass')
        self.final_channels = out_channels
        
    def forward(self, x):
        # 1. 卷积特征提取
        spatial_features = self.conv(x)
        
        # 2. 傅里叶变换
        features_fft = torch.fft.fft2(spatial_features)
        features_fft_shifted = torch.fft.fftshift(features_fft)
        
        # 3. 应用可学习频域滤波器
        filtered_fft = self.freq_filter(features_fft_shifted)
        
        #改进，添加残差链接
        res_filtered_features_fft = features_fft_shifted + filtered_fft
        
        # 4. 反变换回空间域 (可选)
        # filtered_fft_unshifted = torch.fft.ifftshift(filtered_fft)
        # filtered_spatial = torch.fft.ifft2(filtered_fft_unshifted).real
        
        filtered_fft_unshifted = torch.fft.ifftshift(res_filtered_features_fft)
        filtered_spatial = torch.fft.ifft2(filtered_fft_unshifted).real
        
        return {
            'spatial_features': spatial_features,
            'freq_features': features_fft_shifted,
            'filtered_freq': filtered_fft,
            'filtered_spatial': filtered_spatial
        }

def rgb_visualize_results(results, original_img):
    #解决中文标题乱码
    plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    
    """可视化处理结果"""
    # 获取第一个样本的结果
    spatial_feature = results['spatial_features'][0].detach()
    freq_feature = results['freq_features'][0].detach()
    filtered_freq = results['filtered_freq'][0].detach()
    filtered_spatial = results['filtered_spatial'][0].detach()
    
    print(f'滤波提取后的空域特征尺寸是{filtered_spatial.shape}')
    
    # 计算幅度谱
    magnitude = torch.log(torch.abs(freq_feature) + 1)
    filtered_magnitude = torch.log(torch.abs(filtered_freq) + 1)
    
    # 可视化
    plt.figure(figsize=(18, 10))
    
    # 原始图像
    plt.subplot(2, 4, 1)
    plt.imshow(original_img.permute(1, 2, 0))
    plt.title("原始图像")
    plt.axis('off')
    
    # 空间特征 (取第一个通道)
    plt.subplot(2, 4, 2)
    plt.imshow(spatial_feature[0], cmap='viridis')
    plt.title("原始图像空间特征 (通道0)")
    plt.axis('off')
    
    # 原始频域特征 (取第一个通道)
    plt.subplot(2, 4, 3)
    plt.imshow(magnitude[0], cmap='viridis')
    plt.title("原始频域特征 (通道0)")
    plt.axis('off')
    
    # 可学习滤波器权重 (取第一个通道)
    plt.subplot(2, 4, 4)
    plt.imshow(model.freq_filter.filter_weight[0, 0].detach(), cmap='viridis')
    plt.title("可学习滤波器权重 (通道0)")
    plt.colorbar()
    plt.axis('off')
    
    # 滤波后频域特征 (取第一个通道)
    plt.subplot(2, 4, 5)
    plt.imshow(filtered_magnitude[0], cmap='viridis')
    plt.title("滤波后频域特征 (通道0)")
    plt.axis('off')
    
    # 滤波后空间特征 (取第一个通道)
    plt.subplot(2, 4, 6)
    plt.imshow(filtered_spatial[0], cmap='viridis')
    plt.title("滤波后空间特征 (通道0)")
    plt.axis('off')
    
    # 原始空间特征所有通道
    plt.subplot(2, 4, 7)
    all_channels = torch.norm(spatial_feature, dim=0).detach()  # 计算L2范数
    plt.imshow(all_channels / all_channels.max(), cmap='viridis')
    plt.title("原始空间特征所有通道")
    plt.axis('off')
    
    # 滤波后空间特征所有通道
    all_channels_filtered = torch.norm(filtered_spatial, dim=0).detach()
    plt.imshow(all_channels_filtered / all_channels_filtered.max(), cmap='viridis')
    plt.title("滤波后空间特征所有通道")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 加载和预处理图像
    img_path = "C:/Codes/Python/IR_RGB_SR/data/rgbtest.png"  # 替换为你的图像路径
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    
    # 创建模型
    model = FourierFeatureExtractor(in_channels=3, out_channels=16, img_size=(256, 256))
    
    # 前向传播
    results = model(img_tensor)
    
    # 可视化结果
    rgb_visualize_results(results, img_tensor[0])
    
    # 训练示例 (在实际应用中需要定义损失函数和优化器)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = nn.MSELoss()
    
    # 假设我们有一个简单的任务：重建原始图像
    # for epoch in range(10):
    #     optimizer.zero_grad()
    #     results = model(img_tensor)
    #     loss = loss_fn(results['filtered_spatial'], results['spatial_features'])
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")