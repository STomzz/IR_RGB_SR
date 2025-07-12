import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # 使用图注意力网络
from torch_geometric.utils import grid
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

class GraphConvBlock(nn.Module):
    """将2D特征图转换为图结构并进行图卷积"""
    def __init__(self, in_channels, out_channels, img_size=(256, 256)):
        super().__init__()
        self.H, self.W = img_size
        self.num_nodes = self.H * self.W
        
        # 图注意力卷积层
        self.gat = GATConv(in_channels, out_channels, heads=1, concat=True)
        
        # 位置编码增强空间信息感知
        self.pos_encoder = nn.Linear(2, in_channels)
        
        # 用于特征图←→图转换的缓冲区
        self.register_buffer('edge_index', self.create_grid_edges())

    def create_grid_edges(self):
        """创建网格结构的边索引(8邻域)"""
        return grid(self.H, self.W, diagonal=True)  # 包括对角线连接

    def forward(self, x):
        # x形状: [batch, channels, H, W]
        batch_size = x.size(0)
        
        # 1. 位置编码 (添加空间信息)
        pos_x = torch.linspace(-1, 1, self.W, device=x.device)
        pos_y = torch.linspace(-1, 1, self.H, device=x.device)
        grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing='xy')
        pos = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]
        pos_enc = self.pos_encoder(pos).permute(2, 0, 1)  # [C, H, W]
        
        # 2. 添加位置编码到特征
        x = x + pos_enc.unsqueeze(0)
        
        # 3. 特征图转图结构
        x_graph = x.permute(0, 2, 3, 1).reshape(batch_size * self.num_nodes, -1)
        
        # 4. 图卷积处理
        edge_index = self.edge_index.repeat(1, batch_size)
        batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(self.num_nodes)
        x_graph = self.gat(x_graph, edge_index, batch_indices)
        
        # 5. 图结构转回特征图
        x_out = x_graph.view(batch_size, self.H, self.W, -1)
        return x_out.permute(0, 3, 1, 2)  # [batch, channels, H, W]


class InfraredFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size
        self.H, self.W = img_size
        
        # 初始卷积层保留用于低层级特征提取
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels)
        )
        
        # 图神经网络模块替换传统卷积
        self.gnn_layers = nn.Sequential(
            GraphConvBlock(base_channels, base_channels*2, img_size),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*2),
            
            GraphConvBlock(base_channels*2, base_channels*4, img_size),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*4)
        )
        self.final_channels = base_channels * 4
        
        # 通道注意力机制
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.final_channels, self.final_channels // 4),
            nn.ReLU(),
            nn.Linear(self.final_channels // 4, self.final_channels),
            nn.Softmax(dim=1)
        )
        
        # 可学习滤波器
        self.filter = nn.Parameter(torch.ones(1, self.final_channels, self.H, self.W))
        self.filter_bias = nn.Parameter(torch.zeros(1, self.final_channels, 1, 1))
        
        # 特征增强层
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(self.final_channels, self.final_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.final_channels // 2, self.final_channels, 3, padding=1)
        )
        
        # 输出归一化层
        self.norm = nn.BatchNorm2d(self.final_channels)

    def forward(self, x):
        # 1. 初始卷积提取基础特征
        features = self.init_conv(x)
        
        # 2. 图神经网络处理
        features = self.gnn_layers(features)
        
        batch_size, channels, H, W = features.shape
        
        # 3. 通道注意力机制
        pooled = self.global_pool(features).view(batch_size, -1)
        channel_weights = self.channel_mlp(pooled).view(batch_size, channels, 1, 1)
        weighted_features = features * channel_weights
        
        # 4. 应用可学习滤波器
        filtered_features = weighted_features * self.filter + self.filter_bias
        
        # 5. 特征增强
        enhanced_features = self.enhance_conv(filtered_features) + filtered_features
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
        
        
class InfraredFeatureVisualizer:
    def __init__(self, model_path=None, img_size=(256, 256), device='cuda'):
        """
        红外特征可视化工具
        
        参数:
            model_path: 预训练模型路径 (可选)
            img_size: 图像输入尺寸
            device: 计算设备 (cuda/cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # 初始化模型
        self.model = InfraredFeatureExtractor(
            in_channels=1, 
            base_channels=16, 
            img_size=img_size
        ).to(self.device).eval()
        
        # 加载预训练权重 (如果有)
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"已加载预训练模型: {model_path}")
        else:
            print("使用随机初始化的模型")
    
    def preprocess_image(self, image_path):
        """
        预处理红外图像:
          1. 读取为灰度图
          2. 调整尺寸
          3. 归一化
          4. 转换为张量
        """
        # 读取图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 调整尺寸
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        # 归一化 [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 添加通道和批次维度
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return img, img_tensor
    
    def extract_features(self, image_tensor):
        """执行特征提取"""
        with torch.no_grad():
            features = self.model(image_tensor)
        return features
    
    def visualize_features(self, original_img, features, save_path=None):
        """
        可视化特征提取过程
        
        参数:
            original_img: 原始红外图像 (numpy数组)
            features: 模型输出的特征字典
            save_path: 结果保存路径 (可选)
        """
        # 准备可视化
        plt.figure(figsize=(16, 12), dpi=100)
        
        # 原始图像
        plt.subplot(2, 3, 1)
        plt.imshow(original_img, cmap='jet')
        plt.title('原始红外图像')
        plt.axis('off')
        
        # 原始特征图 (取前3个通道的平均)
        raw_feats = features['raw_features'].cpu().numpy()[0]
        raw_vis = np.mean(raw_feats[:3], axis=0)
        plt.subplot(2, 3, 2)
        plt.imshow(raw_vis, cmap='viridis')
        plt.title('GNN提取的原始特征')
        plt.axis('off')
        
        # 通道注意力权重
        channel_weights = features['channel_weights'].cpu().numpy()[0]
        plt.subplot(2, 3, 3)
        plt.bar(range(len(channel_weights)), channel_weights.squeeze())
        plt.title('通道注意力权重分布')
        plt.xlabel('通道索引')
        plt.ylabel('权重')
        
        # 加权特征图
        weighted_feats = features['weighted_features'].cpu().numpy()[0]
        weighted_vis = np.mean(weighted_feats[:3], axis=0)
        plt.subplot(2, 3, 4)
        plt.imshow(weighted_vis, cmap='plasma')
        plt.title('通道加权特征')
        plt.axis('off')
        
        # 滤波后特征图
        filtered_feats = features['filtered_features'].cpu().numpy()[0]
        filtered_vis = np.mean(filtered_feats[:3], axis=0)
        plt.subplot(2, 3, 5)
        plt.imshow(filtered_vis, cmap='cividis')
        plt.title('空间滤波后特征')
        plt.axis('off')
        
        # 最终增强特征
        enhanced_feats = features['enhanced_features'].cpu().numpy()[0]
        enhanced_vis = np.mean(enhanced_feats[:3], axis=0)
        plt.subplot(2, 3, 6)
        plt.imshow(enhanced_vis, cmap='inferno')
        plt.title('增强后的最终特征')
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"可视化结果已保存至: {save_path}")
    
    def visualize_feature_maps(self, features, n_maps=16, save_path=None):
        """
        可视化特征图网格
        
        参数:
            features: 模型输出的特征字典
            n_maps: 显示的特征图数量
            save_path: 结果保存路径 (可选)
        """
        # 提取特征图
        feats = features['enhanced_features'].cpu().numpy()[0]
        
        # 选择部分特征图
        step = max(1, feats.shape[0] // n_maps)
        selected_feats = feats[::step][:n_maps]
        
        # 创建网格
        rows = int(np.ceil(np.sqrt(n_maps)))
        cols = rows
        
        plt.figure(figsize=(15, 15), dpi=100)
        plt.suptitle(f'特征图可视化 (共{len(selected_feats)}张)', fontsize=16)
        
        for i, feat in enumerate(selected_feats):
            plt.subplot(rows, cols, i+1)
            plt.imshow(feat, cmap='hot')
            plt.title(f'通道 {i*step}')
            plt.axis('off')
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"特征图网格已保存至: {save_path}")
    
    def run(self, image_path, result_prefix='/home/zhangshutao/codes/IR_RGB_SR/results/ir_graph'):
        """
        完整处理流程:
          1. 加载和预处理图像
          2. 特征提取
          3. 可视化结果
        """
        # 预处理图像
        orig_img, img_tensor = self.preprocess_image(image_path)
        
        # 特征提取
        features = self.extract_features(img_tensor)
        
        # 可视化特征提取过程
        process_path = f"{result_prefix}/features_process.png"
        self.visualize_features(orig_img, features, save_path=process_path)
        
        # 可视化特征图网格
        maps_path = f"{result_prefix}/feature_maps.png"
        self.visualize_feature_maps(features, save_path=maps_path)
        
        return features


if __name__ == "__main__":
    # 创建可视化工具
    visualizer = InfraredFeatureVisualizer(
        model_path=None,  # 可以替换为实际模型路径
        img_size=(256, 256),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 示例红外图像 (替换为实际路径)
    image_path = "/home/zhangshutao/codes/IR_RGB_SR/data/train/Infrared/00001.png"  # 红外图像路径
    
    # 执行特征提取和可视化
    visualizer.run(image_path, result_prefix="infrared_features")
    
    print("红外特征提取与可视化完成!")