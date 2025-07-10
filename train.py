import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import os
from model.fusion import InfraredFourierFusionModel
import argparse

# 自定义数据集类
class NoisyInfraredDataset(Dataset):
    def __init__(self, IR_img, RGB_img, low_IR ,img_size=(800,600), noise_level=0.1):
        self.ir_files = sorted([os.path.join(IR_img, f) for f in os.listdir(IR_img)])
        self.rgb_files = sorted([os.path.join(RGB_img, f) for f in os.listdir(RGB_img)])
        self.low_ir_files = sorted([os.path.join(low_IR, f) for f in os.listdir(low_IR)])
        assert len(self.ir_files) == len(self.rgb_files) == len(self.low_ir_files), "数据量不匹配"
        
        # 图像转换
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 std=[0.229, 0.224, 0.225])
        self.noise_level = noise_level

    def __len__(self):
        return len(self.ir_files)

    def __getitem__(self, idx):
        # 加载原始红外图像（目标）
        clean_ir = Image.open(self.ir_files[idx]).convert('L')
        clean_ir = self.transform(clean_ir)
        
        # lowir（输入）
        low_ir = Image.open(self.low_ir_files[idx]).convert('L')
        low_ir = self.transform(low_ir)
        
        # 加载RGB图像
        rgb_img = Image.open(self.rgb_files[idx]).convert('RGB')
        rgb_tensor = self.transform(rgb_img)
        rgb_tensor = self.rgb_normalize(rgb_tensor)
        
        return {'low_ir': low_ir, 
                'rgb': rgb_tensor, 
                'clean_ir': clean_ir}

#组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, device='cuda', use_adv=False):
        """
        :param device: 计算设备 (cuda/cpu)
        :param use_adv: 是否使用对抗损失 (需要GAN)
        """
        super().__init__()
        self.device = device
        self.use_adv = use_adv
        
        # L1损失 (保持与原模型一致)
        self.l1_loss = nn.L1Loss()
        
        # 感知损失组件
        self.vgg = self._init_vgg_features()
        
        # 梯度损失组件
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    def _init_vgg_features(self):
        """初始化VGG特征提取器"""
        # 导入权重枚举类
        from torchvision.models import VGG16_Weights
        
        # 使用新的权重API加载模型
        weights = VGG16_Weights.IMAGENET1K_V1  # 或 VGG16_Weights.DEFAULT
        
        # 加载带权重的模型并获取特征部分
        vgg = models.vgg16(weights=weights).features[:16]
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        return vgg.eval().to(self.device)
    
    def perceptual_loss(self, gen, gt):
        """计算感知损失"""
        # 单通道转三通道
        if gen.shape[1] == 1:
            gen = gen.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)
        
        # 获取预训练权重的标准化参数（更严谨的做法）
        from torchvision.models import VGG16_Weights
        weights = VGG16_Weights.IMAGENET1K_V1
        mean = weights.transforms().mean
        std = weights.transforms().std
        
        # 标准化
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(self.device)
        
        gen_norm = (gen - mean_tensor) / std_tensor
        gt_norm = (gt - mean_tensor) / std_tensor
        
        # 提取特征并计算L1损失
        features_gen = self.vgg(gen_norm)
        features_gt = self.vgg(gt_norm)
        return F.l1_loss(features_gen, features_gt)
        
    def gradient_loss(self, gen, gt):
        """计算图像梯度损失"""
        # X方向梯度
        grad_gen_x = F.conv2d(gen, self.sobel_x, padding=1)
        grad_gt_x = F.conv2d(gt, self.sobel_x, padding=1)
        
        # Y方向梯度
        grad_gen_y = F.conv2d(gen, self.sobel_y, padding=1)
        grad_gt_y = F.conv2d(gt, self.sobel_y, padding=1)
        
        loss = F.l1_loss(grad_gen_x, grad_gt_x) + F.l1_loss(grad_gen_y, grad_gt_y)
        return loss / 2
    
    def forward(self, gen_img, gt_img, discriminator=None):
        """
        计算组合损失
        :param gen_img: 生成图像
        :param gt_img: 真实图像
        :param discriminator: 判别器模型 (仅当use_adv=True时需提供)
        """
        # 1. L1损失
        l1 = self.l1_loss(gen_img, gt_img)
        
        # 2. 感知损失
        percep = self.perceptual_loss(gen_img, gt_img)
                                                                                                                                                                                
        # # 3. 对抗损失
        # if self.use_adv and discriminator:
        #     # 使用最小二乘GAN损失
        #     fake_scores = discriminator(gen_img)
        #     adv = torch.mean((fake_scores - 1) ** 2)
        # else:
        #     adv = torch.tensor(0.0, device=self.device)
        
        # 4. 梯度损失
        grad = self.gradient_loss(gen_img, gt_img)
        
        # 5. 组合加权损失
        total_loss = 0.7 * l1  + 0.3 * grad
        
        return {
            "total": total_loss,
            "l1": l1.item(),
            # "perceptual": percep.item(),
            # "adversarial": adv.item() if self.use_adv else 0,
            "gradient": grad.item()
        }

# 训练配置
def train_model(RGB_img, IR_img, low_IR,cudaID = None,pretrained_path=None,batch_size = 4):
    # 初始化配置
    img_size = (800,600)
    # batch_size = 8
    epochs = 100
    
    # 检查可用GPU数量
    if cudaID == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        # num_gpus = 3
    else:
        num_gpus = 1
        device = torch.device(f"cuda:{cudaID}")
        
    
    print(f"检测到 {num_gpus} 个可用的GPU")
    
    # 初始化模型
    model = InfraredFourierFusionModel(
        img_size=img_size,
        origianl_channels=16
    ).to(device)
    if num_gpus > 1:
        print(f"使用 {num_gpus} 个GPU进行并行训练")
        model = nn.DataParallel(model)
    
    
    # if pretrained_path:
    #     try:
    #         model.load_state_dict(torch.load(pretrained_path, map_location=device))
    #         print(f"成功加载预训练权重: {pretrained_path}")
    #     except Exception as e:
    #         print(f"加载预训练权重失败: {str(e)}")
    # 加载预训练权重（如果有）
    if pretrained_path:
        try:
            # 处理多GPU训练模型的状态字典
            state_dict = torch.load(pretrained_path, map_location=device)
            if num_gpus <= 1 and any(key.startswith('module.') for key in state_dict):
                # 如果预训练模型是多GPU训练的，但当前使用单GPU
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            elif num_gpus > 1 and not any(key.startswith('module.') for key in state_dict):
                # 如果预训练模型是单GPU训练的，但当前使用多GPU
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            print(f"成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"加载预训练权重失败: {str(e)}")
    
    # 定义损失函数和优化器
    criterion =  nn.L1Loss()
    # criterion = CombinedLoss(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    #可调整学习率
    def lr_lambda(epoch):
        return 0.1 if epoch >= 20 else 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    
    # 创建数据集和数据加载器
    train_dataset = NoisyInfraredDataset(
        IR_img=IR_img,
        RGB_img=RGB_img,
        low_IR=low_IR,
        img_size=img_size,
        noise_level=0.05  # 可调节噪声强度
    )
    
    # 根据GPU数量调整批次大小
    effective_batch_size = batch_size * max(1, num_gpus)
    print(f"实际批次大小: {effective_batch_size} (每个GPU: {batch_size})")
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=effective_batch_size, 
                             shuffle=True,
                             num_workers=4 * num_gpus,  # 根据GPU数量增加工作线程
                             pin_memory=True)  # 加速数据传输
    
    # train_loader = DataLoader(train_dataset, 
    #                          batch_size=batch_size, 
    #                          shuffle=True,
    #                          num_workers=0)
   
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            low_ir = batch['low_ir'].to(device)
            rgb = batch['rgb'].to(device)
            clean_ir = batch['clean_ir'].to(device)
            
            optimizer.zero_grad()
            outputs = model(low_ir, rgb)  # 输入low红外和RGB
            # loss = criterion(outputs, clean_ir)['total']  # 与干净红外比较(conbined loss)
            loss = criterion(outputs, clean_ir) #L1 loss
            
            loss.backward()
            optimizer.step()
            
            # 统计损失
            running_loss += loss.item()
            
            # 每10个batch打印进度
            if batch_idx % 10 == 9:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                
           
        
        # 每个epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] completed, Average Loss: {epoch_loss:.4f}')
        #当前学习率
        print(f"Epoch [{epoch+1}/{epochs}] - 当前学习率: { optimizer.param_groups[0]['lr']:.6f}")
        #学习率更新
        scheduler.step() 
        
        
        # 每10个epoch结束后保存对比图
        if (epoch + 1) % 10 == 0:
            model.eval()
            
            torch.save(model.state_dict(), f'checkout/temp_model_epoch{epoch+1}.pth')
            print("临时模型已保存")
            with torch.no_grad():
                # 获取一个测试样本
                sample = train_dataset[0]
                low_ir = sample['low_ir'].unsqueeze(0).to(device)
                rgb = sample['rgb'].unsqueeze(0).to(device)
                
                # 生成预测
                pred_ir = model(low_ir, rgb)
                
                # 转换为numpy图像
                clean_img = sample['clean_ir'].squeeze().cpu().numpy()
                noisy_img = sample['low_ir'].squeeze().cpu().numpy()
                pred_img = pred_ir.squeeze().cpu().numpy()
                
                # 绘制对比图
                plt.figure(figsize=(15,5))
                plt.subplot(1,3,1)
                plt.imshow(noisy_img, cmap='gray')
                plt.title(f'Noisy Input (Epoch {epoch+1})')
                plt.axis('off')
                
                plt.subplot(1,3,2)
                plt.imshow(pred_img, cmap='gray')
                plt.title('Enhanced Output')
                plt.axis('off')
                
                plt.subplot(1,3,3)
                plt.imshow(clean_img, cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')
                
                # 保存图像
                plt.savefig(f'./results/compare_epoch_{epoch+1}.png')
                plt.close()
            
            model.train()  # 恢复训练模式
    
    # 保存模型
    torch.save(model.state_dict(), 'checkout/final_model.pth')
    print("训练完成，模型已保存")

def test_model(ir_path, rgb_path, train_ir_path,pretrained_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 600
    
    # 初始化模型
    model = InfraredFourierFusionModel(
        img_size=(img_size, img_size),
        origianl_channels=16
    ).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
    
    # 加载测试图像
    with torch.no_grad():
        # 处理红外图像
        ir_img = Image.open(ir_path).convert('L')
        ir_tensor = transform(ir_img).unsqueeze(0).to(device)
        
        #训练时的ir图像
        ir_train = Image.open(train_ir_path).convert('L')
        ir_train_tensor = transform(ir_train).unsqueeze(0).to(device)
        
        # 处理RGB图像
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_tensor = transform(rgb_img)
        rgb_tensor = rgb_normalize(rgb_tensor).unsqueeze(0).to(device)
        
        # 模型推理
        enhanced_ir = model(ir_tensor, rgb_tensor)
        
        # 转换为numpy
        input_img_ir = ir_tensor.squeeze().cpu().numpy()
        train_img_ir = ir_train_tensor.squeeze().cpu().numpy()
        output_img = enhanced_ir.squeeze().cpu().numpy()
        
        # 绘制对比图
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(input_img_ir, cmap='gray')
        plt.title('gt Infrared')
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(train_img_ir, cmap='gray')
        plt.title('input Infrared')
        plt.axis('off')
        
        plt.subplot(1,3,3)
        plt.imshow(output_img, cmap='gray')
        plt.title('Enhanced Infrared')
        plt.axis('off')
        
        # 保存结果
        plt.savefig('./results/test_comparison.png')
        plt.close()
        print("测试完成，结果已保存至 results/test_comparison.png")          
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='运行模式: train 或 test')
    parser.add_argument('--outdir', type=str, default='./results', help='结果存储results')
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    data_dir = {
        'rgb':'./data/train/Visible',
        'ir' :'./data/train/Infrared_enhance',
        'low_ir':'./data/train/Infrared'
    }
    if args.train:
        train_model(
            RGB_img=data_dir['rgb'],
            IR_img=data_dir['ir'],
            low_IR=data_dir['low_ir'],
        )
    else:
        test_model(
            ir_path=f'{data_dir["ir"]}/00006.png',
            rgb_path=f'{data_dir["rgb"]}/00006.png',
            pretrained_path='./checkout/infrared_fusion_model.pth'
        )
        