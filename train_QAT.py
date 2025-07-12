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
from torch.quantization import prepare_qat, convert
from train import load_multi_GPU_pretrain_model

# 自定义数据集类
class NoisyInfraredDataset(Dataset):
    def __init__(self, IR_img, RGB_img, low_IR ,img_size=(512,512)):
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

    
def train_model_qat(RGB_img, IR_img, low_IR,pretrained_path=None,batch_size = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 512  # 保持与训练尺寸一致
    epochs = 50
    
    # 初始化模型
    model = InfraredFourierFusionModel(
        img_size=(img_size, img_size),
        origianl_channels=16
    ).to(device)
    
    model = load_multi_GPU_pretrain_model(model=model,pretrained_path=pretrained_path,device=device)
    model.train()
    
    #================配置QAT模型============
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    model_qat = prepare_qat(model)
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=0.001)
    # criterion =  nn.L1Loss()
    criterion = CombinedLoss(device=device)
    
    # 冻结BN层参数
    model_qat.apply(torch.quantization.disable_observer)  # 冻结Observer
    model_qat.apply(torch.nn.intrinsic.qat.freeze_bn_stats)  # 冻结BN统计量
    
    # 创建数据集和数据加载器
    train_dataset = NoisyInfraredDataset(
        IR_img=IR_img,
        RGB_img=RGB_img,
        low_IR=low_IR,
        # img_size=img_size,
        
    )
    train_loader = DataLoader(train_dataset, 
                             batch_size=4, 
                             shuffle=True,
                             num_workers=0 ,  # 根据GPU数量增加工作线程
                             pin_memory=True)  # 加速数据传输
    
    # 创建结果目录
    os.makedirs('./results/qat_results', exist_ok=True)
    
    for epoch in range(epochs):
        model_qat.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            low_ir = batch['low_ir'].to(device)
            rgb = batch['rgb'].to(device)
            clean_ir = batch['clean_ir'].to(device)
            
            optimizer.zero_grad()
            outputs = model_qat(low_ir, rgb)  # 输入low红外和RGB
            loss = criterion(outputs, clean_ir)['total']  # 与干净红外比较(conbined loss)
            # loss = criterion(outputs, clean_ir) #L1 loss
            
            loss.backward()
            
            #====打印梯度信息===
            total_norm = 0
            for p in model_qat.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f' Gradient Norm: {total_norm:.4f}')
            
            
            #====更新权重====
            optimizer.step()
            
            # 统计损失
            running_loss += loss.item()
            
            # 每10个batch打印进度
            if batch_idx % 10 == 9:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 每个epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] completed, Average Loss:================ {epoch_loss:.4f}==============')
        # 每10个epoch结束后保存对比图
        if (epoch + 1) % 10 == 0:
            model_qat.eval()
            with torch.no_grad():
                # 获取一个测试样本
                sample = train_dataset[0]
                low_ir = sample['low_ir'].unsqueeze(0).to(device)
                rgb = sample['rgb'].unsqueeze(0).to(device)
                
                # 生成预测
                pred_ir = model_qat(low_ir, rgb)
                
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
                plt.savefig(f'./results/qat_results/compare_epoch_{epoch+1}.png')
                plt.close()
            
            model.train()  # 恢复训练模式
    model_qat.eval()
    model_int8 = convert(model_qat)
    torch.jit.save(torch.jit.script(model_int8),'./checkout/qat_model.pth')
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
        'low_ir':'./data/train/Infrared',
        'ir_test':'./data/test/Infrared',
        'rgb_test':'./data/test/Visible'
    }
    if args.train:
        train_model_qat(
            RGB_img=data_dir['rgb'],
            IR_img=data_dir['ir'],
            low_IR=data_dir['low_ir'],
            pretrained_path = './checkout/final_model.pth'
        )