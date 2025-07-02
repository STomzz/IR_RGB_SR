import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from model.fusion import InfraredFourierFusionModel

# 自定义数据集类
class NoisyInfraredDataset(Dataset):
    def __init__(self, IR_img, RGB_img, img_size=256, noise_level=0.1):
        self.ir_files = sorted([os.path.join(IR_img, f) for f in os.listdir(IR_img)])
        self.rgb_files = sorted([os.path.join(RGB_img, f) for f in os.listdir(RGB_img)])
        assert len(self.ir_files) == len(self.rgb_files), "数据量不匹配"
        
        # 图像转换
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
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
        
        # 添加噪声（输入）
        noisy_ir = clean_ir + torch.randn_like(clean_ir) * self.noise_level
        noisy_ir = torch.clamp(noisy_ir, 0, 1)
        
        # 加载RGB图像
        rgb_img = Image.open(self.rgb_files[idx]).convert('RGB')
        rgb_tensor = self.transform(rgb_img)
        rgb_tensor = self.rgb_normalize(rgb_tensor)
        
        return {'noisy_ir': noisy_ir, 
                'rgb': rgb_tensor, 
                'clean_ir': clean_ir}
# 训练配置
def train_model(RGB_img, IR_img,pretrained_path=None):
    # 初始化配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 256
    batch_size = 8
    epochs = 500
    
    # 初始化模型
    model = InfraredFourierFusionModel(
        img_size=(img_size, img_size),
        origianl_channels=16
    ).to(device)
    
    if pretrained_path:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print(f"成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"加载预训练权重失败: {str(e)}")
    
    # 定义损失函数和优化器
    criterion =  nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建数据集和数据加载器
    train_dataset = NoisyInfraredDataset(
        IR_img=IR_img,
        RGB_img=RGB_img,
        img_size=img_size,
        noise_level=0.05  # 可调节噪声强度
    )
    train_loader = DataLoader(train_dataset, 
                             batch_size=batch_size, 
                             shuffle=True,
                             num_workers=0)
   
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            noisy_ir = batch['noisy_ir'].to(device)
            rgb = batch['rgb'].to(device)
            clean_ir = batch['clean_ir'].to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_ir, rgb)  # 输入带噪红外和RGB
            loss = criterion(outputs, clean_ir)  # 与干净红外比较
            
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
        
        # 每10个epoch结束后保存对比图
        if (epoch + 1) % 10 == 0:
            model.eval()
            if(epoch + 1) % 100 == 0:
                torch.save(model.state_dict(), f'C:/Codes/Python/IR_RGB_SR/checkout/temp_model_epoch{epoch+1}.pth')
                print("训练完成，临时模型已保存")
            with torch.no_grad():
                # 获取一个测试样本
                sample = train_dataset[0]
                noisy_ir = sample['noisy_ir'].unsqueeze(0).to(device)
                rgb = sample['rgb'].unsqueeze(0).to(device)
                
                # 生成预测
                pred_ir = model(noisy_ir, rgb)
                
                # 转换为numpy图像
                clean_img = sample['clean_ir'].squeeze().cpu().numpy()
                noisy_img = sample['noisy_ir'].squeeze().cpu().numpy()
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
                plt.savefig(f'C:/Codes/Python/IR_RGB_SR/results/compare_epoch_{epoch+1}.png')
                plt.close()
            
            model.train()  # 恢复训练模式
    
    # 保存模型
    torch.save(model.state_dict(), 'C:/Codes/Python/IR_RGB_SR/checkout/infrared_fusion_model.pth')
    print("训练完成，模型已保存为 infrared_fusion_model.pth")

def test_model(ir_path, rgb_path, pretrained_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 256
    
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
        
        # 处理RGB图像
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_tensor = transform(rgb_img)
        rgb_tensor = rgb_normalize(rgb_tensor).unsqueeze(0).to(device)
        
        # 模型推理
        enhanced_ir = model(ir_tensor, rgb_tensor)
        
        # 转换为numpy
        input_img = ir_tensor.squeeze().cpu().numpy()
        output_img = enhanced_ir.squeeze().cpu().numpy()
        
        # 绘制对比图
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(input_img, cmap='gray')
        plt.title('Input Infrared')
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(output_img, cmap='gray')
        plt.title('Enhanced Infrared')
        plt.axis('off')
        
        # 保存结果
        plt.savefig('C:/Codes/Python/IR_RGB_SR/results/test_comparison.png')
        plt.close()
        print("测试完成，结果已保存至 results/test_comparison.png")          
    
    
if __name__ == "__main__":
    data_dir_1 = {
        'rgb':'C:/Codes/Python/IR_RGB_SR/data/RoadScene-master/crop_LR_visible',
        'ir' :'C:/Codes/Python/IR_RGB_SR/data/RoadScene-master/cropinfrared'
    }
    data_dir_2 = {
        'rgb':'C:/Codes/Python/IR_RGB_SR/data/train/Visible',
        'ir' :'C:/Codes/Python/IR_RGB_SR/data/train/Infrared'
    }
    train_model(
        RGB_img=data_dir_2['rgb'],
        IR_img=data_dir_2['ir'],
        pretrained_path='C:/Codes/Python/IR_RGB_SR/checkout/infrared_fusion_model.pth'
    )
    # test_model(
    #     ir_path='C:/Codes/Python/IR_RGB_SR/data/RoadScene-master/cropinfrared/FLIR_00006.jpg',
    #     rgb_path='C:/Codes/Python/IR_RGB_SR/data/RoadScene-master/crop_LR_visible/FLIR_00006.jpg',
    #     pretrained_path='C:/Codes/Python/IR_RGB_SR/checkout/infrared_fusion_model.pth'
    # )
        