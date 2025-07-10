import cv2
import numpy as np
import os
from tqdm import tqdm

def enhance_ir_with_color(ir_path, color_path, output_dir, *, alpha=0.9, beta=0.1, channel_index=1):
    """
    使用彩色图像的指定通道增强红外图像
    :param ir_path: 红外图像路径
    :param color_path: 彩色图像路径
    :param output_dir: 输出目录
    :param alpha: 红外图像权重 (0-1)
    :param beta: 彩色通道权重 (0-1)
    :param channel_index: 彩色通道索引 (0:B, 1:G, 2:R)
    """
    # 读取图像
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(color_path)
    
    if ir_img is None:
        print(f"警告: 无法读取红外图像 {ir_path}")
        return
    if color_img is None:
        print(f"警告: 无法读取彩色图像 {color_path}")
        return
    
    # 提取指定通道并转换为单通道
    color_channel = color_img[:, :, channel_index]
    
    # 确保图像尺寸相同（红外图像优先）
    if color_channel.shape != ir_img.shape:
        color_channel = cv2.resize(color_channel, (ir_img.shape[1], ir_img.shape[0]))
    
    # 通道叠加融合
    blended = cv2.addWeighted(ir_img, alpha, color_channel, beta, 0)
    
    # 锐化处理强化线条
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blended, -1, kernel)
    
    # 保存结果
    filename = os.path.basename(ir_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, sharpened)

# 主处理函数
def batch_enhance_ir(ir_dir, color_dir, output_dir, *, channel='G', alpha=0.7, beta=0.3):
    """
    批量处理红外图像增强
    :param ir_dir: 红外图像目录
    :param color_dir: 彩色图像目录
    :param output_dir: 输出目录
    :param channel: 彩色通道选择 ('B', 'G', 'R')
    :param alpha: 红外图像权重
    :param beta: 彩色通道权重
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置通道索引
    channel_map = {'B': 0, 'G': 1, 'R': 2}
    channel_index = channel_map.get(channel.upper(), 1)  # 默认为绿色通道
    
    # 获取文件列表
    ir_files = [f for f in os.listdir(ir_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    color_files = set(os.listdir(color_dir))
    
    print(f"找到 {len(ir_files)} 张红外图像")
    
    # 处理每对图像
    processed = 0
    for ir_file in tqdm(ir_files, desc="处理图像"):
        # 查找匹配的彩色图像（支持不同扩展名）
        name, _ = os.path.splitext(ir_file)
        color_match = None
        
        # 查找同名彩色图像（允许不同扩展名）
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            candidate = name + ext
            if candidate in color_files:
                color_match = candidate
                break
        
        if color_match:
            ir_path = os.path.join(ir_dir, ir_file)
            color_path = os.path.join(color_dir, color_match)
            
            enhance_ir_with_color(
                ir_path, 
                color_path, 
                output_dir,
                alpha=alpha,
                beta=beta,
                channel_index=channel_index
            )
            processed += 1
        else:
            print(f"警告: 未找到 {ir_file} 对应的彩色图像")
    
    print(f"处理完成! 成功增强 {processed} 张图像，结果保存在 {output_dir}")

# 使用示例
if __name__ == "__main__":
    # 配置路径和参数
    data_dir = {
        'rgb':'./data/train/Visible',
        'ir' :'./data/train/Infrared',
        'ir_enhance' : './data/train/Infrared_enhance'
    }
    
    # 执行批量处理
    batch_enhance_ir(
        ir_dir=data_dir['ir'],
        color_dir=data_dir['rgb'],
        output_dir=data_dir['ir_enhance'],
        channel='G',    # 使用绿色通道
        alpha=0.9,      # 红外权重
        beta=0.1        # 彩色通道权重
    )