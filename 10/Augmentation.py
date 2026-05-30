import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 OpenMP 冲突

import PIL.Image as Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from functools import partial

def imshow(img_path, transform):
    """
    Function to show data augmentation
    Param img_path: path of the image
    Param transform: data augmentation technique to apply
    """
    img = Image.open(img_path)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].set_title(f'Original image {img.size}')
    ax[0].imshow(img)
    img = transform(img)
    ax[1].set_title(f'Transformed image {img.size}')
    ax[1].imshow(img)
    plt.show()

def cutout(image, mask_size=30, num_masks=5):
        # 统一转为 RGB，防止灰度图导致索引错误
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image_np = np.array(image).copy()
    h, w = image_np.shape[0], image_np.shape[1]

    for _ in range(num_masks):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        y1 = max(0, y - mask_size // 2)
        y2 = min(h, y + mask_size // 2)
        x1 = max(0, x - mask_size // 2)
        x2 = min(w, x + mask_size // 2)

        # 遮挡区域设置为黑色
        image_np[y1:y2, x1:x2, :] = 0

    return Image.fromarray(image_np)


    # ======== 批量处理并保存所有图片 ========
import os
from functools import partial

folder_path = r"E:\mylab\10\archive\PetImages\Cat"
output_folder = r"E:\MyLabLogs\10"  # 增强后的图片保存到这里

# 如果输出文件夹不存在，自动创建
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(folder_path) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

if image_files:
    print(f"找到 {len(image_files)} 张图片，开始批量增强并保存...")
    for img_file in image_files:
        path = os.path.join(folder_path, img_file)
        print(f"正在处理: {img_file}")
        
        # 读取原始图片
        img = Image.open(path)
        # 应用 Cutout 增强
        augmented_img = cutout(img, mask_size=30, num_masks=5)
        
        # 保存到输出文件夹（保留原文件名）
        save_path = os.path.join(output_folder, img_file)
        augmented_img.save(save_path)
        
    print(f"所有图片处理完毕！请查看文件夹: {output_folder}")
else:
    print("文件夹中没有找到图片！")