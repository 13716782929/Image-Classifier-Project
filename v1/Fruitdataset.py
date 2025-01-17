import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from Config import Config
import random
import numpy as np


class FruitDataset(Dataset):
    def __init__(self, dir, transform=None, is_train=True):
        """
        初始化数据集类
        :param dir: 数据集根目录
        :param transform: 数据增强或预处理方法
        :param is_train: 是否是训练集，训练集需要动态生成 mixed 类
        """
        self.root_dir = dir
        self.transform = transform
        self.is_train = is_train
        self.data = []
        self.labels = []
        self.class_to_idx = Config.CLASS_TO_IDX 
        self.class_counts = {class_name: 0 for class_name in self.class_to_idx.keys()} 
        self.mixed_samples = []  # 用于存储动态生成的 mixed 样本

        # 遍历目录并解析文件名
        for file_name in os.listdir(dir):
            if file_name.endswith('.jpg'):
                # 从文件名中提取类别
                class_name = file_name.split('_')[0]
                if class_name not in self.class_to_idx:
                    raise ValueError(f"Class '{class_name}' not found in Config.CLASS_TO_IDX")

                label = self.class_to_idx[class_name]  # 使用 Config 中定义的映射
                self.data.append((file_name, label))  # 保存文件名和标签
                self.class_counts[class_name] += 1

        # 如果是训练集，动态生成 mixed 类
        if self.is_train:
            self._generate_mixed_samples()

        # 打印信息
        print(f"Class-to-Index Mapping (from Config): {self.class_to_idx}")
        print(f"Class Counts: {self.class_counts}")

    def _generate_mixed_samples(self):
        """
        动态生成 mixed 类样本，直到数量与其他类别平衡
        """
        target_count = max(self.class_counts.values())  # 找到最大类别数量
        mixed_label = self.class_to_idx['mixed']
        while self.class_counts['mixed'] < target_count:
            # 从 apple, orange, banana 三个类别中各随机抽取一张，再额外随机抽取一张
            chosen_samples = []
            for class_name in ['apple', 'orange', 'banana']:
                chosen_samples.append(random.choice(
                    [item for item in self.data if item[1] == self.class_to_idx[class_name]]
                ))
            chosen_samples.append(random.choice(self.data))  # 再随机选择一张任意类别

            # 拼接图像
            mixed_image = self._create_mixed_image(chosen_samples)
            # mixed_name = f"mixed_{len(self.mixed_samples)}.jpg"

            # 添加到 mixed 样本集合
            self.mixed_samples.append((mixed_image, mixed_label))
            self.class_counts['mixed'] += 1

    def _create_mixed_image(self, samples):
        images = []
        for file_name, _ in samples:
            img_path = os.path.join(self.root_dir, file_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w, c = image.shape
            if h > w:
                pad = (h - w) // 2
                image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            elif w > h:
                pad = (w - h) // 2
                image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            image = cv2.resize(image, (Config.SIZE // 2, Config.SIZE // 2))
            images.append(image)

        top_row = np.hstack((images[0], images[1]))
        bottom_row = np.hstack((images[2], images[3]))
        mixed_image = np.vstack((top_row, bottom_row))

        return Image.fromarray(mixed_image)

    def __len__(self):
        """
        数据集大小
        :return: 数据集中样本的数量
        """
        return len(self.data) + (len(self.mixed_samples) if self.is_train else 0)

    def __getitem__(self, index):
        if index < len(self.data):
            file_name, label = self.data[index]
            img_path = os.path.join(self.root_dir, file_name)
            image = cv2.imread(img_path)  
            if image is None:
                raise FileNotFoundError(f"Image {img_path} not found.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w, c = image.shape
            if h > w:
                pad = (h - w) // 2
                image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            elif w > h:
                pad = (w - h) // 2
                image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            image = cv2.resize(image, (Config.SIZE, Config.SIZE))
            image = Image.fromarray(image)
        else:
            mixed_index = index - len(self.data)
            image, label = self.mixed_samples[mixed_index]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (H, W, C) -> (C, H, W)

        return image, label

