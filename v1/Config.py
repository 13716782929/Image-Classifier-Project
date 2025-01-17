import os
from datetime import datetime
import torch
from torchvision import transforms


class Config:
    TRAIN_DIR = "data/train"
    TEST_DIR = "data/test"
    NUM_CLASSES = 4
    CLASS_TO_IDX = {
        'apple': 0,
        'banana': 1,
        'orange': 2,
        'mixed': 3
    }
    NUM_CLASSES = len(CLASS_TO_IDX)

    # 结果存储目录
    RESULT_DIR = "results"
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")  # 当前时间戳
    RUN_DIR = os.path.join(RESULT_DIR, TIMESTAMP)  # 每次运行的单独目录
    MODEL_SAVE_PATH = os.path.join(RUN_DIR, "best_model.pth")
    LOG_PATH = os.path.join(RUN_DIR, "train_log.txt")

    # 超参数
    SIZE = 64
    BATCH_SIZE = 16
    LEARNING_RATE = 0.005
    EPOCHS = 60
    WEIGHT_DECAY = 1e-4  # L2 正则化权重
    LR_SCHEDULE = "step"  # 支持 'step', 'cosine', 'none'
    STEP_SIZE = 10  # 学习率下降的步长
    GAMMA = 0.5  # 学习率衰减系数

    
    DATA_AUGMENTATION = transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomRotation(15), 
        transforms.RandomHorizontalFlip(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])

    # 设备选择
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def prepare_directories():
        """
        创建运行结果存储目录。
        """
        os.makedirs(Config.RUN_DIR, exist_ok=True)
        print(f"Results for this run will be saved in: {Config.RUN_DIR}")
