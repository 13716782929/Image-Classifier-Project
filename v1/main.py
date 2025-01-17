from Config import Config
from Model import FruitClassifierModel
from Trainer import Trainer
from Fruitdataset import FruitDataset
from torch.utils.data import DataLoader
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


    # 数据加载器
    train_dataset = FruitDataset(Config.TRAIN_DIR, transform=Config.DATA_AUGMENTATION)
    val_dataset = FruitDataset(Config.TEST_DIR, transform=Config.DATA_AUGMENTATION, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = FruitClassifierModel(input_shape=(3, Config.SIZE, Config.SIZE), num_classes=Config.NUM_CLASSES)  # 4 分类任务

    # 初始化 Trainer
    trainer = Trainer(model, train_loader, val_loader, Config)

    # 开始训练
    trainer.train()

