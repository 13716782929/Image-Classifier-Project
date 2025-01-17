import torch
from tqdm import tqdm
from Visualizer import Visualizer
import numpy as np
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

        # 初始化可视化工具
        self.visualizer = Visualizer(config.RUN_DIR)

        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器
        if config.LR_SCHEDULE == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
        elif config.LR_SCHEDULE == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.EPOCHS)
        else:
            self.scheduler = None

        # 创建结果目录
        config.prepare_directories()

        # 保存配置文件和模型定义文件
        self.save_source_files()

    def save_source_files(self):
        """
        将 Config.py 和 Model.py 保存到运行结果目录
        """
        try:
            shutil.copy("Config.py", self.config.RUN_DIR)
            shutil.copy("Model.py", self.config.RUN_DIR)
            print("Config.py and Model.py have been copied to the results directory.")
        except FileNotFoundError as e:
            print(f"Error copying source files: {e}")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / len(self.train_loader.dataset)
        accuracy = 100.0 * correct / total
        self.train_loss.append(avg_loss)
        self.train_acc.append(accuracy)
        return avg_loss, accuracy

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Validating Epoch {epoch}"):
                images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(outputs.softmax(dim=1).cpu().numpy())

        avg_loss = running_loss / len(self.val_loader.dataset)
        accuracy = 100.0 * correct / total
        self.val_loss.append(avg_loss)
        self.val_acc.append(accuracy)
        return avg_loss, accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def save_model(self):
        torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")

    def save_log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.config.LOG_PATH, "a") as log_file:
            log_file.write(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%\n"
            )

    def train(self):
        best_accuracy = 0.0

        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"Epoch {epoch}/{self.config.EPOCHS}")
            train_loss, train_accuracy = self.train_one_epoch(epoch)
            val_loss, val_accuracy, val_labels, val_preds, val_probs = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(val_accuracy)
                self.save_model()

            self.save_log(epoch, train_loss, train_accuracy, val_loss, val_accuracy)

        # 绘制曲线
        self.visualizer.plot_metrics(self.train_acc, self.val_acc, self.train_loss, self.val_loss)

        # 绘制 ROC 和混淆矩阵
        self.visualizer.plot_roc_curve(val_labels, val_probs, self.config.NUM_CLASSES)
        self.visualizer.plot_confusion_matrix(val_labels, val_preds, [str(i) for i in range(self.config.NUM_CLASSES)])

        print("\nTraining complete!")
        print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
        print(f"Final Training Accuracy: {self.train_acc[-1]:.2f}%")
        print(f"Final Validation Accuracy: {self.val_acc[-1]:.2f}%")
        print(f"Final Training Loss: {self.train_loss[-1]:.4f}")
        print(f"Final Validation Loss: {self.val_loss[-1]:.4f}")
