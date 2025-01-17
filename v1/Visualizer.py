import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import os


class Visualizer:
    def __init__(self, result_dir):
        """
        初始化可视化工具
        :param result_dir: 保存结果的目录
        """
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def plot_metrics(self, train_acc, val_acc, train_loss, val_loss):
        """
        绘制并保存训练和验证的准确率和损失曲线
        """
        epochs = range(1, len(train_acc) + 1)

        plt.figure(figsize=(12, 5))

        # 准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Train vs Validation Accuracy")
        plt.legend()

        # 损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train vs Validation Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, "metrics_curve.png"))
        plt.show()

    def plot_roc_curve(self, labels, probs, num_classes):
        """
        绘制并保存 ROC 和 AUC 曲线
        """
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.result_dir, "roc_curve.png"))
        plt.show()

    def plot_confusion_matrix(self, labels, preds, class_names):
        """
        绘制并保存混淆矩阵热力图
        """
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.result_dir, "confusion_matrix.png"))
        plt.show()
