import torch
import torch.nn as nn
from Config import Config


class FruitClassifierModel(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), num_classes=4):
        """
        PyTorch 实现的 Fruit Classifier 模型。
        Args:
            input_shape (tuple): 输入图片的形状，默认 (3, 128, 128)。
            num_classes (int): 分类类别数量，默认 4。
        """
        super(FruitClassifierModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # 第一层卷积 + 跳跃连接
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.shortcut1 = nn.Conv2d(3, 32, kernel_size=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + 跳跃连接
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.shortcut2 = nn.Conv2d(32, 64, kernel_size=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层卷积 + 跳跃连接
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.shortcut3 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四层卷积 + 跳跃连接
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.shortcut4 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第五层卷积 + 跳跃连接
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.shortcut5 = nn.Conv2d(256, 512, kernel_size=1, padding=0)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        flattened_size = 512 * (input_shape[1] // 32) * (input_shape[2] // 32)  # 动态计算展平大小
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 第一层卷积 + 跳跃连接
        shortcut = self.shortcut1(x)
        x = torch.relu(self.conv1(x))
        x = torch.add(x, shortcut)
        x = self.pool1(x)

        # 第二层卷积 + 跳跃连接
        shortcut = self.shortcut2(x)
        x = torch.relu(self.conv2(x))
        x = torch.add(x, shortcut)
        x = self.pool2(x)

        # 第三层卷积 + 跳跃连接
        shortcut = self.shortcut3(x)
        x = torch.relu(self.conv3(x))
        x = torch.add(x, shortcut)
        x = self.pool3(x)

        # 第四层卷积 + 跳跃连接
        shortcut = self.shortcut4(x)
        x = torch.relu(self.conv4(x))
        x = torch.add(x, shortcut)
        x = self.pool4(x)

        # 第五层卷积 + 跳跃连接
        shortcut = self.shortcut5(x)
        x = torch.relu(self.conv5(x))
        x = torch.add(x, shortcut)
        x = self.pool5(x)

        # 全连接层
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # 假设 Config 定义了 SIZE 和 NUM_CLASSES
    model = FruitClassifierModel(input_shape=(3, Config.SIZE, Config.SIZE), num_classes=Config.NUM_CLASSES)

    # 打印模型结构
    print(model)

    # 测试输入数据
    sample_input = torch.randn(8, 3, Config.SIZE, Config.SIZE)  # 假设 batch_size=8
    output = model(sample_input)
    print(f"Output shape: {output.shape}")  # 应输出: torch.Size([8, Config.NUM_CLASSES])