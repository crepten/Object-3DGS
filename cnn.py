import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(ImageCNN, self).__init__()

        # 卷积层部分
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层部分
        # 假设输入图像尺寸为32x32，经过三次池化后变为4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # 展平
        x = x.view(-1, 64 * 4 * 4)

        # 全连接层
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

# 创建模型实例
def create_model(num_classes=10, input_channels=3):
    model = ImageCNN(num_classes=num_classes, input_channels=input_channels)
    return model

# 测试模型
if __name__ == "__main__":
    # 测试输入 (批次大小, 通道数, 高度, 宽度)
    x = torch.randn(1, 3, 32, 32)
    model = create_model(num_classes=10)
    output = model(x)
    print(f"输出形状: {output.shape}")
    print(f"示例输出: {output}")