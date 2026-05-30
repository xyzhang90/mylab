import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter   # ← 加上这一行
# 自定义数据集
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)

    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                items = line.strip().split(",")
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels

    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).view(-1)
        image = image / 255.0  # 归一化
        image = (image - 0.1307) / 0.3081  # 标准化
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.images)


# 模型定义
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# 参数设置
batch_size = 64
learning_rate = 0.1
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [NEW] 创建 SummaryWriter，日志保存在指定文件夹
writer = SummaryWriter(log_dir="E:\\MyLabLogs\\MNIST_experiment")

# 数据加载（路径请确保正确）
train_dataset = MNISTDataset(r"E:\mylab\mnist\mnist_train.csv\mnist_train.csv")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset(r"E:\mylab\mnist\mnist_test.csv\mnist_test.csv")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练过程
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total

    # [NEW] 记录到 TensorBoard
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

# [NEW] 测试过程也记录测试准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
test_acc = 100 * correct / total
writer.add_scalar("Accuracy/test", test_acc, 0)  # 只记录最终测试准确率，若想按epoch记录可在每个epoch后测试

print(f"Test Accuracy: {test_acc:.2f}%")

writer.close()