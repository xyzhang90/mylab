import torch
from torch.utils.tensorboard import SummaryWriter

# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 数据生成部分修正 ---
# 修正：确保是整数逗号分隔
inputs = torch.rand(100, 3)

# 定义真实参数 (用于生成数据)
# 注意：为了匹配 inputs(100,3) 和 targets(100,1)，权重必须是 (3,1)
true_w = torch.tensor([[1.0], [2.0], [3.0]])
true_b = torch.tensor([4.0])

# 生成目标值
targets = inputs @ true_w + true_b + 0.1 * torch.randn(100, 1)

# --- 2. SummaryWriter 修正 ---
# 修正：路径使用双斜杠，或者 r"..."
# 建议指定一个具体的文件夹名，方便管理
writer = SummaryWriter(log_dir="E:\\MyLabLogs")

# --- 3. 模型参数初始化 ---
w = torch.rand((3, 1), requires_grad=True, device=device)
b = torch.rand((1,), requires_grad=True, device=device)

# 数据移动到设备
inputs = inputs.to(device)
targets = targets.to(device)

epoch = 10000
lr = 0.003

for i in range(epoch):
    # 前向传播
    outputs = inputs @ w + b

    # 计算损失
    loss = torch.mean(torch.square(outputs - targets))

    # 写入 TensorBoard
    # 修正：去掉了你代码中可能存在的非法特殊符号（看起来像中间点 ·）
    writer.add_scalar("loss/train", loss.item(), i)

    # 反向传播
    loss.backward()

    # 更新参数
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad  # 修正：这里你之前的代码可能是 h，必须是 b

    # 梯度清零
    w.grad.zero_()
    b.grad.zero_()

    if i % 1000 == 0:
        print(f"Epoch {i}, loss: {loss.item()}")

writer.close()

print("训练后的权重 w:", w)
print("训练后的偏置 b:", b)