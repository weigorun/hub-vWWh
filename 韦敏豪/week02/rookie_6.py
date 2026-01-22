#导入所需的库
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#生成sin函数数据
x = np.linspace(0, 2 * np.pi, 100)  # 0到2π的100个点
y = np.sin(x)  # sin值
# 将数据转换为PyTorch张量里
X = torch.tensor(x, dtype=torch.float32).view(-1, 1)  # 添加批次维度
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  #同上
#定义多层网络模型
model = nn.Sequential(
    nn.Linear(1, 64),  # 输入到隐藏
    nn.ReLU(),  # 激活
    nn.Linear(64, 64),  # 隐藏到隐藏
    nn.ReLU(),  # 激活
    nn.Linear(64, 1)  # 隐藏到输出
)
#训练模型
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器

# 训练1000轮
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    # 每100轮打印一次损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
#绘制可视化结果
with torch.no_grad():
    y_pred = model(X).numpy()
# 创建图表
plt.figure(figsize=(12, 8))
# 原始函数和拟合结果
plt.subplot(2, 1, 1)
plt.plot(x, y, 'b-', label='Original sin function')
plt.plot(x, y_pred, 'r--', label='Fitted model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting sin function with a 3-layer neural network')
plt.legend()
plt.grid(True)
# Loss变化
plt.subplot(2, 1, 2)
plt.plot(losses, 'g-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
# 保存并显示图表
plt.tight_layout()
plt.savefig('sin_fit.png')
plt.show()
# 验证特定点
test_x = [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
test_y = np.sin(test_x)
with torch.no_grad():
    test_pred = model(torch.tensor(test_x, dtype=torch.float32).view(-1, 1)).numpy()
# 输出结果
print("\nSin function fitting results:")
for i in range(len(test_x)):
    print(f"sin({test_x[i]:.2f}) = {test_y[i]:.4f}, Predicted: {test_pred[i][0]:.4f}")
