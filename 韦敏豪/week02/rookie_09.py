import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
#创建不同结构的模型
def create_model(model_type, input_dim, hidden_dim, output_dim):
    #设置不同类型输出的结果
    if model_type == "simple":
        # 1层隐藏层
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    elif model_type == "deep":
        # 2层隐藏层
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    elif model_type == "wide":
        # 1层隐藏层，但节点数更多
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    elif model_type == "linear":
        # 无隐藏层
        return nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
# 模拟训练过程
# 模拟数据
input_dim = 500  # 词典大小
output_dim = 3  # 类别数
batch_size = 32  # 批次大小
num_epochs = 8  # 训练轮数
# 创建不同模型
models = {
    "Simple (1 hidden layer)": create_model("simple", input_dim, 128, output_dim),
    "Deep (2 hidden layers)": create_model("deep", input_dim, 128, output_dim),
    "Wide (1 hidden layer, wider)": create_model("wide", input_dim, 64, output_dim),
    "Linear (no hidden layer)": create_model("linear", input_dim, 128, output_dim)
}
# 记录损失
loss_history = {name: [] for name in models.keys()}
# 训练每个模型
for name, model in models.items():
    # 优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    # 模拟训练
    for epoch in range(num_epochs):
        # 生成模拟输入和标签
        inputs = torch.randn(batch_size, input_dim)
        labels = torch.randint(0, output_dim, (batch_size,))
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录损失
        loss_history[name].append(loss.item())
    print(f"{name} 训练完成，最终损失: {loss.item():.4f}")
# 输出可视化结果
plt.figure(figsize=(10, 6))
# 绘制每种模型的损失曲线
for name, losses in loss_history.items():
    plt.plot(losses, label=name)
# 设置图表标题和标签
plt.title("不同模型结构的训练损失对比")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("model_loss_comparison.png")  # 保存图像
plt.show()
print("模型对比完成！结果已保存为 'model_loss_comparison.png'")