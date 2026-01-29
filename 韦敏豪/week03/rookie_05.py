#导入必要的库
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#从指定路径读取CSV文件
dataset = pd.read_csv("../../day01/Week01/dataset.csv", sep="\t", header=None)
# 提取第一列作为文本数据，并转换为Python列表
texts = dataset[0].tolist()
# 提取第二列作为标签数据，并转换为Python列表
string_labels = dataset[1].tolist()
# 创建一个字典，将唯一的字符串标签映射到从0开始的整数索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将原始的字符串标签列表转换为对应的整数索引列表，供模型训练使用
numerical_labels = [label_to_index[label] for label in string_labels]
# 构建字符级别的词汇表 (Character-level Vocabulary)
# 初始化字典，包含一个特殊的填充字符 '<pad>'，其索引为0
char_to_index = {'<pad>': 0}
# 遍历所有文本中的每一个字符
for text in texts:
    for char in text:
        # 如果字符不在词汇表中，则为其分配一个新的索引
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
# 创建反向映射字典，用于从索引查找字符（主要用于调试或展示）
index_to_char = {i: char for char, i in char_to_index.items()}
# 词汇表的总大小（即唯一字符的数量）
vocab_size = len(char_to_index)
# 定义输入序列的最大长度。超过此长度的文本将被截断，不足的将被填充
max_len = 40
#自定义数据集类
# 继承自 PyTorch 的 Dataset 基类，用于自定义数据加载逻辑
class CharLSTMDataset(Dataset):
 # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts  # 存储原始文本
        # 将标签列表转换为PyTorch的LongTensor，用于分类任务
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index  # 存储字符映射字典
        self.max_len = max_len  # 存储最大序列长度

    def __len__(self):
        # 返回数据集样本个数
        return len(self.texts)
        #获取单个样本及其标签
    def __getitem__(self, idx):
        # 获取索引为idx的原始文本
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]
#通用循环神经网络分类器
MODEL_TYPE = 'lstm'  #在此处修改以切换模型 'rnn','gru','lstm'
#如果运行RNN则修改为MODEL_TYPE = 'rnn'
#如果运行GRU则修改为MODEL_TYPE = 'gru'
#如果运行LSTM则修改为MODEL_TYPE = 'lstm'
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, model_type='lstm'):
        # 词表大小 转换后维度的维度
        super(RNNClassifier, self).__init__()
        self.model_type = model_type.lower()  # 转换为小写以方便比较
        #嵌入层
        # 将离散的字符索引转换为连续的、稠密的向量表示（可学习的参数）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #循环神经网络层 (Recurrent Layer)
        #根据model_type选择不同的模型
        if self.model_type == 'rnn':
            #使用的RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif self.model_type == 'lstm':
            #使用LSTM
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.model_type == 'gru':
            #使用GRU
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            #报错未输入指定的训练模型
            raise ValueError("model_type must be 'rnn', 'lstm', or 'gru'")
        #全连接输出层
        #将RNN的最终隐藏状态映射到各个类别的logits
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        #通过嵌入层
        embedded = self.embedding(x)
        #通过RNN层
        rnn_out, hidden = self.rnn(embedded)
        #最终的隐藏状态用于分类
        if self.model_type == 'lstm':
            final_hidden = hidden[0].squeeze(0)
        else:
            final_hidden = hidden.squeeze(0)
        #全连接层进行分类
        output = self.fc(final_hidden)
        return output
#训练与评估流程
#创建数据集实例和数据加载器
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)
#定义模型超参数
embedding_dim = 64  # 嵌入向量的维度
hidden_dim = 128  # RNN隐藏层的维度
output_dim = len(label_to_index)  # 输出维度等于类别数量
#实例化模型，传入之前定义的MODEL_TYPE
model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, MODEL_TYPE)
#定义损失函数：交叉熵损失，适用于多分类任务
criterion = nn.CrossEntropyLoss()
# 定义优化器：Adam优化器，学习率设为0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 开始训练循环
num_epochs = 4  # 训练4个epoch
for epoch in range(num_epochs):
    model.train()  # 将模型设置为训练模式（启用Dropout, BatchNorm等）
    running_loss = 0.0  # 累计当前epoch的总损失
    #遍历数据加载器中的每一个batch
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item()  # 累加损失值
        # 每50个batch打印一次当前batch的损失，用于监控训练过程
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
    #打印当前epoch的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
#预测函数
def classify_text(text, model, char_to_index, max_len, index_to_label):
    #对输入文本进行与训练数据相同的预处理：截断、映射、填充
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    #添加一个batch维度，因为模型期望输入是 (batch_size, seq_len)
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    #输出logits中概率最大的类别索引
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    # 将索引转换回原始的标签字符串
    predicted_label = index_to_label[predicted_index]
    return predicted_label
#创建索引到标签的反向映射字典，用于预测结果展示
index_to_label = {i: label for label, i in label_to_index.items()}
#执行预测示例
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
