# 导入必要的库
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
#加载AG_News数据集
dataset = load_dataset('ag_news')
#将训练集和测试集转换为 pandas DataFrame，便于操作
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
#打印数据集基本信息，确认类别数量
print("训练集前5行：")
print(train_df.head())
print(f"类别数量：{train_df['label'].nunique()}")
#标签编码：将原始标签（0,1,2,3）保留，但记录类别名称
category_names = ['World', 'Sports', 'Business', 'Technology']
train_labels = train_df['label'].values
test_labels = test_df['label'].values
#初始化BERT分词器（使用uncased版本，忽略大小写）
print("加载 BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 对训练文本进行分词和编码
print("正在对训练文本进行分词...")
train_encodings = tokenizer(
    train_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors='pt'
)
#对测试文本进行分词和编码
print("正在对测试文本进行分词...")
test_encodings = tokenizer(
    test_df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors='pt'
)
#构建 Hugging Face Dataset对象
print("构建 Dataset 对象...")
#创建训练数据集
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
#创建测试数据集
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})
#定义评估指标函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}
#加载预训练 BERT 模型，并设置分类头
print("加载 BERT base 预训练模型...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4
)
#配置训练参数
training_args = TrainingArguments(
    output_dir='./assets/weights/ag_news_bert',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42
)
#创建 Trainer 并开始训练
print("创建 Trainer 并开始训练...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
#开始训练
trainer.train()
#评估最终模型性能
print("评估最终模型...")
results = trainer.evaluate()
print(f"测试集准确率: {results['eval_accuracy']:.4f}")
#保存模型和分词器
print("保存微调后的模型和 tokenizer...")
model.save_pretrained('./assets/weights/ag_news_bert')
tokenizer.save_pretrained('./assets/weights/ag_news_bert')
#同时保存 PyTorch 模型权重
torch.save(model.state_dict(), './assets/weights/ag_news_bert.pt')
print("BERT 微调完成！模型已保存至 ./assets/weights/ag_news_bert")