# 导入必要的库
from transformers import BertTokenizer, BertForSequenceClassification
import torch
#定义类别名称
CATEGORY_NAMES = ['World', 'Sports', 'Business', 'Technology']
# 加载微调后的模型和 tokenizer
print("加载微调后的 BERT 模型...")
model = BertForSequenceClassification.from_pretrained('./assets/weights/ag_news_bert')
tokenizer = BertTokenizer.from_pretrained('./assets/weights/ag_news_bert')
#设置模型为评估模式
model.eval()
# 定义预测函数
def predict(text):
    """对单条文本进行分类预测"""
    # 对输入文本进行编码
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=64
    )

    # 禁用梯度计算（推理阶段）
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

    return CATEGORY_NAMES[predicted_class_id]
#测试多个新样本
test_samples = [
    "The stock market surged after the Fed's announcement.",
    "Scientists discover a new exoplanet in the habitable zone.",
    "Manchester United wins the Premier League title.",
    "Peace talks between two countries have resumed in Geneva.",
    "Apple releases new iPhone with advanced AI features."
]
print("\n🔍 测试新样本分类结果：")
for text in test_samples:
    pred = predict(text)
    print(f"文本: {text}")
    print(f"预测类别: {pred}\n")