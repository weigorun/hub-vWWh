import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os
#创建模型目录
os.makedirs("models", exist_ok=True)
#加载数据
df = pd.read_csv("assets/dataset/train.csv")
#构建 TF-IDF + 逻辑回归 pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression())
])
#训练
print("正在训练 TF-IDF 模型...")
pipeline.fit(df['text'], df['label'])
#保存模型
joblib.dump(pipeline, "models/tfidf_model.pkl")
print("TF-IDF 模型已保存至 models/tfidf_model.pkl")