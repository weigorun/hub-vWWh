#导入需要的库
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import List
# 初始化FastAPI应用
app = FastAPI(title="意图识别服务", version="1.0")
#加载TF-IDF模型
try:
    tfidf_model = joblib.load("models/tfidf_model.pkl")
except FileNotFoundError:
    raise RuntimeError("模型未找到！请先运行 training_code/train_tfidf.py")
#定义请求体
class TextInput(BaseModel):
    request_id: str
    request_text: str
#定义响应体
@app.post("/v1/text-cls/tfidf")
async def classify_intent_tfidf(input_data: TextInput):
    try:
        #预测
        pred = tfidf_model.predict([input_data.request_text])[0]
        proba = max(tfidf_model.predict_proba([input_data.request_text])[0])
        return {
            "request_id": input_data.request_id,
            "request_text": input_data.request_text,
            "classify_result": [pred],
            "confidence": float(proba),
            "error_msg": "ok"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测出错: {str(e)}")
#健康检查
@app.get("/health")
async def health_check():
    return {"status": "running"}
