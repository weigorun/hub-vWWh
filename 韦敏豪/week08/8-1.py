from pydantic import BaseModel, Field
import re
from typing import Optional
# 结构化翻译请求参数，明确字段含义和描述
class TranslationRequest(BaseModel):
    src_lang: str = Field(description="原始语种（如：英语、中文）")
    dst_lang: str = Field(description="目标语种（如：英语、中文）")
    text: str = Field(description="需要翻译的文本内容")

def parse_translation_input(user_input: str) -> TranslationRequest:
    # 正则匹配核心内容
    pattern = r"[将|把]?(.*?)[翻译为|翻译成](.*)"
    match = re.search(pattern, user_input.strip())

    if not match:
        raise ValueError("输入格式不合法！请使用示例格式：'帮我将good！翻译为中文'")

    # 提取待翻译文本和目标语种
    text = match.group(1).strip()
    dst_lang = match.group(2).strip()

    # 辅助函数：自动识别原始语种（基础版，可扩展）
    def detect_source_language(text: str) -> str:
        # 中文判断：匹配中文字符范围
        if re.search(r'[\u4e00-\u9fff]', text):
            return "中文"
        # 英语判断：匹配字母
        elif re.search(r'[a-zA-Z]', text):
            return "英语"
        # 可扩展：日语、法语等其他语种识别
        else:
            return "未知"

    # 识别原始语种
    src_lang = detect_source_language(text)
    return TranslationRequest(
        src_lang=src_lang,
        dst_lang=dst_lang,
        text=text
    )



def translate_text(req: TranslationRequest) -> str:
    # 模拟中英互译词典
    translate_dict = {
        "good！": "好的！",
        "hello": "你好",
        "谢谢": "Thank you",
        "再见": "Goodbye",
        "I love you": "我爱你"
    }

    # 中英互译逻辑
    if req.src_lang == "英语" and req.dst_lang == "中文":
        return translate_dict.get(req.text, f"暂不支持翻译：{req.text}")
    elif req.src_lang == "中文" and req.dst_lang == "英语":
        return translate_dict.get(req.text, f"暂不支持翻译：{req.text}")
    # 可扩展：其他语种翻译逻辑
    else:
        return f"暂未支持【{req.src_lang}】→【{req.dst_lang}】的翻译"


# ---------------------- 4. 智能体主逻辑 ----------------------
def translation_agent(user_input: str) -> None:

    try:
        # 步骤1：解析用户输入，生成结构化请求
        trans_request = parse_translation_input(user_input)
        print("===== 解析结果 =====")
        print(f"原始语种：{trans_request.src_lang}")
        print(f"目标语种：{trans_request.dst_lang}")
        print(f"待翻译文本：{trans_request.text}")

        # 步骤2：执行翻译
        trans_result = translate_text(trans_request)
        print("\n===== 翻译结果 =====")
        print(trans_result)

    except ValueError as e:
        print(f"错误：{e}")

if __name__ == "__main__":
    # 测试输入（模拟用户指令）
    user_input = "帮我将good！翻译为中文"
    # 启动智能体
    translation_agent(user_input)