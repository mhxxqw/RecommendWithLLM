import pandas as pd
import json
import  re
from sentence_transformers import SentenceTransformer


with open(r"C:\Users\15529\Desktop\essay\FileData\Beauty_5.json", "r") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
# 展示前几行
# print(df.head())


# 定义文本清洗函数
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # 去除HTML标签
    text = re.sub(r'[^a-z0-9\s]', '', text)  # 去除标点/特殊字符
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
    return text

# 合并并清洗 summary 和 reviewText
def preprocess_row(row):
    merged = f"{row.get('summary', '')} {row.get('reviewText', '')}"
    return clean_text(merged)

# 应用到 DataFrame
df["text"] = df.apply(preprocess_row, axis=1)
df["user_id"] = df["reviewerID"]
df["item_id"] = df["asin"]
df["rating"] = df["overall"]
df["timestamp"] = df["unixReviewTime"]

# 保留所需字段
processed_df = df[["user_id", "item_id", "text", "rating", "timestamp"]]

# 按用户聚合评论文本
user_texts = processed_df.groupby("user_id")["text"].apply(lambda texts: " ".join(texts)).reset_index()
user_texts.columns = ["user_id", "full_text"]

print(user_texts.head(2).to_dict(orient="records"))


# 初始化预训练模型
model = SentenceTransformer(r"C:\Users\15529\Desktop\essay\model\model1\all-MiniLM-L6-v2")
# 编码文本为向量
user_embeddings = model.encode(user_texts["full_text"].tolist(), show_progress_bar=True)

# 合并向量和用户ID
user_embedding_df = pd.DataFrame(user_embeddings)
user_embedding_df["user_id"] = user_texts["user_id"]
print(user_embedding_df.head(2).to_dict(orient="records"))
