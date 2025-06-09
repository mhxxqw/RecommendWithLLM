import pandas as pd
import json
import ast
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# def load_review_data(file_path, max_lines=1000):
#     with open(file_path, "r") as f:
#         data = [json.loads(line) for _, line in zip(range(max_lines), f)]
#     return pd.DataFrame(data)

def load_review_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def preprocess_reviews(df):
    def preprocess_row(row):
        merged = f"{row.get('summary', '')} {row.get('reviewText', '')}"
        return clean_text(merged)

    df["text"] = df.apply(preprocess_row, axis=1)
    df["user_id"] = df["reviewerID"]
    df["item_id"] = df["asin"]
    df["rating"] = df["overall"]
    df["timestamp"] = df["unixReviewTime"]

    return df[["user_id", "item_id", "text", "rating", "timestamp"]]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_user_reviews(df, train_ratio=0.8):
    train_rows = []
    test_rows = []
    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values(by="timestamp")
        n = len(user_df)
        split_point = int(n * train_ratio)
        if split_point == 0:
            # 无法切分时，全部作为训练
            train_rows.append(user_df)
        else:
            train_rows.append(user_df.iloc[:split_point])
            test_rows.append(user_df.iloc[split_point:])
    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows) if test_rows else pd.DataFrame(columns=df.columns)
    return train_df, test_df

def aggregate_user_texts(processed_df):
    user_texts = processed_df.groupby("user_id")["text"].apply(lambda texts: " ".join(texts)).reset_index()
    user_texts.columns = ["user_id", "full_text"]
    return user_texts


def encode_texts(texts, model_path):
    model = SentenceTransformer(model_path)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def generate_user_embeddings(user_texts, model_path):
    embeddings = encode_texts(user_texts["full_text"].tolist(), model_path)
    user_embedding_df = pd.DataFrame(embeddings)
    user_embedding_df["user_id"] = user_texts["user_id"]
    return user_embedding_df

def load_product_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [ast.literal_eval(line) for line in f]
    return pd.DataFrame(data)

# def load_product_data(file_path, max_lines=1000):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = [ast.literal_eval(line) for _, line in zip(range(max_lines), f)]
#     return pd.DataFrame(data)

def preprocess_products(df):
    df["text"] = df.apply(build_product_text, axis=1)
    return df[["asin", "text"]]

def build_product_text(row):
    title = row.get("title", "") or ""
    description = row.get("description", "") or ""

    categories = ""
    if isinstance(row.get("categories"), list) and len(row["categories"]) > 0:
        cat_list = row["categories"][0]
        categories = " > ".join(cat_list)

    return f"{title}. {description}. Category: {categories}".strip()

def get_user_liked_items(df, user_id, threshold=4.0):
    return df[(df['user_id'] == user_id) & (df['rating'] >= threshold)]['item_id'].tolist()


def precision_recall_at_k(recommended_items, liked_items, k):
    recommended_at_k = recommended_items[:k]
    relevant = set(recommended_at_k) & set(liked_items)
    precision = len(relevant) / len(recommended_at_k) if recommended_at_k else 0
    recall = len(relevant) / len(liked_items) if liked_items else 0
    return precision, recall

def generate_product_embeddings(product_df, model_path):
    embeddings = encode_texts(product_df["text"].tolist(), model_path)
    embedding_df = pd.DataFrame(embeddings)
    embedding_df["item_id"] = product_df["asin"].values
    return embedding_df
def recommend_for_user(user_id, user_embedding_df, product_df, product_embedding_df, top_k=5):
    # 1. 找到该用户的向量
    user_row = user_embedding_df[user_embedding_df["user_id"] == user_id]
    if user_row.empty:
        return f"User ID {user_id} not found."

    user_vector = user_row.drop(columns=["user_id"]).values  # shape: (1, 384)

    # 2. 所有商品向量（去除 item_id）
    item_vectors = product_embedding_df.drop(columns=["item_id"]).values  # shape: (N, 384)

    # 3. 计算余弦相似度
    similarities = cosine_similarity(user_vector, item_vectors)[0]  # shape: (N,)

    # 4. 获取 top_k 最大相似度的索引
    top_indices = similarities.argsort()[::-1][:top_k]

    # 5. 提取推荐商品信息
    top_items = product_embedding_df.iloc[top_indices]["item_id"].values
    top_scores = similarities[top_indices]

    recommended = []

    for asin, score in zip(top_items, top_scores):
        prod_row = product_df[product_df["asin"] == asin].iloc[0]
        reason = prod_row.get("text", "")[:300] + "..."  # 推荐理由可取前300字符
        recommended.append({
            "item_id": asin,
            "title": prod_row.get("title", ""),
            "score": round(float(score), 4),
            "reason": reason
        })

    return recommended


def main():
    # 文件路径
    review_path = r"C:\Users\15529\Desktop\essay\FileData\Beauty_5.json"
    product_path = r"C:\Users\15529\Desktop\essay\FileData\meta_Beauty.json"
    model_path = r"C:\Users\15529\Desktop\essay\model\model1\all-MiniLM-L6-v2"

    # 用户评论处理
    review_df = load_review_data(review_path)
    processed_reviews = preprocess_reviews(review_df)

    #  划分训练/测试集
    train_df, test_df = split_user_reviews(processed_reviews)

    # 用训练集生成用户 embedding
    user_texts_train = aggregate_user_texts(train_df)
    user_embedding_df = generate_user_embeddings(user_texts_train, model_path)

    # 商品处理
    product_df = load_product_data(product_path)
    processed_product_df = preprocess_products(product_df)
    product_embedding_df = generate_product_embeddings(processed_product_df, model_path)

    # 推荐和评估
    example_user_id = "A2V5R832QCSOMX"

    recommendations = recommend_for_user(
        user_id=example_user_id,
        user_embedding_df=user_embedding_df,
        product_df=product_df,
        product_embedding_df=product_embedding_df,
        top_k=5
    )

    # 如果用户没有测试集记录
    if test_df[test_df["user_id"] == example_user_id].empty:
        print(f"No test data for user {example_user_id}. Cannot evaluate precision/recall.")
    else:
        liked_items = get_user_liked_items(test_df, example_user_id, threshold=4.0)
        recommended_items = [item["item_id"] for item in recommendations]

        precision, recall = precision_recall_at_k(recommended_items, liked_items, k=5)

        print("\nTop-5 Recommendations:")
        from pprint import pprint
        pprint(recommendations)

        print(f"\nPrecision@5: {precision:.4f}")
        print(f"Recall@5: {recall:.4f}")


if __name__ == "__main__":
    main()
