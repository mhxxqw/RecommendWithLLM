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

def load_product_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [ast.literal_eval(line) for line in f]
    return pd.DataFrame(data)

# def load_product_data(file_path, max_lines=1000):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = [ast.literal_eval(line) for _, line in zip(range(max_lines), f)]
#     return pd.DataFrame(data)




def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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


# def load_product_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = [ast.literal_eval(line) for line in f]
#     return pd.DataFrame(data)
#

def build_product_text(row):
    title = row.get("title", "") or ""
    description = row.get("description", "") or ""

    categories = ""
    if isinstance(row.get("categories"), list) and len(row["categories"]) > 0:
        cat_list = row["categories"][0]
        categories = " > ".join(cat_list)

    return f"{title}. {description}. Category: {categories}".strip()


def preprocess_products(df):
    df["text"] = df.apply(build_product_text, axis=1)
    return df[["asin", "text"]]


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

def get_user_liked_items(df, user_id, threshold=4.0):
    return df[(df['user_id'] == user_id) & (df['rating'] >= threshold)]['item_id'].tolist()


def precision_recall_at_k(recommended_items, liked_items, k):
    # 只考虑前 k 个推荐的 item（因为通常我们只评估前 5 或前 10 的推荐效果
    recommended_at_k = recommended_items[:k]
    # 交集操作：推荐列表中真正命中的用户喜欢的商品
    relevant = set(recommended_at_k) & set(liked_items)

    #如果你推荐了 5 个商品，其中 2 个是用户喜欢的 → Precision = 2/5 = 0.4
    precision = len(relevant) / len(recommended_at_k) if recommended_at_k else 0
    #如果用户喜欢 4 个商品，你命中了其中 2 个 → Recall = 2/4 = 0.5
    recall = len(relevant) / len(liked_items) if liked_items else 0

    return precision, recall


def main():
    # 文件路径
    # review_path = r"C:\Users\15529\Desktop\essay\FileData\Beauty_5.json"
    # product_path = r"C:\Users\15529\Desktop\essay\FileData\meta_Beauty.json"
    # model_path = r"C:\Users\15529\Desktop\essay\model\model1\all-MiniLM-L6-v2"

    review_path = "Beauty_5.json"
    product_path = "meta_Beauty.json"
    model_path = "/root/autodl-tmp/models/all-MiniLM-L6-v2"


    # 用户评论处理
    review_df = load_review_data(review_path)
    processed_reviews = preprocess_reviews(review_df)
    user_texts = aggregate_user_texts(processed_reviews)


    user_embedding_df = generate_user_embeddings(user_texts, model_path)


    # 商品元信息处理
    product_df = load_product_data(product_path)
    processed_product_df = preprocess_products(product_df)


    product_embedding_df = generate_product_embeddings(processed_product_df, model_path)


    # 替换为你要推荐的 user_id
    example_user_id = "A2V5R832QCSOMX"

    recommendations = recommend_for_user(
        user_id=example_user_id,
        user_embedding_df=user_embedding_df,
        product_df=product_df,
        product_embedding_df=product_embedding_df,
        top_k=5
    )

    from pprint import pprint
    pprint(recommendations)

    # Ground truth: 用户真实喜欢的商品（评分 >= 4）
    liked_items = get_user_liked_items(processed_reviews, example_user_id, threshold=4)

    print(liked_items)

    # 推荐的商品
    recommended_items = [item["item_id"] for item in recommendations]

    # 评估
    precision, recall = precision_recall_at_k(recommended_items, liked_items, k=5)

    print(f"Precision@5: {precision:.4f}")
    print(f"Recall@5: {recall:.4f}")


if __name__ == "__main__":
    main()
