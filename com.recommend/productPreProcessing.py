import pandas as pd
import ast
from sentence_transformers import SentenceTransformer

with open(r"C:\Users\15529\Desktop\essay\FileData\meta_Beauty.json", 'r', encoding='utf-8') as f:
    # 使用ast安全转换单引号字符串
    data = [ast.literal_eval(line) for line in f]


product_df = pd.DataFrame(data)
print(f"Total number of products: {product_df['asin'].nunique()}")


def build_product_text(row):
    title = row.get("title", "") or ""
    description = row.get("description", "") or ""

    # 处理类别
    categories = ""
    if isinstance(row.get("categories"), list) and len(row["categories"]) > 0:
        cat_list = row["categories"][0]  # 一级类别路径
        categories = " > ".join(cat_list)

    # 拼接成完整语义文本
    combined_text = f"{title}. {description}. Category: {categories}"
    return combined_text.strip()


# 新增一列 text
product_df["text"] = product_df.apply(build_product_text, axis=1)
print(product_df[["asin", "text"]].head(1).to_dict(orient="records"))


model = SentenceTransformer(r"C:\Users\15529\Desktop\essay\model\model1\all-MiniLM-L6-v2")
# 编码文本为向量
product_embeddings = model.encode(product_df["text"].tolist(), show_progress_bar=True)


embedding_df = pd.DataFrame(product_embeddings)
embedding_df["item_id"] = product_df["asin"].values
print(embedding_df.head(1).to_dict(orient="records")[0])
