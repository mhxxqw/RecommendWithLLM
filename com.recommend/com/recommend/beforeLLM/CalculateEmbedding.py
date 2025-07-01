import os
import pickle
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer

def encode_texts(texts, model_path):
    # Encode a list of product texts using a SentenceTransformer
    model = SentenceTransformer(model_path)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def save_embeddings(embedding_df, save_path):
    # Save embeddings DataFrame to a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(embedding_df, f)

def load_embeddings(load_path):
    # Load embeddings DataFrame from pickle if exists
    if not os.path.exists(load_path):
        return None
    with open(load_path, 'rb') as f:
        return pickle.load(f)

def build_product_text(row):
    # Combine title, description, and categories into a single text
    title = row.get("title", "") or ""
    description = row.get("description", "") or ""

    categories = ""
    if isinstance(row.get("categories"), list) and len(row["categories"]) > 0:
        cat_list = row["categories"][0]
        categories = " > ".join(cat_list)

    return f"{title}. {description}. Category: {categories}".strip()

def preprocess_products(df):
    # Apply text building to each product
    df["text"] = df.apply(build_product_text, axis=1)
    return df[["asin", "text"]]

def load_product_data(file_path):
    # Load product metadata from JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [ast.literal_eval(line) for line in f]
    return pd.DataFrame(data)

def generate_product_embeddings(product_df, model_path):
    # Generate text embeddings for products
    embeddings = encode_texts(product_df["text"].tolist(), model_path)
    embedding_df = pd.DataFrame(embeddings)
    embedding_df["item_id"] = product_df["asin"].values
    return embedding_df

def main():
    product_path = "meta_Beauty.json"
    model_path = os.path.abspath(r"C:\Users\15529\Desktop\essay\model\model1\all-MiniLM-L6-v2")
    embedding_cache = "embedding/product_embeddings.pkl"

    # Load cached embeddings if they exist
    product_embedding_df = load_embeddings(embedding_cache)
    if product_embedding_df is None:
        print("No cached embeddings found, starting computation...")
        product_df = load_product_data(product_path)
        processed_product_df = preprocess_products(product_df)
        product_embedding_df = generate_product_embeddings(processed_product_df, model_path)
        save_embeddings(product_embedding_df, embedding_cache)
        print(f"Embeddings computed and saved to: {embedding_cache}")
    else:
        print(f"Successfully loaded cached embeddings from: {embedding_cache}")

if __name__ == "__main__":
    main()
