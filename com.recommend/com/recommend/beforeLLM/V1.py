import torch
import pandas as pd
import numpy as np
import pickle
import json
import re
from tqdm import tqdm
from sklearn.cluster import KMeans

def load_embeddings(load_path, device="cuda"):
    # load product embeddings from pickle file
    with open(load_path, 'rb') as f:
        df = pickle.load(f)
    item_ids = df['item_id'].values
    embedding_cols = [c for c in df.columns if c != 'item_id']
    embeddings = torch.tensor(df[embedding_cols].values, device=device, dtype=torch.float32)
    return item_ids, embeddings

def load_review_data(file_path):
    # load raw review data from JSON
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def clean_text(text):
    # clean HTML tags and non-alphanumeric characters
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_reviews(df):
    # convert ratings to text sentiment and merge with summary/review text
    def rating_to_sentiment(rating):
        mapping = {
            1: "very dissatisfied",
            2: "dissatisfied",
            3: "neutral",
            4: "satisfied",
            5: "very satisfied"
        }
        return mapping.get(int(rating), "neutral")

    def preprocess_row(row):
        sentiment = rating_to_sentiment(row.get("overall", 3))
        summary = row.get("summary", "")
        review_text = row.get("reviewText", "")
        return clean_text(f"User sentiment: {sentiment}. {summary} {review_text}")

    df['text'] = df.apply(preprocess_row, axis=1)
    df['user_id'] = df['reviewerID']
    df['item_id'] = df['asin']
    df['rating'] = df['overall']
    df['timestamp'] = df['unixReviewTime']
    return df[['user_id', 'item_id', 'text', 'rating', 'timestamp']]

def split_train_test_by_time(df, test_size=0.2):
    # split each user's reviews by time for train/test
    df_sorted = df.sort_values(['user_id', 'timestamp'])
    train, test = [], []
    for user, group in df_sorted.groupby('user_id'):
        n = len(group)
        split = int(n * (1 - test_size))
        train.append(group.iloc[:split])
        test.append(group.iloc[split:])
    return pd.concat(train), pd.concat(test)

def aggregate_user_profiles_multi_interest(
    train_df,
    product_item_ids,
    product_embeddings,
    lambda_decay=0.01,
    history_size=20,
    n_clusters=3
):
    # build multi-interest user profiles using KMeans clustering on embeddings
    id_to_idx = {iid: idx for idx, iid in enumerate(product_item_ids)}
    user_profiles = []
    user_ids = []
    purchased_items = {}
    user_groups = train_df.sort_values(['user_id', 'timestamp']).groupby('user_id')

    for user_id, user_data in tqdm(user_groups, desc="User profiles"):
        user_data = user_data.tail(history_size)  # keep only recent interactions
        items = user_data['item_id'].values
        times = user_data['timestamp'].values
        ratings = user_data['rating'].values
        T = times.max()
        normalized_ratings = ratings / 5.0
        weights = np.exp(-lambda_decay * (T - times)) * normalized_ratings  # time decay + rating
        indices = [id_to_idx[iid] for iid in items if iid in id_to_idx]

        if len(indices) == 0:
            # cold-start, zero vector
            avg_embs = [torch.zeros(product_embeddings.shape[1], device=product_embeddings.device) for _ in range(n_clusters)]
        else:
            emb = product_embeddings[indices].cpu().numpy()
            w = weights
            n_clusters_ = min(n_clusters, len(indices))
            kmeans = KMeans(n_clusters=n_clusters_, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(emb)
            avg_embs = []
            for cid in range(n_clusters_):
                mask = cluster_labels == cid
                if not mask.any():
                    avg_embs.append(torch.zeros(product_embeddings.shape[1], device=product_embeddings.device))
                    continue
                emb_c = emb[mask]
                w_c = w[mask]
                denom = w_c.sum()
                if denom < 1e-8:
                    weighted_emb = emb_c.mean(axis=0)
                else:
                    weighted_emb = (emb_c * w_c[:, None]).sum(axis=0) / denom
                avg_embs.append(torch.tensor(weighted_emb, device=product_embeddings.device, dtype=torch.float32))
            # pad with zeros if cluster number is less than n_clusters
            while len(avg_embs) < n_clusters:
                avg_embs.append(torch.zeros(product_embeddings.shape[1], device=product_embeddings.device))

        user_profiles.append(torch.stack(avg_embs, dim=0))
        user_ids.append(user_id)
        purchased_items[user_id] = set(items)

    user_matrix = torch.stack(user_profiles, dim=0)
    return user_ids, user_matrix, purchased_items

def generate_batch_recommendations_multi_interest(
    user_ids, user_matrix, purchased_items,
    product_item_ids, product_embeddings,
    k=10, batch_size=512
):
    # recommend top-k items for each user based on similarity
    item_norm = torch.nn.functional.normalize(product_embeddings, dim=1)
    recommendations = {}
    n_users, n_clusters, d = user_matrix.shape

    for start in tqdm(range(0, n_users, batch_size), desc="Batch recommend"):
        end = min(start + batch_size, n_users)
        batch_user = user_matrix[start:end]
        batch_user = torch.nn.functional.normalize(batch_user, dim=2)
        batch_user = batch_user.reshape(-1, d)
        sim_matrix = torch.matmul(batch_user, item_norm.T)
        sim_matrix = sim_matrix.reshape(end - start, n_clusters, -1)
        max_sim_matrix, _ = sim_matrix.max(dim=1)  # max pooling over interests
        topk_scores, topk_indices = torch.topk(max_sim_matrix, k + 50, dim=1)

        for i, user_idx in enumerate(range(start, end)):
            user_id = user_ids[user_idx]
            recs = []
            for idx in topk_indices[i].cpu().tolist():
                iid = product_item_ids[idx]
                if iid not in purchased_items[user_id]:
                    recs.append(iid)
                if len(recs) >= k:
                    break
            recommendations[user_id] = recs
    return recommendations

def evaluate_recommendations(recommendations, test_df, k_values=[5, 10]):
    # calculate precision and recall at k
    user_test_items = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    results = {}
    for k in k_values:
        precisions, recalls = [], []
        for uid, rec in recommendations.items():
            if uid not in user_test_items:
                continue
            test_items = user_test_items[uid]
            hits = len(set(rec[:k]) & test_items)
            precisions.append(hits / k)
            recalls.append(hits / len(test_items) if len(test_items) > 0 else 0)
        results[f'Precision@{k}'] = np.mean(precisions)
        results[f'Recall@{k}'] = np.mean(recalls)
    return results

def main():
    review_path = "Beauty_5.json"
    embedding_cache = "product_embeddings.pkl"
    # load product embeddings
    product_item_ids, product_embeddings = load_embeddings(embedding_cache, device="cuda")
    # load and preprocess reviews
    review_df = load_review_data(review_path)
    processed_reviews = preprocess_reviews(review_df)
    train_df, test_df = split_train_test_by_time(processed_reviews)
    # aggregate user multi-interest profiles
    user_ids, user_matrix, purchased_items = aggregate_user_profiles_multi_interest(
        train_df, product_item_ids, product_embeddings, lambda_decay=0.01, history_size=20
    )
    # generate recommendations
    recommendations = generate_batch_recommendations_multi_interest(
        user_ids, user_matrix, purchased_items, product_item_ids,
        product_embeddings, k=10, batch_size=512
    )
    # evaluate
    results = evaluate_recommendations(recommendations, test_df)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
