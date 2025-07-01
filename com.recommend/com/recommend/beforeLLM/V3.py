import torch
import pandas as pd
import numpy as np
import pickle
import json
import re
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering


def load_embeddings(load_path, device="cuda"):
    # Load product/item embeddings from pickle file
    with open(load_path, 'rb') as f:
        df = pickle.load(f)
    item_ids = df['item_id'].values
    embedding_cols = [c for c in df.columns if c != 'item_id']
    embeddings = torch.tensor(df[embedding_cols].values, device=device, dtype=torch.float32)
    return item_ids, embeddings


def load_review_data(file_path):
    # Load review data in JSON lines format
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def clean_text(text):
    # Simple text cleaning (lowercase, remove HTML, keep alphanumerics)
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_reviews(df):
    # Convert numeric ratings to sentiment text
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

    # Add processed text and relevant columns
    df['text'] = df.apply(preprocess_row, axis=1)
    df['user_id'] = df['reviewerID']
    df['item_id'] = df['asin']
    df['rating'] = df['overall']
    df['timestamp'] = df['unixReviewTime']
    return df[['user_id', 'item_id', 'text', 'rating', 'timestamp']]


def split_train_test_by_time(df, test_size=0.2):
    # Time-based train-test split for each user
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
    # Build user profiles with multi-interest representation
    id_to_idx = {iid: idx for idx, iid in enumerate(product_item_ids)}
    user_profiles = []
    user_ids = []
    purchased_items = {}
    user_groups = train_df.sort_values(['user_id', 'timestamp']).groupby('user_id')

    for user_id, user_data in tqdm(user_groups, desc="User profiles"):
        user_data = user_data.tail(history_size)  # only most recent items
        items = user_data['item_id'].values
        times = user_data['timestamp'].values
        ratings = user_data['rating'].values
        T = times.max()
        normalized_ratings = ratings / 5.0
        time_weights = np.exp(-lambda_decay * (T - times))  # time decay
        weights = normalized_ratings * time_weights
        indices = [id_to_idx[iid] for iid in items if iid in id_to_idx]

        if len(indices) == 0:
            # cold start: use global average as user profile
            global_avg = product_embeddings.mean(dim=0)
            avg_embs = [global_avg.clone() for _ in range(n_clusters)]
        elif len(indices) < 5:
            # too few items, fallback to weighted average
            emb = product_embeddings[indices].cpu().numpy()
            w = weights
            avg_emb = (emb * w[:, None]).sum(axis=0) / (w.sum() + 1e-8)
            avg_embs = [torch.tensor(avg_emb, device=product_embeddings.device, dtype=torch.float32)]
            while len(avg_embs) < n_clusters:
                avg_embs.append(torch.zeros(product_embeddings.shape[1], device=product_embeddings.device))
        else:
            # enough history, apply clustering to discover multiple interests
            emb = product_embeddings[indices].cpu().numpy()
            w = weights
            sample_weight = w / (w.sum() + 1e-8)
            # oversample high-weighted items to help clustering
            repeat_count = np.clip((sample_weight * 100).astype(int), 1, 20)
            emb_weighted = np.repeat(emb, repeat_count, axis=0)
            n_clusters_ = min(n_clusters, max(1, len(indices) // 3))
            if len(emb_weighted) < 10:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters_)
                cluster_labels = clusterer.fit_predict(emb_weighted)
            else:
                kmeans = KMeans(n_clusters=n_clusters_, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(emb_weighted)
            avg_embs = []
            for cid in range(n_clusters_):
                mask = cluster_labels == cid
                if not mask.any():
                    avg_embs.append(torch.zeros(product_embeddings.shape[1], device=product_embeddings.device))
                    continue
                emb_c = emb_weighted[mask]
                weighted_emb = emb_c.mean(axis=0)
                avg_embs.append(torch.tensor(weighted_emb, device=product_embeddings.device, dtype=torch.float32))
            # pad with zeros if fewer than n_clusters clusters formed
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
        candidate_pools=None,
        k=10, batch_size=512
):
    # Generate recommendations in batches
    item_norm = torch.nn.functional.normalize(product_embeddings, dim=1)
    recommendations = {}
    n_users, n_clusters, d = user_matrix.shape
    item_id_to_idx = {iid: idx for idx, iid in enumerate(product_item_ids)}
    temperature = 0.5  # for softmax re-weighting
    for start in tqdm(range(0, n_users, batch_size), desc="Batch recommend"):
        end = min(start + batch_size, n_users)
        batch_user = user_matrix[start:end]
        batch_user = torch.nn.functional.normalize(batch_user, dim=2)
        for i, user_idx in enumerate(range(start, end)):
            user_id = user_ids[user_idx]
            pool = candidate_pools.get(user_id, product_item_ids)  # candidate pool for this user
            pool_indices = [item_id_to_idx[iid] for iid in pool if iid in item_id_to_idx]
            pool_item_norm = item_norm[pool_indices]
            user_interest = batch_user[i]
            sim_matrix = torch.matmul(user_interest, pool_item_norm.T)
            user_history_len = len(purchased_items[user_id])
            if user_history_len < 10:
                # short history: rely on max similarity
                final_scores = sim_matrix.max(dim=0).values
            else:
                # long history: re-weight by interest softmax
                interest_weights = torch.softmax(sim_matrix.max(dim=1).values / temperature, dim=0)
                final_scores = torch.sum(sim_matrix * interest_weights[:, None], dim=0)
            # pick top-k avoiding already purchased
            topk_scores, topk_indices = torch.topk(final_scores, k + 50)
            recs = []
            for idx in topk_indices.cpu().tolist():
                iid = product_item_ids[pool_indices[idx]]
                if iid not in purchased_items[user_id]:
                    recs.append(iid)
                if len(recs) >= k:
                    break
            recommendations[user_id] = recs
    return recommendations


def evaluate_recommendations(recommendations, test_df, k_values=[5, 10]):
    # Compute precision and recall at k
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


def build_candidate_pool(train_df, product_item_ids, top_k_popular=10000, per_user_category=2000):
    # Build a candidate pool per user combining top popular items + their own items
    item_counts = train_df['item_id'].value_counts()
    top_popular_items = set(item_counts.head(top_k_popular).index)
    user_candidate_pool = {}
    for user_id, group in train_df.groupby('user_id'):
        user_items = group['item_id'].unique()
        category_pool = set(user_items)
        pool = top_popular_items.union(category_pool)
        user_candidate_pool[user_id] = pool
    return user_candidate_pool


def main():
    review_path = "Beauty_5.json"
    embedding_cache = "product_embeddings.pkl"

    # load embeddings
    product_item_ids, product_embeddings = load_embeddings(embedding_cache, device="cuda")

    # load and preprocess review data
    review_df = load_review_data(review_path)
    processed_reviews = preprocess_reviews(review_df)

    # split by time
    train_df, test_df = split_train_test_by_time(processed_reviews)

    # aggregate user profiles
    user_ids, user_matrix, purchased_items = aggregate_user_profiles_multi_interest(
        train_df, product_item_ids, product_embeddings, lambda_decay=0.01, history_size=20
    )

    # build candidate pools
    candidate_pools = build_candidate_pool(train_df, product_item_ids)

    # generate recommendations
    recommendations = generate_batch_recommendations_multi_interest(
        user_ids, user_matrix, purchased_items, product_item_ids,
        product_embeddings, k=10, batch_size=512, candidate_pools=candidate_pools
    )

    # evaluate
    results = evaluate_recommendations(recommendations, test_df)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
