import torch
import pandas as pd
import numpy as np
import pickle
import json
import re
from tqdm import tqdm
from sklearn.cluster import KMeans

def load_embeddings(load_path, device="cuda"):
    with open(load_path, 'rb') as f:
        df = pickle.load(f)
    item_ids = df['item_id'].values
    emb_cols = [c for c in df.columns if c != 'item_id']
    embeddings = torch.tensor(df[emb_cols].values, device=device, dtype=torch.float32)
    return item_ids, embeddings

def load_review_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_reviews(df):
    def rating_to_sentiment(r):
        return {
            1: "very dissatisfied",
            2: "dissatisfied",
            3: "neutral",
            4: "satisfied",
            5: "very satisfied"
        }.get(int(r), "neutral")

    def row_text(row):
        sentiment = rating_to_sentiment(row["overall"])
        summary = row.get("summary", "")
        review = row.get("reviewText", "")
        return clean_text(f"User sentiment: {sentiment}. {summary} {review}")

    df['text'] = df.apply(row_text, axis=1)
    df['user_id'] = df['reviewerID']
    df['item_id'] = df['asin']
    df['rating'] = df['overall']
    df['timestamp'] = df['unixReviewTime']
    return df[['user_id', 'item_id', 'text', 'rating', 'timestamp']]

def split_train_test_by_time(df, test_size=0.2):
    df_sorted = df.sort_values(['user_id', 'timestamp'])
    train, test = [], []
    for user, group in df_sorted.groupby('user_id'):
        n = len(group)
        split = int(n * (1 - test_size))
        train.append(group.iloc[:split])
        test.append(group.iloc[split:])
    return pd.concat(train), pd.concat(test)

def build_user_profiles(
    train_df, product_item_ids, product_embeddings,
    lambda_decay=0.01, history_size=20, n_clusters=1
):
    id_to_idx = {iid: idx for idx, iid in enumerate(product_item_ids)}
    user_profiles = []
    user_ids = []
    purchased_items = {}

    user_groups = train_df.sort_values(['user_id', 'timestamp']).groupby('user_id')

    for user_id, group in tqdm(user_groups, desc="Building user profiles"):
        group = group[group['rating'] > 3]
        if group.empty:
            avg_embs = [torch.zeros(product_embeddings.shape[1], device=product_embeddings.device) for _ in range(n_clusters)]
            user_profiles.append(torch.stack(avg_embs))
            user_ids.append(user_id)
            purchased_items[user_id] = set()
            continue

        group = group.tail(history_size)
        items = group['item_id'].values
        times = group['timestamp'].values
        ratings = group['rating'].values

        weights = np.exp(-lambda_decay * (times.max() - times)) * (ratings / 5.0)
        indices = [id_to_idx[iid] for iid in items if iid in id_to_idx]

        if not indices:
            avg_embs = [torch.zeros(product_embeddings.shape[1], device=product_embeddings.device) for _ in range(n_clusters)]
        else:
            emb = product_embeddings[indices].cpu().numpy()
            w = weights
            if len(indices) < 5:
                denom = w.sum()
                avg_emb = (emb * w[:, None]).sum(axis=0) / (denom + 1e-8)
                avg_embs = [torch.tensor(avg_emb, device=product_embeddings.device)]
                while len(avg_embs) < n_clusters:
                    avg_embs.append(torch.zeros(product_embeddings.shape[1], device=product_embeddings.device))
            else:
                kmeans = KMeans(n_clusters=min(n_clusters, len(indices)), random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(emb)
                avg_embs = []
                for cid in range(kmeans.n_clusters):
                    mask = cluster_labels == cid
                    emb_c = emb[mask]
                    w_c = w[mask]
                    denom = w_c.sum()
                    if denom < 1e-8:
                        avg = emb_c.mean(axis=0)
                    else:
                        avg = (emb_c * w_c[:, None]).sum(axis=0) / denom
                    avg_embs.append(torch.tensor(avg, device=product_embeddings.device))
                while len(avg_embs) < n_clusters:
                    avg_embs.append(torch.zeros(product_embeddings.shape[1], device=product_embeddings.device))

        user_profiles.append(torch.stack(avg_embs))
        user_ids.append(user_id)
        purchased_items[user_id] = set(items)

    user_matrix = torch.stack(user_profiles).to(torch.float32)
    return user_ids, user_matrix, purchased_items

def recommend_multi_interest(
    user_ids,
    user_matrix,
    purchased_items,
    product_item_ids,
    product_embeddings,
    k=10,
    batch_size=64,
    candidate_pools=None
):
    item_norm = torch.nn.functional.normalize(product_embeddings, dim=1)
    recommendations = {}
    n_users, n_clusters, d = user_matrix.shape
    item_id_to_idx = {iid: idx for idx, iid in enumerate(product_item_ids)}

    for start in tqdm(range(0, n_users, batch_size), desc="Batch recommending"):
        end = min(start + batch_size, n_users)
        batch_user = user_matrix[start:end]
        batch_user = torch.nn.functional.normalize(batch_user, dim=2)

        for i, user_idx in enumerate(range(start, end)):
            user_id = user_ids[user_idx]
            if candidate_pools is not None:
                pool_items = candidate_pools.get(user_id, product_item_ids)
            else:
                pool_items = product_item_ids

            pool_indices = [item_id_to_idx[iid] for iid in pool_items if iid in item_id_to_idx]
            pool_item_norm = item_norm[pool_indices]
            user_interests = batch_user[i]
            sim_matrix = torch.matmul(user_interests, pool_item_norm.T)
            max_sim, _ = sim_matrix.max(dim=0)
            topk_num = min(k + 50, len(max_sim))
            topk_scores, topk_indices = torch.topk(max_sim, topk_num)

            recs = []
            for idx in topk_indices.cpu().tolist():
                iid = pool_items[idx]
                if iid not in purchased_items.get(user_id, set()):
                    recs.append(iid)
                if len(recs) >= k:
                    break

            recommendations[user_id] = recs

    return recommendations

def evaluate(recommendations, test_df, k_values=[5,10]):
    user_test = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    results = {}
    for k in k_values:
        precisions, recalls = [], []
        for uid, rec in recommendations.items():
            if uid not in user_test:
                continue
            test_items = user_test[uid]
            hits = len(set(rec[:k]) & test_items)
            precisions.append(hits / k)
            recalls.append(hits / len(test_items) if len(test_items) else 0)
        results[f"Precision@{k}"] = np.mean(precisions)
        results[f"Recall@{k}"] = np.mean(recalls)
    return results

def sanity_check(product_embeddings):
    idx = np.random.randint(0, product_embeddings.shape[0])
    vec = product_embeddings[idx]
    norm_vec = torch.nn.functional.normalize(vec, dim=0)
    sim = torch.dot(norm_vec, norm_vec).item()
    print(f"[Sanity Check] Cosine of identical vector: {sim:.6f}")

def build_test_candidate_pool(
    test_df,
    train_df,
    product_item_ids,
    user_list,
    num_negatives=5,
    max_positives=5
):
    user_test_items = test_df[test_df['rating'] > 3].groupby("user_id")["item_id"].apply(list).to_dict()
    user_train_items = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    all_items_set = set(product_item_ids)
    user_candidate_pool = {}

    for user_id in user_list:
        pos_items = user_test_items.get(user_id, [])
        if len(pos_items) == 0:
            continue
        pos_items = pos_items[:max_positives]
        seen_items = user_train_items.get(user_id, set())
        available_negatives = list(all_items_set - seen_items - set(pos_items))
        if len(available_negatives) < num_negatives:
            continue
        neg_items = np.random.choice(available_negatives, num_negatives, replace=False)
        pool = list(pos_items) + list(neg_items)
        user_candidate_pool[user_id] = pool

    print(f"[Candidate Pool] built for {len(user_candidate_pool)} users with 10 items each.")
    return user_candidate_pool

def precision_recall_by_threshold(
    user_ids,
    user_matrix,
    purchased_items,
    test_df,
    product_item_ids,
    product_embeddings,
    candidate_pools,
    threshold=0.5,
    eval_name="Evaluation"
):
    item_norm = torch.nn.functional.normalize(product_embeddings, dim=1)
    n_users, n_clusters, d = user_matrix.shape
    item_id_to_idx = {iid: idx for idx, iid in enumerate(product_item_ids)}
    user_test_items = test_df[test_df['rating'] > 3].groupby("user_id")["item_id"].apply(set).to_dict()
    precisions = []
    recalls = []

    for i, user_id in enumerate(user_ids):
        if user_id not in candidate_pools:
            continue
        pool_items = candidate_pools[user_id]
        pool_indices = [item_id_to_idx[iid] for iid in pool_items if iid in item_id_to_idx]
        pool_item_norm = item_norm[pool_indices]
        user_interests = user_matrix[i]
        user_interests = torch.nn.functional.normalize(user_interests, dim=1)
        sim_matrix = torch.matmul(user_interests, pool_item_norm.T)
        max_sim, _ = sim_matrix.max(dim=0)
        predict_positive = (max_sim >= threshold).cpu().numpy()
        true_positive = np.array([1 if iid in user_test_items.get(user_id, set()) else 0 for iid in pool_items])

        tp = ((predict_positive == 1) & (true_positive == 1)).sum()
        fp = ((predict_positive == 1) & (true_positive == 0)).sum()
        fn = ((predict_positive == 0) & (true_positive == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    print(f"[{eval_name}] Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f} over {len(precisions)} users")

def evaluate_hit_at_1(recommendations, test_df):
    user_test_items = test_df[test_df['rating'] > 3].groupby("user_id")["item_id"].apply(set).to_dict()
    hits = 0
    total = 0
    for uid, rec in recommendations.items():
        if uid not in user_test_items:
            continue
        true_items = user_test_items[uid]
        if len(true_items) == 0:
            continue
        total += 1
        if rec and rec[0] in true_items:
            hits += 1
    hit_rate = hits / total if total > 0 else 0
    print(f"[Hit@1 Evaluation] hit@1: {hit_rate:.4f} over {total} users")

def main():
    review_path = "Beauty_5.json"
    emb_path = "product_embeddings.pkl"

    product_item_ids, product_embeddings = load_embeddings(emb_path, device="cuda")
    review_df = load_review_data(review_path)
    processed = preprocess_reviews(review_df)
    train_df, test_df = split_train_test_by_time(processed)

    user_ids, user_matrix, purchased_items = build_user_profiles(
        train_df, product_item_ids, product_embeddings, lambda_decay=0.01, n_clusters=1
    )

    sanity_check(product_embeddings)

    user_pos_counts = (
        test_df[test_df['rating'] > 3]
        .groupby("user_id")["item_id"]
        .count()
    )

    sorted_users = user_pos_counts.sort_values(ascending=False)
    top_active_users = sorted_users.index[:200]

    # Option 1: build candidate pool with 1 positive + 9 negatives
    # This is a hard test setting, because only 1 positive among 10 items
    full_candidate_pool_opt1 = build_test_candidate_pool(
        test_df, train_df, product_item_ids,
        user_list=top_active_users,
        num_negatives=9,
        max_positives=1
    )

    qualified_user_ids = list(full_candidate_pool_opt1.keys())[:50]
    qualified_idx = [user_ids.index(uid) for uid in qualified_user_ids]
    user_ids = qualified_user_ids
    user_matrix = user_matrix[qualified_idx]

    candidate_pool_opt1 = {uid: full_candidate_pool_opt1[uid] for uid in user_ids}

    recs_opt1 = recommend_multi_interest(
        user_ids,
        user_matrix,
        purchased_items,
        product_item_ids,
        product_embeddings,
        k=10,
        batch_size=64,
        candidate_pools=candidate_pool_opt1
    )

    evaluate_hit_at_1(recs_opt1, test_df)

    # Option 2: build candidate pool with 5 positives + 5 negatives
    # This is an easier setting, with more positives available in the pool
    full_candidate_pool_opt2 = build_test_candidate_pool(
        test_df, train_df, product_item_ids,
        user_list=user_ids,
        num_negatives=5,
        max_positives=5
    )

    candidate_pool_opt2 = {uid: full_candidate_pool_opt2[uid] for uid in user_ids if uid in full_candidate_pool_opt2}

    recs_opt2 = recommend_multi_interest(
        user_ids,
        user_matrix,
        purchased_items,
        product_item_ids,
        product_embeddings,
        k=10,
        batch_size=64,
        candidate_pools=candidate_pool_opt2
    )

    precision_recall_by_threshold(
        user_ids, user_matrix, purchased_items,
        test_df, product_item_ids, product_embeddings,
        candidate_pools=candidate_pool_opt2,
        threshold=0.5,
        eval_name="Option2 Threshold Evaluation"
    )

if __name__ == "__main__":
    main()
