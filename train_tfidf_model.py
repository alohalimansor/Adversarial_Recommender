# train_tfidf_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from scipy.sparse import vstack

CLEANED_DATA_PATH = 'electronics_reviews_cleaned.parquet'
TOP_K = 10


# --- Helper functions for metrics ---
def get_rank(item, ranked_list):
    try:
        return ranked_list.index(item) + 1
    except ValueError:
        return float('inf')


def calculate_metrics(recommendations, ground_truth_dict):
    hr, ndcg, mrr = [], [], []
    for uid, recs in recommendations.items():
        true_item = ground_truth_dict.get(uid)
        if true_item:
            # HR@K
            if true_item in recs:
                hr.append(1)
            else:
                hr.append(0)

            # NDCG@K and MRR@K
            rank = get_rank(true_item, recs)
            if rank <= TOP_K:
                ndcg.append(1 / np.log2(rank + 1))
                mrr.append(1 / rank)
            else:
                ndcg.append(0)
                mrr.append(0)

    return np.mean(hr), np.mean(ndcg), np.mean(mrr)


# --- Main script functions (build_weighted_user_profiles, etc.) ---
def build_weighted_user_profiles(train_df, vectorizer):
    print("Building WEIGHTED user profiles...")
    user_profiles, df_vectors = {}, train_df.copy()
    train_tfidf = vectorizer.transform(df_vectors['cleaned_review_text'])
    df_vectors['vector'] = [v for v in train_tfidf]
    user_review_groups = df_vectors.groupby('reviewerID')
    for user_id, group in tqdm(user_review_groups, desc="Processing users"):
        weighted_vectors, total_weight = [], 0
        for _, row in group.iterrows():
            weight = float(row['overall'])
            weighted_vectors.append(row['vector'] * weight)
            total_weight += weight
        if weighted_vectors:
            user_profiles[user_id] = np.asarray(vstack(weighted_vectors).sum(axis=0) / total_weight)
    return user_profiles


def main():
    print("--- TF-IDF Model Evaluation ---")
    df = pd.read_parquet(CLEANED_DATA_PATH)
    df_sorted = df.sort_values(by='unixReviewTime')
    test_df = df_sorted.groupby('reviewerID').tail(1)
    train_df = df.drop(test_df.index)
    test_data_dict = pd.Series(test_df.asin.values, index=test_df.reviewerID).to_dict()

    all_items_df = df.drop_duplicates(subset=['asin']).reset_index(drop=True)
    vectorizer = TfidfVectorizer(max_features=10000)
    vectorizer.fit(all_items_df['cleaned_review_text'])
    all_items_matrix = vectorizer.transform(all_items_df['cleaned_review_text'])

    user_profiles = build_weighted_user_profiles(train_df, vectorizer)

    # Generate recommendations for all test users
    all_recommendations = {}
    print("\nGenerating recommendations for all test users...")
    for uid, profile in tqdm(user_profiles.items(), desc="Recommending"):
        if uid in test_data_dict:
            similarities = cosine_similarity(profile, all_items_matrix)
            top_k_indices = np.argsort(-similarities[0])[:TOP_K]
            recommended_ids = [all_items_df['asin'].iloc[i] for i in top_k_indices]
            all_recommendations[uid] = recommended_ids

    hr10, ndcg10, mrr = calculate_metrics(all_recommendations, test_data_dict)

    print("\n--- TF-IDF FINAL PERFORMANCE ---")
    print(f"Hit Rate @ 10:  {hr10:.4f}")
    print(f"NDCG @ 10:      {ndcg10:.4f}")
    print(f"MRR @ 10:       {mrr:.4f}")
    print("---------------------------------")


if __name__ == '__main__':
    main()