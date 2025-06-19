# train_svd_model.py
import pandas as pd
from surprise import Dataset, Reader, SVD
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random

CLEANED_DATA_PATH = 'electronics_reviews_cleaned.parquet'
TOP_K = 10
NEGATIVE_SAMPLE_SIZE = 99


# --- Metric helpers ---
def get_rank(item, ranked_list):
    try:
        return ranked_list.index(item) + 1
    except ValueError:
        return float('inf')


def main():
    print("--- SVD Model Evaluation ---")
    df = pd.read_parquet(CLEANED_DATA_PATH)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['reviewerID', 'asin', 'overall']], reader)

    df_sorted = df.sort_values(by='unixReviewTime')
    train_df = df_sorted.groupby('reviewerID').head(-1)
    test_df = df_sorted.groupby('reviewerID').tail(1)

    trainset = Dataset.load_from_df(train_df[['reviewerID', 'asin', 'overall']], reader).build_full_trainset()
    test_data_dict = pd.Series(test_df.asin.values, index=test_df.reviewerID).to_dict()
    all_item_ids = list(df['asin'].unique())

    print("\nTraining SVD model...")
    algo = SVD(n_factors=100, n_epochs=30, lr_all=0.005, reg_all=0.02, verbose=True)
    algo.fit(trainset)

    print("\nEvaluating model...")
    hr, ndcg, mrr = [], [], []
    for uid, true_item_id in tqdm(test_data_dict.items(), desc="Evaluating SVD"):
        items_rated_by_user = {trainset.to_raw_iid(iid) for iid, _ in trainset.ur[trainset.to_inner_uid(uid)]}
        candidate_items = {true_item_id}
        while len(candidate_items) < NEGATIVE_SAMPLE_SIZE + 1:
            random_item = random.choice(all_item_ids)
            if random_item not in items_rated_by_user:
                candidate_items.add(random_item)

        predictions = [algo.predict(uid=uid, iid=iid) for iid in candidate_items]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_k_recs = [pred.iid for pred in predictions[:TOP_K]]

        if true_item_id in top_k_recs:
            hr.append(1)
        else:
            hr.append(0)
        rank = get_rank(true_item_id, top_k_recs)
        if rank <= TOP_K:
            ndcg.append(1 / np.log2(rank + 1))
            mrr.append(1 / rank)
        else:
            ndcg.append(0)
            mrr.append(0)

    hr10, ndcg10, mrr_val = np.mean(hr), np.mean(ndcg), np.mean(mrr)

    print("\n--- SVD FINAL PERFORMANCE ---")
    print(f"Hit Rate @ 10:  {hr10:.4f}")
    print(f"NDCG @ 10:      {ndcg10:.4f}")
    print(f"MRR @ 10:       {mrr_val:.4f}")
    print("------------------------------")


if __name__ == '__main__':
    main()