# attack_neumf_poisoning_eval.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from tensorflow.keras.models import load_model

# --- Configuration ---
CLEAN_MODEL_PATH = 'neumf_clean.keras'
POISONED_MODEL_PATH = 'neumf_poisoned.keras'
CLEAN_DATA_PATH = 'electronics_reviews_cleaned.parquet'
USER_MAP_PATH = 'user_map.pkl'
ITEM_MAP_PATH = 'item_map.pkl'
TOP_K = 10
NUM_EVAL_USERS = 1000  # Evaluate on a sample of 1000 users


def get_top_k_recs_for_user(model, user_id, item_ids_to_score, n_items, k):
    """Generates Top-K recommendations for a single user."""
    user_input = np.full(n_items, user_id)

    predictions = model.predict([user_input, item_ids_to_score, user_input, item_ids_to_score], verbose=0)

    item_scores = sorted(list(zip(item_ids_to_score, predictions.flatten())), key=lambda x: x[1], reverse=True)

    return [item_id for item_id, score in item_scores[:k]]


def main():
    # --- Load Models and Mappings ---
    print("Loading models and data mappings...")
    clean_model = load_model(CLEAN_MODEL_PATH)
    poisoned_model = load_model(POISONED_MODEL_PATH)
    with open(USER_MAP_PATH, 'rb') as f:
        user_map = pickle.load(f)
    with open(ITEM_MAP_PATH, 'rb') as f:
        item_map = pickle.load(f)

    # Invert maps for easy lookup
    rev_user_map = {v: k for k, v in user_map.items()}
    rev_item_map = {v: k for k, v in item_map.items()}

    df = pd.read_parquet(CLEAN_DATA_PATH)

    # --- Identify Target Item and Evaluation Users ---
    item_counts = df['asin'].value_counts()
    target_item_asin = item_counts.iloc[100:101].index[0]
    target_item_id = rev_item_map[target_item_asin]
    print(f"Identified Target Item: {target_item_asin} (ID: {target_item_id})")

    # Find users who have NOT rated the target item
    rated_target_users = set(df[df['asin'] == target_item_asin]['reviewerID'])
    all_users = set(df['reviewerID'])
    eval_users_asin = list(all_users - rated_target_users)

    # Take a sample and convert to integer IDs
    eval_users_sample_asin = np.random.choice(eval_users_asin, NUM_EVAL_USERS, replace=False)
    eval_users_sample_id = [rev_user_map[uid] for uid in eval_users_sample_asin]

    print(f"Evaluating attack on {NUM_EVAL_USERS} neutral users...")

    # --- Run Evaluation ---
    clean_hits = 0
    poisoned_hits = 0
    all_item_indices = np.array(list(item_map.keys()))
    n_items_total = len(all_item_indices)

    for user_id in tqdm(eval_users_sample_id, desc="Getting Recommendations"):
        # Get recs from CLEAN model
        clean_recs = get_top_k_recs_for_user(clean_model, user_id, all_item_indices, n_items_total, TOP_K)
        if target_item_id in clean_recs:
            clean_hits += 1

        # Get recs from POISONED model
        poisoned_recs = get_top_k_recs_for_user(poisoned_model, user_id, all_item_indices, n_items_total, TOP_K)
        if target_item_id in poisoned_recs:
            poisoned_hits += 1

    # --- Report Results ---
    print("\n--- DATA POISONING ATTACK RESULTS ---")
    print(f"Target Item: {target_item_asin}")
    print(f"Evaluated on {NUM_EVAL_USERS} users.")
    print("-" * 35)
    print(f"Times Recommended by CLEAN Model:   {clean_hits}")
    print(f"Times Recommended by POISONED Model: {poisoned_hits}")
    print("-" * 35)

    if poisoned_hits > clean_hits:
        print("✅ ATTACK SUCCEEDED: The poisoned model recommends the target item more frequently.")
    else:
        print("❌ ATTACK FAILED: The poisoned model did not promote the target item effectively.")


if __name__ == '__main__':
    main()