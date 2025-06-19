import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import PWWSRen2019
from textattack.attack_results import SuccessfulAttackResult
import logging

# Reduce verbosity for this test
logging.getLogger("textattack").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- Configuration & Wrapper (copied from our main script) ---
VECTORIZER_PATH = 'vectorizer.pkl'
CLEANED_DATA_PATH = 'electronics_reviews_cleaned.parquet'
SIMILARITY_THRESHOLD = 0.15  # We'll use this as a starting point


class RecommenderModelWrapper(ModelWrapper):
    def __init__(self, vectorizer, user_profile):
        self.model = self
        self.vectorizer = vectorizer
        self.user_profile = user_profile

    def __call__(self, text_list):
        vectors = self.vectorizer.transform(text_list)
        similarities = cosine_similarity(vectors, self.user_profile).flatten()
        probs = np.array([[1.0 - (s > SIMILARITY_THRESHOLD), float(s > SIMILARITY_THRESHOLD)] for s in similarities])
        return probs


def verify_attack_direction(vectorizer):
    """Your essential test to validate that attacks reduce similarity."""
    print("\n--- Running Attack Direction Verification ---")

    # Create a simple, ideal test case
    test_text = "this is a great excellent fantastic good quality product"
    test_profile = vectorizer.transform([test_text])  # Profile is the item itself
    wrapper = RecommenderModelWrapper(vectorizer, test_profile)

    # Check original prediction and similarity
    original_probs = wrapper([test_text])
    original_sim = cosine_similarity(vectorizer.transform([test_text]), test_profile)[0][0]
    print(f"Original Similarity: {original_sim:.4f}")
    print(f"Original Wrapper Prediction (Prob [0, 1]): {original_probs[0]}")

    # Run attack
    attack = PWWSRen2019.build(wrapper)
    result = attack.attack(test_text, 1)  # Attack the 'recommended' class

    if isinstance(result, SuccessfulAttackResult):
        print("\nAttack was successful according to TextAttack.")
        new_text = result.perturbed_text()
        new_probs = wrapper([new_text])
        new_sim = cosine_similarity(vectorizer.transform([new_text]), test_profile)[0][0]

        print(f"Perturbed Text: '{new_text}'")
        print(f"After Attack Similarity: {new_sim:.4f}")
        print(f"After Attack Wrapper Prediction: {new_probs[0]}")

        if new_sim < original_sim:
            print("\n✅ VALIDATION PASSED: Attack correctly reduced the cosine similarity.")
        else:
            print(
                "\n❌ VALIDATION FAILED: Attack did NOT reduce the cosine similarity. The wrapper logic must be inverted.")
    else:
        print(
            "\n⚠️ VALIDATION INCONCLUSIVE: Attack failed on this simple case. Wrapper might be too robust or misconfigured.")


def analyze_score_distribution(vectorizer, df, all_items_df):
    """Analyzes score distribution to inform threshold choice."""
    print("\n--- Analyzing Score Distribution for Threshold Validation ---")
    test_df = df.groupby('reviewerID').tail(1)
    test_data_dict = pd.Series(test_df.asin.values, index=test_df.reviewerID).to_dict()

    scores = []
    for user_id, test_item_id in list(test_data_dict.items())[:2000]:  # Scan 2000 users
        user_reviews = df[(df['reviewerID'] == user_id) & (df['asin'] != test_item_id)]
        if user_reviews.empty or test_item_id not in all_items_df.index: continue

        attack_item_text = all_items_df.loc[test_item_id, 'cleaned_review_text']
        attack_item_vector = vectorizer.transform([attack_item_text])
        best_base_review_text, highest_similarity = None, -1

        for review_text in user_reviews['cleaned_review_text']:
            sim = cosine_similarity(vectorizer.transform([review_text]), attack_item_vector)[0][0]
            if sim > highest_similarity:
                highest_similarity, best_base_review_text = sim, review_text

        if best_base_review_text:
            minimal_user_profile = vectorizer.transform([best_base_review_text])
            score = cosine_similarity(attack_item_vector, minimal_user_profile)[0][0]
            scores.append(score)

    scores_series = pd.Series(scores)
    print("\nDistribution of 'Minimal Profile' Scores:")
    print(scores_series.describe(percentiles=[.25, .5, .75, .9, .95]))
    print(
        "\nRECOMMENDATION: Choose a SIMILARITY_THRESHOLD near the 75th percentile to ensure a good number of attackable candidates.")


if __name__ == '__main__':
    print("Loading components for validation...")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer_main = pickle.load(f)
    df_main = pd.read_parquet(CLEANED_DATA_PATH)
    all_items_df_main = df_main.drop_duplicates(subset=['asin']).set_index('asin')

    # Run Quality Checks
    verify_attack_direction(vectorizer_main)
    analyze_score_distribution(vectorizer_main, df_main, all_items_df_main)