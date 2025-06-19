import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm
from scipy.stats import binom, ttest_rel
import types

# --- TextAttack Imports ---
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import PWWSRen2019
from textattack.attack_results import SuccessfulAttackResult
from textattack import Attack
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwapWordNet
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.grammaticality import PartOfSpeech

# --- LIME Imports ---
import lime
from lime.lime_text import LimeTextExplainer

# --- Setup ---
logging.getLogger("textattack").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- Configuration ---
VECTORIZER_PATH = 'vectorizer.pkl'
USER_PROFILES_PATH = 'user_profiles.pkl'
CLEANED_DATA_PATH = 'electronics_reviews_cleaned.parquet'
NUM_ATTACKS_TO_RUN = 50
SIMILARITY_THRESHOLD = 0.12
LIME_WEIGHT_THRESHOLD = 0.01


# --- Model Wrapper ---
class RecommenderModelWrapper(ModelWrapper):
    def __init__(self, vectorizer, user_profile):
        self.model = self
        self.vectorizer = vectorizer
        self.user_profile = user_profile

    def __call__(self, text_list):
        vectors = self.vectorizer.transform(text_list)
        similarities = cosine_similarity(vectors, self.user_profile).flatten()
        clamped_sims = np.clip(similarities, 0, 1)
        return np.array([[1.0 - s, s] for s in clamped_sims])


# --- Custom Method to be Injected ---
def custom_is_goal_complete(self, model_output, attacked_text):
    return model_output[self.ground_truth_output] < self.threshold


# --- FIXED RandomAttack Function ---
def RandomAttack(model_wrapper):
    """
    Builds a simple random attack with proper constraint initialization.
    """
    # Create goal function
    goal_function = UntargetedClassification(model_wrapper)

    # Create transformation
    transformation = WordSwapWordNet()

    # Create constraints with proper initialization
    constraints = [
        RepeatModification(),
        StopwordModification(),
    ]

    # Create search method
    search_method = GreedySearch()

    # Build the attack
    attack = Attack(goal_function, constraints, transformation, search_method)

    # Apply custom settings
    attack.goal_function.threshold = SIMILARITY_THRESHOLD
    attack.goal_function._is_goal_complete = types.MethodType(custom_is_goal_complete, attack.goal_function)

    return attack


# --- Alternative Simple RandomAttack Function ---
def SimpleRandomAttack(model_wrapper):
    """
    Alternative implementation using minimal constraints to avoid compatibility issues.
    """
    try:
        # Start with a working attack recipe
        attack = PWWSRen2019.build(model_wrapper)

        # Replace with simpler transformation
        attack.transformation = WordSwapWordNet()

        # Use only basic constraints that are guaranteed to work
        attack.constraints = []  # Start with no constraints for maximum compatibility

        # Apply custom goal function settings
        attack.goal_function.threshold = SIMILARITY_THRESHOLD
        attack.goal_function._is_goal_complete = types.MethodType(custom_is_goal_complete, attack.goal_function)

        return attack
    except Exception as e:
        print(f"Error creating SimpleRandomAttack: {e}")
        return None


# --- Improved LIME Helper ---
def get_lime_explanation_words_improved(text_to_explain, wrapper):
    explainer = LimeTextExplainer(class_names=['not_recommended', 'recommended'])
    try:
        explanation = explainer.explain_instance(text_to_explain, wrapper, num_features=10, labels=(1,))
        return {word for word, weight in explanation.as_list(label=1) if abs(weight) > LIME_WEIGHT_THRESHOLD}
    except Exception:
        return set()


# --- Jaccard Similarity Helper ---
def calculate_jaccard(set1, set2):
    if not set1 and not set2: return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


def main():
    # --- Phase 1 & 2: Setup and Candidate Finding ---
    print("Loading saved model components...")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(USER_PROFILES_PATH, 'rb') as f:
        full_user_profiles = pickle.load(f)
    df = pd.read_parquet(CLEANED_DATA_PATH)
    all_items_df = df.drop_duplicates(subset=['asin']).set_index('asin')
    print(f"\nScanning users to find best attack candidates (score > {SIMILARITY_THRESHOLD})...")
    test_df = df.groupby('reviewerID').tail(1)
    test_data_dict = pd.Series(test_df.asin.values, index=test_df.reviewerID).to_dict()
    attack_candidates = []
    for user_id, test_item_id in tqdm(list(test_data_dict.items())[:2000], desc="Scanning Candidates"):
        user_reviews = df[(df['reviewerID'] == user_id) & (df['asin'] != test_item_id)]
        if user_reviews.empty or test_item_id not in all_items_df.index: continue
        attack_item_text = all_items_df.loc[test_item_id, 'cleaned_review_text']
        attack_item_vector = vectorizer.transform([attack_item_text])
        best_base_review_text, highest_similarity = None, -1
        for review_text in user_reviews['cleaned_review_text']:
            sim = cosine_similarity(vectorizer.transform([review_text]), attack_item_vector)[0][0]
            if sim > highest_similarity: highest_similarity, best_base_review_text = sim, review_text
        if best_base_review_text:
            minimal_user_profile = vectorizer.transform([best_base_review_text])
            score = cosine_similarity(attack_item_vector, minimal_user_profile)[0][0]
            if score > SIMILARITY_THRESHOLD:
                attack_candidates.append({"user_id": user_id, "item_id": test_item_id, "score": score,
                                          "base_review": best_base_review_text, "attack_text": attack_item_text})
    if not attack_candidates:
        print("Could not find any suitable candidates.")
        return
    candidate_df = pd.DataFrame(attack_candidates).sort_values(by='score', ascending=False)
    print(f"\nFound {len(candidate_df)} promising candidates. Attacking the top {NUM_ATTACKS_TO_RUN}...")

    # --- Phase 3: The Ultimate Attack Loop ---
    results_list = []
    for _, row in tqdm(candidate_df.head(NUM_ATTACKS_TO_RUN).iterrows(),
                       total=min(NUM_ATTACKS_TO_RUN, len(candidate_df)), desc="Running All Attacks"):
        user_id, item_id, original_text = row['user_id'], row['item_id'], row['attack_text']
        result = {"item_id": item_id, "original_score": row['score']}

        # Test 1: Sophisticated Attack on Minimal Profile
        min_profile = vectorizer.transform([row['base_review']])
        min_wrapper = RecommenderModelWrapper(vectorizer, min_profile)
        pwws_attack = PWWSRen2019.build(min_wrapper)
        pwws_attack.goal_function.threshold = SIMILARITY_THRESHOLD
        pwws_attack.goal_function._is_goal_complete = types.MethodType(custom_is_goal_complete,
                                                                       pwws_attack.goal_function)
        try:
            attack_result = pwws_attack.attack(original_text, 1)
            if isinstance(attack_result, SuccessfulAttackResult):
                result["pwws_min_profile_status"] = "Success"
                p_text = attack_result.perturbed_text()
                original_words = get_lime_explanation_words_improved(original_text, min_wrapper)
                perturbed_words = get_lime_explanation_words_improved(p_text, min_wrapper)
                result["jaccard_similarity"] = calculate_jaccard(original_words, perturbed_words)
            else:
                result["pwws_min_profile_status"] = "Failed"
        except Exception as e:
            print(f"PWWS minimal profile attack failed: {e}")
            result["pwws_min_profile_status"] = "Error"

        # Test 2: Sophisticated Attack on Full Profile
        try:
            full_profile = full_user_profiles[user_id]
            full_wrapper = RecommenderModelWrapper(vectorizer, full_profile)
            pwws_attack_full = PWWSRen2019.build(full_wrapper)
            pwws_attack_full.goal_function.threshold = SIMILARITY_THRESHOLD
            pwws_attack_full.goal_function._is_goal_complete = types.MethodType(custom_is_goal_complete,
                                                                                pwws_attack_full.goal_function)
            attack_result_full = pwws_attack_full.attack(original_text, 1)
            result["pwws_full_profile_status"] = "Success" if isinstance(attack_result_full,
                                                                         SuccessfulAttackResult) else "Failed"
        except Exception as e:
            print(f"PWWS full profile attack failed: {e}")
            result["pwws_full_profile_status"] = "Error"

        # Test 3: Simple Random Attack on Minimal Profile
        try:
            # Try the main RandomAttack first
            random_attack = RandomAttack(min_wrapper)
            if random_attack is None:
                # Fallback to SimpleRandomAttack
                random_attack = SimpleRandomAttack(min_wrapper)

            if random_attack is not None:
                attack_result_random = random_attack.attack(original_text, 1)
                result["random_min_profile_status"] = "Success" if isinstance(attack_result_random,
                                                                              SuccessfulAttackResult) else "Failed"
            else:
                result["random_min_profile_status"] = "Error"
        except Exception as e:
            print(f"Random attack failed: {e}")
            # Try the simple fallback
            try:
                simple_attack = SimpleRandomAttack(min_wrapper)
                if simple_attack is not None:
                    attack_result_simple = simple_attack.attack(original_text, 1)
                    result["random_min_profile_status"] = "Success" if isinstance(attack_result_simple,
                                                                                  SuccessfulAttackResult) else "Failed"
                else:
                    result["random_min_profile_status"] = "Error"
            except Exception as e2:
                print(f"Simple random attack also failed: {e2}")
                result["random_min_profile_status"] = "Error"

        results_list.append(result)

    # --- Phase 4: Reporting ---
    print("\n\n--- FINAL MANUSCRIPT RESULTS ---")
    if results_list:
        results_df = pd.DataFrame(results_list).fillna('-')
        print("--- Detailed Results Table ---")
        print(results_df.to_string(index=False))

        print("\n--- Aggregate Success Rates & Confidence Intervals ---")
        n = len(results_df)
        success_vectors = {}
        for attack_type, name in [('pwws_min_profile', 'Sophisticated (Minimal Profile)'),
                                  ('pwws_full_profile', 'Sophisticated (Full Profile)'),
                                  ('random_min_profile', 'Random Baseline (Minimal Profile)')]:
            success_vectors[attack_type] = (results_df[f'{attack_type}_status'] == 'Success').astype(int).tolist()
            rate = np.mean(success_vectors[attack_type])
            ci_low, ci_high = binom.interval(0.95, n, rate)
            print(f"{name} Success Rate: {rate:.2%} [95% CI: {ci_low / n:.2%} - {ci_high / n:.2%}]")

        print("\n--- Significance Testing (Paired T-Tests) ---")

        stat, p_value_full_vs_min = ttest_rel(success_vectors['pwws_full_profile'], success_vectors['pwws_min_profile'])
        print(f"1. Sophisticated (Full) vs. Sophisticated (Minimal): p-value = {p_value_full_vs_min:.4f}")
        if p_value_full_vs_min < 0.05:
            print("   - RESULT: The difference IS statistically significant.")
        else:
            print("   - RESULT: The difference is NOT statistically significant.")

        stat, p_value_pwws_vs_random = ttest_rel(success_vectors['pwws_min_profile'],
                                                 success_vectors['random_min_profile'])
        print(f"2. Sophisticated (Minimal) vs. Random (Minimal): p-value = {p_value_pwws_vs_random:.4f}")
        if p_value_pwws_vs_random < 0.05:
            print("   - RESULT: The difference IS statistically significant.")
        else:
            print("   - RESULT: The difference is NOT statistically significant.")

        jaccard_scores = pd.to_numeric(results_df[results_df['jaccard_similarity'] != '-']['jaccard_similarity'])
        if not jaccard_scores.empty:
            print(f"\nAverage Jaccard Similarity on Successes: {jaccard_scores.mean():.4f}")


if __name__ == '__main__':
    main()