import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import PWWSRen2019
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
import lime
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import logging

# Reduce TextAttack logging verbosity
logging.getLogger("textattack").setLevel(logging.ERROR)

# --- Configuration ---
VECTORIZER_PATH = 'vectorizer.pkl'
USER_PROFILES_PATH = 'user_profiles.pkl'
CLEANED_DATA_PATH = 'electronics_reviews_cleaned.parquet'
NUM_CANDIDATES_TO_TEST = 15
SIMILARITY_THRESHOLD = 0.15  # Lowered threshold for more sensitive attacks


# AFTER
class RecommenderModelWrapper(ModelWrapper):
    """Improved wrapper with better TextAttack compatibility"""

    def __init__(self, vectorizer, user_profile, threshold=0.15):
        self.model = self  # <-- ADD THIS LINE BACK IN
        self.vectorizer = vectorizer
        self.user_profile = user_profile
        self.threshold = threshold

    def __call__(self, text_list):
        """
        Returns probabilities for each class [not_recommended, recommended]
        """
        vectors = self.vectorizer.transform(text_list)
        similarities = cosine_similarity(vectors, self.user_profile).flatten()

        # Convert similarities to probabilities
        # If similarity > threshold, classify as recommended (class 1)
        # Otherwise, classify as not recommended (class 0)
        results = []
        for sim in similarities:
            if sim > self.threshold:
                # High similarity -> recommended (class 1)
                prob_not_rec = 0.1  # Low probability for class 0
                prob_rec = 0.9  # High probability for class 1
            else:
                # Low similarity -> not recommended (class 0)
                prob_not_rec = 0.9  # High probability for class 0
                prob_rec = 0.1  # Low probability for class 1

            results.append([prob_not_rec, prob_rec])

        return np.array(results)


def get_lime_explanation_words(text_to_explain, vectorizer, user_profile, threshold=0.15):
    """Get LIME explanation words for the recommendation"""

    def prediction_fn(texts):
        text_vectors = vectorizer.transform(texts)
        similarities = cosine_similarity(text_vectors, user_profile).flatten()

        # Convert to probabilities like in the wrapper
        probs = []
        for sim in similarities:
            if sim > threshold:
                probs.append([0.1, 0.9])  # [not_rec, rec]
            else:
                probs.append([0.9, 0.1])

        return np.array(probs)

    explainer = LimeTextExplainer(class_names=['not_recommended', 'recommended'])
    explanation = explainer.explain_instance(text_to_explain, prediction_fn, num_features=10)
    return {word for word, weight in explanation.as_list() if weight > 0}


def analyze_attack_potential(text, vectorizer, user_profile, threshold=0.15):
    """Analyze if a text is a good candidate for attack"""
    vector = vectorizer.transform([text])
    similarity = cosine_similarity(vector, user_profile)[0][0]

    # Check if it's currently recommended
    is_recommended = similarity > threshold

    # Get words and their importance
    words = text.split()
    word_importances = []

    for i, word in enumerate(words):
        # Create modified text without this word
        modified_words = words[:i] + words[i + 1:]
        modified_text = ' '.join(modified_words)

        if modified_text.strip():  # Make sure it's not empty
            modified_vector = vectorizer.transform([modified_text])
            modified_similarity = cosine_similarity(modified_vector, user_profile)[0][0]
            importance = abs(similarity - modified_similarity)
            word_importances.append((word, importance))

    # Sort by importance
    word_importances.sort(key=lambda x: x[1], reverse=True)

    return {
        'original_similarity': similarity,
        'is_recommended': is_recommended,
        'top_important_words': word_importances[:5],
        'attackable': is_recommended and len(word_importances) > 0
    }


def main():
    print("Loading saved model components...")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(USER_PROFILES_PATH, 'rb') as f:
        user_profiles = pickle.load(f)
    df = pd.read_parquet(CLEANED_DATA_PATH)
    all_items_df = df.drop_duplicates(subset=['asin']).set_index('asin')

    # Find candidates
    test_df = df.groupby('reviewerID').tail(1)
    test_data_dict = pd.Series(test_df.asin.values, index=test_df.reviewerID).to_dict()

    candidates = []
    for user_id, true_item_id in test_data_dict.items():
        if user_id in user_profiles and true_item_id in all_items_df.index:
            user_profile = user_profiles[user_id]
            item_text = all_items_df.loc[true_item_id, 'cleaned_review_text']
            item_vector = vectorizer.transform([item_text])
            score = cosine_similarity(item_vector, user_profile)[0][0]

            # Only include candidates that are currently recommended
            if score > SIMILARITY_THRESHOLD:
                attack_analysis = analyze_attack_potential(item_text, vectorizer, user_profile, SIMILARITY_THRESHOLD)
                if attack_analysis['attackable']:
                    candidates.append({
                        "user_id": user_id,
                        "item_id": true_item_id,
                        "score": score,
                        "text_length": len(item_text.split()),
                        "top_words": [w[0] for w in attack_analysis['top_important_words']]
                    })

    if not candidates:
        print("No attackable candidates found. Try lowering SIMILARITY_THRESHOLD.")
        return

    candidate_df = pd.DataFrame(candidates).sort_values(by='score', ascending=False)
    print(f"Found {len(candidates)} attackable candidates")

    # Run experiments
    print(f"\n--- Starting Experiment on Top {min(NUM_CANDIDATES_TO_TEST, len(candidates))} Candidates ---")
    results_list = []

    for idx, (_, row) in enumerate(candidate_df.head(NUM_CANDIDATES_TO_TEST).iterrows()):
        user_id = row['user_id']
        item_id = row['item_id']
        original_score = row['score']

        target_user_profile = user_profiles[user_id]
        original_text = all_items_df.loc[item_id, 'cleaned_review_text']

        print(f"\nCandidate {idx + 1}: Item {item_id}")
        print(f"Original score: {original_score:.4f}")
        print(f"Text preview: {original_text[:100]}...")

        # Create model wrapper
        model_wrapper = RecommenderModelWrapper(vectorizer, target_user_profile, SIMILARITY_THRESHOLD)

        # Test the wrapper
        test_output = model_wrapper([original_text])
        print(f"Model wrapper output: {test_output[0]}")

        try:
            # Build and run attack
            attack = PWWSRen2019.build(model_wrapper)
            attack_result = attack.attack(original_text, 1)  # Target class 1 (recommended)

            print(f"Attack result type: {type(attack_result)}")

            if isinstance(attack_result, SuccessfulAttackResult):
                perturbed_text = attack_result.perturbed_text()
                perturbed_vector = vectorizer.transform([perturbed_text])
                new_score = cosine_similarity(perturbed_vector, target_user_profile)[0][0]

                print(f"SUCCESS! New score: {new_score:.4f}")
                print(f"Perturbed text preview: {perturbed_text[:100]}...")

                # Analyze explanations
                try:
                    original_words = get_lime_explanation_words(original_text, vectorizer, target_user_profile,
                                                                SIMILARITY_THRESHOLD)
                    perturbed_words = get_lime_explanation_words(perturbed_text, vectorizer, target_user_profile,
                                                                 SIMILARITY_THRESHOLD)

                    intersection = len(original_words.intersection(perturbed_words))
                    union = len(original_words.union(perturbed_words))
                    jaccard = intersection / union if union != 0 else 0

                    results_list.append({
                        "item_id": item_id,
                        "original_score": original_score,
                        "new_score": new_score,
                        "score_drop": original_score - new_score,
                        "jaccard_similarity": jaccard,
                        "original_words": len(original_words),
                        "perturbed_words": len(perturbed_words),
                        "common_words": intersection
                    })
                except Exception as e:
                    print(f"Error in LIME analysis: {e}")

            elif isinstance(attack_result, FailedAttackResult):
                print(f"Attack failed: {attack_result}")
            else:
                print(f"Unknown attack result: {attack_result}")

        except Exception as e:
            print(f"Error during attack: {e}")

    # Display results
    print("\n\n--- FINAL EXPERIMENT RESULTS ---")
    if not results_list:
        print("No successful attacks were found among the candidates.")
        print("\nDebugging suggestions:")
        print("1. Check if your texts are long enough for meaningful attacks")
        print("2. Verify that similarity scores are well above the threshold")
        print("3. Consider using different attack methods")
        print("4. Try lowering SIMILARITY_THRESHOLD for more sensitive detection")
    else:
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(by='jaccard_similarity', ascending=True)
        print("Successful attacks, sorted by explanation damage (lower Jaccard = more damage):")
        print(results_df.to_string(index=False))

        print(f"\n--- SUMMARY ---")
        print(f"Successful attacks: {len(results_list)}")
        print(f"Average score drop: {results_df['score_drop'].mean():.4f}")
        print(f"Average Jaccard similarity: {results_df['jaccard_similarity'].mean():.4f}")

        best_attack = results_df.iloc[0]
        print(f"\nBest attack (lowest Jaccard):")
        print(f"Item ID: {best_attack['item_id']}")
        print(f"Score drop: {best_attack['original_score']:.4f} -> {best_attack['new_score']:.4f}")
        print(f"Jaccard similarity: {best_attack['jaccard_similarity']:.4f}")


if __name__ == '__main__':
    main()