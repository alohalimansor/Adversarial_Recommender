import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import shap
import warnings

# Suppress a common SHAP warning
warnings.filterwarnings("ignore", message="Passing a numpy array with 'dtype=object' is deprecated")

# --- Configuration ---
CLEAN_MODEL_PATH = 'neumf_clean.keras'
POISONED_MODEL_PATH = 'neumf_poisoned.keras'
CLEAN_DATA_PATH = 'electronics_reviews_cleaned.parquet'
USER_MAP_PATH = 'user_map.pkl'
ITEM_MAP_PATH = 'item_map.pkl'


def find_analysis_candidate(clean_model, poisoned_model, user_map, item_map, df):
    """Finds a user whose recommendation for the target item changed after poisoning."""
    print("Searching for a user affected by the poisoning attack...")

    rev_user_map = {v: k for k, v in user_map.items()}
    rev_item_map = {v: k for k, v in item_map.items()}

    item_counts = df['asin'].value_counts()
    target_item_asin = item_counts.iloc[100:101].index[0]
    target_item_id = rev_item_map[target_item_asin]

    rated_target_users = set(df[df['asin'] == target_item_asin]['reviewerID'])
    all_users = set(df['reviewerID'])
    eval_users_asin = list(all_users - rated_target_users)
    eval_users_sample_asin = np.random.choice(eval_users_asin, 500, replace=False)
    eval_users_sample_id = [rev_user_map[uid] for uid in eval_users_sample_asin]

    for user_id in eval_users_sample_id:
        user_input = np.array([user_id])
        item_input = np.array([target_item_id])

        clean_pred = clean_model.predict([user_input, item_input, user_input, item_input], verbose=0)[0][0]
        poisoned_pred = poisoned_model.predict([user_input, item_input, user_input, item_input], verbose=0)[0][0]

        # Find a user where the prediction flipped from low to high
        if clean_pred < 0.5 and poisoned_pred > 0.5:
            print(f"Found candidate! User: {user_map[user_id]}, Item: {target_item_asin}")
            print(f"Prediction changed from {clean_pred:.4f} (Clean) to {poisoned_pred:.4f} (Poisoned)")
            return user_id, target_item_id, user_map[user_id], target_item_asin

    return None, None, None, None


def main():
    # --- Load Models and Data ---
    print("Loading models and data...")
    clean_model = load_model(CLEAN_MODEL_PATH)
    poisoned_model = load_model(POISONED_MODEL_PATH)
    with open(USER_MAP_PATH, 'rb') as f: user_map = pickle.load(f)
    with open(ITEM_MAP_PATH, 'rb') as f: item_map = pickle.load(f)
    df = pd.read_parquet(CLEAN_DATA_PATH)

    # --- Find a Candidate for Explanation ---
    user_id, item_id, user_asin, item_asin = find_analysis_candidate(clean_model, poisoned_model, user_map, item_map,
                                                                     df)

    if user_id is None:
        print(
            "Could not find a suitable candidate where recommendation flipped. Please re-run or increase sample size.")
        return

    # --- Create a SHAP Explainer ---
    # We'll create a background dataset for the explainer from a sample of the training data
    background_data = df.sample(n=100)
    user_bg = background_data['reviewerID'].map({v: k for k, v in user_map.items()}).values
    item_bg = background_data['asin'].map({v: k for k, v in item_map.items()}).values

    # Ensure background data has the correct shape
    background_array = np.array([user_bg, item_bg, user_bg, item_bg]).T
    print("Background data shape:", background_array.shape)

    # Verify predict_for_shap output
    def predict_for_shap(X):
        return poisoned_model.predict([X[:, 0], X[:, 1], X[:, 2], X[:, 3]], verbose=0)

    # Test predict_for_shap
    test_input = np.array([[user_id, item_id, user_id, item_id]])
    test_output = predict_for_shap(test_input)
    print("Predict output shape:", test_output.shape)

    explainer = shap.KernelExplainer(predict_for_shap, background_array)

    # --- Generate and Plot Explanations ---
    print("\nGenerating SHAP explanation for the POISONED model's prediction...")
    instance_to_explain = np.array([[user_id, item_id, user_id, item_id]])
    shap_values = explainer.shap_values(instance_to_explain)
    print("SHAP values shape:", np.array(shap_values).shape)

    # Create a force plot
    plot = shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][:, 0],  # Extract (4,) array for the single output
        instance_to_explain[0],
        feature_names=['user_id_mf', 'item_id_mf', 'user_id_mlp', 'item_id_mlp'],
        matplotlib=True,
        show=False
    )

    # Save the plot
    plot_path = 'shap_explanation_poisoned.png'
    plot.get_figure().savefig(plot_path, bbox_inches='tight')
    print(f"\n--- SHAP ANALYSIS COMPLETE ---")
    print(f"✅ Explanation plot saved to {plot_path}")

if __name__ == '__main__':
    main()