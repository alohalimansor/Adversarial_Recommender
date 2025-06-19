import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
from scipy.sparse import vstack
import pickle

# --- Configuration ---
CLEANED_DATA_PATH = 'electronics_reviews_cleaned.parquet'
VECTORIZER_PATH = 'vectorizer.pkl'
USER_PROFILES_PATH = 'user_profiles.pkl'


def build_user_profiles(train_df, vectorizer):
    """Creates a profile for each user from the training data."""
    print("Building user profiles...")
    user_profiles = {}

    train_tfidf = vectorizer.transform(train_df['cleaned_review_text'])
    train_df = train_df.reset_index()

    user_vectors_map = {user: [] for user in train_df['reviewerID'].unique()}
    for i, row in train_df.iterrows():
        user_vectors_map[row['reviewerID']].append(train_tfidf[i])

    for user_id, vectors in tqdm(user_vectors_map.items(), desc="Processing users"):
        if vectors:
            user_profiles[user_id] = np.asarray(vstack(vectors).mean(axis=0))

    return user_profiles


def main():
    """Main function to build and save recommender components."""
    print("Loading cleaned data...")
    df = pd.read_parquet(CLEANED_DATA_PATH)

    # --- Data Splitting ---
    print("Splitting data into training and test sets...")
    df = df.sort_values(by='unixReviewTime')
    train_df = df.groupby('reviewerID').head(-1)  # All but the last review for training

    # --- TF-IDF Vectorizer ---
    print("\nFitting TF-IDF vectorizer...")
    all_items_df = df.drop_duplicates(subset=['asin']).reset_index(drop=True)
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(all_items_df['cleaned_review_text'])

    # --- Model Training ---
    user_profiles = build_user_profiles(train_df, vectorizer)

    # --- Saving Components ---
    print(f"\nSaving vectorizer to {VECTORIZER_PATH}...")
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"Saving user profiles to {USER_PROFILES_PATH}...")
    with open(USER_PROFILES_PATH, 'wb') as f:
        pickle.dump(user_profiles, f)

    print("\nModel components saved successfully.")


if __name__ == '__main__':
    main()