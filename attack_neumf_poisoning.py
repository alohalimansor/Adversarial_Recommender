import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Configuration ---
CLEANED_DATA_PATH = 'electronics_reviews_cleaned.parquet'
POISONED_OUTPUT_PATH = 'electronics_reviews_poisoned.parquet'

NUM_FAKE_USERS = 20
NUM_RATINGS_PER_FAKE_USER = 50


def create_poisoned_dataset():
    """
    Injects fake user profiles into the dataset to promote a target item.
    """
    print("Loading clean dataset...")
    df = pd.read_parquet(CLEANED_DATA_PATH)

    # --- Step 1: Select a Target Item to Promote ---
    # We'll pick a moderately popular item that isn't already at the very top.
    item_counts = df['asin'].value_counts()
    target_item = item_counts.iloc[100:101].index[0]  # Pick the 101st most popular item
    print(f"Selected target item to promote: {target_item}")

    # --- Step 2: Select "Filler" Items for Fake Ratings ---
    # These are popular items that our fake users will down-rate.
    popular_items = item_counts.head(NUM_RATINGS_PER_FAKE_USER * 2).index.tolist()
    # Ensure the target item is not in the popular list
    popular_items = [item for item in popular_items if item != target_item]

    # --- Step 3: Generate and Inject Fake User Profiles ---
    print(f"Generating {NUM_FAKE_USERS} fake user profiles...")

    fake_reviews = []
    for i in tqdm(range(NUM_FAKE_USERS), desc="Creating Fake Users"):
        fake_user_id = f'fake_user_{i}'

        # 1. Add a high rating for the target item
        fake_reviews.append({
            'reviewerID': fake_user_id,
            'asin': target_item,
            'overall': 5.0,
            'unixReviewTime': df['unixReviewTime'].max() + 1,  # Add after all real reviews
            'cleaned_review_text': 'promo review'  # Text is irrelevant for NeuMF
        })

        # 2. Add low ratings for other popular items
        items_to_downrate = np.random.choice(popular_items, NUM_RATINGS_PER_FAKE_USER - 1, replace=False)
        for item in items_to_downrate:
            fake_reviews.append({
                'reviewerID': fake_user_id,
                'asin': item,
                'overall': 1.0,
                'unixReviewTime': df['unixReviewTime'].max() + 1,
                'cleaned_review_text': 'filler review'
            })

    fake_df = pd.DataFrame(fake_reviews)

    # --- Step 4: Combine and Save the Poisoned Dataset ---
    poisoned_df = pd.concat([df, fake_df], ignore_index=True)

    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Number of injected fake ratings: {len(fake_df)}")
    print(f"Poisoned dataset size: {len(poisoned_df)}")

    print(f"\nSaving poisoned dataset to {POISONED_OUTPUT_PATH}...")
    poisoned_df.to_parquet(POISONED_OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == '__main__':
    create_poisoned_dataset()