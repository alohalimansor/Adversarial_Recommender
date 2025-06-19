import pandas as pd
import string
from nltk.corpus import stopwords
from tqdm import tqdm

# --- Configuration ---
# Set the path to your downloaded dataset
INPUT_FILE_PATH = 'Electronics_5.json.gz'
# Set the path for the cleaned output file
OUTPUT_FILE_PATH = 'electronics_reviews_cleaned.parquet'

def clean_text(text):
    """
    A helper function to clean a single string of text.
    - Converts to lowercase
    - Removes punctuation
    - Removes stopwords
    """
    # Ensure text is a string
    if not isinstance(text, str):
        return ""

    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)

def main():
    """
    Main function to load, process, and save the dataset.
    """
    print(f"Loading dataset from: {INPUT_FILE_PATH}")
    # Load the gzipped JSON file line by line
    df = pd.read_json(INPUT_FILE_PATH, lines=True)
    print("Dataset loaded successfully.")
    print(f"Initial number of reviews: {len(df)}")

    # --- Filtering ---
    print("Starting filtering process...")

    # 1. Filter reviews by length (50-200 words)
    df['word_count'] = df['reviewText'].str.split().str.len().fillna(0)
    df = df[(df['word_count'] >= 50) & (df['word_count'] <= 200)]
    print(f"Reviews after length filtering: {len(df)}")

    # 2. Filter for products with at least 25 reviews
    product_counts = df['asin'].value_counts()
    products_to_keep = product_counts[product_counts >= 25].index
    df = df[df['asin'].isin(products_to_keep)]
    print(f"Reviews after product filtering (>=25 reviews/product): {len(df)}")

    # 3. Filter for users with at least 10 reviews
    user_counts = df['reviewerID'].value_counts()
    users_to_keep = user_counts[user_counts >= 10].index
    df = df[df['reviewerID'].isin(users_to_keep)]
    print(f"Reviews after user filtering (>=10 reviews/user): {len(df)}")

    if df.empty:
        print("No data left after filtering. Please check your filtering criteria.")
        return

    # --- Cleaning ---
    print("\nStarting text cleaning process (this may take a few minutes)...")
    # Use tqdm to show a progress bar
    tqdm.pandas(desc="Cleaning review text")
    df['cleaned_review_text'] = df['reviewText'].progress_apply(clean_text)

    # --- Saving Output ---
    print(f"\nSaving cleaned data to: {OUTPUT_FILE_PATH}")
    # Select relevant columns to save
    final_df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime', 'cleaned_review_text']]
    # Save to Parquet format for efficiency. It's faster and uses less disk space than CSV.
    final_df.to_parquet(OUTPUT_FILE_PATH, index=False)

    print("\nPreprocessing complete!")
    print(f"Final number of cleaned reviews: {len(final_df)}")
    print(f"Cleaned data saved successfully.")

if __name__ == '__main__':
    main()