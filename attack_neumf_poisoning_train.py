import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
# *** We now load the POISONED data ***
DATA_PATH = 'electronics_reviews_poisoned.parquet'
TOP_K = 10
NEGATIVE_SAMPLE_SIZE = 99

# --- Paths for saving the trained model and mappings ---
POISONED_MODEL_SAVE_PATH = 'neumf_poisoned.keras'
USER_MAP_SAVE_PATH = 'user_map.pkl'
ITEM_MAP_SAVE_PATH = 'item_map.pkl'


def prepare_data(df):
    """Prepares the data for the NeuMF model, including creating user/item indices."""
    print("Preparing poisoned data for NeuMF model...")

    df['user_id'] = df['reviewerID'].astype('category').cat.codes
    df['item_id'] = df['asin'].astype('category').cat.codes

    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()

    user_map = dict(zip(df['user_id'], df['reviewerID']))
    item_map = dict(zip(df['item_id'], df['asin']))

    return df, n_users, n_items, user_map, item_map


def create_train_test(df):
    """Performs a time-based split and generates negative samples for training."""
    print("Splitting data and generating negative samples...")

    # IMPORTANT: We need to exclude our fake users from the test set
    real_users_df = df[~df['reviewerID'].str.startswith('fake_user_')]
    df_sorted = real_users_df.sort_values(by='unixReviewTime')

    test_df = df_sorted.groupby('user_id').tail(1)

    # The training set is the entire poisoned dataset, minus the test rows
    train_df = df.drop(test_df.index)

    # --- Negative Sampling for Training ---
    item_ids = set(df['item_id'].unique())
    user_items = train_df.groupby('user_id')['item_id'].apply(set)
    negatives = []
    for user_id, items in tqdm(user_items.items(), desc="Negative Sampling"):
        num_neg_samples = len(items) * 4
        # Ensure we don't sample more than available
        available_negs = list(item_ids - items)
        if len(available_negs) < num_neg_samples:
            num_neg_samples = len(available_negs)

        user_negatives = random.sample(available_negs, num_neg_samples)
        for item_id in user_negatives:
            negatives.append([user_id, item_id, 0])

    positives = train_df[['user_id', 'item_id']].copy()
    positives['label'] = 1
    neg_df = pd.DataFrame(negatives, columns=['user_id', 'item_id', 'label'])
    final_train_df = pd.concat([positives, neg_df]).sample(frac=1).reset_index(drop=True)

    return final_train_df, test_df


def build_neumf_model(n_users, n_items, mf_dim=16, layers=[64, 32, 16, 8]):
    """Builds the NeuMF model architecture (same as before)."""
    print("Building NeuMF model architecture...")
    user_input_mf = Input(shape=(1,), dtype='int32', name='user_input_mf')
    item_input_mf = Input(shape=(1,), dtype='int32', name='item_input_mf')
    mf_user_embedding = Embedding(input_dim=n_users, output_dim=mf_dim)(user_input_mf)
    mf_item_embedding = Embedding(input_dim=n_items, output_dim=mf_dim)(item_input_mf)
    mf_user_vec, mf_item_vec = Flatten()(mf_user_embedding), Flatten()(mf_item_embedding)
    mf_vec = mf_user_vec * mf_item_vec

    user_input_mlp = Input(shape=(1,), dtype='int32', name='user_input_mlp')
    item_input_mlp = Input(shape=(1,), dtype='int32', name='item_input_mlp')
    mlp_user_embedding = Embedding(input_dim=n_users, output_dim=layers[0] // 2)(user_input_mlp)
    mlp_item_embedding = Embedding(input_dim=n_items, output_dim=layers[0] // 2)(item_input_mlp)
    mlp_user_vec, mlp_item_vec = Flatten()(mlp_user_embedding), Flatten()(mlp_item_embedding)
    concat_vec = Concatenate()([mlp_user_vec, mlp_item_vec])
    for units in layers:
        concat_vec = Dense(units, activation='relu')(concat_vec)
        concat_vec = Dropout(0.2)(concat_vec)

    final_vec = Concatenate()([mf_vec, concat_vec])
    prediction = Dense(1, activation='sigmoid')(final_vec)
    model = Model(inputs=[user_input_mf, item_input_mf, user_input_mlp, item_input_mlp], outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model


def main():
    """Main function to train and save the POISONED NeuMF model."""
    df = pd.read_parquet(DATA_PATH)
    df, n_users, n_items, user_map, item_map = prepare_data(df)
    train_data, test_data = create_train_test(df)

    model = build_neumf_model(n_users, n_items)

    print("\nTraining POISONED NeuMF model...")
    user_train_mf = train_data['user_id'].values
    item_train_mf = train_data['item_id'].values
    labels_train = train_data['label'].values

    model.fit(
        [user_train_mf, item_train_mf, user_train_mf, item_train_mf],
        labels_train,
        epochs=5,
        batch_size=256,
        verbose=1
    )

    # --- Save the trained model and mappings for later analysis ---
    print(f"\nSaving poisoned model to {POISONED_MODEL_SAVE_PATH}...")
    model.save(POISONED_MODEL_SAVE_PATH)

    print(f"Saving user map to {USER_MAP_SAVE_PATH}...")
    with open(USER_MAP_SAVE_PATH, 'wb') as f:
        pickle.dump(user_map, f)

    print(f"Saving item map to {ITEM_MAP_SAVE_PATH}...")
    with open(ITEM_MAP_SAVE_PATH, 'wb') as f:
        pickle.dump(item_map, f)

    print("\nPoisoned model and mappings saved successfully. Ready for analysis.")


if __name__ == '__main__':
    main()