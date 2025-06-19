# train_neumf_model.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
DATA_PATH = 'electronics_reviews_cleaned.parquet' # *** Using CLEAN data ***
MODEL_SAVE_PATH = 'neumf_clean.keras' # *** Saving as CLEAN model ***
# We don't need to save the maps again as they are the same
USER_MAP_SAVE_PATH = 'user_map.pkl'
ITEM_MAP_SAVE_PATH = 'item_map.pkl'

# (All helper functions: prepare_data, create_train_test, build_neumf_model are identical to the poisoned script)
def prepare_data(df):
    df['user_id'] = df['reviewerID'].astype('category').cat.codes
    df['item_id'] = df['asin'].astype('category').cat.codes
    n_users, n_items = df['user_id'].nunique(), df['item_id'].nunique()
    user_map = dict(zip(df['user_id'], df['reviewerID']))
    item_map = dict(zip(df['item_id'], df['asin']))
    return df, n_users, n_items, user_map, item_map
def create_train_test(df):
    df_sorted = df.sort_values(by='unixReviewTime')
    test_df = df_sorted.groupby('user_id').tail(1)
    train_df = df.drop(test_df.index)
    item_ids = set(df['item_id'].unique())
    user_items = train_df.groupby('user_id')['item_id'].apply(set)
    negatives = []
    for user_id, items in tqdm(user_items.items(), desc="Negative Sampling"):
        num_neg_samples = len(items) * 4
        available_negs = list(item_ids - items)
        if len(available_negs) < num_neg_samples: num_neg_samples = len(available_negs)
        user_negatives = random.sample(available_negs, num_neg_samples)
        for item_id in user_negatives: negatives.append([user_id, item_id, 0])
    positives = train_df[['user_id', 'item_id']].copy(); positives['label'] = 1
    neg_df = pd.DataFrame(negatives, columns=['user_id', 'item_id', 'label'])
    final_train_df = pd.concat([positives, neg_df]).sample(frac=1).reset_index(drop=True)
    return final_train_df, test_df
def build_neumf_model(n_users, n_items, mf_dim=16, layers=[64, 32, 16, 8]):
    user_input_mf = Input(shape=(1,), dtype='int32', name='user_input_mf')
    item_input_mf = Input(shape=(1,), dtype='int32', name='item_input_mf')
    mf_user_embedding = Embedding(input_dim=n_users, output_dim=mf_dim)(user_input_mf)
    mf_item_embedding = Embedding(input_dim=n_items, output_dim=mf_dim)(item_input_mf)
    mf_user_vec, mf_item_vec = Flatten()(mf_user_embedding), Flatten()(mf_item_embedding)
    mf_vec = mf_user_vec * mf_item_vec
    user_input_mlp = Input(shape=(1,), dtype='int32', name='user_input_mlp')
    item_input_mlp = Input(shape=(1,), dtype='int32', name='item_input_mlp')
    mlp_user_embedding = Embedding(input_dim=n_users, output_dim=layers[0]//2)(user_input_mlp)
    mlp_item_embedding = Embedding(input_dim=n_items, output_dim=layers[0]//2)(item_input_mlp)
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
    df = pd.read_parquet(DATA_PATH)
    df, n_users, n_items, user_map, item_map = prepare_data(df)
    train_data, _ = create_train_test(df) # We don't need test_data for training
    model = build_neumf_model(n_users, n_items)
    print("\nTraining CLEAN NeuMF model...")
    user_train, item_train, labels_train = train_data['user_id'], train_data['item_id'], train_data['label']
    model.fit([user_train, item_train, user_train, item_train], labels_train, epochs=5, batch_size=256, verbose=1)
    print(f"\nSaving clean model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Clean model saved successfully.")

if __name__ == '__main__':
    main()