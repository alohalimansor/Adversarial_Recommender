The Recommender's Trilemma: A Comparative Analysis of Accuracy, Robustness, and Explainability
This repository contains the full source code and experimental pipeline for the paper, "The Recommender's Trilemma: A Comparative Study of Accuracy, Robustness, and Explainability in E-Commerce AI."

The scripts are organized into a logical, step-by-step pipeline to allow for the complete reproduction of all results presented in the manuscript.

Abstract
The successful deployment of recommender systems is critical to modern e-commerce, yet the trustworthiness of these AI-driven platforms is often evaluated on the single dimension of predictive accuracy. This paper introduces and empirically investigates the Recommender’s Trilemma: the inherent trade-off between (1) predictive accuracy, (2) adversarial robustness, and (3) explanation reliability. We conduct a rigorous comparative analysis of three distinct architectures—a content-based TF-IDF model, a collaborative filtering SVD model, and a state-of-the-art NeuMF deep learning model—to map their performance across this trilemma. Our findings reveal a critical "dual failure" cascade in both text-based and collaborative filtering models: successful attacks on a model's predictions simultaneously induce a catastrophic collapse in the integrity of its explanations. Our work concludes that a model which cannot be reliably understood is not truly trustworthy, regardless of its performance, and that a holistic evaluation across the trilemma is essential for the future of trustworthy AI.

Reproducibility Pipeline
To reproduce all experiments and results from the paper, please follow the steps below.

Step 0: Setup
Download Data: Download the reviews_Electronics_5.json.gz file from the Amazon Review Dataset page and place it in the root of this repository.

Create Environment: Create a Python virtual environment.

Install Dependencies: Install all required libraries using the provided requirements.txt file:

pip install -r requirements.txt

Download NLTK Data: Run the following command in a Python interpreter to pre-download necessary language models:

import nltk
nltk.download('all')

Step 1: Data Preprocessing
Run the first script to clean the raw data.

Script: 01_preprocess_data.py (Use your preprocess_data.py file)

Input: reviews_Electronics_5.json.gz

Output: electronics_reviews_cleaned.parquet

python 01_preprocess_data.py

Step 2: Model Benchmarking
Run the following scripts to train and benchmark the three recommender models. This will generate the performance data for Table 1 in the manuscript.

Script: 02a_benchmark_tfidf.py (Use your train_tfidf_with_full_metrics.py)

Script: 02b_benchmark_svd.py (Use your train_svd_with_full_metrics.py)

Script: 02c_benchmark_neumf.py (Use your train_neumf_with_full_metrics.py)

python 02a_benchmark_tfidf.py
python 02b_benchmark_svd.py
python 02c_benchmark_neumf.py

Step 3: Adversarial Analysis of the TF-IDF Model
This section requires two preliminary steps before running the main attack.

3a. Save TF-IDF Components: Save the trained TF-IDF vectorizer and user profiles for the attack script to use.

Script: 03a_save_tfidf_components.py (Use your save_model_components.py)

python 03a_save_tfidf_components.py

3b. Run Main Attack Script: This performs the comparative attacks (sophisticated vs. random, minimal vs. full profile) and the full explanation stability analysis on the TF-IDF model. This generates the data for Table 2 in the manuscript.

Script: 03b_attack_tfidf_model.py (Use your final, corrected run_final_manuscript_experiment.py file)

python 03b_attack_tfidf_model.py

Step 4: Adversarial Analysis of the NeuMF Model
This multi-step process executes the data poisoning attack on the deep learning model.

4a. Prepare the Attack: Create the poisoned dataset.

Script: 04a_attack_neumf_prep.py (Use your attack_neumf_poisoning.py file)

Output: electronics_reviews_poisoned.parquet

python 04a_attack_neumf_prep.py

4b. Train Models: Train and save both the clean and poisoned NeuMF models.

Script: 04b_train_clean_neumf.py (Use your train_and_save_clean_neumf.py file)

Script: 04c_train_poisoned_neumf.py (Use your train_poisoned_neumf.py file)

Output: neumf_clean.keras, neumf_poisoned.keras, and mapping files.

python 04b_train_clean_neumf.py
python 04c_train_poisoned_neumf.py

4c. Evaluate the Attack: Compare the models to confirm the attack's success rate.

Script: 04d_evaluate_neumf_attack.py (Use your attack_neumf_poisoning_eval.py file)

python 04d_evaluate_neumf_attack.py

Step 5: Explanation Analysis of the NeuMF Model
Generate the final visualization showing how the poisoning attack corrupted the NeuMF model's reasoning. This generates Figure 1 in the manuscript.

Script: 05_analyze_neumf_explanation.py (Use your analyze_neumf_explanation.py file)

Output: shap_explanation_poisoned.png

python 05_analyze_neumf_explanation.py

requirements.txt
# Core Data Science Libraries
pandas
numpy<2.0 # For compatibility with scikit-surprise
scikit-learn

# Recommender System Library
scikit-surprise

# Deep Learning Library
tensorflow

# Adversarial Attack Library
textattack

# Explainability Libraries
lime
shap

# Utilities
tqdm
matplotlib
