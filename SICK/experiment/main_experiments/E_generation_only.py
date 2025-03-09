import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Add parent directory to sys.path if needed
parent_dir = os.path.join(os.getcwd(), '..', '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utility.data import get_loader, EmbeddingDataset
from utility.visuals import *
from mkit.torch_support.tensor_utils import xy_to_tensordataset
from model.gan import GANs
from model.knn import KNN
from model.rf import RF
from model.dnn import DNN
from model.data_augmentation_tech import (
    conditional_smote_sampling, 
    conditional_kde_sampling, 
    conditional_gmm_sampling
)
import torch
from torch.utils.data import Subset
from datasets import load_from_disk

# Device and parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128          
condition_dim = 10        
gan_epochs = 200  
generation_sizes = [1000, 10000]  # Multiple generation sizes for synthetic generation
# generation_sizes = [100]
batch_size = 32                
num_epochs = 10
learning_rate = 0.001

# Load dataset and split into train and test
full_dataset = load_from_disk('../../data/full_dataset_new', keep_in_memory=True)
split_datasets = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets['train']
test_dataset = split_datasets['test']

# For external test evaluation (if desired)
test_x = np.array(test_dataset['embedding'])
test_y = np.array(test_dataset['labels'])

# Full training arrays (will be used for selecting sample sizes)
train_x_full = np.array(train_dataset['embedding'])
train_y_full = np.array(train_dataset['labels'])
train_ds = EmbeddingDataset(train_dataset)
test_ds = EmbeddingDataset(test_dataset)

# Get input dimension from one sample
sample_emb, _ = train_ds[0]
input_dim = sample_emb.shape[0]
hidden_dim = 128
num_classes = 3  

# Define a list of training sample sizes
sample_sizes = [100, 200, 500, 1000, len(train_ds)]

# Number of folds for cross validation
n_splits = 5

# List to collect results; each entry will be a dict for one (sample_size, generation_size) combo.
summary_data = []

# Iterate over each generation size (for synthetic data generation)
for generation_size in generation_sizes:
    print(f"\n=== Generation Size: {generation_size} ===")
    # For each training sample size...
    for size in sample_sizes:
        print(f"\n[Train Data] Sample size: {size}")
        # Get the first "size" samples from the full training set
        X_train = train_x_full[:size]
        y_train = train_y_full[:size]
        
        # Set up StratifiedKFold for cross validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Lists to store scores for each fold for all models/augmentations
        dnn_scores = {"real": [], "smote": [], "kde": [], "gmm": [], "gan": []}
        knn_scores = {"real": [], "smote": [], "kde": [], "gmm": [], "gan": []}
        rf_scores  = {"real": [], "smote": [], "kde": [], "gmm": [], "gan": []}
        
        fold_no = 1
        for train_index, val_index in skf.split(X_train, y_train):
            print(f"  Fold {fold_no}/{n_splits}")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            # Create a loader for the "real" training fold
            real_loader = xy_to_tensordataset(X_train_fold, y_train_fold, return_loader=True)
            
            # ---------- Synthetic Data Generation ----------
            # SMOTE
            synthetic_x_smote, synthetic_y_smote = conditional_smote_sampling(
                X_train_fold, y_train_fold, generation_size
            )
            smote_loader = xy_to_tensordataset(synthetic_x_smote, synthetic_y_smote, return_loader=True)
            
            # KDE (using conditions for classes [0, 1, 2])
            synthetic_x_kde, synthetic_y_kde = conditional_kde_sampling(
                X_train_fold, y_train_fold, generation_size, 
                n_components=min(X_train_fold.shape[1] - 1, generation_size - 1), 
                condition=[0, 1, 2]
            )
            kde_loader = xy_to_tensordataset(synthetic_x_kde, synthetic_y_kde, return_loader=True)
            
            # GMM
            synthetic_x_gmm, synthetic_y_gmm = conditional_gmm_sampling(
                X_train_fold, y_train_fold, generation_size, condition=[0, 1, 2]
            )
            gmm_loader = xy_to_tensordataset(synthetic_x_gmm, synthetic_y_gmm, return_loader=True)
            
            # GAN
            # Note: Here we assume GANs.generate takes (loader, generation_size) and uses the training fold.
            synthetic_x_gen, synthetic_y_gen = GANs(
                batch_size,
                X_train_fold,
                y_train_fold,
                train_y_full,  # using full training labels as in your original code
                latent_dim,
                condition_dim,
                device,
                gan_epochs
            ).generate(real_loader, generation_size)
            gan_loader = xy_to_tensordataset(synthetic_x_gen, synthetic_y_gen, return_loader=True)
            
            # ---------- Model Evaluation on Validation Fold ----------
            # For DNN, use the loader and evaluate on the fold's validation data.
            real_acc_dnn   = DNN(input_dim, hidden_dim, num_classes, num_epochs, device, learning_rate).fit_and_eval(real_loader, X_val_fold, y_val_fold)
            smote_acc_dnn  = DNN(input_dim, hidden_dim, num_classes, num_epochs, device, learning_rate).fit_and_eval(smote_loader, X_val_fold, y_val_fold)
            kde_acc_dnn    = DNN(input_dim, hidden_dim, num_classes, num_epochs, device, learning_rate).fit_and_eval(kde_loader, X_val_fold, y_val_fold)
            gmm_acc_dnn    = DNN(input_dim, hidden_dim, num_classes, num_epochs, device, learning_rate).fit_and_eval(gmm_loader, X_val_fold, y_val_fold)
            gan_acc_dnn    = DNN(input_dim, hidden_dim, num_classes, num_epochs, device, learning_rate).fit_and_eval(gan_loader, X_val_fold, y_val_fold)
            
            dnn_scores["real"].append(real_acc_dnn)
            dnn_scores["smote"].append(smote_acc_dnn)
            dnn_scores["kde"].append(kde_acc_dnn)
            dnn_scores["gmm"].append(gmm_acc_dnn)
            dnn_scores["gan"].append(gan_acc_dnn)
            
            # For KNN and RF, we call fit_and_eval directly.
            real_acc_knn   = KNN().fit_and_eval(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            smote_acc_knn  = KNN().fit_and_eval(synthetic_x_smote, synthetic_y_smote, X_val_fold, y_val_fold)
            kde_acc_knn    = KNN().fit_and_eval(synthetic_x_kde, synthetic_y_kde, X_val_fold, y_val_fold)
            gmm_acc_knn    = KNN().fit_and_eval(synthetic_x_gmm, synthetic_y_gmm, X_val_fold, y_val_fold)
            gan_acc_knn    = KNN().fit_and_eval(synthetic_x_gen, synthetic_y_gen, X_val_fold, y_val_fold)
            
            knn_scores["real"].append(real_acc_knn)
            knn_scores["smote"].append(smote_acc_knn)
            knn_scores["kde"].append(kde_acc_knn)
            knn_scores["gmm"].append(gmm_acc_knn)
            knn_scores["gan"].append(gan_acc_knn)
            
            real_acc_rf   = RF().fit_and_eval(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            smote_acc_rf  = RF().fit_and_eval(synthetic_x_smote, synthetic_y_smote, X_val_fold, y_val_fold)
            kde_acc_rf    = RF().fit_and_eval(synthetic_x_kde, synthetic_y_kde, X_val_fold, y_val_fold)
            gmm_acc_rf    = RF().fit_and_eval(synthetic_x_gmm, synthetic_y_gmm, X_val_fold, y_val_fold)
            gan_acc_rf    = RF().fit_and_eval(synthetic_x_gen, synthetic_y_gen, X_val_fold, y_val_fold)
            
            rf_scores["real"].append(real_acc_rf)
            rf_scores["smote"].append(smote_acc_rf)
            rf_scores["kde"].append(kde_acc_rf)
            rf_scores["gmm"].append(gmm_acc_rf)
            rf_scores["gan"].append(gan_acc_rf)
            
            fold_no += 1

        # Average the scores over folds for the current (sample_size, generation_size) configuration
        avg_dnn = {k: np.mean(v) for k, v in dnn_scores.items()}
        avg_knn = {k: np.mean(v) for k, v in knn_scores.items()}
        avg_rf  = {k: np.mean(v) for k, v in rf_scores.items()}
        
        # Append results to summary_data
        summary_data.append({
            "Train Samples": size,
            "Generation Size": generation_size,
            "KNN Real Dataset Accuracy": avg_knn["real"],
            "KNN GAN Accuracy": avg_knn["gan"],
            "KNN SMOTE Accuracy": avg_knn["smote"],
            "KNN KDE Accuracy": avg_knn["kde"],
            "KNN GMM Accuracy": avg_knn["gmm"],
            
            "DNN Real Dataset Accuracy": avg_dnn["real"],
            "DNN GAN Accuracy": avg_dnn["gan"],
            "DNN SMOTE Accuracy": avg_dnn["smote"],
            "DNN KDE Accuracy": avg_dnn["kde"],
            "DNN GMM Accuracy": avg_dnn["gmm"],
            
            "Random Forest Real Dataset Accuracy": avg_rf["real"],
            "Random Forest GAN Accuracy": avg_rf["gan"],
            "Random Forest SMOTE Accuracy": avg_rf["smote"],
            "Random Forest KDE Accuracy": avg_rf["kde"],
            "Random Forest GMM Accuracy": avg_rf["gmm"]
        })

# Create the summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Save to CSV
summary_df.to_csv("augmentation_performance_gen.csv", index=False)
print(summary_df)
