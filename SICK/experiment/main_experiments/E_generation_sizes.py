import os
import sys
import numpy as np
import pandas as pd
parent_dir = os.path.join(os.getcwd(), '..', '..')
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from utility.data import get_loader, EmbeddingDataset
from utility.visuals import *
from model.gan import GANs
from model.knn import KNN
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
batch_size = 32                

# Load dataset and split into train and test
full_dataset = load_from_disk('../../data/full_dataset_new', keep_in_memory=True)
split_datasets = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets['train']
test_dataset = split_datasets['test']

test_x = np.array(test_dataset['embedding'])
test_y = np.array(test_dataset['labels'])

train_x_full = np.array(train_dataset['embedding'])
train_y_full = np.array(train_dataset['labels'])
train_ds = EmbeddingDataset(train_dataset)
test_ds = EmbeddingDataset(test_dataset)
test_loader = get_loader(test_ds, batch_size=batch_size, shuffle=False)  

# Define a list of training sample sizes
sample_sizes = [20, 50, 100, 200, 1000, len(train_ds)]

# List to collect results; each entry will be a dict for one (sample_size, generation_size) combo.
summary_data = []

# Iterate over each generation size (for synthetic data generation)...
for generation_size in generation_sizes:
    print(f"\n=== Generation Size: {generation_size} ===")
    # For each training sample size...
    for size in sample_sizes:
        print(f"\n[Train Data] Sample size: {size}")
        # Get real training data for current size
        X_train = train_x_full[:size]
        y_train = train_y_full[:size]

        # Evaluate baseline KNN accuracy on real data
        real_acc = KNN().fit_and_eval(X_train, y_train, test_x, test_y)

        synthetic_x_smote, synthetic_y_smote = conditional_smote_sampling(
            X_train, y_train, generation_size
        )
        smote_augmented_x = np.concatenate([X_train, synthetic_x_smote])
        smote_augmented_y = np.concatenate([y_train, synthetic_y_smote])
        smote_acc = KNN().fit_and_eval(smote_augmented_x, smote_augmented_y, test_x, test_y)

        synthetic_x_kde, synthetic_y_kde = conditional_kde_sampling(
            X_train, y_train, generation_size, n_components=min(X_train.shape[1] - 1, generation_size - 1), condition=[0, 1, 2]
        )
        kde_augmented_x = np.concatenate([X_train, synthetic_x_kde])
        kde_augmented_y = np.concatenate([y_train, synthetic_y_kde]) if synthetic_y_kde is not None else y_train
        kde_acc = KNN().fit_and_eval(kde_augmented_x, kde_augmented_y, test_x, test_y)

        synthetic_x_gmm, synthetic_y_gmm = conditional_gmm_sampling(
            X_train, y_train, generation_size, condition=[0, 1, 2]
        )
        gmm_augmented_x = np.concatenate([X_train, synthetic_x_gmm])
        gmm_augmented_y = np.concatenate([y_train, synthetic_y_gmm]) if synthetic_y_gmm is not None else y_train
        gmm_acc = KNN().fit_and_eval(gmm_augmented_x, gmm_augmented_y, test_x, test_y)


        train_subset = Subset(train_ds, range(size))
        gan_loader = get_loader(train_subset, batch_size=batch_size, shuffle=True)
        # Assuming GANs.generate takes (training_data, batch_size, X_train, y_train, full_train_y, latent_dim, condition_dim, device, gan_epochs)
        synthetic_x_gen, synthetic_y_gen = GANs(
            train_ds,
            batch_size,
            X_train,
            y_train,
            train_y_full,
            latent_dim,
            condition_dim,
            device,
            gan_epochs
        ).generate(size, generation_size)
        # Concatenate synthetic samples with the real training data
        gan_augmented_x = np.concatenate([X_train, synthetic_x_gen])
        gan_augmented_y = np.concatenate([y_train, synthetic_y_gen])
        gan_acc = KNN().fit_and_eval(gan_augmented_x, gan_augmented_y, test_x, test_y)
        
        # Append the results for the current configuration
        summary_data.append({
            "Train Samples": size,
            "Generation Size": generation_size,
            "Real Accuracy": real_acc,
            "GAN Accuracy": gan_acc,
            "SMOTE Accuracy": smote_acc,
            "KDE Accuracy": kde_acc,
            "GMM Accuracy": gmm_acc
        })

# Create the summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Save to CSV
summary_df.to_csv("augmentation_performance_sizes.csv", index=False)
print(summary_df)
