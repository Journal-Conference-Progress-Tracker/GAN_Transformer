import os
import sys
import numpy as np
import pandas as pd
parent_dir = os.path.join(os.getcwd(), '..', '..')
if parent_dir not in sys.path: sys.path.append(parent_dir)
from utility.data import get_loader, EmbeddingDataset
from utility.visuals import *
from model.gan import Generator, Discriminator, GANManager

import torch
from torch.utils.data import Subset
from datasets import load_from_disk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128          
condition_dim = 10        
gan_epochs = 20            
generation_sizes = [1000, 10000]  # Multiple generation sizes
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

sample_sizes = [20, 50, 100, 200, 1000, len(train_ds)]

# Initialize dictionaries to store the results
knn_accuracy_before = {}
knn_accuracy_after = []
accuracy_after_title = []


# Iterate through generation sizes
for generation_size in generation_sizes:
    knn_accracy_after_gen = {}
    print(f"\nTraining with Generation Size: {generation_size}")
    
    for size in sample_sizes:
        print(f"\n[Real Data Only] Training size: {size}")
        X_train = train_x_full[:size]
        y_train = train_y_full[:size]

        # KNN on real data
        knn_real = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        knn_real.fit(X_train, y_train)
        pred_before = knn_real.predict(test_x)
        acc_before = accuracy_score(test_y, pred_before)
        knn_accuracy_before[size] = acc_before



        # Prepare GAN data loader
        train_subset = Subset(train_ds, range(size))
        gan_loader = get_loader(train_subset, batch_size=batch_size, shuffle=True)
        input_dim = X_train.shape[1]  
        unique_labels = np.unique(train_y_full)
        num_classes_gan = len(unique_labels)

        # Create and train GAN model
        generator = Generator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            num_classes=num_classes_gan,
            start_dim=latent_dim * 2,
            n_layer=3,
            output_dim=input_dim
        ).to(device)
        discriminator = Discriminator(
            condition_dim=condition_dim,
            num_classes=num_classes_gan,
            start_dim=256,
            n_layer=3,
            input_dim=input_dim
        ).to(device)

        manager = GANManager(generator, discriminator, gan_loader, gan_epochs, latent_dim, device)
        manager.train()
        synthetic_x, synthetic_y = manager.generate(unique_labels, generation_size)

        # Combine real and synthetic data
        train_combined_x = np.concatenate([X_train, synthetic_x], axis=0)
        train_combined_y = np.concatenate([y_train, synthetic_y], axis=0)

        # KNN on augmented data (real + synthetic)
        knn_aug = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn_aug.fit(train_combined_x, train_combined_y)
        pred_after = knn_aug.predict(test_x)
        acc_after = accuracy_score(test_y, pred_after)
        knn_accracy_after_gen[size] = acc_after
    knn_accuracy_after.append(knn_accracy_after_gen)
    accuracy_after_title.append(f"After Concatenation Accuracy with size of {generation_size}")
# Now create the final summary DataFrame
summary_data = []

for size in sample_sizes:
    summary_data.append([
        size,
        knn_accuracy_before[size],
        *[after[size] for after in knn_accuracy_after]
    ])

# Create DataFrame for summary
summary_df = pd.DataFrame(
    summary_data,
    columns=["Train Samples", "Real Only Accuracy", *accuracy_after_title]
)

summary_df.to_csv("g_sizes.csv", index=False)
# Display the accuracy summary
display_accuracy_summary_with_more_than_two_bars(summary_df, file_name="../../figure/generation_sizes_accuracy_plot.png")
save_table_as_image(summary_df, file_name="../../figure/generation_sizes_accuracy.png")