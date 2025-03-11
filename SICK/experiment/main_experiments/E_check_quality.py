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
from utility.eval import compute_seq_similarity
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
from transformers import AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch.utils.data import Subset
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util



model_name = "tasksource/deberta-small-long-nli"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = SentenceTransformer('all-MiniLM-L6-v2')
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
test_text = np.array(test_dataset['input_ids'])
test_y = np.array(test_dataset['labels'])

# Full training arrays (will be used for selecting sample sizes)
train_x_full = np.array(train_dataset['embedding'])
train_text = np.array(train_dataset['input_ids'])
train_y_full = np.array(train_dataset['labels'])
train_ds = EmbeddingDataset(train_dataset)
test_ds = EmbeddingDataset(test_dataset)

# Get input dimension from one sample
sample_emb, _ = train_ds[0]
input_dim = sample_emb.shape[0]
hidden_dim = 128
num_classes = 3  




# Number of folds for cross validation
n_splits = 5
generation_size = 1000
size = len(train_ds)

summary_data = []
print(f"\n[Train Data] Sample size: {size}")
# Get the first "size" samples from the full training set
X_train = train_x_full[:size]
y_train = train_y_full[:size]
y_train_text = train_text[:size]
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_no = 1
for train_index, val_index in skf.split(X_train, y_train):
    print(f"Fold: [{fold_no}/{n_splits}]")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    y_train_text_fold, y_val_text_fold = y_train_text[train_index], y_train_text[val_index]


    att  = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=1, n_jobs=-1, leaf_size=100))
    att.fit(X_train_fold, y_train_text_fold)
    # Generating synthetic data using different methods
    real_loader = xy_to_tensordataset(X_val_fold, y_val_fold, return_loader=True)
            
    synthetic_methods = {
        'SMOTE': conditional_smote_sampling(X_val_fold, y_val_fold, generation_size, condition=[0, 1, 2]),
        'KDE': conditional_kde_sampling(X_val_fold, y_val_fold, generation_size, condition=[0, 1, 2], n_components=min(X_train_fold.shape[1] - 1, generation_size - 1)),
        'GMM': conditional_gmm_sampling(X_val_fold, y_val_fold, generation_size, condition=[0, 1, 2]),
        'GAN':  GANs(
                batch_size,
                X_val_fold,
                y_val_fold,
                train_y_full,  # using full training labels as in your original code
                latent_dim,
                condition_dim,
                device,
                gan_epochs
            ).generate(real_loader, generation_size)
    }

    # Evaluating each synthetic data method
    for method_name, (synthetic_x, synthetic_y) in synthetic_methods.items():
        pred = att.predict(synthetic_x)
        seq_similarity_score = compute_seq_similarity(model, synthetic_y, pred)

        # Append results
        summary_data.append({
            'Fold': fold_no,
            'Method': method_name,
            'Sequence Similarity Score': seq_similarity_score,
            'Generation_Size': generation_size,
            'Sample_Size': len(X_train_fold)
        })

    fold_no += 1

# Convert collected data into DataFrame
results_df = pd.DataFrame(summary_data)

# Export to CSV
results_csv_path = 'cross_validation_results.csv'
results_df = pd.DataFrame(summary_data)
results_df.to_csv(results_csv_path, index=False)

print(f'Results saved to {results_csv_path}')
