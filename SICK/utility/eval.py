

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from IPython.display import clear_output


def jaccard_similarity(y_true, y_pred, pad_id=0):
    """
    Compute Jaccard similarity between two sequences of token IDs or strings.

    Args:
        y_true (list[list[int]] or list[str]): Ground truth token IDs or text.
        y_pred (list[list[int]] or list[str]): Predicted token IDs or text.
        pad_id (int): Padding token ID to ignore. Defaults to 0.

    Returns:
        float: Average Jaccard similarity across all sequences.
    """
    total_jaccard = 0
    count = 0

    for true_seq, pred_seq in zip(y_true, y_pred):
        if isinstance(true_seq, str):  # If input is a list of strings, treat words as tokens
            true_set = set(true_seq.split())
            pred_set = set(pred_seq.split())
        else:  # If input is token IDs, filter out padding and compute set similarity
            true_set = set([token for token in true_seq if token != pad_id])
            pred_set = set([token for token in pred_seq if token != pad_id])

        intersection = len(true_set & pred_set)
        union = len(true_set | pred_set)

        jaccard = intersection / union if union > 0 else 0.0
        total_jaccard += jaccard
        count += 1

    return total_jaccard / count if count > 0 else 0.0

def align_sequences(seq1, seq2, metric='euclidean', one_to_one=True):
    """
    Aligns two sequences of vectors based on their closest counterparts, preserving original format.

    Parameters:
    - seq1: np.ndarray or torch.Tensor, shape (n, d), first sequence of vectors
    - seq2: np.ndarray or torch.Tensor, shape (m, d), second sequence of vectors
    - metric: str, distance metric (default: 'euclidean')
    - one_to_one: bool, whether to perform one-to-one alignment using Hungarian algorithm (default: True)

    Returns:
    - aligned_seq1: np.ndarray or torch.Tensor, aligned vectors from seq1
    - aligned_seq2: np.ndarray or torch.Tensor, corresponding closest vectors from seq2
    """
    # Detect input types and convert to numpy if necessary
    seq1_is_tensor = isinstance(seq1, torch.Tensor)
    seq2_is_tensor = isinstance(seq2, torch.Tensor)

    if seq1_is_tensor:
        seq1 = seq1.cpu().numpy()
    if seq2_is_tensor:
        seq2 = seq2.cpu().numpy()

    # Compute pairwise distance matrix
    cost_matrix = cdist(seq1, seq2, metric=metric)

    if one_to_one:
        # Use Hungarian algorithm for optimal one-to-one alignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        aligned_seq1 = seq1[row_ind]
        aligned_seq2 = seq2[col_ind]
    else:
        # Use Nearest Neighbors (one-to-many alignment)
        nearest_indices = np.argmin(cost_matrix, axis=1)
        aligned_seq1 = seq1
        aligned_seq2 = seq2[nearest_indices]

    # Convert back to tensors if original inputs were tensors
    if seq1_is_tensor:
        aligned_seq1 = torch.tensor(aligned_seq1)
    if seq2_is_tensor:
        aligned_seq2 = torch.tensor(aligned_seq2)

    return aligned_seq1, aligned_seq2

def calculate_hit_rate(y_true, y_pred, pad_id=0):
    """
    Compute hit rate between two sequences of token IDs.

    Args:
        y_true (list[list[int]]): Ground truth token IDs.
        y_pred (list[list[int]]): Predicted token IDs.
        pad_id (int): Padding token ID to ignore. Defaults to 0.

    Returns:
        float: Hit rate (accuracy) over non-padding tokens.
    """
    # Convert to numpy arrays
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)

    # Ensure y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Incompatible shapes: {y_true.shape} vs {y_pred.shape}")

    # Count non-padding positions
    non_pad_mask = (y_true != pad_id)

    # Compute hit rate only over non-padding tokens
    hits = np.sum((y_true == y_pred) & non_pad_mask)
    total = np.sum(non_pad_mask)

    hit_rate = hits / total if total > 0 else 0.0
    return hit_rate

def compute_seq_similarity(model: SentenceTransformer = None, seq_true=None, seq_pred=None, tokenizer=None, device: torch.device = torch.device('cuda')):
    """
    Compute cosine similarity between two sequences of token IDs using SentenceTransformer embeddings.

    Args:
        model (SentenceTransformer): Preloaded SentenceTransformer model.
        seq_true (list[list[int]]): Ground truth sequence of token IDs (list of lists).
        seq_pred (list[list[int]]): Predicted sequence of token IDs (list of lists).
        tokenizer: The tokenizer corresponding to the model that generated token IDs (for decoding).
        device (torch.device): Device to perform computation on ('cpu', 'cuda'). Defaults to automatic selection.
    """
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Set device explicitly if provided
    if device is not None:
        model = model.to(device)

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for decoding token IDs.")

    jaccard = jaccard_similarity(seq_true, seq_pred)
    # Decode the token IDs into strings (handle list of lists)
    seq_true_tokens = [tokenizer.decode(seq, skip_special_tokens=True) for seq in seq_true]
    seq_pred_tokens = [tokenizer.decode(seq, skip_special_tokens=True) for seq in seq_pred]

    # Compute embeddings
    emb_true = model.encode(seq_true_tokens, convert_to_tensor=True, batch_size=64)
    emb_pred = model.encode(seq_pred_tokens, convert_to_tensor=True, batch_size=64)
        
    aligned_pred, aligned_y = align_sequences(emb_pred, emb_true)

    unaligned_cos_sim = util.cos_sim(emb_pred, emb_true).diag().mean().item()
    after_aligned_cos_sim = util.cos_sim(aligned_pred, aligned_y).diag().mean().item()

    return after_aligned_cos_sim, unaligned_cos_sim, jaccard, seq_true_tokens, seq_pred_tokens
