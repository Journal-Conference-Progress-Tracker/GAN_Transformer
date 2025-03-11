import numpy as np
from sentence_transformers import SentenceTransformer, util

import torch

def calculate_hit_rate(y_true, y_pred, pad_id=0):
    # Convert to numpy arrays (if not already)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out padded positions
    non_pad = y_true != pad_id
    filtered_true = y_true[non_pad]
    filtered_pred = y_pred[non_pad]

    # Compute hit rate
    hits = np.sum(filtered_true == filtered_pred)
    total = len(filtered_true)

    hit_rate = hits / total if total > 0 else 0.0
    return hit_rate

def compute_seq_similarity(model: SentenceTransformer = None, seq_true=None, seq_pred=None, device: torch.device = torch.device('cuda')):
    """
    Compute cosine similarity between two sequences of IDs using SentenceTransformer embeddings.

    Args:
        model (SentenceTransformer): Preloaded SentenceTransformer model.
        seq_true (list[int]): Ground truth sequence of IDs.
        seq_pred (list[int]): Predicted sequence of IDs.
        device (torch.device): Device to perform computation on ('cpu', 'cuda'). Defaults to automatic selection.

    Returns:
        float: Cosine similarity score as a float between 0 and 1.
    """
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Set device explicitly if provided
    if device is not None:
        model = model.to(device)

    # Convert ID sequences to space-separated strings
    seq_true_tokens = " ".join(map(str, seq_true))
    seq_pred_tokens = " ".join(map(str, seq_pred))

    # Compute embeddings
    emb_true = model.encode(seq_true_tokens, convert_to_tensor=True, batch_size=32)
    emb_pred = model.encode(seq_pred_tokens, convert_to_tensor=True, batch_size=32)

    # Compute cosine similarity
    cos_sim = util.cos_sim(emb_true, emb_pred).item()

    return cos_sim