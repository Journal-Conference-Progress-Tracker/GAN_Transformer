import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig, 
)
model_name = "tasksource/deberta-small-long-nli"

def tokenize_function(examples):
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(
        examples["sentence_A"],
        examples["sentence_B"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
def compute_deberta_embedding(examples, device):
        
    config = AutoConfig.from_pretrained(model_name)
    deberta = AutoModel.from_pretrained(model_name, config=config)
    # examples["input_ids"] and examples["attention_mask"] will be lists of lists
    input_ids = torch.tensor(examples["input_ids"], device=device)
    attention_mask = torch.tensor(examples["attention_mask"], device=device)
    deberta.to(device)
    with torch.no_grad():
        outputs = deberta(input_ids=input_ids, attention_mask=attention_mask)
        # shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

    # --- Mean Pooling ---
    # Expand mask for broadcasting: shape becomes (batch_size, seq_len, hidden_size)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # Zero out all padded embeddings
    masked_embeddings = last_hidden_state * mask

    # Sum over the sequence dimension
    summed = torch.sum(masked_embeddings, dim=1)

    # Count the actual tokens (excluding padded ones)
    sum_of_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

    # Mean pooling
    mean_pooled = summed / sum_of_mask  # shape: (batch_size, hidden_size)

    # Convert to CPU (if on GPU) and to list (so the Dataset can store them)
    mean_pooled = mean_pooled.cpu().tolist()

    return {"embedding": mean_pooled}


class HFDatasetWrapper(Dataset):
    """
    Wrap a Hugging Face Dataset so it behaves like a PyTorch Dataset.
    """
    def __init__(self, hf_dataset, return_dict=True):
        """
        Args:
            hf_dataset: A Hugging Face Dataset object.
            return_dict (bool):
                If True, __getitem__ returns the raw dictionary (column_name -> value).
                If False, __getitem__ returns a tuple of values in alphabetical column order.
        """
        self.hf_dataset = hf_dataset
        self.column_names = hf_dataset.column_names
        self.return_dict = return_dict

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]  # a dict {column_name: value}
        if self.return_dict:
            return sample
        else:
            # return a tuple in alphabetical order of columns
            # or you could define a custom column order
            sorted_columns = sorted(self.column_names)
            return tuple(sample[col] for col in sorted_columns)