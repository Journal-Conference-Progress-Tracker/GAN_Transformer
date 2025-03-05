import torch
import torch.nn as nn

class EmbeddingDNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.5):
        super(EmbeddingDNNClassifier, self).__init__()
        
        self.embedding_A = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_B = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
       
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, ids_A, ids_B):
        emb_A = self.embedding_A(ids_A)
        emb_B = self.embedding_B(ids_B)
        pooled_A = emb_A.mean(dim=1)
        pooled_B = emb_B.mean(dim=1)
        features = torch.cat([pooled_A, pooled_B], dim=1)  
        logits = self.fc(features)
        return logits