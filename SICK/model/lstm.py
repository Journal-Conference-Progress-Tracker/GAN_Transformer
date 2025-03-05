import torch
import torch.nn as nn

class EmbeddingLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, max_length, dropout=0.5):

        super(EmbeddingLSTMClassifier, self).__init__()
        self.embedding_A = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm_A = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        self.embedding_B = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm_B = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, ids_A, ids_B):

        emb_A = self.embedding_A(ids_A)  # (batch, seq_len, embed_dim)
        _, (h_A, _) = self.lstm_A(emb_A)   # h_A: (num_layers, batch, hidden_dim)
        h_A = h_A[-1]  
        
        emb_B = self.embedding_B(ids_B)
        _, (h_B, _) = self.lstm_B(emb_B)
        h_B = h_B[-1]
 
        features = torch.cat([h_A, h_B], dim=1) 
        logits = self.fc(features)
        return logits