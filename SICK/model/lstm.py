import torch
import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, input_dim) => (batch, seq_len=1, input_dim)
        x = x.unsqueeze(1)  
        out, (h, c) = self.lstm(x)  # out: (batch, 1, hidden_dim), h: (1, batch, hidden_dim)
        h_last = h[-1]
        out = self.dropout(h_last)
        out = self.fc(out)
        return out
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