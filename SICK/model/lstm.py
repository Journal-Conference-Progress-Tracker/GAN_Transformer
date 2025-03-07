import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score

def train_and_evaluate_lstm(model, train_loader, test_loader, num_epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_y.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    test_acc = accuracy_score(all_labels, all_preds)
    return test_acc, all_preds, all_labels


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