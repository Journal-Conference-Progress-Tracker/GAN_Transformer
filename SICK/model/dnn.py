import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig



class DNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(DNNClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.fc(x)


class DeBERTaDNNClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super(DeBERTaDNNClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)

        self.deberta_A = AutoModel.from_pretrained(model_name, config=self.config)
        self.deberta_B = AutoModel.from_pretrained(model_name, config=self.config)
        for param in self.deberta_A.parameters():
            param.requires_grad = False
        for param in self.deberta_B.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_labels)
        )
        
    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        outputs_A = self.deberta_A(input_ids=input_ids_A, attention_mask=attention_mask_A)
        pooled_A = outputs_A.pooler_output  # (batch, hidden_size)
        outputs_B = self.deberta_B(input_ids=input_ids_B, attention_mask=attention_mask_B)
        pooled_B = outputs_B.pooler_output  # (batch, hidden_size)
        features = torch.cat([pooled_A, pooled_B], dim=1)  # (batch, 2 * hidden_size)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits
