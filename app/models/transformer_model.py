import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ExactTransformerModel(nn.Module):
    """Exact model architecture that matches your saved weights"""

    def __init__(self, input_size=35, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super(ExactTransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first=True to match the saved weights
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)  # [batch_size, seq_len, d_model]
        x = self.transformer(x)  # [batch_size, seq_len, d_model]
        x = x.mean(dim=1)  # Global average pooling over sequence
        return self.output(x)  # [batch_size, 1]