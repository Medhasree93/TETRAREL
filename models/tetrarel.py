import torch
import torch.nn as nn
from models.attention import TemporalTransactionalAttention


class TETRAREL(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.attention = TemporalTransactionalAttention(
            hidden_dim,
            hidden_dim
        )

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = x.mean(dim=1)
        out = self.classifier(x)

        return out, x
