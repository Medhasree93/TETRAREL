import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransactionalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = torch.matmul(Q, K.transpose(-2, -1))
        attention = attention / (K.shape[-1] ** 0.5)
        attention = F.softmax(attention, dim=-1)

        output = torch.matmul(attention, V)

        return output
