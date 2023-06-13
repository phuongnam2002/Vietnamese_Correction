import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        max_seq_len = int(max_seq_len)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * np.sqrt(self.d_model)  # make embedding relatively larger
        seq_len = x.size(1)
        pe = self.pe[:, seq_len].clone().detach()
        if x.is_cuda:
            pe = pe.cuda()
        x = x + pe
        return self.dropout(x)  # max_seq_len, d_model


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)  # batch_size, max_seq_len, d_model


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()
        self.size = d_model
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def scale_dot_product_attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)  # batch_size, num_heads, seq_len, seq_len

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = F.dropout(scores)

    output = torch.matmul(scores, v)  # batch_size, num_heads, seq_len, d_model
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        q: Query (batch_size, seq_len, d_model)
        k: Key (batch_size, seq_len, d_model)
        v: Value (batch_size, seq_len, d_model)
        """
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.heads, self.d_k)  # batch_size,seq_len,num_heads,d_k
        k = self.k_linear(k).view(batch_size, -1, self.heads, self.d_k)  # batch_size,seq_len,num_heads,d_k
        v = self.v_linear(v).view(batch_size, -1, self.heads, self.d_k)  # batch_size,seq_len,num_heads,d_k

        q = q.transpose(1, 2)  # batch_size, num_heads, seq_len, d_model
        k = k.transpose(1, 2)  # batch_size, num_heads, seq_len, d_model
        v = v.transpose(1, 2)  # batch_size, num_heads, seq_len, d_model

        scores = scale_dot_product_attention(q, k, v, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # batch_size, seq_len, d_model

        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout, activation=F.relu):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dff, d_model)
        self.activation = activation

    def forward(self, x):
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)

        return x  # batch_size, seq_len, d_model
