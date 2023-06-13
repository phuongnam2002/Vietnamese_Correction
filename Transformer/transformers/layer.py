import torch.nn as nn
from .utillayer import MultiHeadAttention, FeedForward, Norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dff, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.ff = FeedForward(d_model, dff, dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout2(x)
        x = residual + x

        return x  # batch_size, seq_len, d_model


class DecoderLayer(nn.Module):
    def __init__(self, d_model, dff, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.attn1 = MultiHeadAttention(num_heads, d_model, dropout)
        self.attn2 = MultiHeadAttention(num_heads, d_model, dropout)
        self.ff = FeedForward(d_model, dff, dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        residual = x
        x = self.norm1(x)
        x = self.attn1(x, x, x, trg_mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.attn2(x, e_outputs, e_outputs, src_mask)
        x = self.dropout2(x)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = self.dropout3(x)
        x = residual + x

        return x
