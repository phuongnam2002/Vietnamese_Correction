import torch.nn as nn
from fairseq.modules.lightweight_convolution import LightweightConv1dTBC
from .utillayer import Norm, FeedForward, MultiHeadAttention


class LightweightConvLayer(nn.Module):
    def __init__(self, d_model, conv_dim, kernel_size, weight_softmax,
                 num_heads, weight_dropout):
        super(LightweightConvLayer, self).__init__()
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        self.linear1 = nn.Linear(d_model, conv_dim * 2)
        self.activation = nn.GLU()
        self.conv = LightweightConv1dTBC(conv_dim, kernel_size, padding_l,
                                         weight_softmax, num_heads, weight_dropout)
        self.linear2 = nn.Linear(conv_dim, d_model)

    def forward(self, x, mask):
        x = self.linear1(x)  # batch, seq_len, conv_dim*2
        x = self.activation(x)
        conv_mask = mask[:, -1, :]  # batch, d_model
        conv_mask = conv_mask.unsqueeze(-1)  # batch, d_model, 1
        x = x.masked_fill(conv_mask == 0, 0)

        x = x.transpose(0, 1)  # seq_len, batch_size, conv_dim*2
        x = self.conv(x.contiguous())  # seq_len, batch_size, conv_dim
        x = x.transpose(0, 1)  # batch, seq_len, conv_dim
        x = self.linear2(x)  # batch, seq_len, d_model
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dff, num_heads, dropout=0.1,
                 weight_softmax=True, weight_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)

        self.conv_dim = d_model
        self.kernel_size = 3
        self.conv = LightweightConvLayer(d_model, self.conv_dim, self.kernel_size,
                                         weight_softmax, num_heads, weight_dropout)
        self.ff = FeedForward(d_model, dff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)
        x = self.conv(x, mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout2(x)
        x = residual + x

        return x  # batch, seq_len, d_model


class DecoderLayer(nn.Module):
    def __init__(self, d_model, dff, num_heads, dropout=0.1,
                 weight_softmax=True, weight_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.ff = FeedForward(d_model, dff, dropout)

        self.conv_dim = d_model
        self.kernel_size = 3

        self.conv = LightweightConvLayer(d_model, self.conv_dim, self.kernel_size,
                                         weight_softmax, num_heads, weight_dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        residual = x
        x = self.norm1(x)
        x = self.conv(x, trg_mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.attn(x, e_outputs, e_outputs, src_mask)
        x = self.dropout2(x)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = self.dropout3(x)
        x = residual + x

        return x  # batch, seq_len, d_model
