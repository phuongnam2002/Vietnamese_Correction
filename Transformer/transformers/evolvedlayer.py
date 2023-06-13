import torch
import torch.nn as nn
import torch.nn.functional as F
from .utillayer import FeedForward, Norm, MultiHeadAttention


class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(SeparableConv1D, self).__init__()
        self.deep_wise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   padding, groups=in_channels)
        self.point_wise = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        x: batch, seq_len, in_channels
        output: batch, seq_len, out_channels
        """
        x = self.deep_wise(x)
        x = self.point_wise(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dff, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.norm_glu = Norm(d_model)
        self.glu1 = nn.Linear(d_model, d_model)
        self.glu2 = nn.Linear(d_model, d_model)

        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model * 4)

        self.left_conv = nn.Linear(d_model, d_model * 4)
        self.left_dropout = nn.Dropout(dropout)

        self.right_conv = nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1)
        self.right_dropout = nn.Dropout(dropout)

        self.sep_conv = SeparableConv1D(d_model * 4, d_model // 2, kernel_size=9, padding=4)

        self.norm_attn = Norm(d_model)
        self.attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm_ff = Norm(d_model)
        self.ff = FeedForward(d_model, dff, dropout)

    def forward(self, x, mask):
        residual = x
        x = self.norm_glu(x)  # batch, seq_len, d_model
        values = self.glu1(x)  # batch, seq_len, d_model
        gates = torch.sigmoid(self.glu2(x))
        hidden_state = values * gates
        x = residual + hidden_state  # batch, seq_len, d_model

        conv_mask = mask[:, -1, :]  # batch, d_model
        conv_mask = conv_mask.unsqueeze(-1)  # batch, d_model, 1

        residual = x
        x = self.norm1(x)
        x = x.masked_fill(conv_mask == 0, 0)  # batch, seq_len, d_model

        left_state = self.left_conv(x)  # batch, seq_len, d_model * 4
        left_state = F.relu(left_state)
        left_state = self.left_dropout(left_state)  # batch, seq_len, d_model * 4

        right_state = self.right_conv(x.transpose(-1, -2)).transpose(-1, -2)  # batch, seq_len, d_model / 2
        right_state = F.relu(right_state)
        right_state = self.right_dropout(right_state)  # batch, seq_len, d_model / 2
        right_state = self.right_dropout(right_state)

        hidden_state = left_state + right_state  # batch, seq_len, d_model * 4
        hidden_state = self.norm2(hidden_state)
        hidden_state = hidden_state.masked_fill(conv_mask == 0, 0)
        hidden_state = self.sep_conv(hidden_state.transpose(-1, -2)).transpose(-1, -2)  # batch, seq_len, d_model / 2
        hidden_state = F.pad(hidden_state, (0, self.d_model // 2))  # batch, seq_len, d_model

        x = residual + hidden_state

        residual = x
        x = self.norm_attn(x)
        attn = self.attn(x, x, x, mask)
        attn = self.attn_dropout(attn)
        x = residual + attn

        residual = x
        x = self.norm_ff(x)
        hidden_state = self.ff(x)
        x = residual + hidden_state  # batch, seq_len, d_model

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, dff, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn1 = MultiHeadAttention(num_heads, d_model, dropout)
        self.attn2 = MultiHeadAttention(num_heads, d_model, dropout)
        self.ff = FeedForward(d_model, dff, dropout)

        self.heads = num_heads
        self.d_model = d_model

        self.norm_attn1 = Norm(d_model)
        self.self_attn1 = MultiHeadAttention(num_heads * 2, d_model, dropout)
        self.self_attn_dropout1 = nn.Dropout(dropout)
        self.enc_attn1 = MultiHeadAttention(num_heads, d_model, dropout)
        self.enc_attn_dropout1 = nn.Dropout(dropout)

        self.norm_conv1 = Norm(d_model)
        self.norm_conv2 = Norm(d_model * 2)

        self.left_sep_conv = SeparableConv1D(d_model, d_model * 2, kernel_size=11)
        self.right_sep_conv = SeparableConv1D(d_model, d_model // 2, kernel_size=7)
        self.sep_conv = SeparableConv1D(d_model * 2, d_model, kernel_size=7)

        self.norm_attn2 = Norm(d_model)
        self.self_attn2 = MultiHeadAttention(num_heads, d_model, dropout)

        self.norm_attn3 = Norm(d_model)
        self.enc_attn3 = MultiHeadAttention(num_heads, d_model, dropout)

        self.norm_ff = Norm(d_model)
        self.ff = FeedForward(d_model, dff, dropout, activation=lambda x: x * torch.sigmoid(x))

    def forward(self, x, e_outputs, src_mask, trg_mask):
        residual = x
        x = self.norm_attn1(x)

        self_attn = self.self_attn1(x, x, x, trg_mask)
        self_attn = self.self_attn_dropout1(self_attn)  # batch, seq_len, d_model

        enc_attn = self.enc_attn1(x, e_outputs, e_outputs, src_mask)
        enc_attn = self.enc_attn_dropout1(enc_attn)  # batch, seq_len, d_model

        hidden_state = self_attn + enc_attn
        x = residual + hidden_state  # batch, seq_len, d_model

        conv_mask = trg_mask[:, -1, :]  # batch, d_model
        conv_mask = conv_mask.unsqueeze(-1)  # batch, d_model, 1

        residual = x
        x = self.norm_conv1(x)
        x = x.masked_fill(conv_mask == 0, 0)  # batch, seq_len, d_model
        x_pad = F.pad(x.transpose(-1, -2), (10, 0))

        left_state = self.left_sep_conv(x_pad).transpoese(-1, -2)  # batch, seq_len, d_model * 2
        left_state = F.relu(left_state)

        x_pad = F.pad(x.transpose(-1, -2), (6, 0))

        right_state = self.right_sep_conv(x_pad).transpose(-1, -2)  # batch, seq_len, d_model / 2
        right_state = F.pad(right_state, (0, self.d_model * 2 - self.d_model // 2))  # batch, seq_len, d_model * 2

        hidden_state = left_state + right_state  # batch, seq_len, d_model * 2
        hidden_state = self.norm_conv2(hidden_state)
        hidden_state = hidden_state.masked_fill(conv_mask == 0, 0)
        hidden_state = F.pad(hidden_state.transpose(-1, -2), (6, 0))
        hidden_state = self.sep_conv(hidden_state).transpose(-1, -2)

        x = residual + hidden_state  # batch, seq_len, d_model

        residual = x
        x = self.norm_attn2(x)
        self_attn = self.self_attn2(x, x, x, trg_mask)
        x = residual + self_attn

        residual = x
        x = self.norm_attn3(x)
        enc_attn = self.enc_attn3(x, e_outputs, e_outputs, src_mask)
        x = residual + enc_attn

        residual = x
        x = self.norm_ff(x)
        hidden_state = self.ff(x)
        x = residual + hidden_state

        return x
