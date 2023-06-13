import copy
import torch.nn as nn
from .layer import EncoderLayer, DecoderLayer
from .utillayer import Embedding, PositionalEncoding, Norm


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, dff, num_layers, num_heads, dropout, encoder_layer):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        self.layers = get_clones(encoder_layer(d_model, dff, num_heads, dropout), num_layers)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)

        return self.norm(x)  # batch_size, seq_len, d_model


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, dff, num_layers, num_heads, dropout, decoder_layer):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        self.layers = get_clones(decoder_layer(d_model, dff, num_heads, dropout), num_layers)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)

        return self.norm(x)  # batch_size, target_len, d_model


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, dff, num_layers,
                 num_heads, dropout, encoder_layer=EncoderLayer, decoder_layer=DecoderLayer):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, dff, num_layers, num_heads, dropout, encoder_layer)
        self.decoder = Decoder(trg_vocab_size, d_model, dff, num_layers, num_heads, dropout, decoder_layer)
        self.out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_outputs)
        return output  # batch_size, target_len, d_model
