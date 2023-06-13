import os
import torch
import torch.nn as nn

from .transformer import Transformer


def get_model(src_vocab_size, trg_vocab_size, d_model=512, dff=2048, num_layers=6,
              num_heads=8, dropout=0.1, encoder_layer=None, decoder_layer=None):
    assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
    assert dropout < 1, "dropout should be less than 1"

    if encoder_layer is None and decoder_layer is not None:
        model = Transformer(src_vocab_size, trg_vocab_size, d_model, dff, num_layers, num_heads, dropout,
                            decoder_layer=decoder_layer)
    elif encoder_layer is not None and decoder_layer is None:
        model = Transformer(src_vocab_size, trg_vocab_size, d_model, dff, num_layers, num_heads, dropout,
                            encoder_layer=encoder_layer)
    elif encoder_layer is None and decoder_layer is None:
        model = Transformer(src_vocab_size, trg_vocab_size, d_model, dff, num_layers, num_heads, dropout)
    else:
        model = Transformer(src_vocab_size, trg_vocab_size, d_model, dff, num_layers, num_heads, dropout, encoder_layer,
                            decoder_layer)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def load_model(model, optim, sched, model_path):
    if os.path.isfile(model_path):
        state = torch.load(model_path)
        model.load_state_dict(state['model'])
        if optim is not None:
            optim.load_state_dict(state['optim'])
        if sched is not None:
            sched.load_state_dict(state['sched'])
    else:
        raise Exception("Invalid model path")
