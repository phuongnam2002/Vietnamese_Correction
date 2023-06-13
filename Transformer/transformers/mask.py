import numpy as np
import torch


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')  # triangle matrix
    np_mask = np_mask == 0
    np_mask = torch.from_numpy(np_mask)
    return np_mask  # 1, seq_len, seq_len


def create_src_mask(src, pad_token):
    """
    src: batch_size, seq_len
    pad_token: index of pad_token token

    output: batch_size, 1, seq_len
    """
    src_mask = (src != pad_token).unsqueeze(-2)
    return src_mask  # batch_size, 1, seq_len


def create_trg_mask(trg, pad_token):
    trg_mask = (trg != pad_token).unsqueeze(-2)
    seq_len = trg.size(1)
    np_mask = nopeak_mask(seq_len)
    if trg.is_cuda():
        np_mask = np_mask.cuda()
    trg_mask = trg_mask & np_mask
    return trg_mask


def create_mask(src, trg, src_pad_token, trg_pad_token):
    src_mask = create_src_mask(src, src_pad_token)
    trg_mask = create_trg_mask(trg, trg_pad_token)
    return src_mask, trg_mask
