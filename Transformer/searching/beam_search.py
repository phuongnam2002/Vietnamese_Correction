import math as mt
import torch
import torch.nn.functional as F
from fvac.transformers.mask import nopeak_mask


def init_vars(src, model, SRC, TRG, beam_width=3, max_len=1000, device=-1):
    # src: 1, seq_len
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)  # 1, seq_len, d_model

    outputs = torch.LongTensor([[init_tok]])  # 1, 1
    if device != -1:
        outputs = outputs.cuda()

    trg_mask = nopeak_mask(1)
    if device != -1:
        trg_mask = trg_mask.cuda()

    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)  # 1, 1, trg_vocab

    probs, ix = out[:, -1].data.topk(beam_width)  # 1, beam_width
    log_scores = torch.Tensor([mt.log(prob) for prob in probs.data[0]]).unsqueeze(0)  # 1, beam_width

    outputs = torch.zeros(beam_width, max_len).long()  # beam_width, max_len
    if device != -1:
        outputs = outputs.cuda()

    outputs[:, 0] = init_tok  # first token is sos
    outputs[:, 1] = ix[0]  # second token is top beam_width token

    e_outputs = torch.zeros(beam_width, e_output.size(-2), e_output.size(-1))  # beam_width, seq_len, d_model
    if device != -1:
        e_outputs = e_outputs.cuda()

    e_outputs[:, :] = e_output[0]  # broadcast 1, seq_len, d_model to beam_width, seq_len, d_model

    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    """
    outputs: k, seq_len - keep top_k result
    out: k, seq_len, trg_vocab - output of decoder
    log_scores: 1, k - k best log scores
    """
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([mt.log(p) for p in probs.data.view(-1)]).view(k, -1)  # k, k

    # broadcast add [sent1_score, sent2_score, sent3_score] to log_probs
    log_probs = log_probs + log_scores.transpose(0, 1)

    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unqueeze(0)

    return outputs, log_scores


def beam_search(src, model, SRC, TRG, beam_width=3, max_len=1000, device=-1):
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, beam_width=3, max_len=1000, device=device)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unqueeze(-2)
    ind = None
    for i in range(2, max_len):
        trg_mask = nopeak_mask(i)
        if device != -1:
            trg_mask = trg_mask.cuda()

        out = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)

        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, beam_width)

        eos_ix = (outputs == eos_tok).nonzero()  # n index of elements = eos_tok, shape: n, 1

        if len(eos_ix[:, 0].unique()) == beam_width:  # all candidate is reached <eos>
            alpha = 0.7

            eos_pos = []
            for ki in range(beam_width):
                temp = (eos_ix[:, 0] == ki).nonzero().view(-1)
                eos_pos += [temp[0].item()]
            div = 1 / ((eos_ix[eos_pos])[:, 1].type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.item()
            break

    if ind is None:
        length = (outputs[0] == eos_tok).nonzero()[0]
        return " ".join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    else:
        length = (outputs[ind] == eos_tok).nonzero()[0]
        return " ".join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])


def translate_sentence(sentence, model, SRC, TRG, device=-1):
    model.eval()
    sentence = SRC.preprocess(sentence)
    sentence = SRC.pad([sentence])
    sent_tensor = SRC.numericalize(sentence)  # seq_len, 1
    sent_tensor = sent_tensor.transpose(0, 1)  # 1, seq_len

    if device != -1:
        sent_tensor = sent_tensor.cuda()

    sentence = beam_search(sent_tensor, model, SRC, TRG, beam_width=3, max_len=1000, device=device)

    return sentence
