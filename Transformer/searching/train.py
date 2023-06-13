import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchtext.legacy.data import BucketIterator
from transformers.support import get_model, load_model
from transformers.mask import create_mask
from transformers.optimizer import CosineWithRestarts
from load_data import load_multi30k
from lion_pytorch import Lion


def forward_and_loss(model, src, trg, loss_fn, src_pad, trg_pad):
    trg_input = None
    if trg is not None:
        trg_input = trg[:, :-1]

    src_mask, trg_mask = create_mask(src, trg_input, src_pad_token=src_pad, trg_pad_token=trg_pad)
    preds = model(src, trg_input, src_mask, trg_mask)
    ys = trg[:, 1:].contiguous().view(-1)
    loss = loss_fn(preds.view(-1, preds.size(-1)), ys, ignore_index=trg_pad)
    return preds, loss


def train_model(model, optimizer, train_iter, src_pad, trg_pad, scheduler=None, save_path=None):
    total_loss = 0.0
    total_item = 0

    model.train()

    with tqdm(total=len(train_iter)) as pbar:
        for batch in train_iter:
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)

            optimizer.zero_grad()
            _, loss = forward_and_loss(model, src, trg, F.cross_entropy, src_pad, trg_pad)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_item += batch.batch_size

            pbar.update(1)
            pbar.set_description("train_loss = %.6f" % (total_loss / total_item))

    if save_path is not None:
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": None if scheduler is None else scheduler.state_dict()
        }
        torch.save(state, save_path)


def evaluate_model(model, val_iter, src_pad, trg_pad):
    model.eval()
    with torch.no_grad(), tqdm(total=len(val_iter)) as pbar:
        total_loss = 0.0
        total_item = 0
        for batch in val_iter:
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)

            _, loss = forward_and_loss(model, src, trg, F.cross_entropy, src_pad, trg_pad)

            total_loss += loss.item()
            total_item += batch.batch_size

            pbar.update(1)
            pbar.set_description("val_loss = %.6f" % (total_loss / total_item))


if __name__ == '__main__':
    print("__________Loading data__________")
    (SRC, TRG), (train_ds, val_ds, test_ds) = load_multi30k()
    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']

    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iter = BucketIterator(
        train_ds,
        batch_size=BATCH_SIZE,
        pin_memory=(device == "cuda")
    )

    val_iter = BucketIterator(
        val_ds,
        batch_size=BATCH_SIZE,
        pin_memory=(device == "cuda")
    )

    SRC_NUMWORD = len(SRC.vocab.itos)
    TRG_NUMWORD = len(TRG.vocab.itos)
    learning_rate = 0.0001

    print("__________Init Model__________")
    model = get_model(SRC_NUMWORD, TRG_NUMWORD)

    if device.type == "cuda":
        model = model.cuda()

    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    sched = CosineWithRestarts(optimizer, T_max=len(train_iter))

    if not os.path.isdir("weight"):
        os.makedirs("weight")

    for i in tqdm(range(30)):
        weight_path = "/home/namdp/ViAC/vietnamese-auto-correction/weight/epoch_%d.h5" % (i + 1)
        print("\nEpoch %d" % (i + 1), flush=True)
        train_model(model, optimizer, train_iter, sched, src_pad, trg_pad, weight_path)
        evaluate_model(model, val_iter, src_pad, trg_pad)
