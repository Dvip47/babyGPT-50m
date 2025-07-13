# dataset.py
import os
import pickle
import numpy as np
import torch

# --------------------------------------------------------------
def load_bins(dataset="multitask"):
    """Return train‑tokens, val‑tokens, and block_size from meta.pkl."""
    data_dir = os.path.join("data", dataset)

    train = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    val   = np.fromfile(os.path.join(data_dir, "val.bin"),   dtype=np.uint16)

    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    return train, val, meta["block_size"]


# Will be filled lazily on first call
train_data, val_data, BLOCK = None, None, None

# --------------------------------------------------------------
def get_batch(split, cfg):
    """
    split : 'train' or 'val'
    cfg   : TrainerConfig (needs .batch_size and .dataset)
    Returns tensors x (input) and y (target) of shape [batch, BLOCK]
    """
    global train_data, val_data, BLOCK

    # Lazy‑load binaries the first time
    if train_data is None:
        train_data, val_data, BLOCK = load_bins(cfg.dataset)

    data = train_data if split == "train" else val_data

    # Safety check
    if len(data) <= BLOCK + 1:
        raise ValueError(
            f"❌ Data too small for block size.\n"
            f"   Need > {BLOCK + 1} tokens, but got {len(data)}.\n"
            f"   Either add more text or reduce block_size."
        )

    # Pick random starting indices
    ix = np.random.randint(0, len(data) - BLOCK - 1, size=(cfg.batch_size,))

    # Build tensors
    x = torch.stack([
        torch.tensor(data[i : i + BLOCK], dtype=torch.long) for i in ix
    ])
    y = torch.stack([
        torch.tensor(data[i + 1 : i + 1 + BLOCK], dtype=torch.long) for i in ix
    ])

    return x, y
