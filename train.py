# train.py

import os
import pickle
from config.config_50m import *
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import get_batch
import torch

print("📦 Loading metadata...")
with open(f'data/{dataset}/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
block_size = meta['block_size']
print(f"📚 Vocab size: {vocab_size}, block size: {block_size}")

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias
)

model = GPT(config)
print(f"🧠 Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

tconf = TrainerConfig(
    max_iters=max_iters,
    batch_size=batch_size,
    block_size=block_size,
    learning_rate=learning_rate,
    lr_decay_iters=lr_decay_iters,
    warmup_iters=warmup_iters,
    min_lr=min_lr,
    device=device,
    gradient_accumulation_steps=gradient_accumulation_steps,
    out_dir=out_dir,
    eval_interval=eval_interval,
    eval_iters=eval_iters,
    log_interval=log_interval,
        dataset=dataset  # ✅ Yeh line add karni thi!

)

trainer = Trainer(model, get_batch, tconf)

print("🚀 Training begins...")
trainer.train()
