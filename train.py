# train.py  — BabyGPT training script with progress bar + ETA
# -----------------------------------------------------------
import os, time, pickle
from pathlib import Path
from tqdm.auto import tqdm

import torch

torch.set_num_threads(os.cpu_count())          # let Torch use all cores 
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

# ─── project imports ─────────────────────────────────────────
from config.config_50m import *          # your hyper‑params
from model   import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import get_batch

# ─── load tokenizer metadata ─────────────────────────────────
print("📦 Loading metadata...")
with open(f"data/{dataset}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

vocab_size  = meta["vocab_size"]
block_size_ = meta["block_size"]
print(f"📚 Vocab size: {vocab_size}, block size: {block_size_}")

# ─── model ───────────────────────────────────────────────────
gpt_cfg = GPTConfig(
    vocab_size  = vocab_size,
    block_size  = block_size_,
    n_layer     = n_layer,
    n_head      = n_head,
    n_embd      = n_embd,
    dropout     = dropout,
    bias        = bias,
)
model = GPT(gpt_cfg)
print(f"🧠 Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f} M params")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ─── trainer config ──────────────────────────────────────────
tconf = TrainerConfig(
    max_iters                  = max_iters,
    batch_size                 = batch_size,
    block_size                 = block_size,
    learning_rate              = learning_rate,
    lr_decay_iters             = lr_decay_iters,
    warmup_iters               = warmup_iters,
    min_lr                     = min_lr,
    device                     = device,
    gradient_accumulation_steps= gradient_accumulation_steps,
    out_dir                    = out_dir,
    eval_interval              = eval_interval,
    eval_iters                 = eval_iters,
    log_interval               = log_interval,
    dataset                    = dataset,
)

trainer = Trainer(model, get_batch, tconf)

# ─── training loop with tqdm ─────────────────────────────────
print("🚀 Training begins...")
start_wall = time.time()

pbar = tqdm(range(trainer.iter_num, max_iters), initial=trainer.iter_num,
            total=max_iters,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} • {elapsed} < {remaining}")

for _ in pbar:
    # single forward/backward/step
    loss = trainer.train_step()                 # <-- needs to exist in Trainer

    if trainer.iter_num % log_interval == 0:
        pbar.set_description(f"loss {loss:.4f}")

    # periodic eval + checkpoint
    if (trainer.iter_num % eval_interval == 0) or (trainer.iter_num == max_iters):
        val_loss = trainer.evaluate()           # <-- needs to exist in Trainer
        elapsed  = (time.time() - start_wall) / 3600
        pbar.write(f"✅ iter {trainer.iter_num} | train {loss:.4f} • "
                   f"val {val_loss:.4f} • elapsed {elapsed:.2f} h")

    if trainer.iter_num >= max_iters:
        break
