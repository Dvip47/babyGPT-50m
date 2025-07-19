# -----------------------------
# BabyGPT 30M Config (Custom)
# -----------------------------

# ---------- IO ----------
out_dir   = 'out/babygpt_mac_30m'  # New output folder
dataset   = 'multitask'            # Folder under data/

# ---------- Logging ----------
eval_interval = 250         # Evaluate more frequently
log_interval  = 50
eval_iters    = 100

# ---------- Training Schedule ----------
batch_size  = 1              # Small batch for CPU/MPS
gradient_accumulation_steps = 8   # 1x8 = 8 effective batch
block_size  = 128            # Sequence length

learning_rate   = 3e-4
max_iters       = 5000       # Enough for 13K samples
lr_decay_iters  = 5000
warmup_iters    = 500
min_lr          = 1e-5
grad_clip       = 1.0
weight_decay    = 0.1
adamw_beta2     = 0.95

# ---------- Model (30M Parameters Approx) ----------
n_layer = 6
n_head  = 6
n_embd  = 384
dropout = 0.2                 # Regularization to prevent overfit
bias    = False               # No bias in Linear layers
vocab_size = None             # Will be read from meta.pkl

# ---------- Speedups ----------
# CPU or Intel Mac
dtype   = 'float32'
compile = False  # Set True if using PyTorch 2.0+

# ---------- Notes ----------
# For Apple M1/M2, run training with:
# python3 train.py config/train_babygpt_30m.py --device=mps --compile=False
#
# For CPU (Intel Mac/Linux), run:
# python3 train.py config/train_babygpt_30m.py --device=cpu
