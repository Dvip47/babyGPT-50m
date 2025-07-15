# ---------- IO ----------
out_dir   = 'out/babygpt_mac_50m'
dataset   = 'multitask'

# ---------- logging ----------
eval_interval = 500          # evaluate every 500 iters
log_interval  = 50
eval_iters    = 200

# ---------- training schedule ----------
batch_size  = 1              # tiny to fit CPU/MPS RAM
gradient_accumulation_steps = 8   # global batch = 8
block_size  = 128

learning_rate   = 3e-4
max_iters       = 60000       # let it run as long as you like
max_iters       = 20000       # let it run as long as you like
lr_decay_iters  = 60000
warmup_iters    = 600
min_lr          = 1e-5
grad_clip       = 1.0
weight_decay    = 0.1
adamw_beta2     = 0.95

# ---------- model ----------
n_layer = 10
n_head  = 8
n_embd  = 384
dropout = 0.2                 # a bit more regularisation
bias    = False
vocab_size = None             # auto from meta.pkl

# ---------- speedups ----------
# Intel/older Macs: leave float32 + CPU
# Apple Silicon:  set dtype='float16', device='mps' via train.py arg
dtype    = 'float32'
compile  = False
