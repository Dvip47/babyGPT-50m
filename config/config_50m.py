# config/config_50m.py

out_dir = 'out/babygpt_50m'
eval_interval = 250
log_interval = 50
eval_iters = 200

wandb_log = False
wandb_project = 'babygpt-50m'
wandb_run_name = 'ft-run'

dataset = 'multitask'
gradient_accumulation_steps = 4
batch_size = 2
block_size = 64
# block_size = 256

n_layer = 10
n_head = 8
n_embd = 384
dropout = 0.1

bias = False
vocab_size = None  # get from meta.pkl
learning_rate = 3e-4
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-5
warmup_iters = 200

compile = False
