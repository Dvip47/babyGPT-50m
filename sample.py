# sample.py
import torch, pickle, argparse
from model import GPT, GPTConfig
import tiktoken

# ── CLI ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",   default="out/babygpt_50m/ckpt.pt")
parser.add_argument("--max_new", type=int, default=120)
parser.add_argument("--start",  default="[CODE]\n# Add two numbers\n")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k",  type=int, default=None)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── tokenizer meta ─────────────────────────────
with open("data/multitask/meta.pkl", "rb") as f:
    meta = pickle.load(f)
enc = tiktoken.get_encoding(meta["tokenizer"])

# ── load checkpoint ────────────────────────────
ckpt = torch.load(args.ckpt, map_location=device)
cfg  = GPTConfig(**ckpt["model_args"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# ── encode prompt ──────────────────────────────
idx = torch.tensor([enc.encode(args.start)], dtype=torch.long).to(device)

# ── generate loop ──────────────────────────────
for _ in range(args.max_new):
    # crop to block_size context
    if idx.size(1) > cfg.block_size:
        idx_cond = idx[:, -cfg.block_size:]
    else:
        idx_cond = idx

    logits, _ = model(idx_cond)
    logits = logits[:, -1, :] / args.temperature

    if args.top_k:
        v, _ = torch.topk(logits, args.top_k)
        logits[logits < v[:, [-1]]] = -float("Inf")

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
    idx = torch.cat((idx, next_id), dim=1)

# ── decode & print ─────────────────────────────
generated = enc.decode(idx[0].tolist())
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
print(generated)
