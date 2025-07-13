# sample.py
import torch, pickle, argparse
from model import GPT, GPTConfig
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="out/babygpt_50m/ckpt.pt")
parser.add_argument("--max_new", type=int, default=120)
parser.add_argument("--start", default="[CODE]\n# Add two numbers\n")
args = parser.parse_args()

# ── load tokenizer meta ─────────────────────────
with open("data/multitask/meta.pkl", "rb") as f:
    meta = pickle.load(f)
enc = tiktoken.get_encoding(meta["tokenizer"])

# ── load checkpoint ─────────────────────────────
ckpt = torch.load(args.ckpt, map_location="cpu")
cfg  = GPTConfig(**ckpt["model_args"])
model = GPT(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

# ── encode prompt ───────────────────────────────
idx = torch.tensor([enc.encode(args.start)], dtype=torch.long)

# ── generate ────────────────────────────────────
for _ in range(args.max_new):
    logits, _ = model(idx)
    probs = torch.softmax(logits[0, -1], dim=-1)
    next_id = torch.multinomial(probs, 1)
    idx = torch.cat([idx, next_id.unsqueeze(0)], dim=1)

# ── decode and print ────────────────────────────
print(enc.decode(idx[0].tolist()))
