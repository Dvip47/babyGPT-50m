# sample.py
# ------------------------------------------------------------
# Minimal inference script for your BabyGPT‑50M checkpoint.
# ------------------------------------------------------------
import torch, pickle, tiktoken
from types import MethodType

# ---------- 1.  Load model + checkpoint ---------------------
from model import GPTConfig, GPT            # <-- make sure model.py exists

ckpt_path = "out/babygpt_50m/ckpt.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

config = GPTConfig(**ckpt["model_args"])
model  = GPT(config)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# ---------- 2.  Add a generate() helper if GPT lacks one ----
@torch.no_grad()
def _generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.config.block_size:]  # crop context length
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_tok), dim=1)
    return idx

# attach only if not already present
if not hasattr(GPT, "generate"):
    model.generate = MethodType(_generate, model)

# ---------- 3.  Tokenizer -----------------------------------
enc = tiktoken.get_encoding("gpt2")         # same tokenizer used at training

def encode(text: str):  return enc.encode_ordinary(text)
def decode(tokens):     return enc.decode(tokens)

# ---------- 4.  Prompt & Generate ---------------------------
prompt = "[EMAIL]\nQ: Follow‑up mail after job interview?\nA:"

idx = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
out = model.generate(idx, max_new_tokens=120)[0].tolist()
print(decode(out))
