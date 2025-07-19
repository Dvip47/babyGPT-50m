import torch, tiktoken, gradio as gr
from types import MethodType
from pathlib import Path
from model import GPT, GPTConfig

# ------------------------------------------------------------
# 1. Locate the latest checkpoint in out/babygpt_mac_50m
ckpt_dir = Path("out/babygpt_mac_50m")
ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
if not ckpts:
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
ckpt_path = ckpts[-1]          # newest
print(f"📦 Loading {ckpt_path.name}")

ckpt   = torch.load(ckpt_path, map_location="cpu")
config = GPTConfig(**ckpt["model_args"])
model  = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()

# ------------------------------------------------------------
# 2. Add generate() if needed
@torch.no_grad()
def generate_fn(self, idx, max_new_tokens=100, temperature=0.5, top_k=40):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        v, _    = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("Inf")
        probs   = torch.softmax(logits, dim=-1)
        next_t  = torch.multinomial(probs, num_samples=1)
        idx     = torch.cat((idx, next_t), dim=1)
    return idx

if not hasattr(model, "generate"):
    model.generate = MethodType(generate_fn, model)

# ------------------------------------------------------------
# 3. Tokenizer helpers
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)
decode = enc.decode

# ------------------------------------------------------------
# 4. Text generator for Gradio
def generate_text(prompt, max_tokens=80, temperature=0.5, top_k=40):
    if not prompt.strip().endswith("A:"):
        prompt = prompt.rstrip() + "\nA:"
    idx = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())

# ------------------------------------------------------------
# 5. Build UI
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=6, label="Prompt",
                   placeholder="[CODE]\nQ: Reverse a list in Python?\nA:"),
        gr.Slider(10, 200, value=80, step=10, label="Max New Tokens"),
        gr.Slider(0.1, 1.2, value=0.5, step=0.1, label="Temperature"),
        gr.Slider(10, 100, value=40, step=10, label="Top‑K"),
    ],
    outputs=gr.Textbox(lines=8, label="Generated Answer"),
    title="🍼 BabyGPT‑50M (Mac) – Gradio UI",
    description="Give a tag-based prompt, e.g.:\n[WHATSAPP]\nQ: GF ko good morning text bhejo.\nA:"
)

if __name__ == "__main__":
    demo.launch()
