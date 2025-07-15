import torch
import gradio as gr
import pickle
import tiktoken
from model import GPT, GPTConfig
from types import MethodType

# Load checkpoint
ckpt = torch.load("out/babygpt_50m/ckpt.pt", map_location="cpu")
config = GPTConfig(**ckpt["model_args"])
model = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()

# Add .generate method if not available
@torch.no_grad()
def generate_fn(self, idx, max_new_tokens=100, temperature=0.5, top_k=40):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("Inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx
if not hasattr(model, "generate"):
    model.generate = MethodType(generate_fn, model)

# Tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode_ordinary(s)
decode = enc.decode

# Main generation function
def generate_text(prompt, max_tokens=80, temperature=0.5, top_k=40):
    if not prompt.strip().endswith("A:"):
        prompt += "\nA:"
    idx = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
    out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())

# Gradio UI
gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="[EMAIL]\\nQ: Follow-up mail after interview?\nA:"),
        gr.Slider(10, 200, value=80, step=10, label="Max New Tokens"),
        gr.Slider(0.1, 1.2, value=0.5, step=0.1, label="Temperature"),
        gr.Slider(10, 100, value=40, step=10, label="Top-K"),
    ],
    outputs=gr.Textbox(label="Generated Answer"),
    title="🍼 BabyGPT - Gradio Web UI",
    description="Type a prompt like: [CODE]\\nQ: Reverse a list in Python?\\nA:"
).launch()
