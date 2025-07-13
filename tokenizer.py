# tokenizer.py
import os
import tiktoken
import numpy as np
from tqdm import tqdm

def prepare_tokenizer_and_bins(
    input_path, 
    output_dir, 
    vocab_model="gpt2", 
    train_split=0.9, 
    block_size=256
):
    print("🔠 Loading tokenizer...")
    enc = tiktoken.get_encoding(vocab_model)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"📄 Read {len(data)} characters")

    tokens = enc.encode(data)
    print(f"🔢 Encoded to {len(tokens)} tokens")

    split_idx = int(len(tokens) * train_split)
    train_ids = tokens[:split_idx]
    val_ids = tokens[split_idx:]

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "train.bin")
    val_out = os.path.join(output_dir, "val.bin")

    train_ids.tofile(train_out)
    val_ids.tofile(val_out)

    meta = {
        'vocab_size': enc.n_vocab,
        'block_size': block_size,
        'tokenizer': vocab_model
    }

    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        import pickle
        pickle.dump(meta, f)

    print("✅ Tokenizer finished. Files created:")
    print(f"  - {train_out} ({len(train_ids)} tokens)")
    print(f"  - {val_out} ({len(val_ids)} tokens)")
    print(f"  - meta.pkl with vocab and block_size")

if __name__ == "__main__":
    prepare_tokenizer_and_bins(
        input_path="data/multitask/input.txt",
        output_dir="data/multitask/",
        vocab_model="gpt2",
        block_size=256
    )
