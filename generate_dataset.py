    # generate_dataset.py

    from pathlib import Path
    import csv

    # ─────────────────────────────────────────────────────────────
    # 1. Hugging Face sources
    #    Tuple layout:
    #    (dataset_id, split, tag, q_key, a_key [, config_name])
    # ─────────────────────────────────────────────────────────────
    HF_DATASETS = [
        ("squad",                   "train",      "EN_GK", "question", "answers.text"),
        ("squad",                   "validation", "EN_GK", "question", "answers.text"),

        # Hindi subset: we pass config="hi" when calling load_dataset
        ("ai4bharat/indicqa",       "train",      "HI_GK", "question", "answer", "hi"),

        ("iamtarun/upsc-csat",      "train",      "UPSC",  "question", "correct_answer"),

        # Code snippets (no answer field)
        ("codeparrot/github-code",  "train",      "CODE",  "code",     None),
    ]

    # ─────────────────────────────────────────────────────────────
    # 2. Optional manual CSV directory (kept but commented)
    # ─────────────────────────────────────────────────────────────
    # MANUAL_DIR = Path("data/manual")  # CSVs with columns: tag,question,answer

    # ─────────────────────────────────────────────────────────────
    # 3. Output file
    # ─────────────────────────────────────────────────────────────
    OUTPUT = Path("data/multitask/input.txt")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    def write_block(f, tag, prompt, answer=""):
        """Write one block in BabyGPT format."""
        if tag == "CODE":
            f.write(f"[{tag}]\n{prompt.strip()}\n\n")
        else:
            f.write(f"[{tag}]\nQ: {prompt.strip()}\nA: {answer.strip()}\n\n")

    # ─────────────────────────────────────────────────────────────
    def build(max_per_source=None):
        """Combine all sources into input.txt."""
        try:
            from datasets import load_dataset
            HF_OK = True
        except ModuleNotFoundError:
            HF_OK = False
            print("⚠️  datasets library not installed – skipping HF sources.")

        with OUTPUT.open("w", encoding="utf-8") as out:

            # 1️⃣ Hugging Face loop
            if HF_OK:
                for entry in HF_DATASETS:
                    if len(entry) == 6:
                        ds_id, split, tag, q_key, a_key, config = entry
                    else:
                        ds_id, split, tag, q_key, a_key = entry
                        config = None

                    cfg_msg = f" ({config})" if config else ""
                    print(f"🔹 Loading {ds_id}:{split}{cfg_msg} …")

                    try:
                        ds = load_dataset(ds_id, config, split=split) if config else load_dataset(ds_id, split=split)
                    except Exception as e:
                        print(f"⚠️  Could not load {ds_id}:{split}: {e}")
                        continue

                    for i, row in enumerate(ds):
                        if max_per_source and i >= max_per_source:
                            break

                        prompt = row[q_key]
                        if a_key:  # Q‑A style
                            ans = row
                            for k in a_key.split("."):
                                ans = ans[k]
                            ans = ans[0] if isinstance(ans, list) else ans
                            write_block(out, tag, prompt, ans)
                        else:      # CODE snippet
                            write_block(out, tag, prompt)

            # 2️⃣ Manual CSVs (uncomment if needed)
            # if MANUAL_DIR.exists():
            #     for csv_file in MANUAL_DIR.glob("*.csv"):
            #         print(f"🔹 Reading {csv_file}")
            #         with csv_file.open(newline='', encoding='utf-8') as f:
            #             reader = csv.DictReader(f)
            #             for i, row in enumerate(reader):
            #                 if max_per_source and i >= max_per_source:
            #                     break
            #                 write_block(out,
            #                             row["tag"].strip().upper(),
            #                             row["question"],
            #                             row.get("answer", ""))

        print(f"✅  Dataset written to {OUTPUT}")

    # ─────────────────────────────────────────────────────────────
    if __name__ == "__main__":
        build(max_per_source=100000)   # set None for full dataset
