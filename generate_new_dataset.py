from datasets import load_dataset
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 0. File‑paths
# ─────────────────────────────────────────────────────────────
INPUT_FILE  = Path("data/multitask/input-gemini.txt")
OUTPUT_FILE = INPUT_FILE          # append in‑place
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. Target sizes
# ─────────────────────────────────────────────────────────────
TARGET_QUOTA = dict(CODE=2000, HI_GK=1000, UPSC=1000)

# ─────────────────────────────────────────────────────────────
# 2. 100 % parquet / arrow sources
#    (no remote scripts ⇒ no trust_remote_code headaches)
# ─────────────────────────────────────────────────────────────
HF_DATASETS = [
    # 2 000 random code snippets (Python & friends, 5 M rows total)
    ("codeparrot/codeparrot-clean",           # id
     "train",                                 # split
     "CODE",                                  # tag
     "content",                               # question field
     None,                                    # answer field
     None,                                    # filter‑fn
     dict(streaming=True)),                   # load_dataset kwargs

    # 1 000 Hindi general‑knowledge / history Q‑A
    ("kaifahmad/indian-history-hindi-QA-3.4k",
     "train",
     "HI_GK",
     "Question",
     "Answer",
     None,
     {}),

    # 1 000 UPSC policy / exam FAQs (310 rows × repeat until 1000)
    ("prnv19/UPSC_FAQ",
     "train",
     "UPSC",
     "Question",
     "Answer",
     None,
     {}),
]

# ─────────────────────────────────────────────────────────────
# 3. Helpers
# ─────────────────────────────────────────────────────────────
def get_current_counts():
    counts  = {k: 0 for k in TARGET_QUOTA}
    seen_qs = set()
    if INPUT_FILE.exists():
        current_tag = None
        for line in INPUT_FILE.read_text(encoding="utf‑8").splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current_tag = line[1:-1]
            elif line.startswith("Q: ") and current_tag in counts:
                q = line[3:]
                if q not in seen_qs:
                    counts[current_tag] += 1
                    seen_qs.add(q)
    return counts, seen_qs


def write_block(fp, tag, q, a=""):
    """Uniform format for every tag."""
    fp.write(f"[{tag}]\nQ: {q.strip()}\nA: {a.strip()}\n\n")


# ─────────────────────────────────────────────────────────────
# 4. Main routine
# ─────────────────────────────────────────────────────────────
def generate_missing():
    current_counts, seen = get_current_counts()
    print("📊 Existing counts:", current_counts)

    with OUTPUT_FILE.open("a", encoding="utf‑8") as out:
        for ds_id, split, tag, qk, ak, filt_fn, ld_kwargs in HF_DATASETS:
            need = TARGET_QUOTA[tag] - current_counts[tag]
            if need <= 0:
                print(f"✅ {tag} already satisfied")
                continue

            print(f"🔹 Fetching {tag} → need {need}")
            try:
                ds = load_dataset(ds_id, split=split, **ld_kwargs)
            except Exception as e:
                print(f"⚠️  Could not load {ds_id}: {e}")
                continue

            added = 0
            for row in ds:
                if added >= need:
                    break
                if filt_fn and not filt_fn(row):
                    continue

                q = row.get(qk, "").strip()
                if not q or q in seen:
                    continue

                a = ""
                if ak:
                    a = row.get(ak, "")
                    if isinstance(a, list):
                        a = a[0] if a else ""

                write_block(out, tag, q, a)
                seen.add(q)
                added += 1

            current_counts[tag] += added
            print(f"   → added {added} new rows for {tag}")

    print("\n✅  All done – data appended to", OUTPUT_FILE)


if __name__ == "__main__":
    generate_missing()


# #!/usr/bin/env python3
# """
# generate_dataset_quota.py  —  DEDUP EDITION
# Build a BabyGPT training file with exact per‑tag quotas
# while ensuring every question (or code prompt) is unique.
# """

# from pathlib import Path
# from collections import defaultdict
# from random import shuffle, choice, randint
# from datasets import load_dataset
# import csv
# from tqdm import tqdm

# # ───── 1) Desired quotas ─────────────────────────────────
# QUOTA = {
#     "EN_GK"   : 10_000,
#     "CODE"    : 2_500,
#     "HI_GK"   : 1_000,
#     "UPSC"    : 1_000,
#     "WHATSAPP":   500,
#     "YOUTUBE" :   500,
#     "RESUME"  :   500,
#     "EMAIL"   :   500,
#     "STARTUP" :   500,
#     "ENGLISH" :   500,
# }

# # ───── 2) HuggingFace sources (tested, public) ───────────
# HF_SOURCES = [
#     ("squad", "train",      "EN_GK", "question", "answers.text", None),
#     ("squad", "validation", "EN_GK", "question", "answers.text", None),

#     ("mbpp", "train", "CODE", "text", None, None),

#     ("ai4bharat/indicqa", "train", "HI_GK", "question", "answer",
#         lambda r: r.get("lang", "") == "hi"),
#     ("iisc-adlearn/inquisitive-qahindi", "train", "HI_GK",
#         "question", "short_answer", None),

#     ("thepanacealab/cafps", "train", "UPSC", "question", "answer", None),
# ]

# # ───── 3) Mini template seeds for small tags ─────────────
# WHATSAPP_WISHES = [
#     ("Wish Happy Holi in English.", "Happy Holi! May your life be filled with vibrant colors and joy."),
#     ("Wish Happy Diwali in Hindi.", "दिवाली की हार्दिक शुभकामनाएँ! यह पर्व आपके जीवन में खुशियाँ लाए।"),
#     ("Wish Happy New Year to a friend.", "Happy New Year! May our friendship grow stronger in 2026."),
# ]

# YOUTUBE_IDEAS = [
#     "Vlog: Day in the Life of a Commerce Student",
#     "Python Quick‑Tips (60‑second shorts)",
#     "Street Food Review in Mumbai",
#     "Budget Travel Series (₹500 Per Day)",
# ]

# STARTUP_AREAS = [
#     "AI‑powered personal finance advisor for tier‑2 cities.",
#     "Hyperlocal grocery delivery on electric cycles.",
#     "Voice‑first English learning app for Hindi speakers.",
#     "Battery swap network for two‑wheelers.",
# ]

# EMAIL_TEMPLATES = [
#     ("Apology email to customer for delayed shipment",
#      "Subject: Apology for Delay in Delivery\n\nDear {name},\nWe regret the delay of your order #{id}. …"),
#     ("Leave application for family function",
#      "Subject: Leave Request (25–27 July)\n\nDear {manager},\nI would like to apply for leave …"),
# ]

# RESUME_TEMPLATES = [
#     ("Fresher software engineer resume summary",
#      "Enthusiastic Computer Science graduate proficient in Python, React, and SQL …"),
#     ("Cover letter for digital marketing intern",
#      "Dear Hiring Manager,\nI am writing to apply for the Digital Marketing Internship at …"),
# ]

# ENGLISH_DRILLS = [
#     ("Correct this sentence: \"She don't like ice‑cream.\"", "She doesn't like ice‑cream."),
#     ("Translate to English: \"मुझे पढ़ना पसंद है।\"", "I like to read."),
# ]

# # ───── 4) Output  ────────────────────────────────────────
# OUT = Path("data/multitask/input.txt")
# OUT.parent.mkdir(parents=True, exist_ok=True)

# # ───── Helpers ───────────────────────────────────────────
# seen_questions: set[str] = set()              # global dedup set
# tag_counts      = defaultdict(int)            # live quota tracker

# def unique(question: str) -> bool:
#     """Return True if question is new; else False (duplicate)."""
#     q = question.strip()
#     if q in seen_questions:
#         return False
#     seen_questions.add(q)
#     return True

# def write_block(file, tag: str, q: str, a: str = ""):
#     file.write(f"[{tag}]\n")
#     if tag == "CODE":
#         file.write(q.strip() + "\n\n")
#     else:
#         file.write(f"Q: {q.strip()}\nA: {a.strip()}\n\n")

# # ───── 5) Add HuggingFace data (with dedup) ─────────────
# def load_hf_blocks(out):
#     for ds_id, split, tag, qk, ak, filtr in HF_SOURCES:
#         if tag_counts[tag] >= QUOTA[tag]:
#             continue
#         need = QUOTA[tag] - tag_counts[tag]
#         print(f"🔹 HF → {ds_id}:{split}  (need {need})")
#         try:
#             ds = load_dataset(ds_id, split=split)
#         except Exception as e:
#             print(f"   ⚠️  Skip {ds_id}: {e}")
#             continue

#         # Shuffle once for diversity
#         rows = ds.shuffle(seed=42)

#         for row in rows:
#             if tag_counts[tag] >= QUOTA[tag]:
#                 break
#             if filtr and not filtr(row):
#                 continue

#             q = row[qk]
#             if not unique(q):
#                 continue

#             # Resolve answer
#             a = ""
#             if ak:
#                 ref = row
#                 for key in ak.split("."):
#                     ref = ref[key]
#                 a = ref[0] if isinstance(ref, list) else ref

#             write_block(out, tag, q, a)
#             tag_counts[tag] += 1

# # ───── 6) Template generator with dedup ─────────────────
# def fill_from_templates(out, tag, pairs):
#     while tag_counts[tag] < QUOTA[tag]:
#         q, a = pairs[tag_counts[tag] % len(pairs)]  # simple cycling
#         # personalise placeholders
#         a_mod = (a.replace("{name}", choice(["Akash", "Sana", "Ravi"]))
#                    .replace("{manager}", choice(["Sir", "Ma'am"]))
#                    .replace("{id}", str(randint(1000, 9999))))
#         if not unique(q):
#             # modify question slightly to remain unique
#             q = q + f" ({randint(1,9999)})"
#         write_block(out, tag, q, a_mod)
#         tag_counts[tag] += 1

# # ───── 7) Main routine ──────────────────────────────────
# def main():
#     with OUT.open("w", encoding="utf-8") as out:
#         # A) Big HF quotas
#         load_hf_blocks(out)

#         # B) Small template quotas
#         fill_from_templates(out, "WHATSAPP", WHATSAPP_WISHES)
#         fill_from_templates(out, "YOUTUBE",
#             [(f"YouTube channel idea: {idea}", idea) for idea in YOUTUBE_IDEAS])
#         fill_from_templates(out, "STARTUP",
#             [(f"Startup idea using AI: {idea}", idea) for idea in STARTUP_AREAS])
#         fill_from_templates(out, "EMAIL", EMAIL_TEMPLATES)
#         fill_from_templates(out, "RESUME", RESUME_TEMPLATES)
#         fill_from_templates(out, "ENGLISH", ENGLISH_DRILLS)

#     # Summary
#     print("\n✅ Final counts (deduplicated)")
#     for tg in QUOTA:
#         print(f"  {tg:<10}: {tag_counts[tg]} / {QUOTA[tg]}")
#     print(f"\n🚀 TOTAL blocks: {sum(tag_counts.values()):,} (all unique)")

# if __name__ == "__main__":
#     main()





# # #!/usr/bin/env python3
# # # generate_data.py  (with block counter)

# # from datasets import load_dataset
# # from pathlib import Path
# # import csv, pandas as pd

# # # ─────────────────────────────
# # # HF datasets: (id, split, tag, q_key, a_key, optional_filter_fn)
# # # ─────────────────────────────
# # HF_DATASETS = [
# #     ("squad", "train",        "EN_GK", "question", "answers.text"),
# #     ("squad", "validation",   "EN_GK", "question", "answers.text"),
# #     ("ai4bharat/indicqa", "train", "HI_GK", "question", "answer",
# #         lambda r: r.get("lang", "") == "hi"),
# #     ("iamtarun/upsc-csat", "train", "UPSC", "question", "correct_answer"),
# #     ("codeparrot/github-code", "train", "CODE", "code", None),
# # ]

# # # ─────────────────────────────
# # # Kaggle CSV sources (manually downloaded)
# # # ─────────────────────────────
# # KAGGLE_CSV_FILES = [
# #     ("data/kaggle/resume_samples.csv",  "RESUME"),
# #     ("data/kaggle/youtube_ideas.csv",   "YOUTUBE"),
# #     ("data/kaggle/email_templates.csv", "EMAIL"),
# # ]

# # # ─────────────────────────────
# # # Manual CSVs (optional)
# # # ─────────────────────────────
# # MANUAL_DIR = Path("data/manual")

# # # ─────────────────────────────
# # # Final output
# # # ─────────────────────────────
# # OUTPUT = Path("data/multitask/input-gemini.txt")
# # OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# # def write(f, tag, q, a=""):
# #     """Write one block and return 1 (so we can sum later)."""
# #     q, a = str(q).strip(), str(a).strip()
# #     if tag == "CODE":
# #         f.write(f"[{tag}]\n{q}\n\n")
# #     else:
# #         f.write(f"[{tag}]\nQ: {q}\nA: {a}\n\n")
# #     return 1


# # def generate(max_per=5000):
# #     total = 0  # ← block counter
# #     with OUTPUT.open("w", encoding="utf-8") as out:

# #         # 1️⃣ Hugging Face datasets
# #         for ds_id, split, tag, qk, ak, *filter_fn in HF_DATASETS:
# #             print(f"🔹 HF: {ds_id}:{split}")
# #             try:
# #                 ds = load_dataset(ds_id, split=split)
# #             except Exception as e:
# #                 print(f"  ⚠️ Skipped {ds_id}: {e}")
# #                 continue
# #             filt = filter_fn[0] if filter_fn else None
# #             for i, r in enumerate(ds):
# #                 if max_per and i >= max_per:
# #                     break
# #                 if filt and not filt(r):
# #                     continue
# #                 q = r.get(qk)
# #                 if ak:
# #                     a = r
# #                     for k in ak.split("."):
# #                         a = a[k]
# #                     a = a[0] if isinstance(a, list) else a
# #                     total += write(out, tag, q, a)
# #                 else:
# #                     total += write(out, tag, q)

# #         # 2️⃣ Kaggle CSVs
# #         for path, tag in KAGGLE_CSV_FILES:
# #             try:
# #                 df = pd.read_csv(path)
# #                 print(f"🔹 Kaggle: {path}")
# #                 for i, row in df.iterrows():
# #                     if max_per and i >= max_per:
# #                         break
# #                     total += write(out, tag, row["question"], row.get("answer", ""))
# #             except Exception as e:
# #                 print(f"  ⚠️ Skipped Kaggle {path}: {e}")

# #         # 3️⃣ Manual CSVs
# #         if MANUAL_DIR.exists():
# #             for file in MANUAL_DIR.glob("*.csv"):
# #                 print(f"🔹 Manual: {file.name}")
# #                 with file.open("r", encoding="utf-8") as fcsv:
# #                     reader = csv.DictReader(fcsv)
# #                     for i, row in enumerate(reader):
# #                         if max_per and i >= max_per:
# #                             break
# #                         total += write(out, row["tag"].upper(),
# #                                        row["question"], row.get("answer", ""))

# #     print(f"\n✅ Dataset written to → {OUTPUT}  |  {total:,} blocks generated")


# # if __name__ == "__main__":
# #     generate(max_per=5000)


# # # # generate_data.py

# # # from datasets import load_dataset
# # # from pathlib import Path
# # # import csv, pandas as pd

# # # # ─────────────────────────────
# # # # HF datasets: (id, split, tag, q_key, a_key, optional_filter_fn)
# # # # ─────────────────────────────
# # # HF_DATASETS = [
# # #     ("squad", "train", "EN_GK", "question", "answers.text"),
# # #     ("squad", "validation", "EN_GK", "question", "answers.text"),
# # #     ("ai4bharat/indicqa", "train", "HI_GK", "question", "answer", lambda r: r.get("lang", "") == "hi"),
# # #     ("iamtarun/upsc-csat", "train", "UPSC", "question", "correct_answer"),
# # #     ("codeparrot/github-code", "train", "CODE", "code", None),
# # # ]

# # # # ─────────────────────────────
# # # # Kaggle CSV sources (manually downloaded)
# # # # ─────────────────────────────
# # # KAGGLE_CSV_FILES = [
# # #     ("data/kaggle/resume_samples.csv", "RESUME"),    # tag, file_path
# # #     ("data/kaggle/youtube_ideas.csv", "YOUTUBE"),
# # #     ("data/kaggle/email_templates.csv", "EMAIL"),
# # # ]

# # # # ─────────────────────────────
# # # # Manual CSVs (optional)
# # # # ─────────────────────────────
# # # MANUAL_DIR = Path("data/manual")  # must contain CSVs with tag,question,answer

# # # # ─────────────────────────────
# # # # Final output
# # # # ─────────────────────────────
# # # OUTPUT = Path("data/multitask/input.txt")
# # # OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# # # def write(f, tag, q, a=""):
# # #     q, a = q.strip(), a.strip()
# # #     if tag == "CODE":
# # #         f.write(f"[{tag}]\n{q}\n\n")
# # #     else:
# # #         f.write(f"[{tag}]\nQ: {q}\nA: {a}\n\n")

# # # def generate(max_per=5000):
# # #     with OUTPUT.open("w", encoding="utf-8") as out:

# # #         # 1️⃣ Hugging Face datasets
# # #         for ds_id, split, tag, qk, ak, *filter_fn in HF_DATASETS:
# # #             print(f"🔹 HF: {ds_id}:{split}")
# # #             try:
# # #                 ds = load_dataset(ds_id, split=split)
# # #             except Exception as e:
# # #                 print(f"⚠️ Skipped {ds_id}: {e}")
# # #                 continue
# # #             filt = filter_fn[0] if filter_fn else None
# # #             for i, r in enumerate(ds):
# # #                 if max_per and i >= max_per:
# # #                     break
# # #                 if filt and not filt(r): continue
# # #                 q = r.get(qk)
# # #                 if ak:
# # #                     a = r
# # #                     for k in ak.split("."):
# # #                         a = a[k]
# # #                     a = a[0] if isinstance(a, list) else a
# # #                     write(out, tag, q, a)
# # #                 else:
# # #                     write(out, tag, q)

# # #         # 2️⃣ Kaggle CSVs
# # #         for path, tag in KAGGLE_CSV_FILES:
# # #             try:
# # #                 df = pd.read_csv(path)
# # #                 print(f"🔹 Kaggle: {path}")
# # #                 for i, row in df.iterrows():
# # #                     if max_per and i >= max_per:
# # #                         break
# # #                     write(out, tag, row["question"], row.get("answer", ""))
# # #             except Exception as e:
# # #                 print(f"⚠️ Skipped Kaggle {path}: {e}")

# # #         # 3️⃣ Manual CSVs
# # #         if MANUAL_DIR.exists():
# # #             for file in MANUAL_DIR.glob("*.csv"):
# # #                 print(f"🔹 Manual: {file.name}")
# # #                 with file.open("r", encoding="utf-8") as fcsv:
# # #                     reader = csv.DictReader(fcsv)
# # #                     for i, row in enumerate(reader):
# # #                         if max_per and i >= max_per:
# # #                             break
# # #                         write(out, row["tag"].upper(), row["question"], row.get("answer", ""))

# # #     print(f"\n✅ Dataset written to → {OUTPUT}")

# # # # ─────────────────────────────
# # # if __name__ == "__main__":
# # #     generate(max_per=5000)
