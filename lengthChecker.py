#!/usr/bin/env python3
"""
lengthChecker.py
----------------------------------
Count Q‑A blocks in a BabyGPT training file and
report only the whitelisted tags.

Assumed block format:

[TAG]
Q: ...
A: ...

(blank line)
"""

import sys
import re
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────
DEFAULT_FILE = Path("data/multitask/input.txt")

ALLOWED_TAGS = {
    "EN_GK", "CODE", "HI_GK", "UPSC",
    "WHATSAPP", "YOUTUBE", "RESUME", "EMAIL",
    "STARTUP", "ENGLISH", "COLLEGE"
}

TAG_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]$")   # e.g. [EN_GK]

# ─────────────────────────────────────────────────────────────
# 2. Main logic
# ─────────────────────────────────────────────────────────────
def main(file_path: Path):
    if not file_path.exists():
        sys.exit(f"❌ File not found: {file_path}")

    raw = file_path.read_text(encoding="utf-8").strip()
    blocks = raw.split("\n\n")          # blank line separates blocks
    tag_counter = Counter()

    for block in blocks:
        if not block.strip():
            continue                    # skip empty split artefacts

        first_line = block.strip().splitlines()[0].strip()
        m = TAG_RE.match(first_line)
        if not m:
            continue                    # malformed tag → ignore block

        tag = m.group(1)
        if tag not in ALLOWED_TAGS:
            continue                    # not whitelisted → skip block

        tag_counter[tag] += 1

    total = sum(tag_counter.values())
    print(f"📊  Total Q‑A blocks: {total}\n")
    for tag in sorted(ALLOWED_TAGS):
        if tag in tag_counter:
            print(f"  {tag:<10}: {tag_counter[tag]}")

    print(f"\n✅  Total questions counted: {total}")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    main(path)
