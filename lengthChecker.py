#!/usr/bin/env python3
"""
lengthChecker.py
----------------------------------
Count how many Q‑A *blocks* (i.e. questions) are in a BabyGPT
training file.  It also prints a per‑TAG breakdown.

Assumed format for each block:

[EN_GK]
Q: ...text...
A: ...text...

(blank line)
"""

import sys
from collections import Counter
from pathlib import Path

# Default file if no argument provided
DEFAULT_FILE = Path("data/multitask/input.txt")

def main(file_path: Path):
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    raw = file_path.read_text(encoding="utf-8").strip()

    blocks = raw.split("\n\n")  # each block separated by a blank line
    tag_counter = Counter()

    for block in blocks:
        # Tag line is first non‑empty line, e.g. [EN_GK]
        first_line = block.strip().splitlines()[0].strip()
        if first_line.startswith("[") and first_line.endswith("]"):
            tag = first_line[1:-1]  # remove brackets
        else:
            tag = "UNKNOWN"
        tag_counter[tag] += 1

    total = sum(tag_counter.values())
    print(f"📊  Total Q‑A blocks: {total}\n")
    finaltotal=0
    for tag, count in tag_counter.most_common():
        print(f"  {tag:<10} : {count}")
        finaltotal+=count

    print(f"\n✅  Total blocks counted: {finaltotal}")

if __name__ == "__main__":
    # Allow:  python3 lengthChecker.py  <optional_path>
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    main(path)
