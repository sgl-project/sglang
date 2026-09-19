#!/usr/bin/env python3
"""Patch dsa_backend.py: fix AssertionError in draft_extend_v2 AND extend paths.
During EAGLE/NEXTN draft-extend, extend_prefix_lens_cpu may be None.
Set defaults instead of asserting.
Also patch dsa_indexer.py: fix seq_lens_cpu None during draft CG capture.
"""
import re

# ---- Patch 1: dsa_backend.py - replace ALL assert blocks with "All of them must not be None" ----
FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"

with open(FILE, "r") as f:
    content = f.read()

old_pattern = re.compile(
    r'assert \(\s*'
    r'forward_batch\.extend_seq_lens_cpu is not None\s*'
    r'and forward_batch\.extend_seq_lens is not None\s*'
    r'and forward_batch\.extend_prefix_lens_cpu is not None\s*'
    r'\), "All of them must not be None"',
    re.MULTILINE
)

replacement = (
    "if forward_batch.extend_seq_lens_cpu is None:\n"
    "                forward_batch.extend_seq_lens_cpu = forward_batch.extend_seq_lens.cpu() if forward_batch.extend_seq_lens is not None else torch.zeros(forward_batch.batch_size, dtype=torch.int32)\n"
    "            if forward_batch.extend_prefix_lens_cpu is None:\n"
    "                forward_batch.extend_prefix_lens_cpu = torch.zeros(forward_batch.batch_size, dtype=torch.int32)"
)

new_content, count = old_pattern.subn(replacement, content)

if count > 0:
    with open(FILE, "w") as f:
        f.write(new_content)
    print(f"PATCHED: dsa_backend.py - {count} assertion(s) replaced with safe defaults")
else:
    if "if forward_batch.extend_seq_lens_cpu is None:" in content:
        print("dsa_backend.py: already patched")
    else:
        print("WARNING: dsa_backend.py pattern not found")

# ---- Patch 2: dsa_indexer.py - fix seq_lens_cpu None during draft CG ----
FILE2 = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"

with open(FILE2, "r") as f:
    lines = f.readlines()

patched2 = False
new_lines2 = []
for line in lines:
    stripped = line.lstrip()
    if stripped == "assert forward_batch.seq_lens_cpu is not None\n":
        indent = line[:len(line) - len(stripped)]
        new_lines2.append(indent + "if forward_batch.seq_lens_cpu is None:\n")
        new_lines2.append(indent + "    forward_batch.seq_lens_cpu = forward_batch.seq_lens.cpu() if forward_batch.seq_lens is not None else torch.ones(forward_batch.batch_size, dtype=torch.int32)\n")
        patched2 = True
    else:
        new_lines2.append(line)

if patched2:
    with open(FILE2, "w") as f:
        f.writelines(new_lines2)
    print("PATCHED: dsa_indexer.py seq_lens_cpu None fix applied")
else:
    content2 = "".join(lines)
    if "if forward_batch.seq_lens_cpu is None:" in content2:
        print("dsa_indexer.py: already patched")
    else:
        print("WARNING: dsa_indexer.py pattern not found")
