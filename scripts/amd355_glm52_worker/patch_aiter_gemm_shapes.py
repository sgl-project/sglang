#!/usr/bin/env python3
"""Patch AITER BF16 GEMM tuned config with missing K=6144 shapes.

After AITER startup, the glm5_bf16_tuned_gemm.csv is merged into
/tmp/aiter_configs/bf16_tuned_gemm.csv. This script appends missing
M values for K=6144 using the appropriate template:
  - N=256: flydsl kernel template (actual performance improvement)
  - N=32:  torch fallback template (suppresses warning only)

Run AFTER container startup, BEFORE first inference request.
"""
import csv
import sys

CONFIG_FILE = "/tmp/aiter_configs/bf16_tuned_gemm.csv"

with open(CONFIG_FILE) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

existing = set()
flydsl_256_small = None  # M=256 template (for M < 1000)
flydsl_256_large = None  # M=16384 template (for M >= 1000)
torch_32_small = None
torch_32_large = None

for row in rows:
    try:
        M = int(row[2]); N = int(row[3]); K = int(row[4])
    except (IndexError, ValueError):
        continue
    existing.add((M, N, K))
    if K == 6144:
        if N == 256 and row[10] == "flydsl":
            if M == 256: flydsl_256_small = row
            if M == 16384: flydsl_256_large = row
        if N == 32 and row[10] == "torch":
            if M == 256: torch_32_small = row
            if M == 16384: torch_32_large = row

if flydsl_256_small is None:
    print("[ERROR] No flydsl template found for N=256 K=6144")
    sys.exit(1)

added = 0
for M in range(1, 50001):
    # N=256: use flydsl kernel (performance improvement)
    if (M, 256, 6144) not in existing:
        tmpl = flydsl_256_small if M < 1000 else (flydsl_256_large or flydsl_256_small)
        new_row = tmpl.copy()
        new_row[2] = str(M)
        rows.append(new_row)
        added += 1
    # N=32: use torch fallback (warning suppression only)
    if (M, 32, 6144) not in existing:
        tmpl = torch_32_small if M < 1000 else (torch_32_large or torch_32_small)
        if tmpl is None:
            continue
        new_row = tmpl.copy()
        new_row[2] = str(M)
        rows.append(new_row)
        added += 1

with open(CONFIG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"[DONE] Added {added} entries. Total: {len(rows)}")
print(f"  N=256 K=6144: flydsl kernel (perf improvement)")
print(f"  N=32  K=6144: torch fallback (warning suppression)")
