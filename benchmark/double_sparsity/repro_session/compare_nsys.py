"""Side-by-side per-kernel CUDA-time comparison between two nsys-rep files.

Usage:
  PYTHONPATH=python python3 compare_nsys.py \\
      /workspace/nsys_reports/ds_native_off_131072_c32.nsys-rep \\
      /workspace/nsys_reports/ds_native_on_131072_c32.nsys-rep

Useful for proving "the captured graph replays the expected kernels":
DS-on should show the four native kernels (`_ds_native_score_kernel`,
`_ds_native_build_selected_physical_kernel`,
`_ds_native_sparse_attn_stage2_kernel`,
`_ds_native_sparse_attn_stage3_kernel`); DS-off should NOT. Conversely,
DS-on should NOT show `_ds_select_stage2_merge_kernel` /
`_ds_select_stage1_block_topk_kernel` (the legacy stage2/union path).
"""

import argparse
import csv
import subprocess


def load(path):
    cmd = [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        "--output",
        "-",
        path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
    rows = []
    for r in csv.reader(out.splitlines()):
        if not r or r[0] == "Time (%)":
            continue
        try:
            time_pct = float(r[0])
            tot_ns = int(r[1])
            inst = int(r[2])
            avg_ns = float(r[3])
        except Exception:
            continue
        name_raw = r[8]
        short = name_raw
        for sep in ["<", "(", ":", "_object_at"]:
            if sep in short:
                short = short.split(sep, 1)[0]
        short = short.strip()
        rows.append((time_pct, tot_ns, inst, avg_ns, short, name_raw))
    return rows


parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
parser.add_argument("off_report", help="DS-off nsys-rep")
parser.add_argument("on_report", help="DS-on nsys-rep")
args = parser.parse_args()

off = load(args.off_report)
on = load(args.on_report)

# Group by short name across both
all_keys = set()


def key_for(short, raw):
    """Aggregate canonical key — strip variant suffixes for matmuls."""
    return short


def aggregate(rows):
    agg = {}  # short -> [tot_ns, inst, raw_sample]
    for tp, t, i, a, s, r in rows:
        k = key_for(s, r)
        if k not in agg:
            agg[k] = [0, 0, r]
        agg[k][0] += t
        agg[k][1] += i
    return agg


agg_off = aggregate(off)
agg_on = aggregate(on)
all_keys = set(agg_off) | set(agg_on)

# Compute totals
tot_off = sum(v[0] for v in agg_off.values())
tot_on = sum(v[0] for v in agg_on.values())

# Find kernels exclusive to DS-on (pure DS overhead)
ds_only = sorted(
    [
        (k, agg_on[k][0], agg_on[k][1])
        for k in all_keys
        if k in agg_on and k not in agg_off
    ],
    key=lambda x: -x[1],
)
# Find kernels in both (shared workload)
shared = sorted(
    [
        (k, agg_off[k][0], agg_off[k][1], agg_on[k][0], agg_on[k][1])
        for k in all_keys
        if k in agg_off and k in agg_on
    ],
    key=lambda x: -(x[3] + x[1]),
)
# Kernels exclusive to DS-off (dense-only)
off_only = sorted(
    [
        (k, agg_off[k][0], agg_off[k][1])
        for k in all_keys
        if k in agg_off and k not in agg_on
    ],
    key=lambda x: -x[1],
)

print(f"DS-off total GPU time:  {tot_off/1e9:7.2f} s   ({len(agg_off)} unique kernels)")
print(f"DS-on  total GPU time:  {tot_on /1e9:7.2f} s   ({len(agg_on)} unique kernels)")
print(f"DS-on  / DS-off total:  {tot_on/tot_off:7.2f}x")
print()

print("=" * 110)
print("KERNELS EXCLUSIVE TO DS-ON (pure DS overhead — would not exist without DS)")
print("=" * 110)
print(f"  {'Kernel':70s} {'GPU time':>12s} {'% on':>7s} {'inst':>7s}")
ds_only_total = 0
for k, t, i in ds_only[:25]:
    ds_only_total += t
    print(f"  {k[:68]:70s} {t/1e9:8.3f} s   {t/tot_on*100:5.1f}% {i:7d}")
print(
    f"  {'... ALL DS-only kernels':70s} {ds_only_total/1e9:8.3f} s   {ds_only_total/tot_on*100:5.1f}%"
)
print()

print("=" * 110)
print(f"SHARED KERNELS (run in both modes; expected to be nearly identical wall-clock)")
print("=" * 110)
print(
    f"  {'Kernel':50s} {'off time':>10s} {'on time':>10s} {'ratio':>7s} {'off inst':>8s} {'on inst':>8s}"
)
shared_off_t = 0
shared_on_t = 0
for k, t_off, i_off, t_on, i_on in shared[:20]:
    shared_off_t += t_off
    shared_on_t += t_on
    ratio = t_on / t_off if t_off > 0 else 0
    print(
        f"  {k[:48]:50s} {t_off/1e9:6.3f} s   {t_on/1e9:6.3f} s   {ratio:5.2f}x {i_off:8d} {i_on:8d}"
    )
print(
    f"  {'... ALL shared kernels':50s} {shared_off_t/1e9:6.3f} s   {shared_on_t/1e9:6.3f} s   {shared_on_t/shared_off_t:5.2f}x"
)
print()

print("=" * 110)
print(
    f"KERNELS EXCLUSIVE TO DS-OFF (likely just FA3 variants the sparse path doesn't trigger)"
)
print("=" * 110)
off_only_total = 0
for k, t, i in off_only[:10]:
    off_only_total += t
    print(f"  {k[:68]:70s} {t/1e9:8.3f} s   {t/tot_off*100:5.1f}% {i:7d}")
print()

print("=" * 110)
print("SUMMARY")
print("=" * 110)
print(f"  DS-off total GPU time:           {tot_off/1e9:7.2f} s")
print(f"  DS-on  total GPU time:           {tot_on/1e9:7.2f} s")
print(
    f"  Shared workload (matmuls, FA3):  off={shared_off_t/1e9:.2f}s, on={shared_on_t/1e9:.2f}s  (delta={tot_on - tot_off - ds_only_total + off_only_total:+.0f}ns)"
)
print(
    f"  Pure DS overhead (DS-only):      {ds_only_total/1e9:7.2f} s   ({ds_only_total/tot_on*100:.1f}% of DS-on GPU time)"
)
print(f"  Excess GPU time (DS-on - DS-off):{(tot_on - tot_off)/1e9:7.2f} s")
print(
    f"  → DS-only kernels explain        {ds_only_total / (tot_on - tot_off) * 100:.1f}% of the GPU-time gap"
)
