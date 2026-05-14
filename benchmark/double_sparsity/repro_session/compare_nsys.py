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

Attribution caveats:
* `nsys stats --report cuda_gpu_kern_sum` reports CUDA-stream-time
  (i.e. how long each kernel kept its stream busy). It is **structural**
  evidence — "this kernel ran N times for T ns of stream time" — not
  wall-clock attribution. Captured CUDA graphs replay many kernels
  concurrently across streams, so summed GPU time can exceed wall-clock
  even when the workload runs faster. Use this diff to ask "did the
  legacy path run?" or "did `torch.topk` decompose into N CUB kernels?",
  NOT "did kernel X cost Y ms of wall-clock".
* Kernel-name aggregation preserves the full `ns1::ns2::name`
  identifier so that ATen / CUB / topk / FA kernels separate cleanly.
  Template params and call args are stripped; trailing `_object_at_*`
  hashes from flashinfer normalizer factories are stripped too.
"""

import argparse
import csv
import re
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
        rows.append((time_pct, tot_ns, inst, avg_ns, name_raw))
    return rows


def canonicalize(name_raw: str) -> str:
    """Reduce a demangled kernel name to a stable aggregation key.

    Goals:
    * Drop the leading `void ` return-type prefix (uninformative).
    * Drop the function-call argument list at the trailing `(...)`.
    * Drop all template parameters `<...>` (matched balanced).
    * Drop the `_object_at_<hash>...` suffix flashinfer/jit factories use.
    * Preserve `ns1::ns2::name` so ATen subspecialties (e.g. `mbtopk`,
      `cub::detail::scan_by_key`) are visible.

    Examples:
      "void at::native::mbtopk::computeBlockDigitCounts<float, ...>(...)"
        → "at::native::mbtopk::computeBlockDigitCounts"
      "void at_cuda_detail::cub::detail::scan_by_key::DeviceScanByKeyKernel<...>(...)"
        → "at_cuda_detail::cub::detail::scan_by_key::DeviceScanByKeyKernel"
      "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<...>>>(...)"
        → "cutlass::device_kernel" (template payload dropped on purpose;
          if FA-specific attribution is needed, peek the template
          payload below)
      "kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensor..."
        → "kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel"
    """
    s = name_raw.strip()
    # 1) drop leading `void ` (sometimes `const void `, etc.)
    s = re.sub(r"^(const\s+)?void\s+", "", s)
    # 2) drop _object_at_<hash> suffix used by flashinfer jit factories.
    #    The hash payload looks like "_object_at__tensorptr..._0".
    s = re.sub(r"_object_at_.*$", "", s)
    # 3) strip balanced template params `<...>` (might be nested).
    out = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            if depth > 0:
                depth -= 1
            continue
        if depth == 0:
            out.append(ch)
    s = "".join(out).strip()
    # 4) strip the trailing call-argument list.
    paren = s.find("(")
    if paren >= 0:
        s = s[:paren]
    s = s.strip()
    # 5) collapse the FA3 template-stripped form to something readable.
    #    "cutlass::device_kernel" hides whether it's FlashAttnFwdSm90 vs
    #    a generic gemm. We peek `flash::FlashAttn` / `flash::FlashAttnSm`
    #    out of the raw template payload to keep that visible.
    if s == "cutlass::device_kernel" and "FlashAttnFwd" in name_raw:
        return "cutlass::device_kernel:FlashAttnFwd"
    if s == "cutlass::device_kernel" and "FlashAttnBwd" in name_raw:
        return "cutlass::device_kernel:FlashAttnBwd"
    return s


parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
parser.add_argument("off_report", help="DS-off nsys-rep")
parser.add_argument("on_report", help="DS-on nsys-rep")
args = parser.parse_args()

off = load(args.off_report)
on = load(args.on_report)


def aggregate(rows):
    agg = {}  # canonical key -> [tot_ns, inst, one_raw_sample]
    for tp, t, i, a, r in rows:
        k = canonicalize(r)
        if k not in agg:
            agg[k] = [0, 0, r]
        agg[k][0] += t
        agg[k][1] += i
    return agg


agg_off = aggregate(off)
agg_on = aggregate(on)
all_keys = set(agg_off) | set(agg_on)

tot_off = sum(v[0] for v in agg_off.values())
tot_on = sum(v[0] for v in agg_on.values())

# Kernels exclusive to DS-on (pure DS-introduced work)
ds_only = sorted(
    [
        (k, agg_on[k][0], agg_on[k][1])
        for k in all_keys
        if k in agg_on and k not in agg_off
    ],
    key=lambda x: -x[1],
)
# Kernels in both legs.
shared = sorted(
    [
        (k, agg_off[k][0], agg_off[k][1], agg_on[k][0], agg_on[k][1])
        for k in all_keys
        if k in agg_off and k in agg_on
    ],
    key=lambda x: -(x[3] + x[1]),
)
# Kernels exclusive to DS-off.
off_only = sorted(
    [
        (k, agg_off[k][0], agg_off[k][1])
        for k in all_keys
        if k in agg_off and k not in agg_on
    ],
    key=lambda x: -x[1],
)

print("NOTE: nsys cuda_gpu_kern_sum is per-stream kernel time, not wall-clock.")
print("      Use this for STRUCTURAL evidence (which kernels ran, how often).")
print("      Do not attribute wall-clock TBT differences to GPU-time deltas here.")
print()
print(
    f"DS-off total GPU time:  {tot_off / 1e9:7.2f} s   ({len(agg_off)} unique kernels)"
)
print(f"DS-on  total GPU time:  {tot_on / 1e9:7.2f} s   ({len(agg_on)} unique kernels)")
print(f"DS-on  / DS-off total:  {tot_on / tot_off:7.2f}x")
print()

print("=" * 120)
print("KERNELS EXCLUSIVE TO DS-ON (pure DS overhead — would not exist without DS)")
print("=" * 120)
print(f"  {'Kernel':80s} {'GPU time':>12s} {'% on':>7s} {'inst':>9s}")
ds_only_total = 0
for k, t, i in ds_only[:40]:
    ds_only_total += t
    print(f"  {k[:78]:80s} {t / 1e9:8.3f} s   {t / tot_on * 100:5.2f}% {i:9d}")
print(
    f"  {'... ALL DS-only kernels':80s} {ds_only_total / 1e9:8.3f} s   "
    f"{ds_only_total / tot_on * 100:5.2f}%"
)
print()

print("=" * 120)
print("SHARED KERNELS (run in both modes)")
print("=" * 120)
print(
    f"  {'Kernel':70s} {'off time':>10s} {'on time':>10s} {'ratio':>7s} "
    f"{'off inst':>9s} {'on inst':>9s}"
)
shared_off_t = 0
shared_on_t = 0
for k, t_off, i_off, t_on, i_on in shared[:30]:
    shared_off_t += t_off
    shared_on_t += t_on
    ratio = t_on / t_off if t_off > 0 else 0
    print(
        f"  {k[:68]:70s} {t_off / 1e9:6.3f} s   {t_on / 1e9:6.3f} s   "
        f"{ratio:5.2f}x {i_off:9d} {i_on:9d}"
    )
print(
    f"  {'... ALL shared kernels':70s} {shared_off_t / 1e9:6.3f} s   "
    f"{shared_on_t / 1e9:6.3f} s   "
    f"{(shared_on_t / shared_off_t) if shared_off_t > 0 else 0:5.2f}x"
)
print()

print("=" * 120)
print("KERNELS EXCLUSIVE TO DS-OFF (FA3 variants the sparse path doesn't trigger)")
print("=" * 120)
print(f"  {'Kernel':80s} {'GPU time':>12s} {'% off':>7s} {'inst':>9s}")
off_only_total = 0
for k, t, i in off_only[:25]:
    off_only_total += t
    print(f"  {k[:78]:80s} {t / 1e9:8.3f} s   {t / tot_off * 100:5.2f}% {i:9d}")
print()

# Highlight the topk decomposition specifically — PLAN.md called this out.
print("=" * 120)
print("torch.topk DECOMPOSITION (key selector hotspot)")
print("=" * 120)
topk_patterns = (
    "mbtopk",  # at::native::mbtopk::*
    "scan_by_key::DeviceScan",
    "gatherTopK",
)
topk_total = 0
topk_rows = []
for k, v in agg_on.items():
    if any(p in k for p in topk_patterns):
        topk_total += v[0]
        topk_rows.append((k, v[0], v[1]))
topk_rows.sort(key=lambda x: -x[1])
for k, t, i in topk_rows:
    print(f"  {k[:78]:80s} {t / 1e9:8.3f} s   {t / tot_on * 100:5.2f}% {i:9d}")
print(
    f"  {'topk-related total (DS-on)':80s} {topk_total / 1e9:8.3f} s   "
    f"{topk_total / tot_on * 100:5.2f}% of DS-on GPU-time"
)
print()

print("=" * 120)
print("SUMMARY")
print("=" * 120)
print(f"  DS-off total GPU time:           {tot_off / 1e9:7.2f} s")
print(f"  DS-on  total GPU time:           {tot_on / 1e9:7.2f} s")
print(
    f"  Pure DS-only kernels:            {ds_only_total / 1e9:7.2f} s   "
    f"({ds_only_total / tot_on * 100:.2f}% of DS-on GPU time)"
)
print(
    f"  torch.topk decomposition (on):   {topk_total / 1e9:7.2f} s   "
    f"({topk_total / tot_on * 100:.2f}% of DS-on GPU time)"
)
print()
print("Reminder: these are summed per-stream kernel times. Wall-clock TBT for")
print("this workload is reported in the bench-decode JSON, not here.")
