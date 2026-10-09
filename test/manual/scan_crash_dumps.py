"""Scan IDEDD exception dumps of the HcPreSinkhorn MTE fault.

File structure: [uint64 LE index_len][protobuf index][tensor data], with
data length == sum of the ten tensor sizes. The section ORDER inside the
data region is derived two ways:

  1. protobuf-aware: sibling messages in the index that carry one of the
     known tensor sizes as a varint field, taken in message order;
  2. schema-free fallback: byte ranges that are identical across ALL dumps
     must be the constant weight sections (hc_scale 12B / hc_base 96B) --
     their offsets pin the order.

A candidate order is accepted only if the weight sections then cluster into
<=4 distinct (scale, base) pairs across dumps (attn vs FFN call sites); a
wrong order scatters into one pair per file.

At the validated order this reports the cross-dump tiling/workspace
comparison plus fp32/a5a5 sanity per section.

Usage:
  python3 test/manual/scan_crash_dumps.py <exception_info files / globs>
"""

import argparse
import glob
import hashlib
import struct
import sys

LAYOUT_SIZES = {
    "mixes": 196608, "rsqrt": 8192, "hc_scale": 12, "hc_base": 96,
    "x": 67108864, "y": 16777216, "post": 32768, "comb": 131072,
    "workspace": 32, "tiling": 144,
}
TENSORS_SUM = sum(LAYOUT_SIZES.values())
A5X4 = b"\xa5\xa5\xa5\xa5"
SIZE_TO_NAME = {v: k for k, v in LAYOUT_SIZES.items()}


def read_varint(buf, i):
    val, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def pb_fields(buf):
    """Yield (field_no, wire_type, value) for one protobuf message."""
    i = 0
    while i < len(buf):
        try:
            key, i = read_varint(buf, i)
        except IndexError:
            return
        fn, wt = key >> 3, key & 7
        try:
            if wt == 0:
                val, i = read_varint(buf, i)
            elif wt == 2:
                ln, i = read_varint(buf, i)
                val = buf[i:i + ln]
                i += ln
            elif wt == 5:
                val = buf[i:i + 4]
                i += 4
            elif wt == 1:
                val = buf[i:i + 8]
                i += 8
            else:
                return
        except IndexError:
            return
        yield fn, wt, val


def size_varints(msg):
    return [v for _, wt, v in pb_fields(msg) if wt == 0 and v in SIZE_TO_NAME]


def orders_from_index(index):
    """Candidate section orders from sibling index messages carrying sizes."""
    found = []

    def visit(msg):
        entries = [v for _, wt, v in pb_fields(msg) if wt == 2]
        for e in entries:
            visit(e)
        hits = []
        for e in entries:
            sizes = size_varints(e)
            if not sizes:
                for sub in (v for _, wt, v in pb_fields(e) if wt == 2):
                    sizes += size_varints(sub)
            if sizes:
                hits.append(sizes[-1])
        if len(hits) >= 5:
            found.append(hits)

    visit(index)
    out = []
    for hits in found:
        if sorted(hits) == sorted(LAYOUT_SIZES.values()):
            out.append([SIZE_TO_NAME[s] for s in hits])
    return out


def orders_from_constancy(dumps, base_data=None):
    """Schema-free: the weight sections (hc_scale 12B / hc_base 96B) are
    byte-constant WITHIN a call-site group (attn vs FFN) and differ across
    groups. So a weight island is a byte range over which the same FILE
    SUBSET agrees with file0 throughout (constant agreement count) and the
    run length is exactly 12 or 96. Reconstruct the section order whose
    cumulative offsets pin those islands; each placed section must also pass
    its content-type check (bf16 / fp32 / small-int tiling), which makes the
    order unique since all section sizes differ."""
    keys = list(dumps)
    if base_data is None:
        base_data = dumps[keys[0]]
    base = dumps[keys[0]]
    n = len(keys)
    try:
        import numpy as np

        arr = np.frombuffer(base, dtype=np.uint8)
        cnt = np.ones(len(base), dtype=np.int32)
        for k in keys[1:]:
            cnt += (arr == np.frombuffer(dumps[k], dtype=np.uint8)).astype(np.int32)
        cnt_list = cnt.tolist()
    except ImportError:
        cnt_list = [1] * len(base)
        for k in keys[1:]:
            other = dumps[k]
            for j in range(len(base)):
                if base[j] == other[j]:
                    cnt_list[j] += 1
    # maximal runs where cnt stays constant (same agreeing subset) and the
    # subset is a real minority/majority island, not the whole jittery mass
    runs = []
    start = 0
    for j in range(1, len(cnt_list) + 1):
        if j == len(cnt_list) or cnt_list[j] != cnt_list[start]:
            ln = j - start
            if ln >= 12 and 3 <= cnt_list[start] <= n - 1 or (ln >= 12 and cnt_list[start] == n):
                runs.append((start, ln, cnt_list[start]))
            start = j
    pinned = [
        (off, ln)
        for off, ln, _ in runs
        if ln in set(LAYOUT_SIZES.values())
    ]
    if len(pinned) < 2:
        return [], runs[:20]

    used = set()
    results = []

    def dfs(off, pi, acc):
        if len(results) > 100:
            return
        if len(acc) == 10:
            if off == TENSORS_SUM and pi == len(pinned):
                results.append(list(acc))
            return
        rest_min = min(
            (s for nm, s in LAYOUT_SIZES.items() if nm not in used), default=0
        )
        for name, size in LAYOUT_SIZES.items():
            if name in used:
                continue
            end = off + size
            if end > TENSORS_SUM:
                continue
            if not SECTION_TYPE_CHECK[name](base_data[off:end]):
                continue
            if pi < len(pinned):
                poff, pln = pinned[pi]
                if end <= poff and poff - end < rest_min and end != poff:
                    continue
                hits = off == poff and size == pln
                if poff < end and not hits:
                    continue
                used.add(name)
                dfs(end, pi + 1 if hits else pi, acc + [name])
                used.discard(name)
            else:
                used.add(name)
                dfs(end, pi, acc + [name])
                used.discard(name)

    dfs(0, 0, [])
    return results, runs[:20]


def _sampled(section, stride_bytes, take):
    """First take samples of stride_bytes across the section."""
    n = len(section)
    if n <= take * stride_bytes:
        return [section]
    step = (n - stride_bytes) // (take - 1)
    return [section[i:i + stride_bytes] for i in range(0, n - stride_bytes + 1, step)][:take]


def bf16_plausible(section):
    """Fraction of LE bf16 high bytes in the +-normal exponent band; real
    activations concentrate there, random bytes do not. Sampled."""
    hits = tot = 0
    for chunk in _sampled(section, 8192, 16):
        hi = chunk[1::2]
        hits += sum(1 for b in hi if 0x38 <= b <= 0x48 or 0xB8 <= b <= 0xC8)
        tot += len(hi)
    return hits / tot if tot else 0.0


def fp32_plausible(section):
    """Fraction of LE fp32 MSBs consistent with small-magnitude real values
    (zero or +-normal exponent); random bytes scatter. Sampled."""
    hits = tot = 0
    for chunk in _sampled(section, 8192, 16):
        msb = chunk[3::4]
        hits += sum(
            1 for b in msb
            if b == 0 or 0x30 <= b <= 0x4F or 0xB0 <= b <= 0xCF
        )
        tot += len(msb)
    return hits / tot if tot else 0.0


def tiling_plausible(section):
    """Tiling words are small structured ints (dims, blocks, counts)."""
    words = struct.unpack(f"<{len(section) // 4}I", section[: len(section) // 4 * 4])
    small = sum(1 for w in words if w < (1 << 21))
    return small / len(words) if words else 0.0


SECTION_TYPE_CHECK = {
    "x": lambda b: bf16_plausible(b) >= 0.5,
    "y": lambda b: bf16_plausible(b) >= 0.5,
    # fp32 sections must NOT look like bf16 (bf16 data passes a naive fp32
    # band check because every 4th byte is a bf16 high byte)
    "mixes": lambda b: fp32_plausible(b) >= 0.6 and bf16_plausible(b) < 0.75,
    "rsqrt": lambda b: fp32_plausible(b) >= 0.6 and bf16_plausible(b) < 0.75,
    "post": lambda b: fp32_plausible(b) >= 0.6 and bf16_plausible(b) < 0.75,
    "comb": lambda b: fp32_plausible(b) >= 0.6 and bf16_plausible(b) < 0.75,
    # weights are already pinned by the constancy islands; no type check
    "hc_scale": lambda b: True,
    "hc_base": lambda b: True,
    "tiling": lambda b: tiling_plausible(b) >= 0.5,
    "workspace": lambda b: True,
}


def slice_by_order(data, order):
    parts, off = {}, 0
    for name in order:
        size = LAYOUT_SIZES[name]
        parts[name] = data[off:off + size]
        off += size
    return parts if off == len(data) else None


def anchor_clusters(parsed):
    return len({(p["hc_scale"], p["hc_base"]) for p in parsed})


def scan_fp32(data, label):
    n = len(data) // 4
    vals = struct.unpack(f"<{n}f", data[: n * 4])
    bad = sum(1 for v in vals if v != v or abs(v) == float("inf") or abs(v) > 1e6)
    amax = max((abs(v) for v in vals if v == v), default=0.0)
    print(f"  {label:8s} fp32 n={n} bad={bad} |v|max={amax:.3e} a5a5={data.count(A5X4)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        sys.exit("no dump files matched")

    dumps, metas = {}, {}
    for path in files:
        with open(path, "rb") as f:
            blob = f.read()
        if len(blob) < 8 + TENSORS_SUM:
            print(f"{path}: too small; skip")
            continue
        idx_len = struct.unpack("<Q", blob[:8])[0]
        dumps[path] = blob[8 + idx_len:]
        metas[path] = blob[8:8 + idx_len]
    if len(dumps) < 2:
        sys.exit("need at least 2 usable dumps")

    print(f"usable dumps: {len(dumps)}")

    candidates = []
    keys0 = next(iter(dumps))
    idx_orders = orders_from_index(next(iter(metas.values())))
    for o in idx_orders:
        candidates.append((f"index:{' '.join(o)}", o))
    const_orders, runs = orders_from_constancy(dumps)
    for o in idx_orders:
        candidates.append((f"index:{' '.join(o)}", o))
    const_orders, runs = orders_from_constancy(dumps)
    for o in const_orders:
        candidates.append((f"constancy:{' '.join(o)}", o))
    if not candidates:
        print("equal-run sample (offset,len):", runs)
        sys.exit("no candidate order; paste the run list back for analysis")

    chosen = None
    for desc, order in candidates:
        parsed = {}
        ok = True
        for path, data in dumps.items():
            parts = slice_by_order(data, order)
            if parts is None:
                ok = False
                break
            parsed[path] = parts
        if not ok or not (0 < anchor_clusters(parsed) <= 4):
            continue
        # disambiguate interchangeable sizes: x/y must look like real bf16
        if bf16_plausible(parsed[keys0]["x"]) < 0.5:
            continue
        if bf16_plausible(parsed[keys0]["y"]) < 0.5:
            continue
        chosen = (desc, order, parsed)
        break
    if chosen is None:
        sys.exit(f"no candidate validated ({len(candidates)} tried)")

    desc, order, parsed = chosen
    keys0 = next(iter(parsed))
    print(f"section order: {desc}")
    print(f"hc weight groups across dumps: {anchor_clusters(parsed)}")

    tilings, workspaces = {}, {}
    for path, parts in parsed.items():
        tilings[path] = parts["tiling"]
        workspaces[path] = parts["workspace"]
        words = struct.unpack("<36I", parts["tiling"])
        print(f"\n{path}")
        print(f"  tiling u32[0..8] : {' '.join(f'{w:#x}' for w in words[:8])}")
        print(f"  tiling u32[8..16]: {' '.join(f'{w:#x}' for w in words[8:16])}")
        print(f"  tiling sha256    : {hashlib.sha256(parts['tiling']).hexdigest()[:16]}")
        print(f"  workspace bytes  : {parts['workspace'].hex()}")
        scan_fp32(parts["mixes"], "mixes")
        scan_fp32(parts["rsqrt"], "rsqrt")
        for name in ("x", "y", "post", "comb"):
            print(f"  {name:8s} a5a5-hits={parts[name].count(A5X4)}")

    uniq_t = {bytes(v) for v in tilings.values()}
    uniq_w = {bytes(v) for v in workspaces.values()}
    print(f"\n=== cross-dump verdict over {len(tilings)} dumps ===")
    print(f"distinct tiling contents   : {len(uniq_t)}")
    print(f"distinct workspace contents: {len(uniq_w)}")
    if len(uniq_t) > 1:
        print("TILING DIFFERS between same-shape crashes -> nondeterministic/"
              "cross-written tiling: look INSIDE the op package first.")
    elif len(tilings) > 1:
        print("tiling identical across crashes (deterministic; corruption "
              "hypothesis weakened but not excluded).")


if __name__ == "__main__":
    main()
