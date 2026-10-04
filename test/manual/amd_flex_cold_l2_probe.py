"""Diagnostic only: decoder FlexAttention compile options, numerical checks and cold L2 timings."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

CASES = [
    (10, 14, 24, 32, (3, 7, 7)),
    (10, 28, 48, 16, (3, 7, 7)),
    (19, 28, 48, 8, (3, 5, 5)),
    (16, 56, 96, 8, (3, 5, 5)),
    (31, 112, 192, 4, (11, 11, 11)),
    (32, 112, 192, 4, (11, 11, 11)),
]


def worker(index):
    import torch
    from torch.nn.attention.flex_attention import flex_attention
    from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
        _neighborhood_block_mask,
    )

    assert torch.version.hip, "This probe targets the AMD CI failure."
    torch.manual_seed(20261004)
    frames, height, width, heads, kernel = CASES[index]
    tokens = frames * height * width
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print("PROBE_ENV", json.dumps({"torch": torch.__version__, "hip": torch.version.hip,
        "device": str(props), "case": CASES[index], "options": [True, False],
        "flush_bytes": 512 * 1024**2,
        "flush_scope": "512 MiB read/write on the same stream before EVERY measured attention invocation, outside CUDA event timing; compilation warmed separately"}), flush=True)
    # Match the decoder's transposed (B, heads, sequence, head_dim) layout.
    q = (torch.randn(1, tokens, heads, 64, dtype=torch.bfloat16, device=device) * 0.125).transpose(1, 2)
    k = torch.randn(1, tokens, heads, 64, dtype=torch.bfloat16, device=device).transpose(1, 2)
    v = torch.randn(1, tokens, heads, 64, dtype=torch.bfloat16, device=device).transpose(1, 2)
    t = time.monotonic()
    mask = _neighborhood_block_mask(frames, height, width, kernel, device)
    torch.cuda.synchronize()
    print("PROBE_MASK", json.dumps({"seconds": time.monotonic() - t,
        "partial_shape": list(mask.kv_indices.shape),
        "partial_count_max": mask.kv_num_blocks.max().item(),
        "full_count_max": mask.full_kv_num_blocks.max().item()}), flush=True)
    functions = {}
    outputs = {}
    # At 666k/688k tokens the unchanged baseline exceeds startup1200 in model CI.
    # Do not label a candidate-only run as baseline performance coverage.
    options = [True, False] if index < 4 else [False]
    for autotune in options:
        name = "baseline" if autotune else "candidate"
        fn = torch.compile(flex_attention, dynamic=False, options={"max_autotune": autotune})
        t = time.monotonic()
        outputs[name] = fn(q, k, v, block_mask=mask, scale=1.0)
        torch.cuda.synchronize()
        functions[name] = fn
        print("PROBE_COMPILE", json.dumps({"name": name, "seconds": time.monotonic() - t}), flush=True)
    if "baseline" in outputs:
        torch.testing.assert_close(outputs["candidate"], outputs["baseline"], rtol=1e-2, atol=1e-3)
        print("PROBE_PARITY", json.dumps({"max_abs": (outputs["candidate"].float() - outputs["baseline"].float()).abs().max().item()}), flush=True)
    # Independent FP32 attention over exact neighborhoods for boundary + random rows.
    sample = sorted(set([0, width - 1, width, height * width - 1, tokens - 1]
                        + torch.randint(tokens, (59,), device="cpu").tolist()))
    kt, kh, kw = kernel
    neighborhoods = []
    for position in sample:
        t, rem = divmod(position, height * width)
        h, w = divmod(rem, width)
        starts = [max(0, min(c - n // 2, size - n)) for c, n, size in zip((t, h, w), kernel, (frames, height, width))]
        neighborhoods.append([(tt * height + hh) * width + ww
            for tt in range(starts[0], starts[0] + kt)
            for hh in range(starts[1], starts[1] + kh)
            for ww in range(starts[2], starts[2] + kw)])
    indices = torch.tensor(neighborhoods, device=device)
    qs = q[:, :, sample, :].float()
    ks = k[:, :, indices, :].float()
    vs = v[:, :, indices, :].float()
    probs = (qs.unsqueeze(-2) * ks).sum(-1).softmax(-1)
    ref = (probs.unsqueeze(-1) * vs).sum(-2)
    actual = outputs["candidate"][:, :, sample, :].float()
    torch.testing.assert_close(actual, ref, rtol=1e-2, atol=1e-3)
    print("PROBE_REFERENCE", json.dumps({"sample_rows": len(sample), "max_abs": (actual-ref).abs().max().item()}), flush=True)
    del qs, ks, vs, probs, ref, actual, outputs
    flush = torch.zeros(512 * 1024**2 // 4, dtype=torch.float32, device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = {name: [] for name in functions}
    for fn in functions.values():
        for _ in range(10):
            flush.add_(1)
            fn(q, k, v, block_mask=mask, scale=1.0)
        torch.cuda.synchronize()
    order = ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate", "candidate", "baseline"] if "baseline" in functions else ["candidate"] * 4
    for name in order:
        for _ in range(25):
            flush.add_(1)
            start.record()
            output = functions[name](q, k, v, block_mask=mask, scale=1.0)
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end))
    result = {"case_index": index, "case": CASES[index], "timings_ms": timings,
        "median_ms": {k: statistics.median(v) for k,v in timings.items()},
        "baseline_coverage": index < 4,
        "flush_bytes": flush.numel() * flush.element_size(),
        "flush_scope": "each attention invocation; excluded from event timing"}
    Path(f"/sglang-checkout/diffusion-failures/flex-probe-{index}.json").write_text(json.dumps(result, indent=2))
    print("PROBE_RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    Path("/sglang-checkout/diffusion-failures").mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        worker(int(sys.argv[1]))
    else:
        results = []
        for index in range(len(CASES)):
            try:
                r = subprocess.run([sys.executable, __file__, str(index)], timeout=900)
                results.append(r.returncode)
            except subprocess.TimeoutExpired:
                print("PROBE_TIMEOUT", index, flush=True)
                results.append(124)
        print("PROBE_EXITS", results, flush=True)
        sys.exit(1 if any(results) else 0)
