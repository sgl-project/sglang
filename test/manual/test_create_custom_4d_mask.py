"""
Unit tests for _create_custom_4d_mask (commit a475156d).

Verifies:
  1. Numerical accuracy of the new vectorised implementation against the
     original loop-based reference.
  2. Wall-clock performance improvement on a range of (batch, seq_len) sizes.
     On CUDA the benchmark uses cuda events for precise GPU timing.
  3. Optional PyTorch profiler trace capture (--profile / PROFILE_TRACES=1).
     CPU + CUDA activities are captured when a GPU is available.

Usage
-----
# accuracy + perf only (auto-selects CUDA if available):
  python test_create_custom_4d_mask.py

# force CPU regardless of CUDA availability:
  python test_create_custom_4d_mask.py --device cpu

# with profiler traces written to ./pt_traces/:
  python test_create_custom_4d_mask.py --profile
  # or:
  PROFILE_TRACES=1 python test_create_custom_4d_mask.py

# run through pytest (no profiling, CUDA used if available):
  pytest test_create_custom_4d_mask.py -v
"""

import argparse
import os
import sys
import time
import unittest

import torch

# ---------------------------------------------------------------------------
# Global device selection – overridden by --device CLI flag before unittest.main
# ---------------------------------------------------------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Standalone reference implementation (original loop-based code, pre-a475156d)
# ---------------------------------------------------------------------------


def _create_custom_4d_mask_reference(
    sequence_length, dtype, device, batch_size, token_type_ids
):
    """Original O(B*S) Python loop implementation (pre-commit reference)."""
    min_dtype = torch.finfo(dtype).min
    masks = []
    for b in range(batch_size):
        mask = torch.full(
            (sequence_length, sequence_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        type_ids = token_type_ids[b]
        image_positions = (type_ids == 0).nonzero(as_tuple=True)[0]
        text_positions = (type_ids == 1).nonzero(as_tuple=True)[0]

        if len(image_positions) > 0:
            mask[image_positions[:, None], image_positions] = 0.0

        for i, text_pos in enumerate(text_positions):
            if len(image_positions) > 0:
                mask[text_pos, image_positions] = 0.0
            mask[text_pos, text_positions[: i + 1]] = 0.0

        masks.append(mask)

    return torch.stack(masks, dim=0).unsqueeze(1)


# ---------------------------------------------------------------------------
# New vectorised implementation (copy of the production code for self-contained
# testing — keep in sync with CustomQwen2ModelInner._create_custom_4d_mask in
# python/sglang/srt/models/deepseek_ocr.py)
# ---------------------------------------------------------------------------


def _create_custom_4d_mask_new(
    sequence_length, dtype, device, batch_size, token_type_ids
):
    min_dtype = torch.finfo(dtype).min

    is_image = token_type_ids == 0  # [B, S]
    is_text = token_type_ids == 1  # [B, S]

    mask = torch.full(
        (batch_size, sequence_length, sequence_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )

    img_outer = is_image.unsqueeze(2) & is_image.unsqueeze(1)  # [B, S, S]

    idx = torch.arange(sequence_length, device=device)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)  # [S, S]

    text_causal = (
        is_text.unsqueeze(2)  # [B, S, 1]
        & is_text.unsqueeze(1)  # [B, 1, S]
        & causal.unsqueeze(0)  # [1, S, S]
    )  # [B, S, S]

    text_to_img = is_text.unsqueeze(2) & is_image.unsqueeze(1)  # [B, S, S]

    allow = img_outer | text_causal | text_to_img  # [B, S, S]
    mask.masked_fill_(allow, 0.0)

    return mask.unsqueeze(1)  # [B, 1, S, S]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token_type_ids(batch_size, seq_len, image_fraction, device):
    """First `image_fraction` tokens per sequence are image (0), rest are text (1).

    Always produces at least one image token (n_image = max(1, int(seq_len *
    image_fraction))), so passing image_fraction=0 still yields one image token.
    """
    n_image = max(1, int(seq_len * image_fraction))
    ids = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    ids[:, :n_image] = 0
    return ids


def _make_random_token_type_ids(batch_size, seq_len, device, seed=42):
    """Random interleaving of image/text tokens (stress test)."""
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    return torch.randint(0, 2, (batch_size, seq_len), device=device, generator=rng)


def _bench_cuda_events(fn, n, **kwargs):
    """Time `fn` on CUDA using cuda events (excludes H2D launch overhead)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # warmup
    for _ in range(5):
        fn(**kwargs)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n):
        fn(**kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / n  # seconds per iteration


def _bench_wall(fn, n, **kwargs):
    """Time `fn` on CPU using perf_counter."""
    for _ in range(5):
        fn(**kwargs)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(**kwargs)
    return (time.perf_counter() - t0) / n


def _bench(fn, run_device, n=50, **kwargs):
    if "cuda" in str(run_device):
        return _bench_cuda_events(fn, n, **kwargs)
    return _bench_wall(fn, n, **kwargs)


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------


class TestAccuracy(unittest.TestCase):
    """Verify new implementation produces identical masks to the reference."""

    @classmethod
    def setUpClass(cls):
        cls.device = _DEVICE
        cls.dtype = torch.float32

    def _check(self, batch_size, seq_len, token_type_ids):
        ref = _create_custom_4d_mask_reference(
            seq_len, self.dtype, self.device, batch_size, token_type_ids
        )
        new = _create_custom_4d_mask_new(
            seq_len, self.dtype, self.device, batch_size, token_type_ids
        )
        self.assertEqual(ref.shape, new.shape, "shape mismatch")
        ref_cpu, new_cpu = ref.cpu(), new.cpu()
        if not torch.equal(ref_cpu, new_cpu):
            diff = (ref_cpu - new_cpu).abs().max().item()
            self.fail(
                f"mask mismatch for batch={batch_size} seq={seq_len}\n"
                f"max abs diff = {diff}"
            )

    # --- fixed patterns ---

    def test_all_image(self):
        ids = torch.zeros(2, 16, dtype=torch.long, device=self.device)
        self._check(2, 16, ids)

    def test_all_text(self):
        ids = torch.ones(2, 16, dtype=torch.long, device=self.device)
        self._check(2, 16, ids)

    def test_image_then_text(self):
        ids = _make_token_type_ids(4, 32, image_fraction=0.5, device=self.device)
        self._check(4, 32, ids)

    def test_single_image_token(self):
        ids = torch.ones(3, 20, dtype=torch.long, device=self.device)
        ids[:, 0] = 0
        self._check(3, 20, ids)

    def test_single_text_token(self):
        ids = torch.zeros(2, 20, dtype=torch.long, device=self.device)
        ids[:, -1] = 1
        self._check(2, 20, ids)

    def test_batch_size_1(self):
        ids = _make_token_type_ids(1, 64, image_fraction=0.25, device=self.device)
        self._check(1, 64, ids)

    def test_large_seq(self):
        ids = _make_token_type_ids(2, 512, image_fraction=0.6, device=self.device)
        self._check(2, 512, ids)

    # --- random / stress ---

    def test_random_interleaving(self):
        ids = _make_random_token_type_ids(8, 128, device=self.device)
        self._check(8, 128, ids)

    def test_random_large(self):
        ids = _make_random_token_type_ids(4, 1024, device=self.device)
        self._check(4, 1024, ids)

    def test_batch_heterogeneous(self):
        """Different image/text ratios per batch item."""
        ids = torch.ones(4, 64, dtype=torch.long, device=self.device)
        ids[0, :10] = 0
        ids[1, :32] = 0
        ids[2, :63] = 0
        ids[3, :] = 1
        self._check(4, 64, ids)

    # --- output shape ---

    def test_output_shape(self):
        B, S = 3, 48
        ids = _make_token_type_ids(B, S, 0.4, device=self.device)
        out = _create_custom_4d_mask_new(S, self.dtype, self.device, B, ids)
        self.assertEqual(out.shape, (B, 1, S, S))

    # --- value semantics ---

    def test_allowed_entries_are_zero(self):
        """Every position must be 0.0 (allowed) or min_dtype (blocked)."""
        ids = _make_token_type_ids(2, 32, 0.5, device=self.device)
        out = _create_custom_4d_mask_new(32, self.dtype, self.device, 2, ids)
        min_val = torch.finfo(self.dtype).min
        unique = out.cpu().unique()
        for v in unique:
            self.assertIn(
                v.item(),
                {0.0, min_val},
                f"unexpected mask value {v.item()}",
            )

    def test_causal_text_ordering(self):
        """Text token i must NOT attend to text token j > i."""
        B, S = 1, 8
        ids = torch.ones(B, S, dtype=torch.long, device=self.device)
        out = _create_custom_4d_mask_new(S, self.dtype, self.device, B, ids)
        min_val = torch.finfo(self.dtype).min
        mask2d = out.cpu()[0, 0]
        for q in range(S):
            for k in range(S):
                if k <= q:
                    self.assertEqual(
                        mask2d[q, k].item(),
                        0.0,
                        f"text[{q}] should attend to text[{k}]",
                    )
                else:
                    self.assertEqual(
                        mask2d[q, k].item(),
                        min_val,
                        f"text[{q}] should NOT attend to text[{k}]",
                    )

    def test_image_full_attention(self):
        """Image tokens must attend to all other image tokens (bidirectional)."""
        B, S = 1, 12
        n_img = 6
        ids = torch.ones(B, S, dtype=torch.long, device=self.device)
        ids[:, :n_img] = 0
        out = _create_custom_4d_mask_new(S, self.dtype, self.device, B, ids)
        mask2d = out.cpu()[0, 0]
        for q in range(n_img):
            for k in range(n_img):
                self.assertEqual(
                    mask2d[q, k].item(), 0.0, f"image[{q}] should attend to image[{k}]"
                )

    def test_text_attends_to_image(self):
        """Every text token must attend to every image token."""
        B, S = 1, 12
        n_img = 4
        ids = torch.ones(B, S, dtype=torch.long, device=self.device)
        ids[:, :n_img] = 0
        out = _create_custom_4d_mask_new(S, self.dtype, self.device, B, ids)
        mask2d = out.cpu()[0, 0]
        for q in range(n_img, S):
            for k in range(n_img):
                self.assertEqual(
                    mask2d[q, k].item(), 0.0, f"text[{q}] should attend to image[{k}]"
                )

    # --- dtype coverage ---

    def test_float16(self):
        ids = _make_token_type_ids(2, 64, 0.5, device=self.device)
        ref = _create_custom_4d_mask_reference(64, torch.float16, self.device, 2, ids)
        new = _create_custom_4d_mask_new(64, torch.float16, self.device, 2, ids)
        self.assertTrue(torch.equal(ref.cpu(), new.cpu()))

    def test_bfloat16(self):
        ids = _make_token_type_ids(2, 64, 0.5, device=self.device)
        ref = _create_custom_4d_mask_reference(64, torch.bfloat16, self.device, 2, ids)
        new = _create_custom_4d_mask_new(64, torch.bfloat16, self.device, 2, ids)
        self.assertTrue(torch.equal(ref.cpu(), new.cpu()))


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

BENCHMARK_CASES = [
    # (batch_size, seq_len, image_fraction)
    (1, 256, 0.5),
    (4, 512, 0.5),
    (8, 1024, 0.5),
    (16, 2048, 0.5),
    (4, 4096, 0.75),
]
BENCH_ITERS = 50
SPEEDUP_FLOOR = 1.0  # new must be at least as fast as reference


class TestPerformance(unittest.TestCase):
    """New vectorised implementation must not be slower than the reference."""

    @classmethod
    def setUpClass(cls):
        cls.device = _DEVICE
        cls.dtype = torch.float32

    def _run_case(self, batch_size, seq_len, image_fraction):
        ids = _make_token_type_ids(
            batch_size, seq_len, image_fraction, device=self.device
        )
        kwargs = dict(
            sequence_length=seq_len,
            dtype=self.dtype,
            device=self.device,
            batch_size=batch_size,
            token_type_ids=ids,
        )
        t_ref = _bench(
            _create_custom_4d_mask_reference,
            run_device=self.device,
            n=BENCH_ITERS,
            **kwargs,
        )
        t_new = _bench(
            _create_custom_4d_mask_new, run_device=self.device, n=BENCH_ITERS, **kwargs
        )
        speedup = t_ref / t_new
        dev_tag = "CUDA" if "cuda" in str(self.device) else "CPU"
        print(
            f"  [{dev_tag}] B={batch_size:3d} S={seq_len:5d} img%={int(image_fraction*100):3d}%"
            f"  ref={t_ref*1e3:.2f}ms  new={t_new*1e3:.2f}ms  speedup={speedup:.2f}x"
        )
        self.assertGreaterEqual(
            speedup,
            SPEEDUP_FLOOR,
            f"New impl is slower than reference for B={batch_size} S={seq_len} "
            f"(speedup={speedup:.2f}x < required {SPEEDUP_FLOOR}x)",
        )
        return t_ref, t_new, speedup

    def test_performance_small(self):
        print()
        self._run_case(1, 256, 0.5)

    def test_performance_medium(self):
        print()
        self._run_case(4, 512, 0.5)

    def test_performance_large(self):
        print()
        self._run_case(8, 1024, 0.5)

    def test_performance_xlarge(self):
        print()
        self._run_case(16, 2048, 0.5)

    def test_performance_sweep(self):
        """Full sweep over all benchmark cases."""
        print(f"\n--- Performance sweep (device={_DEVICE}) ---")
        for batch_size, seq_len, img_frac in BENCHMARK_CASES:
            self._run_case(batch_size, seq_len, img_frac)


# ---------------------------------------------------------------------------
# PyTorch profiler (optional – triggered by --profile or PROFILE_TRACES=1)
# ---------------------------------------------------------------------------


def run_profiler_traces(output_dir: str = "./pt_traces", device: str = _DEVICE):
    """
    Capture Chrome-trace JSON files for both implementations.

    CPU activity is always recorded.  When `device` is a CUDA device,
    ProfilerActivity.CUDA is added so GPU kernels appear in the trace.
    Traces are written to `output_dir` and can be opened in
    chrome://tracing or the PyTorch TensorBoard plugin.
    """
    os.makedirs(output_dir, exist_ok=True)

    use_cuda = "cuda" in str(device) and torch.cuda.is_available()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Single small case profiled for both implementations.
    # Keep seq_len modest so the Python-loop reference finishes quickly under profiling.
    batch_size, seq_len, img_frac = 4, 128, 0.5
    ids = _make_token_type_ids(batch_size, seq_len, img_frac, device=device)
    kwargs = dict(
        sequence_length=seq_len,
        dtype=torch.float32,
        device=device,
        batch_size=batch_size,
        token_type_ids=ids,
    )

    device_tag = "CUDA" if use_cuda else "CPU"
    print(f"[profiler] device={device}  CUDA_activities={use_cuda}")

    for label, fn in [
        ("reference", _create_custom_4d_mask_reference),
        ("new", _create_custom_4d_mask_new),
    ]:
        trace_path = os.path.join(
            output_dir,
            f"trace_{label}_B{batch_size}_S{seq_len}.json",
        )
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            # warmup inside the profile scope so kernel shapes are recorded
            with torch.profiler.record_function(f"{label}_warmup"):
                for _ in range(3):
                    fn(**kwargs)
            if use_cuda:
                torch.cuda.synchronize()
            # measured iterations — clearly labelled in the Chrome trace
            with torch.profiler.record_function(f"{label}_measured"):
                for _ in range(20):
                    fn(**kwargs)
            if use_cuda:
                torch.cuda.synchronize()

        prof.export_chrome_trace(trace_path)
        print(f"[profiler/{device_tag}] {label} trace written → {trace_path}")

        sort_key = "cuda_time_total" if use_cuda else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=12))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test and benchmark _create_custom_4d_mask"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=bool(int(os.environ.get("PROFILE_TRACES", "0"))),
        help="Capture PyTorch profiler traces (also enabled via PROFILE_TRACES=1)",
    )
    parser.add_argument(
        "--trace-dir",
        default="./pt_traces",
        help="Directory to write profiler JSON traces (default: ./pt_traces)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on: 'cuda', 'cuda:0', 'cpu', etc. "
        "Defaults to CUDA if available, otherwise CPU.",
    )
    args, remaining = parser.parse_known_args()

    # Propagate device choice to global so test classes pick it up
    if args.device is not None:
        _DEVICE = args.device
    print(f"[config] device={_DEVICE}  cuda_available={torch.cuda.is_available()}")

    if args.profile:
        print(f"\n=== PyTorch profiler traces ({_DEVICE}) → {args.trace_dir} ===")
        run_profiler_traces(output_dir=args.trace_dir, device=_DEVICE)
        print()

    sys.argv = [sys.argv[0]] + remaining
    unittest.main(verbosity=2)
