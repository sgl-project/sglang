"""UNO compact target support under a CUDA graph captured after eager warmup (the post-verify graph's lazy capture).

FlashInfer 0.7.3's top_k picks its row-state scratch on torch.cuda.is_current_stream_capturing(), so a compiled
_build_sparse_target_support_tensors that traced top_k outside capture retraces at the first capture, which
torch.compile refuses ("cannot JIT compile during CUDA graph capture"). uno_utils keeps top_k out
of the compiled region. These tests check that the wrapper returns FlashInfer's top_k unchanged, that the compiled
support is bit for bit the same with and without it, that a late capture now works and replays the eager bits, and
(in a subprocess, on 0.7.3) that the unwrapped top_k reproduces the failure.

Only the radix path reads that branch. High-concurrency recipes select it with FLASHINFER_TOPK_ALGO=filtered; with the
variable unset, SM10x takes the clusters path, which never does. So every case runs under both.
"""

import os
import subprocess
import sys
import textwrap
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

VOCAB = 248320
# (requests, tokens per request, max_top_k, top_p): a 128-request post-verify key and small ones.
CASES = ((128, 4, 20, True), (128, 4, 20, False), (4, 4, 128, True), (1, 4, 20, False))
# FLASHINFER_TOPK_ALGO: the high-concurrency recipes' value (radix path), and unset (FlashInfer's default).
ALGOS = ("filtered", None)


@contextmanager
def _topk_algo(algo):
    with patch.dict(os.environ):
        os.environ.pop("FLASHINFER_TOPK_ALGO", None)
        if algo is not None:
            os.environ["FLASHINFER_TOPK_ALGO"] = algo
        yield


def _inputs(requests: int, width: int, max_top_k: int, top_p: bool, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = requests * width
    # Distinct logits in every row, so top_k's order does not depend on tie handling.
    logits = torch.rand(rows, VOCAB, generator=g, device="cuda").argsort(dim=1)
    logits = logits.float() * 0.004 - 400.0
    temperatures = 0.5 + torch.rand(requests, 1, generator=g, device="cuda")
    top_ks = torch.randint(
        1, max_top_k + 1, (requests,), generator=g, device="cuda", dtype=torch.int32
    )
    top_ps = (
        0.5 + 0.5 * torch.rand(requests, generator=g, device="cuda") if top_p else None
    )
    return logits, temperatures, top_ks, top_ps


def _orig(fn):
    return getattr(fn, "_torchdynamo_orig_callable", None) or fn.__wrapped__


class TestUnoSparseSupportLateCapture(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")

    def test_wrapper_returns_flashinfer_top_k_unchanged(self):
        import flashinfer

        from sglang.srt.speculative import uno_utils

        logits = _inputs(16, 4, 128, False, seed=1)[0]
        for algo in ALGOS:
            for k in (20, 128):
                with self.subTest(algo=algo, k=k), _topk_algo(algo):
                    got = uno_utils._flashinfer_top_k(
                        logits, k, sorted=True, deterministic=False
                    )
                    ref = flashinfer.top_k(logits, k, sorted=True, deterministic=False)
                    self.assertTrue(torch.equal(got[0], ref[0]))
                    self.assertTrue(torch.equal(got[1], ref[1]))

    def test_compiled_support_is_the_same_with_and_without_the_wrapper(self):
        import flashinfer

        from sglang.srt.speculative import uno_utils

        orig = _orig(uno_utils._build_sparse_target_support_tensors)
        cases = [(a, i, c) for a in ALGOS for i, c in enumerate(CASES)]
        for algo, i, (requests, width, k, top_p) in cases:
            with (
                self.subTest(algo=algo, requests=requests, max_top_k=k, top_p=top_p),
                _topk_algo(algo),
            ):
                x = _inputs(requests, width, k, top_p, seed=10 + i)
                torch._dynamo.reset()
                with patch.object(uno_utils, "_flashinfer_top_k", flashinfer.top_k):
                    ref = torch.compile(orig, dynamic=True)(*x, requests, width, k)
                torch._dynamo.reset()
                out = torch.compile(orig, dynamic=True)(*x, requests, width, k)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out[0], ref[0]))
                self.assertTrue(torch.equal(out[1], ref[1]))

    def test_late_capture_after_eager_warmup_replays_the_eager_bits(self):
        from sglang.srt.speculative import uno_utils

        fn = uno_utils._build_sparse_target_support_tensors
        cases = [(a, i, c) for a in ALGOS for i, c in enumerate(CASES)]
        for algo, i, (requests, width, k, top_p) in cases:
            with (
                self.subTest(algo=algo, requests=requests, max_top_k=k, top_p=top_p),
                _topk_algo(algo),
            ):
                torch._dynamo.reset()
                static = _inputs(requests, width, k, top_p, seed=20 + i)
                # The post-verify graphs run each key eagerly twice before capturing it.
                for _ in range(2):
                    fn(*static, requests, width, k)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = fn(*static, requests, width, k)
                fresh = _inputs(requests, width, k, top_p, seed=40 + i)
                for dst, src in zip(static, fresh):
                    if dst is not None:
                        dst.copy_(src)
                graph.replay()
                eager = fn(*fresh, requests, width, k)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(captured[0], eager[0]))
                self.assertTrue(torch.equal(captured[1], eager[1]))

    def test_unwrapped_top_k_fails_a_late_capture_on_flashinfer_073(self):
        import flashinfer

        if flashinfer.__version__ != "0.7.3":
            self.skipTest(
                "the capture-dependent row-state branch is FlashInfer 0.7.3's"
            )
        script = textwrap.dedent(
            """
            import torch, flashinfer
            from sglang.srt.speculative import uno_utils
            uno_utils._flashinfer_top_k = flashinfer.top_k
            fn = uno_utils._build_sparse_target_support_tensors
            logits = torch.randn(8, 248320, device="cuda")
            args = (logits, torch.ones(2, 1, device="cuda"),
                    torch.full((2,), 20, dtype=torch.int32, device="cuda"), None, 2, 4, 20)
            fn(*args); fn(*args); torch.cuda.synchronize()
            try:
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    fn(*args)
            except Exception as e:
                print("REPRO_RAISED", type(e).__name__, " ".join(str(e).split())[:4000])
            else:
                print("REPRO_NO_RAISE")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "FLASHINFER_TOPK_ALGO": "filtered"},
        )
        lines = [x for x in result.stdout.splitlines() if x.startswith("REPRO_")]
        self.assertTrue(lines, result.stdout[-2000:] + result.stderr[-4000:])
        self.assertIn("REPRO_RAISED", lines[-1])
        self.assertIn(
            "cannot JIT compile during CUDA graph capture",
            result.stdout + result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
