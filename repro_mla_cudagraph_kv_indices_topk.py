#!/usr/bin/env python3
"""Deterministic repro for the MLA EAGLE draft CUDA-graph kv_indices topk under-allocation.

Exercises the exact sizing + always-on invariant in
`FlashInferMLAMultiStepDraftBackend`
(`python/sglang/srt/layers/attention/flashinfer_mla_backend.py`):

  - EAGER `init_forward_metadata` sizes kv_indices as
    `batch_size * topk * max_context_len` -- HAS the topk factor.
  - CUDA-GRAPH `init_cuda_graph_state` allocates `self.cuda_graph_kv_indices`.
    Before this fix it was `(num_steps, max_bs * max_context_len)` -- MISSING the
    topk factor, so for topk>1 the buffer is topk-times too narrow.
  - `common_template` writes per-step slices `kv_indices[i][: seq_lens_sum*topk +
    bs*(i+1)]` and the always-on invariant asserts the widest required width
    `seq_lens_sum*topk + bs*num_steps <= kv_indices_buffer.shape[1]`.

In production MLA spec decode is gated to topk=1: the ctor raises
`ValueError("Currently Flashinfer MLA only supports topk=1 ...")`. So the
undersized CUDA-graph path is currently UNREACHABLE -- this is a latent /
defensive landmine that mirrors the non-MLA bug fixed in #27338. The moment MLA
tree-mask / topk>1 support lands, the missing factor would overflow the buffer.

This repro deliberately bypasses the ctor gate (a tiny subclass that constructs
just the state `init_cuda_graph_state` + the invariant need) to exercise the
otherwise-unreachable topk>1 sizing path. No model, no server, no download.

Two modes (mirrors the real revert toggle):
  (default)  with the fix: buffer width includes topk -> invariant passes.
  SGLANG_DEBUG_REVERT_PR=27460  re-narrows init_cuda_graph_state (drops the
             `* topk`) via the source patcher, so the buffer is topk-times too
             narrow and the always-on invariant fires deterministically.

Run on any host (CUDA only used for the int32 zeros buffers), e.g. via rdev:
  cd /host_home/common_sync/sglang && \
  PYTHONPATH=/host_home/common_sync/sglang/python \
  python repro_mla_cudagraph_kv_indices_topk.py                       # passes
  SGLANG_DEBUG_REVERT_PR=27460 PYTHONPATH=... \
  python repro_mla_cudagraph_kv_indices_topk.py                       # invariant fires
"""

import argparse
import types

import torch

from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAMultiStepDraftBackend,
)


def build_backend(topk: int, num_steps: int, max_bs: int, max_context_len: int):
    """Construct a minimal backend instance that skips the ctor topk=1 gate.

    Only the fields read by `init_cuda_graph_state` and the `common_template`
    invariant are populated. We bypass the real `__init__` (which would raise for
    topk>1 and needs a full ModelRunner) via `__new__`.
    """
    backend = FlashInferMLAMultiStepDraftBackend.__new__(
        FlashInferMLAMultiStepDraftBackend
    )
    backend.topk = topk
    backend.speculative_num_steps = num_steps
    backend.max_context_len = max_context_len
    # attn_backends is iterated by init_cuda_graph_state to seed per-step buffers;
    # stub each step with a no-op init_cuda_graph_state so we exercise only the
    # parent buffer sizing (the bug site), not the child MLA backend.
    backend.attn_backends = [
        types.SimpleNamespace(
            init_cuda_graph_state=lambda *a, **k: None
        )
        for _ in range(num_steps - 1)
    ]
    return backend


def required_kv_indices_len(topk, num_seqs, seq_lens_sum, num_steps):
    """The widest required kv_indices width -- exactly the invariant formula in
    common_template: seq_lens_sum*topk + bs*num_steps (bs = topk*num_seqs)."""
    bs = topk * num_seqs
    return seq_lens_sum * topk + bs * num_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-seqs", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--max-bs", type=int, default=8)
    parser.add_argument("--max-context-len", type=int, default=4096)
    parser.add_argument(
        "--seq-lens-sum",
        type=int,
        default=10500,
        help="aggregate prefill length across the active batch",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA device (int32 zeros buffers)"

    topk = args.topk
    num_seqs = args.num_seqs
    num_steps = args.num_steps
    max_bs = args.max_bs
    max_context_len = args.max_context_len
    seq_lens_sum = args.seq_lens_sum

    # SGLANG_DEBUG_REVERT_PR=27460 re-narrows init_cuda_graph_state (drops * topk).
    maybe_revert_pr_fix()

    backend = build_backend(topk, num_steps, max_bs, max_context_len)
    backend.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_bs)

    row_width = backend.cuda_graph_kv_indices.shape[1]
    required = required_kv_indices_len(topk, num_seqs, seq_lens_sum, num_steps)

    print(
        f"cuda_graph_kv_indices row width = {row_width} | required = {required} "
        f"(topk={topk}, num_seqs={num_seqs}, seq_lens_sum={seq_lens_sum}, "
        f"num_steps={num_steps}, max_bs={max_bs}, max_context_len={max_context_len})"
    )

    # The always-on invariant from common_template, run here against the real
    # init_cuda_graph_state buffer width. With the fix the row includes topk and
    # holds; with the revert it is topk-times too narrow and this fires.
    try:
        assert required <= row_width, (
            f"EAGLE draft kv_indices row too small: need {required} "
            f"but row width is {row_width} (topk={topk}, "
            f"num_seqs={num_seqs}, seq_lens_sum={seq_lens_sum}, "
            f"num_steps={num_steps}); the buffer must be sized "
            f"max_bs * topk * max_context_len."
        )
    except AssertionError as e:
        print("INVARIANT FIRED (buffer topk-times too narrow):")
        print(f"  AssertionError: {e}")
        raise

    print("invariant held: buffer width includes the topk factor -- no overflow")


if __name__ == "__main__":
    main()
