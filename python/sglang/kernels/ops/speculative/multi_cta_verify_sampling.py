"""The EAGLE verify's softmax and target-only tree sampling as multi-CTA JIT kernels (default off;
SGLANG_ENABLE_MULTI_CTA_VERIFY_SAMPLING=1). They keep every floating-point operation of the stock kernels (torch's
cunn_SoftMaxForward and tree_speculative_sampling_target_only) in the same order, so their outputs are bitwise
equal; jit/csrc/speculative/multi_cta_verify_sampling.cuh holds the kernels and the argument."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

ENABLED = envs.SGLANG_ENABLE_MULTI_CTA_VERIFY_SAMPLING.get()
# Rows = requests x draft tokens; 64 covers 1-16 requests at 4 draft tokens, the range checked bitwise.
MAX_ROWS = 64
# torch launches cunn_SoftMaxForward (1024 threads, ILP 4) for every fp32 row this long; the kernels
# reproduce only that launch, not the persistent, register or shared-memory variants of shorter rows.
_MIN_VOCAB = 65536
_ROW_CTAS = 32
_STOCK_THREADS = 1024
_TREE_THREADS = 1024
_MAX_CHUNKS = 512  # multi_cta_verify_sampling.cuh kMaxChunks

_logged = False
# Per-device scratch, allocated once and never freed: the post-verify CUDA graphs share one
# memory pool across batch sizes and slots, and a slot's outputs must outlive the other slot's
# replay, so the captured body allocates exactly what the stock softmax and tree sampling do.
_scratch: dict = {}


def _scratch_for(device: torch.device) -> dict:
    bufs = _scratch.get(device.index)
    if bufs is None:
        f32 = dict(dtype=torch.float32, device=device)
        bufs = _scratch[device.index] = {
            "maxpart": torch.empty((MAX_ROWS, _ROW_CTAS), **f32),
            "sumpart": torch.empty((MAX_ROWS, _STOCK_THREADS), **f32),
            "meta": torch.empty((MAX_ROWS, 3), dtype=torch.int64, device=device),
            "chunk_sums": torch.empty((2 * MAX_ROWS * _MAX_CHUNKS,), **f32),
            "chunk_max_valid": torch.empty(
                (MAX_ROWS * _MAX_CHUNKS,), dtype=torch.int32, device=device
            ),
        }
    return bufs


@cache_once
def _jit_module(vec_size: int, deterministic: bool) -> Module:
    deterministic_cpp = "true" if deterministic else "false"
    return load_jit(
        "multi_cta_verify_sampling",
        vec_size,
        deterministic,
        cuda_files=["speculative/multi_cta_verify_sampling.cuh"],
        cuda_wrappers=[
            ("softmax", "MultiCtaVerifySoftmax::run"),
            ("tree", f"MultiCtaVerifyTree<{vec_size}, {deterministic_cpp}>::run"),
        ],
        extra_dependencies=["flashinfer"],
    )


@cache_once
def _cake_capable(device_index: int) -> bool:
    # sampling.softmax routes fp32 rows to the (non-bitwise) Cake kernel on SM103 only.
    return torch.cuda.get_device_capability(device_index) == (10, 3)


def applies(logits: torch.Tensor) -> bool:
    """Whether eagle_sample's stock softmax of these verify logits is torch's cunn_SoftMaxForward
    (CUDA, not HIP, not the SM103 Cake path) on a shape the multi-CTA kernels reproduce."""
    return (
        ENABLED
        and logits.is_cuda
        and torch.version.hip is None
        and logits.dtype == torch.float32
        and logits.dim() == 2
        and 0 < logits.shape[0] <= MAX_ROWS
        and logits.shape[1] >= _MIN_VOCAB
        and logits.shape[1] % 4 == 0
        and not _cake_capable(logits.device.index)
    )


def _log_once(rows: int, vocab: int) -> None:
    global _logged
    if not _logged:
        _logged = True
        logger.debug(
            "Multi-CTA verify softmax and tree sampling engaged (rows %d, vocab %d, %d CTAs per row)",
            rows,
            vocab,
            _ROW_CTAS,
        )


def softmax(scaled_logits: torch.Tensor) -> torch.Tensor:
    """torch.softmax(scaled_logits, dim=-1), bitwise, for a tensor applies() accepted."""
    x = scaled_logits.contiguous()
    rows, vocab = x.shape
    _log_once(rows, vocab)
    out = torch.empty_like(x)
    bufs = _scratch_for(x.device)
    _jit_module(4, True).softmax(x, out, bufs["maxpart"][:rows], bufs["sumpart"][:rows])
    return out


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """sglang.kernels.ops.speculative.sampling.tree_speculative_sampling_target_only, bitwise:
    the same arguments, outputs and draft_probs writes."""
    bs, _, vocab = target_probs.shape
    vec_size = math.gcd(4, vocab)
    nchunks = -(-vocab // (_TREE_THREADS * vec_size))
    if bs > MAX_ROWS or nchunks > _MAX_CHUNKS:
        raise ValueError(
            f"multi-CTA tree sampling: batch {bs} or {nchunks} chunks past the scratch"
        )
    bufs = _scratch_for(target_probs.device)
    meta = bufs["meta"][:bs]
    chunk_sums = bufs["chunk_sums"][: 2 * bs * nchunks].view(2, bs, nchunks)
    chunk_max_valid = bufs["chunk_max_valid"][: bs * nchunks].view(bs, nchunks)
    _jit_module(vec_size, deterministic).tree(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        meta,
        chunk_sums,
        chunk_max_valid,
        threshold_single,
        threshold_acc,
    )
