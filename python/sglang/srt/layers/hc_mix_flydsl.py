"""Optional gfx950 HC mix using AITER's FlyDSL fused GEMM epilogues."""

from functools import lru_cache
from importlib.util import find_spec

import torch

# AITER's kernel limit; it stays ahead of the torch.compile fallback up to it
# (K=10240, R=320, cold weights: 26 vs 42 us at 128 rows).
_MAX_ROWS = 128


@lru_cache(maxsize=1)
def _aiter_hc_mix_available():
    # Check only after device/shape guards; optional packages stay lazy.
    try:
        return (
            find_spec("flydsl") is not None
            and find_spec("aiter.ops.flydsl.hc_mix") is not None
        )
    except ModuleNotFoundError:
        return False


def flydsl_hc_mix_weights_supported(down, up, hc):
    """Device and weight-shape half of the check.

    None of it changes after the weights are loaded, so callers evaluate it
    once per layer and pass the result to ``flydsl_hc_mix_supported``.
    """
    # Keep FlyDSL/AITER imports off the NVIDIA path.
    if torch.version.hip is None:
        return False
    r, k = down.shape
    # pack_hc_weights tiles K by 64 and R by 32, and makes its own contiguous
    # copies, so the source weights' layout does not matter.
    return (
        hc == 4
        and r > 0
        and r % 32 == 0
        and k > 0
        and k % 64 == 0
        and up.shape == (k, r)
        and down.dtype == up.dtype == torch.bfloat16
        and down.is_cuda
        and up.device == down.device
        and torch.cuda.get_device_properties(down.device).gcnArchName.split(":")[0]
        == "gfx950"
        and _aiter_hc_mix_available()
    )


def flydsl_hc_mix_supported(x, weights_supported):
    """Per-call half: the activation's dtype, row count and layout.

    ``x`` is the [rows, HC*HS] normed input, which the caller has already
    matched to the weights' K.
    """
    return (
        weights_supported
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 1 <= x.shape[0] <= _MAX_ROWS
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
    )


def weight_cache_key(down, up):
    # Inference tensors do not have version counters. As in the CuTe weight
    # cache, in-place updates of those tensors require reinitializing the cache.
    def version(w):
        return None if w.is_inference() else w._version

    return (down.data_ptr(), version(down), up.data_ptr(), version(up))
