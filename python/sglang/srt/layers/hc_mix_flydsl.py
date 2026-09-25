"""Optional gfx950 HC mix using AITER's FlyDSL fused GEMM epilogues."""

from functools import lru_cache
from importlib.util import find_spec

import torch

from sglang.srt.environ import envs


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


def flydsl_hc_mix_supported(x, down, up, hc, hs):
    # Keep FlyDSL/AITER imports off the NVIDIA and disabled paths.
    return (
        envs.SGLANG_AITER_HC_MIX.get()
        and torch.version.hip is not None
        and x.is_cuda
        and x.ndim == 2
        and 1 <= x.shape[0] <= 16
        and hc == 4
        and hs > 0
        and x.shape[1] == hc * hs
        and x.shape[1] % 2048 == 0
        and down.ndim == 2
        and down.shape[1] == x.shape[1]
        and down.shape[0] > 0
        and down.shape[0] % 32 == 0
        and up.shape == (x.shape[1], down.shape[0])
        and all(t.dtype == torch.bfloat16 for t in (x, down, up))
        and all(t.device == x.device for t in (down, up))
        and all(t.is_contiguous() for t in (x, down, up))
        and x.data_ptr() % 16 == 0
        and torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        == "gfx950"
        and _aiter_hc_mix_available()
    )


def weight_cache_key(down, up):
    # Inference tensors do not have version counters. As in the CuTe weight
    # cache, in-place updates of those tensors require reinitializing the cache.
    def version(w):
        return None if w.is_inference() else w._version

    return (down.data_ptr(), version(down), up.data_ptr(), version(up))
