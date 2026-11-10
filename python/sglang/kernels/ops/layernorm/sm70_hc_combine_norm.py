"""Fuse HC residual application with the following grouped norm on Volta."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs


def supported(layer, y, residuals, norm, partials=None):
    """Whether the fused combine+norm may take this (row, norm, residual) triple.

    ``residuals`` is the gated-connection pair ``(streams, normed_streams)``
    produced by the attention read; ``partials`` is the pre-computed gate
    reduction the read stashed for this same normed residual, if any.
    """
    if not envs.SGLANG_SM70_HC_COMBINE_NORM.get():
        return False
    if (
        layer.hc_count != 4
        or layer.hidden_size != 2560
        or getattr(norm, "group_size", None) != 2560
        or y.ndim != 2
        or y.shape[0] not in (1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32)
        or y.shape[1] != 2560
        or len(residuals) != 2
        or not y.is_cuda
        or y.dtype != torch.float16
        or torch.cuda.get_device_capability(y.device) != (7, 0)
    ):
        return False
    r, n = residuals
    w, gamma = layer.block_inject_weight.weight, norm.weight
    if r.shape != (y.shape[0], 10240) or n.shape != r.shape:
        return False
    if w.shape != (4, 10240) or gamma.shape != (10240,):
        return False
    if any(
        t.dtype != y.dtype
        or t.device != y.device
        or not t.is_contiguous()
        or t.data_ptr() % 16 != 0
        for t in (y, r, n, w, gamma)
    ):
        return False
    if partials is not None:
        return (
            partials.shape == (y.shape[0], 8, 4)
            and partials.dtype == torch.float32
            and partials.device == y.device
            and partials.is_contiguous()
        )
    return True


@cache_once
def _module():
    return load_jit(
        "sm70_hc_combine_norm",
        cuda_files=["elementwise/sm70_hc_combine_norm.cuh"],
        cuda_wrappers=[("run", "sglang::sm70_hc_combine_norm::run")],
    )


def combine_norm(layer, y, residuals, norm, partials=None):
    r, n = residuals
    precomputed = partials is not None
    if partials is None:
        partials = torch.empty((y.shape[0], 8, 4), device=y.device, dtype=torch.float32)
    output, normalized = torch.empty_like(r), torch.empty_like(r)
    _module().run(
        y,
        r,
        n,
        layer.block_inject_weight.weight,
        norm.weight,
        partials,
        output,
        normalized,
        norm.variance_epsilon,
        precomputed,
    )
    return output, normalized
