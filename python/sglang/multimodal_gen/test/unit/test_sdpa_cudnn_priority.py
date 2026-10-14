# SPDX-License-Identifier: Apache-2.0
"""``allow_cudnn_sdp`` has to override torch's own backend choice.

The backend list is written cuDNN-first, but ``sdpa_kernel`` treats it as an
allow-set unless ``set_priority`` is passed -- and it is the same set torch
already chooses from, so without that flag the context is inert. Only the kernel
that actually ran distinguishes the two, so that is what these assert on.

Where the allow-set alone already lands on cuDNN (Hopper, for one) the flag has
nothing left to do and the check skips rather than asserting something it cannot
observe.
"""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    _PYTORCH_DEFAULT_CUDA_SDP_BACKENDS,
    SDPAImpl,
)

NUM_HEADS, HEAD_DIM, SEQ_LEN = 8, 128, 512

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="attention dispatch needs CUDA"
)


def _qkv():
    shape = (1, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return tuple(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )


def _kernel_under(context, q, k, v) -> str:
    """Which family of attention kernel ran inside ``context``."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with context:
            F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
    for event in prof.events():
        if event.device_type.name != "CUDA" or not event.self_device_time_total:
            continue
        name = event.key.lower()
        # cuDNN names its own kernels `cudnn_generated_..._flash_...`, so cuDNN
        # has to be checked first.
        if "cudnn" in name:
            return "cudnn"
        if "flash" in name or "fmha" in name:
            return "flash"
    return "unknown"


def _impl(allow_cudnn_sdp: bool) -> SDPAImpl:
    return SDPAImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=HEAD_DIM**-0.5,
        allow_cudnn_sdp=allow_cudnn_sdp,
    )


def test_allow_cudnn_sdp_beats_the_backend_torch_would_pick():
    q, k, v = _qkv()
    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            F.scaled_dot_product_attention(q, k, v)
    except RuntimeError:
        pytest.skip("no cuDNN attention kernel for this shape on this GPU")

    # The allow-set on its own is what this code did before priority was passed.
    without_priority = sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
    if _kernel_under(without_priority, q, k, v) == "cudnn":
        pytest.skip("the allow-set alone already lands on cuDNN on this GPU")

    assert _kernel_under(_impl(True)._sdpa_context(q), q, k, v) == "cudnn"


def test_opting_out_leaves_backend_selection_alone():
    q, _, _ = _qkv()
    assert isinstance(_impl(False)._sdpa_context(q), type(nullcontext()))


def test_cpu_tensors_do_not_get_a_cuda_context():
    cpu_q = torch.randn(1, NUM_HEADS, 16, HEAD_DIM)
    assert isinstance(_impl(True)._sdpa_context(cpu_q), type(nullcontext()))
