# SPDX-License-Identifier: Apache-2.0
"""``allow_cudnn_sdp`` has to override torch's own backend choice.

The backend list is written cuDNN-first, but ``sdpa_kernel`` treats it as an
allow-set unless ``set_priority`` is passed. Start with a Flash-first ambient
priority so the override is observable even where torch normally prefers cuDNN.
Assert on the dispatched operator: cuDNN's CUDA kernel names and CUPTI timing
availability vary across GPU architectures and profiler installations.
"""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    _PYTORCH_DEFAULT_CUDA_SDP_BACKENDS,
    SDPAImpl,
    torch_default_sdpa_backends,
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
    """Which backend operator executed on the CUDA inputs inside ``context``."""
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with context:
            F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
    operators = {event.key for event in prof.events()}
    for name, backend in (
        ("aten::_scaled_dot_product_cudnn_attention", "cudnn"),
        ("aten::_scaled_dot_product_flash_attention", "flash"),
    ):
        if name in operators:
            return backend
    pytest.fail(f"SDPA backend operator not observed: {sorted(operators)}")


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
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        if not torch.backends.cuda.can_use_cudnn_attention(params):
            pytest.skip("no cuDNN attention kernel for this shape on this GPU")

    flash_first = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    with sdpa_kernel(flash_first, set_priority=True):
        # the allow-set alone must leave the ambient preference unchanged
        without_priority = sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
        assert _kernel_under(without_priority, q, k, v) == "flash"
        assert _kernel_under(_impl(True)._sdpa_context(q), q, k, v) == "cudnn"


def test_opting_out_leaves_backend_selection_alone():
    q, _, _ = _qkv()
    assert isinstance(_impl(False)._sdpa_context(q), type(nullcontext()))


def test_cpu_tensors_do_not_get_a_cuda_context():
    cpu_q = torch.randn(1, NUM_HEADS, 16, HEAD_DIM)
    assert isinstance(_impl(True)._sdpa_context(cpu_q), type(nullcontext()))


def test_reference_sdpa_does_not_inherit_a_previous_attention_priority():
    q, k, v = _qkv()
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    with sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS, set_priority=True):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            if not torch.backends.cuda.can_use_cudnn_attention(params):
                pytest.skip("no cuDNN attention kernel for this shape on this GPU")
            F.scaled_dot_product_attention(q, k, v)
        for priority in (
            _PYTORCH_DEFAULT_CUDA_SDP_BACKENDS,
            list(reversed(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)),
        ):
            with sdpa_kernel(priority, set_priority=True):
                assert _kernel_under(torch_default_sdpa_backends(), q, k, v) == "cudnn"
