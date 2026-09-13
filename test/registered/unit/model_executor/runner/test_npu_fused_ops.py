from __future__ import annotations

import pytest
import torch

pytest.importorskip("sgl_kernel_npu")

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from sglang.srt.hardware_backend.npu.fused_ops import (
    _split_qkv_rmsnorm_rope_fake,
    _split_qkv_rmsnorm_rope_kernel,
    split_qkv_rmsnorm_rope,
)
from sglang.srt.utils import get_compiler_backend


def test_split_qkv_rmsnorm_rope_fake_preserves_output_metadata():
    with FakeTensorMode():
        qkv = torch.empty((1, 4096), dtype=torch.bfloat16)
        sin = torch.empty((1, 128), dtype=torch.bfloat16)
        cos = torch.empty((1, 128), dtype=torch.bfloat16)
        weight = torch.empty((128,), dtype=torch.bfloat16)

        q, k, v = _split_qkv_rmsnorm_rope_fake(
            qkv,
            sin,
            cos,
            q_hidden_size=3072,
            kv_hidden_size=512,
            head_dim=128,
            eps=1e-6,
            q_weight=weight,
            k_weight=weight,
        )

    assert q.shape == (1, 3072)
    assert k.shape == (1, 512)
    assert v.shape == (1, 512)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16


def test_split_qkv_rmsnorm_rope_is_registered_as_custom_op():
    assert hasattr(torch.ops.sglang, "npu_split_qkv_rmsnorm_rope")


def test_split_qkv_rmsnorm_rope_is_opaque_to_symbolic_trace():
    def call_op(qkv, sin, cos, weight):
        return split_qkv_rmsnorm_rope(
            qkv,
            sin,
            cos,
            q_hidden_size=3072,
            kv_hidden_size=512,
            head_dim=128,
            eps=1e-6,
            q_weight=weight,
            k_weight=weight,
        )

    graph = make_fx(call_op, tracing_mode="fake")(
        torch.empty((1, 4096), dtype=torch.bfloat16),
        torch.empty((1, 128), dtype=torch.bfloat16),
        torch.empty((1, 128), dtype=torch.bfloat16),
        torch.empty((128,), dtype=torch.bfloat16),
    )

    call_targets = [
        node.target for node in graph.graph.nodes if node.op == "call_function"
    ]
    assert torch.ops.sglang.npu_split_qkv_rmsnorm_rope.default in call_targets


@pytest.mark.parametrize("batch_size", [1, 2])
def test_compiled_split_qkv_rmsnorm_rope_matches_external_kernel(batch_size):
    """The opaque boundary must preserve values, not only fake metadata.

    This is an NPU qualification test.  It intentionally compares the exact
    installed ``sgl_kernel_npu`` implementation with the registered custom op
    after Dynamo/TorchAir compilation for the two Qwen3-ASR compile buckets.
    """
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is required")

    device = torch.device("npu", torch_npu.npu.current_device())
    q_hidden_size = 3072
    kv_hidden_size = 512
    head_dim = 128
    total_hidden_size = q_hidden_size + 2 * kv_hidden_size

    torch.manual_seed(0)
    qkv = torch.randn(
        (batch_size, total_hidden_size), dtype=torch.bfloat16, device=device
    )
    sin = torch.randn((batch_size, head_dim), dtype=torch.bfloat16, device=device)
    cos = torch.randn((batch_size, head_dim), dtype=torch.bfloat16, device=device)
    weight = torch.randn((head_dim,), dtype=torch.bfloat16, device=device)

    def call_op(qkv, sin, cos, weight):
        return split_qkv_rmsnorm_rope(
            qkv,
            sin,
            cos,
            q_hidden_size=q_hidden_size,
            kv_hidden_size=kv_hidden_size,
            head_dim=head_dim,
            eps=1e-6,
            q_weight=weight,
            k_weight=weight,
        )

    expected = _split_qkv_rmsnorm_rope_kernel(
        qkv,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps=1e-6,
        q_weight=weight,
        k_weight=weight,
    )
    eager_wrapped = call_op(qkv, sin, cos, weight)
    compiled = torch.compile(
        call_op,
        fullgraph=True,
        dynamic=False,
        backend=get_compiler_backend("npugraph_ex"),
    )
    actual = compiled(qkv, sin, cos, weight)
    torch_npu.npu.synchronize()

    for wrapped_tensor, expected_tensor in zip(eager_wrapped, expected):
        torch.testing.assert_close(wrapped_tensor, expected_tensor, rtol=0, atol=0)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
