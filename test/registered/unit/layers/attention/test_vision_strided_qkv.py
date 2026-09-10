"""CPU coverage for handing the packed qkv projection output to the backend uncopied."""

import os
import socket

import pytest
import torch
import torch.nn as nn

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention import vision
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

EMBED_DIM = 32
NUM_HEADS = 4
TOKENS = 6
CU_SEQLENS = torch.tensor([0, TOKENS], dtype=torch.int32)


class _RecordingBackend(nn.Module):
    """stands in for a real backend to observe what the layer hands it"""

    def __init__(self, **kwargs):
        super().__init__()
        self.received = None

    def forward(self, q, k, v, **kwargs):
        self.received = (q, k, v)
        return q


@pytest.fixture(scope="module")
def gloo_world():
    """a one-rank cpu process group, so the row-parallel proj can run"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.fixture
def single_rank(monkeypatch, gloo_world):
    # only the resolved backend name is consulted for gating, so a backend does
    # not need its platform present to be built here
    monkeypatch.setattr(
        vision.VisionAttention,
        "_determine_attention_backend",
        lambda self, passed_backend: passed_backend,
    )
    with (
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
        get_context().override_server_args(),
    ):
        yield


def _build(backend, **kwargs):
    """VisionAttention leaves its parameters uninitialized, so fill them in here"""
    layer = vision.VisionAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        projection_size=EMBED_DIM,
        use_qkv_parallel=True,
        qkv_backend=backend,
        flatten_batch=True,
        **kwargs,
    )
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return layer


@pytest.mark.parametrize(
    ("backend", "extra", "expected"),
    [
        ("fa3", {}, True),
        ("fa4", {}, True),
        ("triton_attn", {}, True),
        ("amx_attn", {}, True),
        ("sdpa", {}, False),
        ("flashinfer_cudnn", {}, False),
        ("xpu_attn", {}, False),
        ("aiter_attn", {}, False),
        # internvl normalizes q/k in place through a view that assumes a dense layout
        ("fa3", {"qk_normalization": True}, False),
        # the model-supplied applier owns the q/k layout, so it must get dense inputs
        ("fa3", {"customized_position_embedding_applier": lambda *args: None}, False),
        # this variant reshapes q/k into a dense copy of its own, so views are fine
        ("fa3", {"qk_normalization_by_head_size": True}, True),
    ],
)
def test_pass_strided_qkv_gating(monkeypatch, single_rank, backend, extra, expected):
    monkeypatch.setitem(vision.QKV_BACKEND_IMPL, backend, _RecordingBackend)

    assert _build(backend, **extra).pass_strided_qkv is expected


@pytest.mark.parametrize("num_kv_heads", [NUM_HEADS, 1])
def test_strided_qkv_carries_the_dense_values(monkeypatch, single_rank, num_kv_heads):
    """the strided views must hold exactly what the copies used to hold"""
    monkeypatch.setitem(vision.QKV_BACKEND_IMPL, "fa3", _RecordingBackend)
    monkeypatch.setitem(vision.QKV_BACKEND_IMPL, "sdpa", _RecordingBackend)
    torch.manual_seed(0)
    strided = _build("fa3", num_kv_heads=num_kv_heads)
    dense = _build("sdpa", num_kv_heads=num_kv_heads)
    dense.load_state_dict(strided.state_dict())
    assert strided.pass_strided_qkv and not dense.pass_strided_qkv
    x = torch.randn(1, TOKENS, EMBED_DIM)

    with torch.no_grad():
        strided(x, cu_seqlens=CU_SEQLENS)
        dense(x, cu_seqlens=CU_SEQLENS)

    packed = strided.qkv_backend.received[0].untyped_storage().data_ptr()
    for strided_tensor, dense_tensor in zip(
        strided.qkv_backend.received, dense.qkv_backend.received
    ):
        # one non-unit stride, over tokens, into the single buffer qkv_proj wrote
        assert not strided_tensor.is_contiguous()
        assert strided_tensor.stride(-1) == 1
        assert strided_tensor.untyped_storage().data_ptr() == packed
        assert dense_tensor.is_contiguous()
        assert torch.equal(strided_tensor, dense_tensor)


def test_strided_qkv_keeps_attention_output_bit_identical(monkeypatch, single_rank):
    monkeypatch.setitem(vision.QKV_BACKEND_IMPL, "fa3", vision.VisionSdpaAttention)
    torch.manual_seed(0)
    strided = _build("fa3")
    dense = _build("sdpa")
    dense.load_state_dict(strided.state_dict())
    assert strided.pass_strided_qkv and not dense.pass_strided_qkv
    x = torch.randn(1, TOKENS, EMBED_DIM)

    with torch.no_grad():
        strided_output = strided(x, cu_seqlens=CU_SEQLENS)
        dense_output = dense(x, cu_seqlens=CU_SEQLENS)

    torch.testing.assert_close(strided_output, dense_output, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
