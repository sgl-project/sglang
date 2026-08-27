"""Which buffer the offloaded PLE prefetch writes into.

The prefetch gather runs on a side stream and writes into a buffer chosen
before the launch. Choosing the shared eager buffer while a CUDA graph is being
recorded bakes that buffer's address into the graph; the next eager forward
that needs more tokens replaces the buffer, and the graph then writes into a
freed block. The choice must therefore follow the driver's capture state, not
only the runner flag, because not every graph runner in the tree sets it.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.hyperconnection import GatedResidual
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpModel,
    Qwen4ExpPLEGroupedNorm,
    Qwen4ExpPLELayer,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

EMBED_DIM = 32


class _Stub(Qwen4ExpPLELayer):
    """A PLE layer reduced to its prefetch-buffer bookkeeping.

    The real constructor needs a checkpoint config and a distributed group, so
    only the module machinery and the two buffer fields are set up here; the
    buffer-selection code under test is inherited unchanged.
    """

    def __init__(self):
        torch.nn.Module.__init__(self)
        self._prefetch_stream = object()
        self._graph_prefetch_buffer = None
        self._graph_prefetch_buffers = {}
        self._eager_prefetch_buffer = None

    def _allocate_prefetch_buffer(self, lookup_tokens, device):
        return torch.zeros(
            lookup_tokens, EMBED_DIM, dtype=torch.bfloat16, device=device
        )


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("the PLE prefetch buffers are a CUDA path")


def test_graph_capture_does_not_bind_the_shared_eager_buffer():
    _require_cuda()
    stub = _Stub()
    ids = torch.zeros(4, dtype=torch.long, device="cuda")
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    stub.prepare_cuda_graph_prefetch_buffer(4, ids.device)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.cuda.graph(graph):
        during_capture = stub._get_prefetch_buffer(4, ids)
    torch.cuda.current_stream().wait_stream(side)

    assert stub._eager_prefetch_buffer is None
    assert stub._graph_prefetch_buffer.data_ptr() == during_capture.data_ptr()

    # The eager path still grows its single buffer and hands out a prefix of it,
    # which is exactly what must never be captured.
    small = stub._get_prefetch_buffer(2, ids)
    large = stub._get_prefetch_buffer(8, ids)
    assert small.shape[0] == 2 and large.shape[0] == 8
    assert stub._eager_prefetch_buffer.shape[0] == 8
    assert large.data_ptr() != during_capture.data_ptr()


def test_recapture_reuses_the_preallocated_graph_prefetch_buffer(monkeypatch):
    _require_cuda()
    stub = _Stub()
    ids = torch.zeros(8, dtype=torch.long, device="cuda")

    stub.prepare_cuda_graph_prefetch_buffer(4, ids.device)
    first = stub._graph_prefetch_buffer
    stub.prepare_cuda_graph_prefetch_buffer(4, ids.device)
    second = stub._graph_prefetch_buffer

    assert second.data_ptr() == first.data_ptr()

    monkeypatch.setattr(stub, "_is_capturing", lambda: True)
    monkeypatch.setattr(
        stub,
        "_allocate_prefetch_buffer",
        lambda *_: pytest.fail("capture allocated a PLE prefetch buffer"),
    )
    assert stub._get_prefetch_buffer(4, ids).data_ptr() == second.data_ptr()


def test_capture_rejects_an_unprepared_prefetch_size(monkeypatch):
    _require_cuda()
    stub = _Stub()
    ids = torch.zeros(8, dtype=torch.long, device="cuda")
    stub.prepare_cuda_graph_prefetch_buffer(4, ids.device)
    monkeypatch.setattr(stub, "_is_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="4 tokens available, 8 needed"):
        stub._get_prefetch_buffer(8, ids)


def test_zero_sized_prewarm_budget_uses_stable_capture_buffer(monkeypatch):
    stub = _Stub()
    ids = torch.zeros(8, dtype=torch.long)
    monkeypatch.setattr(stub, "_is_capturing", lambda: True)
    monkeypatch.setattr("sglang.srt.models.qwen4_exp.is_sm120_supported", lambda: True)

    first = stub._get_prefetch_buffer(8, ids)
    second = stub._get_prefetch_buffer(8, ids)

    assert first.data_ptr() == second.data_ptr()
    assert stub._graph_prefetch_buffers[8].data_ptr() == first.data_ptr()


def test_capture_probe_uses_cuda_runtime_status_on_nvidia(monkeypatch):
    stream = object()
    monkeypatch.setattr(
        "sglang.srt.models.qwen4_exp.get_is_capture_mode", lambda: False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: pytest.fail("torch capture probe is unreliable on NVIDIA"),
    )
    monkeypatch.setattr(
        "sglang.srt.models.qwen4_exp._is_stream_capturing",
        lambda current: current is stream,
        raising=False,
    )

    assert _Stub._is_capturing()


def test_model_prewarm_does_not_allocate_on_non_sm120(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(gather_dp_tokens=False)
    layer.prepare_cuda_graph_prefetch_buffer = MagicMock()
    model = torch.nn.Module()
    model.add_module("ple", layer)
    model._prewarm_cuda_graph_jit_kernels = MagicMock()
    runner = SimpleNamespace(
        device="cuda",
        model_config=SimpleNamespace(quantization=None),
        server_args=SimpleNamespace(
            speculative_num_draft_tokens=4,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full", bs=[8, 16], max_bs=16),
                decode=SimpleNamespace(backend="full", bs=[1, 8], max_bs=8),
            ),
        ),
        decode_num_tokens_per_req=lambda **_: 4,
    )
    monkeypatch.setattr("sglang.srt.models.qwen4_exp.is_sm120_supported", lambda: False)

    Qwen4ExpModel.prewarm_cuda_graphs(model, runner, capture_decode_cuda_graph=True)

    layer.prepare_cuda_graph_prefetch_buffer.assert_not_called()


def test_model_prewarm_does_not_allocate_on_sm121(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(gather_dp_tokens=False)
    layer.prepare_cuda_graph_prefetch_buffer = MagicMock()
    model = torch.nn.Module()
    model.add_module("ple", layer)
    model._prewarm_cuda_graph_jit_kernels = MagicMock()
    runner = SimpleNamespace(
        device="cuda",
        model_config=SimpleNamespace(quantization=None),
        server_args=SimpleNamespace(
            speculative_num_draft_tokens=4,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full", bs=[8, 16], max_bs=16),
                decode=SimpleNamespace(backend="full", bs=[1, 8], max_bs=8),
            ),
        ),
        decode_num_tokens_per_req=lambda **_: 4,
    )
    monkeypatch.setattr("sglang.srt.models.qwen4_exp.is_sm120_supported", lambda: True)
    monkeypatch.setattr(
        "sglang.srt.models.qwen4_exp.is_sm121", lambda: True, raising=False
    )

    Qwen4ExpModel.prewarm_cuda_graphs(model, runner, capture_decode_cuda_graph=True)

    layer.prepare_cuda_graph_prefetch_buffer.assert_not_called()


def test_model_prewarm_allocates_the_largest_sm120_capture_shape(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(gather_dp_tokens=False)
    layer.prepare_cuda_graph_prefetch_buffer = MagicMock()
    model = torch.nn.Module()
    model.add_module("ple", layer)
    model._prewarm_cuda_graph_jit_kernels = MagicMock()
    runner = SimpleNamespace(
        device="cuda",
        model_config=SimpleNamespace(quantization=None),
        server_args=SimpleNamespace(
            speculative_num_draft_tokens=4,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full", bs=[8, 16], max_bs=16),
                decode=SimpleNamespace(backend="full", bs=[1, 8], max_bs=8),
            ),
        ),
        decode_num_tokens_per_req=lambda **_: 4,
    )
    monkeypatch.setattr("sglang.srt.models.qwen4_exp.is_sm120_supported", lambda: True)

    Qwen4ExpModel.prewarm_cuda_graphs(model, runner, capture_decode_cuda_graph=True)

    layer.prepare_cuda_graph_prefetch_buffer.assert_called_once_with(
        32, torch.device("cuda")
    )


def test_jit_prewarm_uses_runtime_eligible_specializations(monkeypatch):
    _require_cuda()
    calls = []
    from sglang.kernels.ops.attention import qsa_indexer
    from sglang.kernels.ops.elementwise import fast_topk, hc_combine
    from sglang.kernels.ops.gemm import fp8_blockwise_gemm
    from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm

    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.block_topk = 512
    indexer.index_head_dim = 128
    indexer.q_layernorm = torch.nn.Module()
    indexer.q_layernorm.weight = torch.nn.Parameter(
        torch.zeros(128, dtype=torch.bfloat16, device="cuda")
    )
    indexer.rotary_emb = SimpleNamespace(
        rotary_dim=128,
        is_neox_style=False,
        mrope_interleaved_glm=False,
        mrope_section=None,
        cos_sin_cache=torch.empty(1, 128, dtype=torch.float32, device="cuda"),
    )

    combine = GatedResidual.__new__(GatedResidual)
    torch.nn.Module.__init__(combine)
    combine.hc_count = 4
    combine.hidden_size = 512
    combine._jit_combine_ok = True
    combine.block_inject_weight = torch.nn.Linear(
        2048, 4, bias=False, dtype=torch.bfloat16, device="cuda"
    )

    norm = Qwen4ExpPLEGroupedNorm(1024, group_size=512).to(
        device="cuda", dtype=torch.bfloat16
    )
    model = torch.nn.Module()
    model.add_module("indexer", indexer)
    model.add_module("combine", combine)
    model.add_module("norm", norm)

    monkeypatch.setattr(
        fast_topk, "_jit_fast_topk_module", lambda *a: calls.append(("topk", a))
    )
    monkeypatch.setattr(
        qsa_indexer, "_jit_qsa_indexer_module", lambda *a: calls.append(("qsa", a))
    )
    monkeypatch.setattr(
        hc_combine, "_jit_hc_combine_module", lambda *a: calls.append(("hc", a))
    )
    monkeypatch.setattr(
        grouped_gemma_rmsnorm,
        "_jit_grouped_gemma_rmsnorm_module",
        lambda *a: calls.append(("rms", a)),
    )
    monkeypatch.setattr(
        fp8_blockwise_gemm,
        "_jit_fp8_blockwise_module",
        lambda: calls.append(("fp8", ())),
    )

    Qwen4ExpModel._prewarm_cuda_graph_jit_kernels(model, quantization="fp8")

    assert calls == [
        ("topk", (512,)),
        ("qsa", (torch.bfloat16, 128, False)),
        ("hc", (4, 512, torch.bfloat16)),
        ("rms", (512, torch.bfloat16)),
        ("fp8", ()),
    ]


def test_jit_prewarm_skips_runtime_fallback_specializations(monkeypatch):
    _require_cuda()
    from sglang.kernels.ops.attention import qsa_indexer
    from sglang.kernels.ops.elementwise import fast_topk, hc_combine
    from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm

    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.block_topk = 512
    indexer.index_head_dim = 96
    indexer.q_layernorm = torch.nn.Module()
    indexer.q_layernorm.weight = torch.nn.Parameter(
        torch.zeros(96, dtype=torch.bfloat16, device="cuda")
    )
    indexer.rotary_emb = SimpleNamespace(
        rotary_dim=96,
        is_neox_style=True,
        mrope_interleaved_glm=False,
        mrope_section=None,
        cos_sin_cache=torch.empty(1, 96, dtype=torch.float32, device="cuda"),
    )
    combine = GatedResidual.__new__(GatedResidual)
    torch.nn.Module.__init__(combine)
    combine.hc_count = 3
    combine.hidden_size = 384
    combine._jit_combine_ok = False
    combine.block_inject_weight = torch.nn.Linear(
        1152, 3, bias=False, dtype=torch.bfloat16, device="cuda"
    )
    norm = Qwen4ExpPLEGroupedNorm(1152, group_size=384).to(
        device="cuda", dtype=torch.bfloat16
    )
    model = torch.nn.Module()
    model.add_module("indexer", indexer)
    model.add_module("combine", combine)
    model.add_module("norm", norm)

    monkeypatch.setattr(
        qsa_indexer,
        "_jit_qsa_indexer_module",
        lambda *_: pytest.fail("ineligible QSA indexer was prewarmed"),
    )
    monkeypatch.setattr(
        hc_combine,
        "_jit_hc_combine_module",
        lambda *_: pytest.fail("ineligible HC combine was prewarmed"),
    )
    monkeypatch.setattr(
        grouped_gemma_rmsnorm,
        "_jit_grouped_gemma_rmsnorm_module",
        lambda *_: pytest.fail("ineligible grouped norm was prewarmed"),
    )
    # The indexer is eligible, so its fast-topk prewarm is expected; stub the
    # JIT module so a dispatch test does not pay a real compile.
    prewarmed_topk = []
    monkeypatch.setattr(
        fast_topk, "_jit_fast_topk_module", lambda topk: prewarmed_topk.append(topk)
    )

    Qwen4ExpModel._prewarm_cuda_graph_jit_kernels(model, quantization=None)
    assert prewarmed_topk == [512]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
