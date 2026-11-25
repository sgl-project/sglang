# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 request-batched block AdaLN (SGLANG_DIFFUSION_H3_ADALN_REQUEST_BATCH)."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTModel,
)

pytestmark = pytest.mark.usefixtures("single_process_model_parallel")

_SMALL = MiniMaxH3DiTArchConfig(
    num_layers=3,
    hidden_size=4,
    time_embed_dim=8,
    adaln_out_features=6 * MINIMAX_H3_ADALN_MODALITY_NUM * 4,
)


def _is_gfx950() -> bool:
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    )


requires_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="needs a ROCm gfx950 GPU")


def _fake_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    freqs = torch.arange(1, dim + 1, device=timesteps.device, dtype=torch.float32)
    return 3 * torch.sin(7 * timesteps.float()[:, None] * freqs)


def _model(arch, num_blocks: int, device, scale: float = 0.5) -> MiniMaxH3DiTModel:
    """A DiT shell holding only the block AdaLN projections."""
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = arch
    model.adaln_cache = None
    model.adaln_t_table = None
    generator = torch.Generator().manual_seed(0)
    blocks = nn.ModuleList()
    for index in range(num_blocks):
        block = MiniMaxH3DiTBlock.__new__(MiniMaxH3DiTBlock)
        nn.Module.__init__(block)
        block.layer_index = index
        block.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            None,
            prefix=f"blocks.{index}.adaln_proj",
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        )
        with torch.no_grad():
            for param in block.adaln_proj.parameters():
                param.copy_(scale * torch.randn(param.shape, generator=generator))
        blocks.append(block)
    model.blocks = blocks.to(device)
    model._time_embedding = lambda t: _fake_time_embedding(t, arch.time_embed_dim)
    return model


def _step_timesteps(num_steps: int, device) -> list[torch.Tensor]:
    """Per-step unique timesteps like the H3 denoise loop: 1 to 3 values per step."""
    steps = []
    for step in range(num_steps):
        t = step / num_steps
        candidates = [t, max(t, 0.25), t, max(t, 0.5 if step % 5 else 1.0)]
        unique = torch.unique(
            torch.tensor(candidates, dtype=torch.float32), sorted=True
        )
        steps.append(unique.to(device))
    return steps


def _bits(timesteps: torch.Tensor) -> tuple[int, ...]:
    return tuple(timesteps.cpu().view(torch.int32).tolist())


def _adaln_input(model, timesteps: torch.Tensor) -> torch.Tensor:
    return F.silu(model._time_embedding(timesteps)).to(torch.bfloat16)


def _per_step_reference(model, timesteps: torch.Tensor):
    """Each block projecting this step alone, as the per-step path does."""
    x = _adaln_input(model, timesteps)
    return x, tuple(
        block.adaln_proj.split_output(
            MiniMaxH3AdalnProj.project_local(block.adaln_proj, x)
        )
        for block in model.blocks
    )


def _assert_same_params(got, expected):
    assert len(got) == len(expected)
    for got_block, expected_block in zip(got, expected):
        assert len(got_block) == len(expected_block) == 6
        for g, e in zip(got_block, expected_block):
            assert g.shape == e.shape and g.stride() == e.stride()
            assert torch.equal(g, e)


@pytest.fixture
def request_batch(monkeypatch):
    monkeypatch.setattr(h3, "_H3_ADALN_REQUEST_BATCH", True)
    monkeypatch.setattr(h3, "_H3_ADALN_REQUEST_CHECK", False)
    if not torch.cuda.is_available():
        # The capture query needs a device runtime; there is nothing to capture here.
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _record_rows(model) -> list[list[int]]:
    rows = [[] for _ in model.blocks]
    for index, block in enumerate(model.blocks):
        project = block.adaln_proj.project_local

        def recording(x, project=project, sink=rows[index]):
            sink.append(int(x.shape[0]))
            return project(x)

        block.adaln_proj.project_local = recording
    return rows


@torch.inference_mode()
def test_prepare_records_unique_plans_keyed_like_forward(request_batch):
    model = _model(_SMALL, 2, "cpu")
    distinct = _step_timesteps(6, "cpu")
    steps = distinct + [t.clone() for t in reversed(distinct)]
    assert model.prepare_adaln_plans(steps) is None
    request = model.__dict__["_h3_adaln_request"]
    assert list(request["plans"]) == [_bits(t) for t in distinct]
    assert request["inputs"] is None and request["table"] is None


@torch.inference_mode()
def test_env_off_keeps_the_per_step_path(monkeypatch):
    monkeypatch.setattr(h3, "_H3_ADALN_REQUEST_BATCH", False)
    model = _model(_SMALL, 2, "cpu")
    steps = _step_timesteps(4, "cpu")
    model.prepare_adaln_plans(steps)
    assert model.__dict__["_h3_adaln_request"] is None
    x = _adaln_input(model, steps[0])
    assert model._h3_adaln_request_params(x, _bits(steps[0])) is None


@torch.inference_mode()
def test_batched_projection_equals_per_step_projection(request_batch):
    model = _model(_SMALL, _SMALL.num_layers, "cpu")
    steps = _step_timesteps(49, "cpu")
    model.prepare_adaln_plans(steps)
    for timesteps in steps:
        x, expected = _per_step_reference(model, timesteps)
        got = model._h3_adaln_request_params(x, _bits(timesteps))
        assert got is not None
        _assert_same_params(got, expected)


@torch.inference_mode()
def test_plans_of_equal_width_share_gemms_of_at_most_16_rows(request_batch):
    model = _model(_SMALL, 2, "cpu")
    rows = _record_rows(model)
    width2 = [torch.tensor([0.01 * i, 0.5 + 0.01 * i]) for i in range(20)]
    width3 = [torch.tensor([0.02 * i, 0.4, 0.9]) for i in range(1, 5)]
    width17 = [torch.linspace(0.0, 1.0, 17)]
    steps = width2 + width3 + width17
    model.prepare_adaln_plans(steps)
    first = steps[0]
    model._h3_adaln_request_params(_adaln_input(model, first), _bits(first))
    # 8 width-2 plans per GEMM, 5 width-3 plans per GEMM, a wider plan alone
    assert rows == [[16, 16, 8, 12, 17]] * 2
    for timesteps in steps:
        x, expected = _per_step_reference(model, timesteps)
        _assert_same_params(
            model._h3_adaln_request_params(x, _bits(timesteps)), expected
        )
    # built once per request
    assert rows == [[16, 16, 8, 12, 17]] * 2


@torch.inference_mode()
def test_unknown_timestep_key_falls_back_without_building(request_batch):
    model = _model(_SMALL, 2, "cpu")
    model.prepare_adaln_plans(_step_timesteps(4, "cpu"))
    other = torch.tensor([0.123, 0.456])
    assert (
        model._h3_adaln_request_params(_adaln_input(model, other), _bits(other)) is None
    )
    assert model._h3_adaln_request_params(_adaln_input(model, other), None) is None
    assert model.__dict__["_h3_adaln_request"]["table"] is None


@torch.inference_mode()
def test_next_request_replaces_the_plan(request_batch):
    model = _model(_SMALL, 2, "cpu")
    first = _step_timesteps(4, "cpu")
    model.prepare_adaln_plans(first)
    model._h3_adaln_request_params(_adaln_input(model, first[1]), _bits(first[1]))
    assert model.__dict__["_h3_adaln_request"]["table"] is not None
    second = [torch.tensor([0.9, 0.95])]
    model.prepare_adaln_plans(second)
    request = model.__dict__["_h3_adaln_request"]
    assert list(request["plans"]) == [_bits(second[0])]
    assert request["table"] is None


class _ForeignBlock(MiniMaxH3DiTBlock):
    pass


def _fallback(case, model, monkeypatch):
    if case == "tensor_parallel":
        monkeypatch.setattr(h3, "get_tp_world_size", lambda: 2)
    elif case == "graph_capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif case == "torch_compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif case == "layerwise_offload":
        monkeypatch.setattr(h3, "is_layerwise_offloaded_module", lambda module: True)
    elif case == "cache_dit_env":
        monkeypatch.setattr(h3.envs, "SGLANG_CACHE_DIT_ENABLED", True, raising=False)
    elif case == "cache_dit_adapter":
        model._sglang_cache_dit_adapter = object()
    elif case == "adaln_cache":
        model.adaln_cache = object()
    elif case == "fp32_weight":
        weight = model.blocks[1].adaln_proj.linear.weight
        weight.data = weight.data.float()
    elif case == "missing_adaln_proj":
        model.blocks[1].adaln_proj = None
    elif case == "foreign_block":
        block = _ForeignBlock.__new__(_ForeignBlock)
        nn.Module.__init__(block)
        block.adaln_proj = model.blocks[1].adaln_proj
        model.blocks[1] = block


@torch.no_grad()
@pytest.mark.parametrize(
    "case",
    [
        "tensor_parallel",
        "graph_capture",
        "torch_compile",
        "layerwise_offload",
        "cache_dit_env",
        "cache_dit_adapter",
        "adaln_cache",
        "fp32_input",
        "fp32_weight",
        "missing_adaln_proj",
        "foreign_block",
    ],
)
def test_unsupported_configurations_fall_back_without_building(
    case, request_batch, monkeypatch
):
    model = _model(_SMALL, 2, "cpu")
    steps = _step_timesteps(4, "cpu")
    model.prepare_adaln_plans(steps)
    request = model.__dict__["_h3_adaln_request"]
    x = _adaln_input(model, steps[0])
    if case == "fp32_input":
        x = x.float()
    _fallback(case, model, monkeypatch)
    assert model._h3_adaln_request_params(x, _bits(steps[0])) is None
    assert request["table"] is None


@torch.inference_mode()
def test_check_mode_rejects_an_input_that_differs_from_the_plan(
    request_batch, monkeypatch
):
    monkeypatch.setattr(h3, "_H3_ADALN_REQUEST_CHECK", True)
    model = _model(_SMALL, 2, "cpu")
    steps = _step_timesteps(4, "cpu")
    model.prepare_adaln_plans(steps)
    x = _adaln_input(model, steps[0])
    assert model._h3_adaln_request_params(x, _bits(steps[0])) is not None
    with pytest.raises(RuntimeError, match="differs from forward's"):
        model._h3_adaln_request_params(x + 1, _bits(steps[0]))


@requires_gfx950
@torch.inference_mode()
def test_batched_projection_is_bit_identical_at_h3_size_on_gfx950(request_batch):
    """hipBLASLt keeps the single-plan kernel up to 16 rows at the real shape."""
    arch = MiniMaxH3DiTArchConfig()
    model = _model(arch, 3, "cuda", scale=0.02)
    steps = _step_timesteps(49, "cuda")
    rows = _record_rows(model)
    model.prepare_adaln_plans(steps)
    for timesteps in steps:
        x, expected = _per_step_reference(model, timesteps)
        _assert_same_params(
            model._h3_adaln_request_params(x, _bits(timesteps)), expected
        )
    assert max(rows[0]) <= 16 and len(rows[0]) < len(steps)
