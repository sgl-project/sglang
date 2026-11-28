# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.distributed import pipeline_parallel as pp_module
from sglang.multimodal_gen.runtime.distributed.pipeline_parallel import (
    PipelineStagePlaceholder,
    overlap_stage_outputs,
    owned_by_other_stage,
    prune_pipeline_stage,
    run_stage_blocks,
    stage_layer_range,
)


class _Stage:
    """one pipeline stage of a fake group; stages share a wire in order"""

    def __init__(self, rank, world_size, wire):
        self.rank_in_group = rank
        self.world_size = world_size
        self.ranks = list(range(world_size))
        self.device_group = None
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1
        self.wire = wire

    def send(self, tensor):
        self.wire.append(tensor.clone())

    def recv(self, shape, dtype):
        tensor = self.wire.pop(0)
        assert tensor.shape == shape and tensor.dtype == dtype
        return tensor


class _Work:
    def __init__(self, log):
        self.log = log

    def wait(self):
        self.log.append("wait")


def _fake_broadcast(monkeypatch, group, log):
    """the last stage keeps its output; other stages receive group.wire_output"""

    def broadcast(tensor, src, group=None, async_op=False):
        log.append("broadcast")
        if active.rank_in_group != src:
            tensor.copy_(active.wire_output)
        return _Work(log)

    active = group
    monkeypatch.setattr(pp_module.torch.distributed, "broadcast", broadcast)


class _Toy(nn.Module):
    _pp_layers = "blocks"

    def __init__(self, num_layers=5):
        super().__init__()
        self.embed = nn.Linear(4, 8)
        self.blocks = nn.ModuleList(nn.Linear(8, 8) for _ in range(num_layers))
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        hidden = run_stage_blocks(
            self,
            self.blocks,
            self.embed(x),
            lambda index, block, h: h + torch.tanh(block(h)),
        )
        return self.head(hidden)


@pytest.mark.parametrize(
    "layers, stages, expected",
    [
        (30, 2, [range(0, 15), range(15, 30)]),
        (30, 4, [range(0, 8), range(8, 16), range(16, 23), range(23, 30)]),
        (5, 3, [range(0, 2), range(2, 4), range(4, 5)]),
    ],
)
def test_stage_layer_range_is_contiguous_and_balanced(layers, stages, expected):
    assert [stage_layer_range(layers, s, stages) for s in range(stages)] == expected


def test_stage_layer_range_rejects_more_stages_than_blocks():
    with pytest.raises(ValueError):
        stage_layer_range(2, 0, 3)


def test_prune_is_a_no_op_without_pipeline_groups(monkeypatch):
    monkeypatch.setattr(parallel_state, "_PP", None)
    model = _Toy()
    prune_pipeline_stage(model)
    assert not hasattr(model, "_pp_owned")
    assert all(isinstance(block, nn.Linear) for block in model.blocks)


@pytest.mark.parametrize("stages", [2, 3])
def test_stages_reproduce_the_unsplit_model_bitwise(monkeypatch, stages):
    torch.manual_seed(0)
    reference = _Toy()
    x = torch.randn(2, 6, 4)
    expected = reference(x)

    wire = []
    outputs = []
    for rank in range(stages):
        group = _Stage(rank, stages, wire)
        monkeypatch.setattr(parallel_state, "_PP", group)
        monkeypatch.setattr(pp_module, "get_pp_group", lambda group=group: group)
        model = _Toy()
        prune_pipeline_stage(model)
        owned = model._pp_owned
        assert owned == stage_layer_range(5, rank, stages)
        for index, block in enumerate(model.blocks):
            assert isinstance(block, PipelineStagePlaceholder) == (index not in owned)
            assert owned_by_other_stage(model, f"blocks.{index}.weight") == (
                index not in owned
            )
        assert not owned_by_other_stage(model, "head.weight")
        # same weights as the reference for this stage's blocks and the shared ends
        state = {
            name: value
            for name, value in reference.state_dict().items()
            if not owned_by_other_stage(model, name)
        }
        model.load_state_dict(state)
        group.wire_output = None if rank == stages - 1 else expected
        _fake_broadcast(monkeypatch, group, [])
        outputs.append(model(x))
    assert not wire
    # the forward hook replaces earlier stages' outputs with the last stage's
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_overlapped_forwards_wait_for_outputs_on_exit(monkeypatch):
    log = []
    group = _Stage(0, 2, [])
    group.wire_output = torch.ones(2, 6, 3)
    monkeypatch.setattr(parallel_state, "_PP", group)
    monkeypatch.setattr(pp_module, "get_pp_group", lambda: group)
    _fake_broadcast(monkeypatch, group, log)
    model = _Toy()
    prune_pipeline_stage(model)
    x = torch.randn(2, 6, 4)

    model(x)
    assert log == ["broadcast", "wait"]
    log.clear()
    with overlap_stage_outputs():
        model(x)
        model(x)
        assert log == ["broadcast", "broadcast"]
    assert log == ["broadcast", "broadcast", "wait", "wait"]
    assert len(group.wire) == 3


def test_pruned_blocks_refuse_to_run():
    with pytest.raises(RuntimeError, match="another pipeline stage"):
        PipelineStagePlaceholder(3)(torch.zeros(1))


def test_boundary_rejects_a_stream_that_changes_shape(monkeypatch):
    group = _Stage(0, 2, [])
    monkeypatch.setattr(pp_module, "get_pp_group", lambda: group)
    model = SimpleNamespace(_pp_owned=range(0, 1))
    with pytest.raises(RuntimeError, match="shape and dtype"):
        run_stage_blocks(
            model,
            [nn.Identity(), nn.Identity()],
            torch.zeros(2, 3),
            lambda index, block, h: torch.zeros(2, 4),
        )
