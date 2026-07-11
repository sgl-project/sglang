from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

from sglang.srt.compilation.compile import (
    _infer_dynamic_arg_dims_from_annotations,
    _mark_dynamic_forward_batch,
    _runtime_dynamic_dim_for_argument,
)


class _MropeModel:
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
    ):
        return input_ids, positions, forward_batch


class _StringAnnotatedMropeModel:
    def forward(
        self,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        forward_batch,
    ):
        return input_ids, positions, forward_batch


def test_positions_marks_the_token_axis_dynamic_for_mrope_and_1d_rope():
    dynamic_dims = _infer_dynamic_arg_dims_from_annotations(_MropeModel.forward)
    string_dynamic_dims = _infer_dynamic_arg_dims_from_annotations(
        _StringAnnotatedMropeModel.forward
    )

    assert dynamic_dims["input_ids"] == 0
    assert dynamic_dims["positions"] == -1
    assert string_dynamic_dims["positions"] == -1


def test_runtime_dynamic_dim_uses_the_token_axis_for_mrope_metadata():
    assert _runtime_dynamic_dim_for_argument("positions") == -1
    assert _runtime_dynamic_dim_for_argument("position_ids") == -1
    assert _runtime_dynamic_dim_for_argument("mrope_positions") == -1
    assert _runtime_dynamic_dim_for_argument("input_ids") == 0


def test_forward_batch_marks_token_and_batch_metadata_dynamic():
    batch = SimpleNamespace(
        input_embeds=torch.empty(8, 16),
        seq_lens=torch.empty(2, dtype=torch.int64),
        mrope_positions=torch.empty(3, 8, dtype=torch.int64),
        scalar=torch.tensor(1),
    )
    marked = []

    def record_mark_dynamic(value, dims):
        marked.append((id(value), tuple(dims)))

    with patch("torch._dynamo.maybe_mark_dynamic", side_effect=record_mark_dynamic):
        _mark_dynamic_forward_batch(batch)

    assert (id(batch.input_embeds), (0,)) in marked
    assert (id(batch.seq_lens), (0,)) in marked
    assert (id(batch.mrope_positions), (1,)) in marked
    assert all(value_id != id(batch.scalar) for value_id, _ in marked)
