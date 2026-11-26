import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

from sglang.srt.compilation import torch_compile_decoration
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


def _patch_model_compile_kwargs(*, is_hip):
    captured = {}

    def record_compile(fn, **kwargs):
        captured.update(kwargs)
        return fn

    with (
        patch.object(torch_compile_decoration, "_is_hip", is_hip),
        patch("torch.compile", side_effect=record_compile),
        torch_compile_decoration.patch_model(
            torch.nn.Linear(1, 1),
            enable_compile=True,
            num_tokens=1,
            tp_group=SimpleNamespace(ca_comm=None),
        ),
    ):
        pass
    return captured


def test_patch_model_takes_its_compile_knobs_from_the_environment():
    """`SGLANG_TORCH_DYNAMIC_SHAPE` must accept every boolean spelling the env
    registry does, as the CPU graph runner already does for the same variable,
    and stays a ROCm-only switch on this path."""
    with patch.dict(os.environ):
        os.environ.pop("SGLANG_TORCH_COMPILE_MODE", None)
        os.environ.pop("SGLANG_TORCH_DYNAMIC_SHAPE", None)
        assert _patch_model_compile_kwargs(is_hip=True) == {
            "mode": "max-autotune-no-cudagraphs",
            "dynamic": False,
        }
        os.environ["SGLANG_TORCH_COMPILE_MODE"] = "reduce-overhead"
        os.environ["SGLANG_TORCH_DYNAMIC_SHAPE"] = "yes"
        assert _patch_model_compile_kwargs(is_hip=True) == {
            "mode": "reduce-overhead",
            "dynamic": True,
        }
        assert _patch_model_compile_kwargs(is_hip=False) == {
            "mode": "reduce-overhead",
            "dynamic": False,
        }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
