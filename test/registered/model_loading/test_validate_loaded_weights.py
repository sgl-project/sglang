"""Unit tests for the post-load weight validator that defends against the
silent NaN-corruption failure mode reported in sgl-project/sglang#26745.

The validator inspects every floating-point parameter each module owns
directly, on a bounded prefix per tensor, looking for the
``torch.empty()`` byte-pattern signature (NaN / Inf / |x| > 1e30) that
uninitialized memory exposes on float dtypes.

These tests run on CPU only and do not require any model checkpoint.
"""

import unittest

import torch

from sglang.srt.model_executor.model_runner import (
    _UNINITIALIZED_MAGNITUDE_THRESHOLD,
    _UNINITIALIZED_SAMPLE_BUDGET,
    _has_uninitialized_signature,
    _validate_loaded_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# _has_uninitialized_signature — per-tensor heuristic
# ---------------------------------------------------------------------------


class TestHasUninitializedSignature(unittest.TestCase):
    """Pin the contract of the per-tensor heuristic.

    Returns ``True`` iff the sampled prefix contains NaN, Inf, or any
    element above :data:`_UNINITIALIZED_MAGNITUDE_THRESHOLD` — the trio
    of values that ``torch.empty()`` typically yields when its bit
    pattern is decoded as a float.
    """

    def test_zeros_are_initialized(self) -> None:
        assert _has_uninitialized_signature(torch.zeros(8)) is False

    def test_normal_weights_are_initialized(self) -> None:
        assert _has_uninitialized_signature(torch.randn(8)) is False

    def test_nan_in_prefix_is_uninitialized(self) -> None:
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET + 64)
        tensor[3] = float("nan")
        assert _has_uninitialized_signature(tensor) is True

    def test_inf_in_prefix_is_uninitialized(self) -> None:
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET + 64)
        tensor[2] = float("inf")
        assert _has_uninitialized_signature(tensor) is True

    def test_neg_inf_in_prefix_is_uninitialized(self) -> None:
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET + 64)
        tensor[1] = float("-inf")
        assert _has_uninitialized_signature(tensor) is True

    def test_huge_magnitude_in_prefix_is_uninitialized(self) -> None:
        # 1e35 is well above the threshold, mirroring the absurd
        # magnitudes ``torch.empty()`` can produce when raw memory bytes
        # are interpreted as a float.
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET + 64)
        tensor[0] = 1e35
        assert _has_uninitialized_signature(tensor) is True

    def test_threshold_boundary_just_over(self) -> None:
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET)
        tensor[0] = 1.5 * _UNINITIALIZED_MAGNITUDE_THRESHOLD
        assert _has_uninitialized_signature(tensor) is True

    def test_threshold_boundary_just_under(self) -> None:
        # Just below the threshold must not flip the heuristic, even
        # though it is still numerically large.
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET)
        tensor[0] = _UNINITIALIZED_MAGNITUDE_THRESHOLD / 10.0
        assert _has_uninitialized_signature(tensor) is False

    def test_empty_tensor_is_initialized(self) -> None:
        # A zero-sized tensor has nothing to inspect — treating it as
        # uninitialized would force callers to special-case empty
        # buffers.
        assert _has_uninitialized_signature(torch.empty(0)) is False

    def test_nan_outside_sample_prefix_not_inspected(self) -> None:
        # Sampling is bounded to ``_UNINITIALIZED_SAMPLE_BUDGET``
        # leading elements. A NaN past the budget is intentionally
        # missed — the heuristic trades completeness for an O(budget)
        # per-tensor cost. uninitialized memory is dense enough in
        # practice (NaN/Inf occupy ~50% of decoded float bytes) that
        # this rarely matters; this test pins the contract so it does
        # not silently change.
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET * 2)
        tensor[_UNINITIALIZED_SAMPLE_BUDGET + 10] = float("nan")
        assert _has_uninitialized_signature(tensor) is False

    def test_bfloat16_dtype_supported(self) -> None:
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET, dtype=torch.bfloat16)
        tensor[5] = float("nan")
        assert _has_uninitialized_signature(tensor) is True

    def test_float16_dtype_supported(self) -> None:
        tensor = torch.zeros(_UNINITIALIZED_SAMPLE_BUDGET, dtype=torch.float16)
        tensor[5] = float("inf")
        assert _has_uninitialized_signature(tensor) is True


# ---------------------------------------------------------------------------
# _validate_loaded_weights — module-level walk
# ---------------------------------------------------------------------------


def _make_initialized_param(
    *shape: int, dtype: torch.dtype = torch.float32
) -> torch.nn.Parameter:
    """Allocate a parameter and fill it with sane (initialized) values."""
    return torch.nn.Parameter(torch.zeros(*shape, dtype=dtype))


def _make_uninitialized_param(
    *shape: int, dtype: torch.dtype = torch.float32
) -> torch.nn.Parameter:
    """Allocate a parameter pre-filled with NaN to simulate the
    ``torch.empty()`` signature deterministically (raw ``torch.empty``
    can occasionally produce sane bytes, which would make the test
    flaky).
    """
    tensor = torch.empty(*shape, dtype=dtype)
    tensor.fill_(float("nan"))
    return torch.nn.Parameter(tensor)


class _SyntheticVLM(torch.nn.Module):
    """Two-tower model emulating the VLM-with-missing-encoder shape from #26745.

    Both towers are populated by leaf ``Linear`` modules. By default the
    visual tower is left uninitialized (NaN-filled) and the language
    tower is fully initialized — matching the production failure where
    Megatron only saved the language model.
    """

    def __init__(
        self,
        visual_initialized: bool = False,
        language_initialized: bool = True,
    ) -> None:
        super().__init__()
        make_v = (
            _make_initialized_param if visual_initialized else _make_uninitialized_param
        )
        make_l = (
            _make_initialized_param
            if language_initialized
            else _make_uninitialized_param
        )

        self.visual = torch.nn.Module()
        self.visual.proj = torch.nn.Linear(4, 4)
        self.visual.proj.weight = make_v(4, 4)
        self.visual.proj.bias = make_v(4)

        self.language = torch.nn.Module()
        self.language.embed = torch.nn.Linear(4, 4)
        self.language.embed.weight = make_l(4, 4)
        self.language.embed.bias = make_l(4)


class TestValidateLoadedWeights(unittest.TestCase):
    """Pin the module-walk semantics of ``_validate_loaded_weights``.

    Walks ``named_modules()`` and for every module inspects only its
    direct floating params (``recurse=False``). This catches
    non-leaf modules that own ``torch.empty()`` parameters of their
    own (e.g. some decoder layers that mix child sub-modules with
    directly owned expert weights), while still avoiding double
    inspection of any single parameter.
    """

    def test_clean_model_returns_empty_with_zero_total(self) -> None:
        model = _SyntheticVLM(visual_initialized=True, language_initialized=True)
        sample, total = _validate_loaded_weights(model)
        assert sample == []
        assert total == 0

    def test_uninitialized_visual_tower_is_flagged(self) -> None:
        model = _SyntheticVLM(visual_initialized=False, language_initialized=True)
        sample, total = _validate_loaded_weights(model)
        # Visual proj has both weight and bias uninitialized; the
        # validator should surface both (recurse=False per module
        # walks all direct params, not just the first).
        assert any("visual.proj.weight" in name for name in sample)
        assert any("visual.proj.bias" in name for name in sample)
        assert not any("language" in name for name in sample)
        assert total >= 2

    def test_both_towers_uninitialized_reports_both(self) -> None:
        model = _SyntheticVLM(visual_initialized=False, language_initialized=False)
        sample, total = _validate_loaded_weights(model)
        assert any("visual.proj" in name for name in sample)
        assert any("language.embed" in name for name in sample)
        assert total >= 4

    def test_max_report_caps_sample_but_not_total(self) -> None:
        # Construct ``max_report + 5`` leaf modules, each with two
        # uninitialized params, to confirm that ``sample`` is capped
        # at ``max_report`` while ``total`` keeps counting past it.
        class _ManyLeaves(torch.nn.Module):
            def __init__(self, n: int) -> None:
                super().__init__()
                for i in range(n):
                    leaf = torch.nn.Linear(2, 2)
                    leaf.weight = _make_uninitialized_param(2, 2)
                    leaf.bias = _make_uninitialized_param(2)
                    self.add_module(f"leaf_{i}", leaf)

        model = _ManyLeaves(15)
        sample, total = _validate_loaded_weights(model, max_report=10)
        assert len(sample) == 10
        assert total == 30  # 15 leaves * 2 params each

    def test_meta_tensors_are_skipped(self) -> None:
        # ``meta`` tensors are placeholders by construction (they have
        # no storage), so the heuristic must not flag them — otherwise
        # FSDP / lazy-init paths would always raise.
        class _MetaModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(2, 2, device="meta")

        sample, total = _validate_loaded_weights(_MetaModel())
        assert sample == []
        assert total == 0

    def test_integer_only_module_is_skipped(self) -> None:
        # Quantized layers expose int8/int4 weights as ``Parameter``;
        # these do not carry the ``torch.empty()`` NaN-bit signature
        # reliably, so the validator must skip non-floating dtypes
        # rather than emit false positives.
        class _IntOnly(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q = torch.nn.Module()
                self.q.weight = torch.nn.Parameter(
                    torch.zeros(2, 2, dtype=torch.int8), requires_grad=False
                )

        sample, total = _validate_loaded_weights(_IntOnly())
        assert sample == []
        assert total == 0

    def test_secondary_param_uninitialized_is_caught(self) -> None:
        # Direct contract test for the recurse=False per-module walk:
        # ``leaf.weight`` is clean, but ``leaf.bias`` is dirty. Both
        # are direct params of the same leaf, and the validator must
        # surface the dirty one (an earlier first-param-only sampling
        # design would have missed this).
        leaf = torch.nn.Linear(2, 2)
        leaf.weight = _make_initialized_param(2, 2)
        leaf.bias = _make_uninitialized_param(2)
        model = torch.nn.Module()
        model.add_module("leaf", leaf)
        sample, total = _validate_loaded_weights(model)
        assert total == 1
        assert len(sample) == 1
        assert "leaf.bias" in sample[0]

    def test_non_leaf_direct_param_is_caught(self) -> None:
        # Real models like ``DeepseekV4DecoderLayer`` mix child
        # sub-modules with parameters they own directly. The validator
        # must inspect those direct params even though the module also
        # has children.
        class _MixedContainer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = torch.nn.Linear(2, 2)  # has its own params
                self.shared_bias = _make_uninitialized_param(2)  # direct

        model = torch.nn.Module()
        model.add_module("layer", _MixedContainer())
        sample, total = _validate_loaded_weights(model)
        assert any("layer.shared_bias" in name for name in sample)
        assert total == 1

    def test_bf16_uninitialized_is_caught(self) -> None:
        leaf = torch.nn.Module()
        leaf.weight = _make_uninitialized_param(2, 2, dtype=torch.bfloat16)
        model = torch.nn.Module()
        model.add_module("leaf", leaf)
        sample, total = _validate_loaded_weights(model)
        assert total == 1
        assert "leaf.weight" in sample[0]

    def test_requires_grad_false_floating_param_is_inspected(self) -> None:
        # Some loaders register float buffers as ``Parameter(...,
        # requires_grad=False)``. The validator does not filter on
        # ``requires_grad``, so an uninitialized inference-only param
        # still raises.
        leaf = torch.nn.Module()
        bad = torch.nn.Parameter(torch.full((4,), float("nan")), requires_grad=False)
        leaf.scale = bad
        model = torch.nn.Module()
        model.add_module("leaf", leaf)
        sample, total = _validate_loaded_weights(model)
        assert total == 1
        assert "leaf.scale" in sample[0]


if __name__ == "__main__":
    unittest.main()
