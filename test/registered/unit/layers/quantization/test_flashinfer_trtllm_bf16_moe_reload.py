"""CPU unit tests for reloading BF16 MoE weights under the TRT-LLM backends.

``UnquantizedFusedMoEMethod`` rewrites BF16 expert weights into the flashinfer
TRT-LLM BlockMajorK layout after loading, so the weight loader has to undo that
layout to copy canonical checkpoint tensors in and something has to put it back.
The disk and checkpoint-engine paths re-run ``process_weights_after_loading``;
``update_weights_from_tensor`` / ``_from_distributed`` do not.

Pins both halves of issue #27787: the undo was gated on
``is_flashinfer_trtllm_routed()`` while the rewrite was gated on
``use_flashinfer_trtllm_moe`` (routed *or* non-routed), and widening that gate
alone just trades the crash for a parameter left canonical while the kernel
reads BlockMajorK.

The layout arithmetic lives in flashinfer and needs SM100, so its entry points
are replaced by a deterministic stand-in that is a genuine reordering. Whether
the stand-in matches BlockMajorK byte for byte is the kernel's contract and is
covered on-device.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _repack_weights_after_hot_update,
)

NUM_EXPERTS = 2
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 128


def _fake_permute_indices(weight_u8):
    """A real, order-changing permutation of the expert tile's rows.

    Deliberately not an involution: a reversal is its own inverse and would let
    the restore re-apply the forward permutation instead of inverting it.
    """
    return torch.roll(torch.arange(weight_u8.shape[0]), 1)


def _fake_convert_to_block_layout(weight_u8, block_k):
    """Stand-in for flashinfer's [N, K] -> [K // block_k, N, block_k] rewrite."""
    n, k = weight_u8.shape
    return weight_u8.view(n, k // block_k, block_k).permute(1, 0, 2).contiguous()


def _mock_flashinfer():
    core = ModuleType("flashinfer.fused_moe.core")
    core._maybe_get_cached_w3_w1_permute_indices = (
        lambda cache, weight_u8, tile_m, **kwargs: _fake_permute_indices(weight_u8)
    )
    core.get_w2_permute_indices_with_cache = (
        lambda cache, weight_u8, tile_m: _fake_permute_indices(weight_u8)
    )
    core.convert_to_block_layout = _fake_convert_to_block_layout

    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.__path__ = []
    fused_moe.core = core
    flashinfer.fused_moe = fused_moe
    return patch.dict(
        sys.modules,
        {
            "flashinfer": flashinfer,
            "flashinfer.fused_moe": fused_moe,
            "flashinfer.fused_moe.core": core,
        },
    )


class _FakeMoELayer(torch.nn.Module):
    """The surface of ``FusedMoE`` that the layout code actually touches."""

    def __init__(self, method, seed: int):
        super().__init__()
        self.quant_method = method
        self.num_local_experts = NUM_EXPERTS
        self.hidden_size = HIDDEN_SIZE
        self.intermediate_size_per_partition = INTERMEDIATE_SIZE
        self.moe_runner_config = SimpleNamespace(is_gated=True)
        generator = torch.Generator().manual_seed(seed)
        self.w13_weight = torch.nn.Parameter(
            torch.randn(
                NUM_EXPERTS,
                2 * INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(
                NUM_EXPERTS,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                generator=generator,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )


class _FakeModel(torch.nn.Module):
    def __init__(self, layer: _FakeMoELayer):
        super().__init__()
        self.layer = layer


def _make_method(use_flashinfer_trtllm_moe: bool = True):
    return UnquantizedFusedMoEMethod(
        use_flashinfer_trtllm_moe=use_flashinfer_trtllm_moe
    )


def _canonical_shapes():
    return {
        "w13_weight": (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "w2_weight": (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
    }


class TestFlashInferTrtllmBf16MoEReload(unittest.TestCase):
    def _cold_load(self, seed: int):
        """A layer loaded from disk and post-processed, i.e. in kernel layout."""
        method = _make_method()
        layer = _FakeMoELayer(method, seed=seed)
        with _mock_flashinfer():
            method._maybe_apply_flashinfer_trtllm_bf16_block_layout(layer)
        return method, layer

    def _restore_for_load(self, method, layer, param_names=("w13", "w2")):
        with _mock_flashinfer():
            for short in param_names:
                name = f"{short}_weight"
                method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                    layer=layer,
                    param=getattr(layer, name),
                    weight_name=f"model.layers.0.mlp.experts.{name}",
                )

    def test_process_weights_after_loading_applies_block_layout(self):
        """Pins the wiring: the split-out helper is still called on the load path."""
        method = _make_method()
        layer = _FakeMoELayer(method, seed=0)
        with patch.object(
            method, "_maybe_apply_flashinfer_trtllm_bf16_block_layout"
        ) as apply_layout:
            method.process_weights_after_loading(layer)
        apply_layout.assert_called_once_with(layer)

    def test_block_layout_changes_shape(self):
        """Guards the premise of every other test in this file."""
        _, layer = self._cold_load(seed=0)
        for name, canonical in _canonical_shapes().items():
            self.assertNotEqual(tuple(getattr(layer, name).data.shape), canonical)

    def test_restore_runs_for_non_routed_backend(self):
        """Regression for #27787: the undo must fire for plain flashinfer_trtllm.

        No moe-runner-backend global is set, so gating the undo on
        ``is_flashinfer_trtllm_routed()`` leaves the weights blocked -- the state
        that made the hot copy raise "size of tensor a (64) ... b (2048)".
        """
        method, layer = self._cold_load(seed=0)
        self._restore_for_load(method, layer)
        for name, canonical in _canonical_shapes().items():
            self.assertEqual(tuple(getattr(layer, name).data.shape), canonical)

    def test_reload_reproduces_cold_load_layout(self):
        """A hot update must leave the weights exactly as a cold load would.

        A gate-widening-only fix does not pass this: the copy succeeds, but
        without the re-derive the bytes stay canonical.
        """
        _, reference = self._cold_load(seed=1)

        method, layer = self._cold_load(seed=0)
        self._restore_for_load(method, layer)

        new_weights = _FakeMoELayer(method, seed=1)
        for name in _canonical_shapes():
            getattr(layer, name).data.copy_(getattr(new_weights, name).data)

        with _mock_flashinfer():
            _repack_weights_after_hot_update(_FakeModel(layer))

        for name in _canonical_shapes():
            got, want = getattr(layer, name).data, getattr(reference, name).data
            self.assertEqual(tuple(got.shape), tuple(want.shape))
            self.assertTrue(torch.equal(got, want), f"{name} differs from cold load")

    def test_repack_is_noop_when_no_weight_was_reverted(self):
        """Re-deriving an intact layout would double-block it."""
        method, layer = self._cold_load(seed=0)
        before = {n: getattr(layer, n).data.clone() for n in _canonical_shapes()}

        with _mock_flashinfer():
            method.repack_weights_after_hot_update(layer)
            method.repack_weights_after_hot_update(layer)

        for name, want in before.items():
            self.assertTrue(torch.equal(getattr(layer, name).data, want))

    def test_repack_is_noop_when_backend_inactive(self):
        method = _make_method(use_flashinfer_trtllm_moe=False)
        layer = _FakeMoELayer(method, seed=0)
        before = {n: getattr(layer, n).data.clone() for n in _canonical_shapes()}

        method.repack_weights_after_hot_update(layer)

        for name, want in before.items():
            self.assertEqual(tuple(getattr(layer, name).data.shape), want.shape)
            self.assertTrue(torch.equal(getattr(layer, name).data, want))

    def test_restore_inverts_the_block_layout(self):
        """The restore must undo the data, not just reinterpret the shape.

        A shape-only restore leaves block-layout bytes for the next re-derive to
        block a second time.
        """
        method = _make_method()
        layer = _FakeMoELayer(method, seed=3)
        original = {n: getattr(layer, n).data.clone() for n in _canonical_shapes()}

        with _mock_flashinfer():
            method._maybe_apply_flashinfer_trtllm_bf16_block_layout(layer)
        self._restore_for_load(method, layer)

        for name, want in original.items():
            got = getattr(layer, name).data
            self.assertEqual(tuple(got.shape), tuple(want.shape))
            self.assertTrue(torch.equal(got, want), f"{name} round trip is lossy")

    def test_bucketed_update_reproduces_cold_load_layout(self):
        """A refit split across several update RPCs must still be correct.

        RL callers batch weights (``weight_sync/utils.py``), so one refit is many
        update calls; expert slots carried by a later bucket must survive the
        earlier buckets' re-derives untouched.
        """
        _, reference = self._cold_load(seed=1)
        method, layer = self._cold_load(seed=0)
        new_weights = _FakeMoELayer(method, seed=1)

        # Bucket 1: w13, expert 0 only. Bucket 2: w13 expert 1. Bucket 3: w2.
        buckets = [
            ("w13_weight", 0),
            ("w13_weight", 1),
            ("w2_weight", None),
        ]
        for name, expert in buckets:
            short = name[: -len("_weight")]
            self._restore_for_load(method, layer, param_names=(short,))
            dst, src = getattr(layer, name).data, getattr(new_weights, name).data
            if expert is None:
                dst.copy_(src)
            else:
                dst[expert].copy_(src[expert])
            with _mock_flashinfer():
                _repack_weights_after_hot_update(_FakeModel(layer))

        for name in _canonical_shapes():
            got, want = getattr(layer, name).data, getattr(reference, name).data
            self.assertEqual(tuple(got.shape), tuple(want.shape))
            self.assertTrue(torch.equal(got, want), f"{name} differs from cold load")

    def test_restore_rejects_non_bijective_permutation(self):
        """The inverse is an argsort, which is only valid for a bijection.

        Guards against a future flashinfer permutation that pads or duplicates
        rows: without the check the restore would silently scramble them.
        """
        method, layer = self._cold_load(seed=0)
        not_a_bijection = lambda w: torch.zeros(w.shape[0], dtype=torch.long)

        with patch(
            f"{__name__}._fake_permute_indices", not_a_bijection
        ), self.assertRaises(RuntimeError) as ctx:
            self._restore_for_load(method, layer, param_names=("w13",))
        self.assertIn("not a bijection", str(ctx.exception))

    def test_restore_rejects_unexpected_numel(self):
        method, layer = self._cold_load(seed=0)
        layer.hidden_size = HIDDEN_SIZE + 64

        with self.assertRaises(RuntimeError):
            self._restore_for_load(method, layer, param_names=("w13",))


class TestRepackRunsOnFailedUpdate(unittest.TestCase):
    """The layout must be re-derived even when a weight update fails.

    With the re-derive on the success path only, a `load_weights` that raised
    part way through left parameters in load layout while the kernel read the
    transformed one -- turning a reported failure into silently wrong numerics.
    """

    class _LoadFailed(RuntimeError):
        pass

    class _FailingModel(torch.nn.Module):
        def load_weights(self, named_tensors):
            raise TestRepackRunsOnFailedUpdate._LoadFailed("load failed mid-update")

    def _make_updater(self, model):
        return weight_updater.WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=SimpleNamespace(),
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: SimpleNamespace(
                server_args=SimpleNamespace(weight_cache_mode="off")
            ),
        )

    def test_repack_runs_when_load_weights_raises(self):
        model = self._FailingModel()
        repacked = []

        with patch.object(
            weight_updater,
            "_repack_weights_after_hot_update",
            lambda m: repacked.append(m),
        ), patch.object(
            weight_updater, "_unsupported_derived_weight_cache_error", lambda: None
        ), patch.object(
            weight_updater, "monkey_patch_torch_reductions", lambda: None
        ):
            with self.assertRaises(self._LoadFailed):
                self._make_updater(model).update_weights_from_tensor(
                    named_tensors=[("w", torch.zeros(1))]
                )

        self.assertEqual(repacked, [model], "layout not re-derived on the error path")


if __name__ == "__main__":
    unittest.main()
