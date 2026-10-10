"""CPU unit tests for reloading BF16 MoE weights under the TRT-LLM backends.

Checkpoint copies need canonical values; session finalization restores kernel
layout. FlashInfer's SM100 layout entry points use an order-changing CPU stand-in;
these tests verify the lifecycle and inverse, not on-device kernel parity.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.lora.layers import BaseLayerWithLoRA, FusedMoEWithLoRA
from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.test_utils import CustomTestCase

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
    core.get_w2_permute_indices_with_cache = lambda cache, weight_u8, tile_m: (
        _fake_permute_indices(weight_u8)
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
        self.attention_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def load_weights(self, named_tensors):
        for name, tensor in named_tensors:
            if name == "attention_weight":
                self.attention_weight.data.copy_(tensor)
                continue
            param_name, _, expert = name.partition(".")
            param = dict(_weights(self.layer))[param_name]
            self.layer.quant_method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                layer=self.layer,
                param=param,
                weight_name=f"model.layers.0.mlp.experts.{param_name}",
            )
            if expert:
                param.data[int(expert)].copy_(tensor)
            else:
                param.data.copy_(tensor)


def _weights(layer):
    return (("w13_weight", layer.w13_weight), ("w2_weight", layer.w2_weight))


def _make_updater(model):
    return weight_updater.WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: SimpleNamespace(server_args=get_server_args()),
    )


def _update_bucket(updater, named_tensors, load_format):
    if load_format == "distributed":
        return updater.load_weights_from_distributed(named_tensors)
    if load_format == "flattened_bucket":
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        named_tensors = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
    else:
        load_format = None
    return updater.update_weights_from_tensor(named_tensors, load_format=load_format)


def _make_method(use_flashinfer_trtllm_moe: bool = True):
    return UnquantizedFusedMoEMethod(
        use_flashinfer_trtllm_moe=use_flashinfer_trtllm_moe
    )


def _canonical_shapes():
    return {
        "w13_weight": (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "w2_weight": (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
    }


class TestFlashInferTrtllmBf16MoEReload(CustomTestCase):
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
                    param=dict(_weights(layer))[name],
                    weight_name=f"model.layers.0.mlp.experts.{name}",
                )

    def test_post_load_preserves_already_packed_weights(self):
        """Repeated post-load processing must preserve the cold-load layout."""
        _, reference = self._cold_load(seed=0)
        method = _make_method()
        layer = _FakeMoELayer(method, seed=0)

        with _mock_flashinfer():
            for _ in range(3):
                method.process_weights_after_loading(layer)
                for (name, param), (_, ref_param) in zip(
                    _weights(layer), _weights(reference)
                ):
                    got = param.data
                    want = ref_param.data
                    self.assertEqual(tuple(got.shape), tuple(want.shape))
                    self.assertTrue(torch.equal(got, want), name)

    def test_post_load_repacks_only_canonical_weights(self):
        """Final post-load must pack a changed weight without repacking its peer."""
        _, reference = self._cold_load(seed=1)
        method, layer = self._cold_load(seed=0)
        untouched_w2 = layer.w2_weight.data.clone()
        self._restore_for_load(method, layer, param_names=("w13",))
        layer.w13_weight.data.copy_(_FakeMoELayer(method, seed=1).w13_weight.data)
        expected = {
            "w13_weight": reference.w13_weight.data,
            "w2_weight": untouched_w2,
        }

        with _mock_flashinfer():
            for _ in range(2):
                method.process_weights_after_loading(layer)
                for name, param in _weights(layer):
                    want = expected[name]
                    got = param.data
                    self.assertEqual(tuple(got.shape), tuple(want.shape))
                    self.assertTrue(torch.equal(got, want), name)

    def test_repack_is_noop_when_no_weight_was_reverted(self):
        """Re-deriving an intact layout would double-block it."""
        method, layer = self._cold_load(seed=0)
        before = {name: param.data.clone() for name, param in _weights(layer)}

        with _mock_flashinfer():
            method._maybe_apply_flashinfer_trtllm_bf16_block_layout(layer)
            method._maybe_apply_flashinfer_trtllm_bf16_block_layout(layer)

        for name, param in _weights(layer):
            self.assertTrue(torch.equal(param.data, before[name]))

    def test_repack_is_noop_when_backend_inactive(self):
        method = _make_method(use_flashinfer_trtllm_moe=False)
        layer = _FakeMoELayer(method, seed=0)
        before = {name: param.data.clone() for name, param in _weights(layer)}

        method._maybe_apply_flashinfer_trtllm_bf16_block_layout(layer)

        for name, param in _weights(layer):
            self.assertEqual(tuple(param.data.shape), before[name].shape)
            self.assertTrue(torch.equal(param.data, before[name]))

    def test_restore_inverts_the_block_layout(self):
        """The restore must undo the data, not just reinterpret the shape.

        A shape-only restore leaves block-layout bytes for the next re-derive to
        block a second time.
        """
        method = _make_method()
        layer = _FakeMoELayer(method, seed=3)
        original = {name: param.data.clone() for name, param in _weights(layer)}

        with _mock_flashinfer():
            method._maybe_apply_flashinfer_trtllm_bf16_block_layout(layer)
        self._restore_for_load(method, layer)

        for name, param in _weights(layer):
            got, want = param.data, original[name]
            self.assertEqual(tuple(got.shape), tuple(want.shape))
            self.assertTrue(torch.equal(got, want), f"{name} round trip is lossy")

    def test_bucketed_update_reproduces_cold_load_layout(self):
        """Restore each weight once across buckets, then pack once at session end."""
        for load_format in ("tensor", "flattened_bucket", "distributed"):
            with self.subTest(load_format=load_format):
                _, reference = self._cold_load(seed=1)
                method, layer = self._cold_load(seed=0)
                new_weights = _FakeMoELayer(method, seed=1)
                updater = _make_updater(_FakeModel(layer))
                pointers = {name: param.data_ptr() for name, param in _weights(layer)}
                buckets = [
                    ("w13_weight.0", new_weights.w13_weight.data[0]),
                    ("w13_weight.1", new_weights.w13_weight.data[1]),
                    ("w2_weight", new_weights.w2_weight.data),
                ]
                with (
                    _mock_flashinfer(),
                    get_context().override_server_args(weight_cache_mode="off"),
                    get_parallel().override(tp_rank=0),
                    patch.object(
                        weight_updater, "monkey_patch_torch_reductions", lambda: None
                    ),
                    patch.object(
                        method,
                        "_restore_trtllm_bf16_canonical_layout",
                        wraps=method._restore_trtllm_bf16_canonical_layout,
                    ) as restore,
                    patch.object(
                        method,
                        "_apply_trtllm_bf16_block_layout",
                        wraps=method._apply_trtllm_bf16_block_layout,
                    ) as pack,
                ):
                    updater.begin_weight_update()
                    for bucket in buckets:
                        success, message = _update_bucket(
                            updater, [bucket], load_format
                        )
                        self.assertTrue(success, message)
                        self.assertEqual(
                            tuple(layer.w13_weight.shape),
                            _canonical_shapes()["w13_weight"],
                        )
                    self.assertEqual(restore.call_count, 2)
                    self.assertEqual(pack.call_count, 0)
                    updater.end_weight_update(run_post_load=False)
                    self.assertEqual(pack.call_count, 2)
                    for (name, param), (_, ref_param) in zip(
                        _weights(layer), _weights(reference)
                    ):
                        self.assertTrue(torch.equal(param.data, ref_param.data), name)
                        self.assertEqual(param.data_ptr(), pointers[name])

    def test_restore_rejects_non_bijective_permutation(self):
        """The inverse is an argsort, which is only valid for a bijection.

        Guards against a future flashinfer permutation that pads or duplicates
        rows: without the check the restore would silently scramble them.
        """
        method, layer = self._cold_load(seed=0)
        not_a_bijection = lambda w: torch.zeros(w.shape[0], dtype=torch.long)

        with (
            patch(f"{__name__}._fake_permute_indices", not_a_bijection),
            self.assertRaises(RuntimeError) as ctx,
        ):
            self._restore_for_load(method, layer, param_names=("w13",))
        self.assertIn("not a bijection", str(ctx.exception))

    def test_restore_rejects_unexpected_numel(self):
        method, layer = self._cold_load(seed=0)
        layer.hidden_size = HIDDEN_SIZE + 64

        with self.assertRaises(RuntimeError):
            self._restore_for_load(method, layer, param_names=("w13",))


class TestWeightUpdateSessionFinalization(CustomTestCase):
    """Session finalization for loader-free copies and LoRA-wrapped models."""

    def test_loader_free_refits_reproduce_cold_load_layout(self):
        """Raw copies bypass loaders: session start must undo the kernel layout."""
        for load_format in ("direct", "p2p"):
            with self.subTest(load_format=load_format):
                method = _make_method()
                layer = _FakeMoELayer(method, seed=0)
                model = _FakeModel(layer)
                updater = _make_updater(model)
                with (
                    _mock_flashinfer(),
                    get_context().override_server_args(weight_cache_mode="off"),
                    get_parallel().override(tp_rank=0),
                    patch.object(
                        weight_updater, "monkey_patch_torch_reductions", lambda: None
                    ),
                    patch.object(
                        model,
                        "load_weights",
                        side_effect=AssertionError(
                            "A loader-free refit called the loader"
                        ),
                    ),
                ):
                    method.process_weights_after_loading(layer)
                    pointers = {
                        name: param.data_ptr() for name, param in _weights(layer)
                    }
                    for seed in (1, 2, 3):
                        new_weights = _FakeMoELayer(method, seed=seed)
                        reference = _FakeMoELayer(method, seed=seed)
                        method.process_weights_after_loading(reference)
                        updater.begin_weight_update()
                        for name, param in _weights(layer):
                            self.assertEqual(
                                tuple(param.shape), _canonical_shapes()[name]
                            )
                        if load_format == "p2p":
                            # Model direct writes as flat copies, without any loader.
                            for (_, param), (_, new_param) in zip(
                                _weights(layer), _weights(new_weights)
                            ):
                                param.data.view(-1).copy_(new_param.data.view(-1))
                        else:
                            success, message = updater.update_weights_from_tensor(
                                [
                                    (f"layer.{name}", param.data)
                                    for name, param in _weights(new_weights)
                                ],
                                load_format="direct",
                            )
                            self.assertTrue(success, message)
                        updater.end_weight_update(run_post_load=load_format == "p2p")
                        for (name, param), (_, ref_param) in zip(
                            _weights(layer), _weights(reference)
                        ):
                            self.assertEqual(tuple(param.shape), tuple(ref_param.shape))
                            self.assertTrue(
                                torch.equal(param.data, ref_param.data), name
                            )
                            self.assertEqual(param.data_ptr(), pointers[name])

    def test_attention_only_update_with_lora_wrapper(self):
        method = _make_method()
        layer = _FakeMoELayer(method, seed=0)
        # Only the wrapper's ownership/traversal matters, not its GPU runner setup.
        wrapper = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
        BaseLayerWithLoRA.__init__(wrapper, layer, SimpleNamespace())
        wrapper.quant_method = method
        model = _FakeModel(wrapper)
        updater = _make_updater(model)
        with (
            _mock_flashinfer(),
            get_context().override_server_args(weight_cache_mode="off"),
            get_parallel().override(tp_rank=0),
            patch.object(weight_updater, "monkey_patch_torch_reductions", lambda: None),
        ):
            method.process_weights_after_loading(layer)
            before = {name: param.data.clone() for name, param in _weights(layer)}
            updater.begin_weight_update()
            success, message = updater.update_weights_from_tensor(
                [("attention_weight", torch.ones(1))]
            )
            self.assertTrue(success, message)
            updater.end_weight_update(run_post_load=False)
            self.assertTrue(torch.equal(model.attention_weight.data, torch.ones(1)))
            for name, param in _weights(layer):
                self.assertTrue(torch.equal(param.data, before[name]), name)


if __name__ == "__main__":
    unittest.main()
