"""Weight updates on NVFP4 fused-MoE layers.

`process_weights_after_loading` permutes FP4 weights and block scales in place
(the TRT-LLM row shuffle and scale interleave, or the CUTLASS scale swizzle).
A weight update post-processes every MoE layer again, so experts it does not
rewrite must first be restored to the checkpoint layout by
`restore_weights_before_loading`, or they end up permuted twice.

These tests run the real post-processing on small, aligned shapes and check
that restore is its exact inverse, so a partial update leaves the untouched
experts bit-identical. Removing or breaking the restore turns them red.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

# Import modelopt_quant before flashinfer_trtllm (circular import through the
# compressed_tensors schemes package otherwise).
# isort: off
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod

# isort: on
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.runtime_context import get_flags
from sglang.srt.utils import get_device_sm
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 4
HIDDEN = 512
INTERMEDIATE = 256
REPACKED = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
BACKENDS = (MoeRunnerBackend.FLASHINFER_CUTLASS, MoeRunnerBackend.FLASHINFER_TRTLLM)


def _raw_tensors(seed, hidden=HIDDEN):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(*shape):
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    rows = 2 * INTERMEDIATE
    return {
        "w13_weight": u8(NUM_EXPERTS, rows, hidden // 2),
        "w13_weight_scale": u8(NUM_EXPERTS, rows, hidden // 16).view(
            torch.float8_e4m3fn
        ),
        "w2_weight": u8(NUM_EXPERTS, hidden, INTERMEDIATE // 2),
        "w2_weight_scale": u8(NUM_EXPERTS, hidden, INTERMEDIATE // 16).view(
            torch.float8_e4m3fn
        ),
    }


def _layer(raw):
    config = MoeRunnerConfig(is_gated=True)
    method = ModelOptNvFp4FusedMoEMethod(
        SimpleNamespace(use_per_token_activation=False)
    )
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.quant_method = method
    layer.moe_runner_config = config
    layer.num_local_experts = NUM_EXPERTS
    layer.intermediate_size_per_partition = INTERMEDIATE
    layer.dispatcher = SimpleNamespace(set_quant_config=lambda config: None)
    layer._fp4_moe_repack = modelopt_quant.Fp4MoeRepack.CHECKPOINT  # create_weights
    params = {name: t.clone() for name, t in raw.items()}
    params.update(
        w13_weight_scale_2=torch.full((NUM_EXPERTS, 2), 0.01, device="cuda"),
        w2_weight_scale_2=torch.full((NUM_EXPERTS,), 0.01, device="cuda"),
        w13_input_scale=torch.full((NUM_EXPERTS, 2), 0.5, device="cuda"),
        w2_input_scale=torch.full((NUM_EXPERTS,), 0.5, device="cuda"),
    )
    for name, data in params.items():
        layer.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    if not method.enable_flashinfer_trtllm_moe:
        for name in ("w13", "w2"):
            setattr(
                layer,
                f"{name}_blockscale_swizzled",
                torch.nn.Parameter(
                    modelopt_quant.swizzle_blockscale(params[f"{name}_weight_scale"]),
                    requires_grad=False,
                ),
            )
    return layer


def _bytes(layer, name):
    return getattr(layer, name).data.view(torch.uint8).clone()


@unittest.skipIf(
    not torch.cuda.is_available() or get_device_sm() < 100,
    "NVFP4 MoE layouts need SM100+",
)
class TestNvFp4MoeRestore(CustomTestCase):
    def _backend(self, backend):
        return get_flags().moe.override(
            runner_backend=backend, a2a_backend=MoeA2ABackend.NONE
        )

    def _process(self, layer):
        with mock.patch.object(
            modelopt_quant,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ):
            layer.quant_method.process_weights_after_loading(layer)

    def test_restore_returns_checkpoint_layout(self):
        raw = _raw_tensors(seed=0)
        for backend in BACKENDS:
            with self.subTest(backend=backend.value), self._backend(backend):
                layer = _layer(raw)
                self._process(layer)
                self.assertFalse(
                    torch.equal(
                        _bytes(layer, "w13_weight_scale"),
                        raw["w13_weight_scale"].view(torch.uint8),
                    )
                )
                storage = {n: getattr(layer, n).data_ptr() for n in REPACKED}
                layer.quant_method.restore_weights_before_loading(layer)
                for name in REPACKED:
                    self.assertTrue(
                        torch.equal(_bytes(layer, name), raw[name].view(torch.uint8)),
                        name,
                    )
                    self.assertEqual(getattr(layer, name).data_ptr(), storage[name])

    def test_layers_share_one_scale_interleave_map(self):
        # One copy per scale shape for all MoE layers, none in a layer's cache.
        flashinfer_trtllm._FP4_SCALE_INTERLEAVE_SOURCES.clear()
        raw = _raw_tensors(seed=3)
        with self._backend(MoeRunnerBackend.FLASHINFER_TRTLLM):
            layers = [_layer(raw), _layer(raw)]
            for layer in layers:
                self._process(layer)
                layer.quant_method.restore_weights_before_loading(layer)
        self.assertEqual(len(flashinfer_trtllm._FP4_SCALE_INTERLEAVE_SOURCES), 2)
        for layer in layers:
            cache = layer.quant_method._cache_permute_indices
            names = {key[0] for key in cache if isinstance(key, tuple)}
            self.assertNotIn("fp4_block_scale_interleave", names)

    def test_partial_update_keeps_untouched_experts(self):
        raw = _raw_tensors(seed=1)
        update = _raw_tensors(seed=2)
        updated_raw = {name: t.clone() for name, t in raw.items()}
        for name in REPACKED:
            updated_raw[name][1] = update[name][1]
        for backend in BACKENDS:
            with self.subTest(backend=backend.value), self._backend(backend):
                layer = _layer(raw)
                self._process(layer)
                before = {name: _bytes(layer, name) for name in REPACKED}
                expected = _layer(updated_raw)
                self._process(expected)

                layer.quant_method.restore_weights_before_loading(layer)
                for name in REPACKED:
                    getattr(layer, name).data[1] = update[name][1]
                self._process(layer)

                for name in REPACKED:
                    after = _bytes(layer, name)
                    for expert in (0, 2, 3):
                        self.assertTrue(
                            torch.equal(after[expert], before[name][expert]),
                            f"{name} expert {expert}",
                        )
                    self.assertTrue(torch.equal(after, _bytes(expected, name)), name)

    def test_restore_is_a_no_op_without_a_repack(self):
        raw = _raw_tensors(seed=3)
        for backend in BACKENDS:
            with self.subTest(backend=backend.value), self._backend(backend):
                layer = _layer(raw)
                layer.quant_method.restore_weights_before_loading(layer)
                self._process(layer)
                layer.quant_method.restore_weights_before_loading(layer)
                layer.quant_method.restore_weights_before_loading(layer)
                for name in REPACKED:
                    self.assertTrue(
                        torch.equal(_bytes(layer, name), raw[name].view(torch.uint8))
                    )

    def test_failed_restore_leaves_the_layer_untouched(self):
        import flashinfer.fused_moe.core as fused_moe_core

        def not_a_bijection(cache, tensor, epilogue_tile_m, **kwargs):
            return torch.zeros(tensor.shape[0], dtype=torch.int64)

        with self._backend(MoeRunnerBackend.FLASHINFER_TRTLLM):
            layer = _layer(_raw_tensors(seed=4))
            self._process(layer)
            before = {name: _bytes(layer, name) for name in REPACKED}
            # w13 plans fine; w2 fails its check after that.
            with mock.patch.object(
                fused_moe_core, "get_w2_permute_indices_with_cache", not_a_bijection
            ):
                with self.assertRaises(RuntimeError):
                    layer.quant_method.restore_weights_before_loading(layer)
            for name in REPACKED:
                self.assertTrue(torch.equal(_bytes(layer, name), before[name]), name)
            self.assertIs(layer._fp4_moe_repack, modelopt_quant.Fp4MoeRepack.TRTLLM)

    def test_padded_scale_layouts(self):
        # hidden 192: w2 block-scale rows are not a multiple of 128, so the
        # scale interleave/swizzle pads them.
        raw = _raw_tensors(seed=5, hidden=192)
        update = _raw_tensors(seed=6, hidden=192)

        with self._backend(MoeRunnerBackend.FLASHINFER_TRTLLM):
            layer = _layer(raw)
            try:
                self._process(layer)
            except AssertionError:
                # FlashInfer's TRT-LLM row shuffle needs 128-row multiples, so
                # this layout fails at load today; restore must stay off if not.
                pass
            self.assertIs(
                layer._fp4_moe_repack, modelopt_quant.Fp4MoeRepack.NOT_RESTORABLE
            )

        # CUTLASS keeps the padded w2 swizzle in a separate buffer, so the
        # source scale stays in checkpoint layout and a partial update works.
        with self._backend(MoeRunnerBackend.FLASHINFER_CUTLASS):
            updated_raw = {name: t.clone() for name, t in raw.items()}
            for name in REPACKED:
                updated_raw[name][1] = update[name][1]
            layer = _layer(raw)
            self._process(layer)
            expected = _layer(updated_raw)
            self._process(expected)
            layer.quant_method.restore_weights_before_loading(layer)
            for name in REPACKED:
                getattr(layer, name).data[1] = update[name][1]
            self._process(layer)
            for name in REPACKED + ("w2_blockscale_swizzled",):
                self.assertTrue(
                    torch.equal(_bytes(layer, name), _bytes(expected, name)), name
                )


if __name__ == "__main__":
    unittest.main()
