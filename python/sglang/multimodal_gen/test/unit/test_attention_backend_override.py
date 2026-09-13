# SPDX-License-Identifier: Apache-2.0
"""Per-request attention backend override: fail-fast validation and the
two-phase layer switching (CPU-only, layer/selector boundaries patched)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention import layer as layer_module
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    flash_attn as flash_attn_module,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    denoising as denoising_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _batch(override=None) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=SimpleNamespace(attention_backend_override=override)
    )


def _fake_backend_cls(enum, *, ring_capable=True):
    return SimpleNamespace(
        get_enum=lambda: enum,
        supports_ring_rotation=lambda: ring_capable,
        get_impl_cls=lambda: lambda **kwargs: f"{enum.name.lower()}_impl",
    )


def _fake_layer(
    default=AttentionBackendEnum.FA, *, is_cross_attention=False
) -> SimpleNamespace:
    return SimpleNamespace(
        backend=default,
        _default_attn_backend=default,
        _attn_impl_by_backend={default: f"{default.name.lower()}_impl"},
        _supported_attention_backends=None,
        _required_attention_backend=None,
        _attn_impl_ctor_kwargs={"num_heads": 2},
        attn_impl=f"{default.name.lower()}_impl",
        head_size=128,
        dtype=torch.bfloat16,
        is_cross_attention=is_cross_attention,
    )


class TestSkipSoftmaxDispatch(unittest.TestCase):
    def test_dense_trtllm_precedes_skip_threshold(self):
        impl = FlashAttentionImpl(
            num_heads=2,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
        )
        params = SimpleNamespace(start_step=14, threshold_scale_factor=500.0)
        with patch.object(
            flash_attn_module, "get_request_skip_softmax_params", return_value=params
        ):
            with patch.object(
                flash_attn_module,
                "get_forward_context",
                return_value=SimpleNamespace(current_timestep=13),
            ):
                self.assertEqual(impl._request_skip_softmax_threshold(), (True, None))
            with patch.object(
                flash_attn_module,
                "get_forward_context",
                return_value=SimpleNamespace(current_timestep=14),
            ):
                self.assertEqual(impl._request_skip_softmax_threshold(), (True, 500.0))


class TestMaybeOverrideAttentionBackend(unittest.TestCase):
    def setUp(self):
        self.default_backend_cls = _fake_backend_cls(AttentionBackendEnum.FA)
        self.stage = DenoisingStage.__new__(DenoisingStage)
        self.stage.server_args = SimpleNamespace(
            enable_breakable_cuda_graph=False,
            enable_torch_compile=False,
            ring_degree=None,
        )
        self.stage.transformer = object()
        self.stage.transformer_2 = None
        self.stage.attn_backend = self.default_backend_cls
        self.stage._attn_backend_default = self.default_backend_cls
        self.stage._attn_metadata_head_size = 64
        self.stage._attention_backend_active_override = None
        self.stage._skip_softmax_forced_fa = False

        self.layers = [_fake_layer(), _fake_layer()]
        self.prepare_calls = []
        self.apply_calls = []
        self.resolved_backend_cls = _fake_backend_cls(AttentionBackendEnum.SAGE_ATTN)

        patchers = [
            patch.object(
                DenoisingStage,
                "_request_switchable_attention_layers",
                lambda stage: self.layers,
            ),
            patch.object(
                denoising_module,
                "prepare_attention_backend_override",
                lambda layer, target: self.prepare_calls.append((layer, target)),
            ),
            patch.object(
                denoising_module,
                "apply_attention_backend_override",
                lambda layer, target: self.apply_calls.append((layer, target)),
            ),
            patch.object(
                denoising_module,
                "get_attn_backend",
                lambda **kwargs: self.resolved_backend_cls,
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_default_is_a_no_op(self):
        self.stage._maybe_override_attention_backend(_batch(None))
        self.assertEqual(self.prepare_calls, [])
        self.assertEqual(self.apply_calls, [])
        self.assertIs(self.stage.attn_backend, self.default_backend_cls)

    def test_override_switches_all_layers_then_fast_paths(self):
        self.stage._maybe_override_attention_backend(_batch("sage_attn"))
        self.assertEqual(len(self.prepare_calls), 2)
        self.assertEqual(len(self.apply_calls), 2)
        self.assertTrue(
            all(t is AttentionBackendEnum.SAGE_ATTN for _, t in self.apply_calls)
        )
        self.assertIs(self.stage.attn_backend, self.resolved_backend_cls)
        self.assertIs(
            self.stage._attention_backend_active_override,
            AttentionBackendEnum.SAGE_ATTN,
        )

        self.stage._maybe_override_attention_backend(_batch("sage_attn"))
        self.assertEqual(len(self.apply_calls), 2)  # unchanged: fast path

    def test_default_batch_restores_server_backend(self):
        self.stage._maybe_override_attention_backend(_batch("sage_attn"))
        self.stage._maybe_override_attention_backend(_batch(None))
        self.assertEqual(
            self.apply_calls[-2:], [(self.layers[0], None), (self.layers[1], None)]
        )
        self.assertIs(self.stage.attn_backend, self.default_backend_cls)
        self.assertIsNone(self.stage._attention_backend_active_override)

    def test_skip_softmax_forces_fa_only_for_self_attention(self):
        self.layers[1] = _fake_layer(is_cross_attention=True)
        self.stage._maybe_override_attention_backend(
            _batch(None), force_fa_for_self_attention=True
        )
        self.assertEqual(
            self.prepare_calls,
            [(self.layers[0], AttentionBackendEnum.FA)],
        )
        self.assertEqual(
            self.apply_calls,
            [
                (self.layers[0], AttentionBackendEnum.FA),
                (self.layers[1], None),
            ],
        )

    def test_skip_softmax_rejects_non_fa_override(self):
        with self.assertRaisesRegex(ValueError, "requires the FA attention backend"):
            self.stage._maybe_override_attention_backend(
                _batch("sage_attn"), force_fa_for_self_attention=True
            )

    def test_skip_softmax_uses_custom_model_fa_without_shared_layers(self):
        self.layers.clear()
        self.stage._maybe_override_attention_backend(
            _batch(None), force_fa_for_self_attention=True
        )
        self.assertEqual(self.prepare_calls, [])
        self.assertEqual(self.apply_calls, [])
        self.assertIs(self.stage.attn_backend, self.default_backend_cls)

    def test_unknown_backend_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown attention_backend_override"):
            self.stage._maybe_override_attention_backend(_batch("bogus_attn"))

    def test_non_switchable_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "not switchable per"):
            self.stage._maybe_override_attention_backend(_batch("video_sparse_attn"))

    def test_rejected_under_breakable_cuda_graph(self):
        self.stage.server_args.enable_breakable_cuda_graph = True
        with self.assertRaisesRegex(ValueError, "breakable CUDA graphs"):
            self.stage._maybe_override_attention_backend(_batch("sage_attn"))
        self.assertEqual(self.apply_calls, [])

    def test_rejected_under_torch_compile(self):
        self.stage.server_args.enable_torch_compile = True
        with self.assertRaisesRegex(ValueError, "torch.compile"):
            self.stage._maybe_override_attention_backend(_batch("sage_attn"))

    def test_rejected_when_server_backend_is_sparse(self):
        self.layers[1] = _fake_layer(default=AttentionBackendEnum.VIDEO_SPARSE_ATTN)
        with self.assertRaisesRegex(ValueError, "sparse backend"):
            self.stage._maybe_override_attention_backend(_batch("sage_attn"))

    def test_rejected_when_no_switchable_layers(self):
        self.layers.clear()
        with self.assertRaisesRegex(ValueError, "no switchable attention layers"):
            self.stage._maybe_override_attention_backend(_batch("sage_attn"))

    def test_rejected_when_target_not_ring_capable_under_ring(self):
        self.stage.server_args.ring_degree = 2
        self.resolved_backend_cls = _fake_backend_cls(
            AttentionBackendEnum.SAGE_ATTN_3, ring_capable=False
        )
        with self.assertRaisesRegex(ValueError, "ring-capable"):
            self.stage._maybe_override_attention_backend(_batch("sage_attn_3"))

    def test_prepare_failure_leaves_layers_unswitched(self):
        def failing_prepare(layer, target):
            if layer is self.layers[1]:
                raise ValueError("unsupported on this layer")
            self.prepare_calls.append((layer, target))

        with patch.object(
            denoising_module, "prepare_attention_backend_override", failing_prepare
        ):
            with self.assertRaisesRegex(ValueError, "unsupported on this layer"):
                self.stage._maybe_override_attention_backend(_batch("sage_attn"))
        self.assertEqual(self.apply_calls, [])
        self.assertIsNone(self.stage._attention_backend_active_override)


class TestLayerPrepareApply(unittest.TestCase):
    def _patch_selector(self, backend_cls):
        return patch.object(
            layer_module, "get_attn_backend", lambda *args, **kwargs: backend_cls
        )

    def test_prepare_builds_and_caches_impl(self):
        layer = _fake_layer()
        backend_cls = _fake_backend_cls(AttentionBackendEnum.SAGE_ATTN)
        with (
            self._patch_selector(backend_cls),
            patch.object(
                layer_module, "wrap_attention_impl_forward", lambda impl: impl
            ),
        ):
            layer_module.prepare_attention_backend_override(
                layer, AttentionBackendEnum.SAGE_ATTN
            )
        self.assertEqual(
            layer._attn_impl_by_backend[AttentionBackendEnum.SAGE_ATTN],
            "sage_attn_impl",
        )
        # prepare must not mutate the active impl
        self.assertEqual(layer.attn_impl, "fa_impl")
        self.assertIs(layer.backend, AttentionBackendEnum.FA)

    def test_prepare_rejects_silent_fallback(self):
        layer = _fake_layer()
        fallback_cls = _fake_backend_cls(AttentionBackendEnum.TORCH_SDPA)
        with self._patch_selector(fallback_cls):
            with self.assertRaisesRegex(ValueError, "refusing the request"):
                layer_module.prepare_attention_backend_override(
                    layer, AttentionBackendEnum.SAGE_ATTN
                )
        self.assertNotIn(AttentionBackendEnum.SAGE_ATTN, layer._attn_impl_by_backend)

    def test_apply_flips_and_restores(self):
        layer = _fake_layer()
        layer._attn_impl_by_backend[AttentionBackendEnum.SAGE_ATTN] = "sage_attn_impl"

        layer_module.apply_attention_backend_override(
            layer, AttentionBackendEnum.SAGE_ATTN
        )
        self.assertEqual(layer.attn_impl, "sage_attn_impl")
        self.assertIs(layer.backend, AttentionBackendEnum.SAGE_ATTN)

        layer_module.apply_attention_backend_override(layer, None)
        self.assertEqual(layer.attn_impl, "fa_impl")
        self.assertIs(layer.backend, AttentionBackendEnum.FA)

    def test_required_backend_ignores_request_override(self):
        layer = _fake_layer(default=AttentionBackendEnum.TORCH_SDPA)
        layer._required_attention_backend = AttentionBackendEnum.TORCH_SDPA

        layer_module.prepare_attention_backend_override(
            layer, AttentionBackendEnum.SAGE_ATTN
        )
        layer_module.apply_attention_backend_override(
            layer, AttentionBackendEnum.SAGE_ATTN
        )

        self.assertEqual(
            layer._attn_impl_by_backend,
            {AttentionBackendEnum.TORCH_SDPA: "torch_sdpa_impl"},
        )
        self.assertEqual(layer.attn_impl, "torch_sdpa_impl")
        self.assertIs(layer.backend, AttentionBackendEnum.TORCH_SDPA)


if __name__ == "__main__":
    unittest.main()
