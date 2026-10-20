# SPDX-License-Identifier: Apache-2.0
"""Quant precision guard: timestep resolution and Cache-DiT invalidation.

Covers the cross-family fallback in ``resolve_precision`` (the linear
W4A8/W4A4 boundary schedule must never leak into the attention family, which
only accepts FLOAT/MXFP4) and the guard's invalidation behavior when the
effective per-step precision changes mid-run.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.cache.quant_precision_guard import (
    install_quant_precision_guard,
)
from sglang.multimodal_gen.runtime.layers.quantization.attention_quant_scheme import (
    AttentionQuantScheme,
)
from sglang.multimodal_gen.runtime.layers.quantization.mxfp4_utils import (
    default_mxfp4_policy,
    resolve_precision,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.test.test_utils import CustomTestCase

NUM_STEPS = 8


class _FakeLinearScheme:
    """Linear layer scheme (mirrors ModelSlimMXFP4Scheme's policy surface)."""

    precision_policy_target = "w4a4_linear"
    precision_policy_default = "W4A4"

    def __init__(self, description):
        self.description = description

    def resolve_precision(self, default_policy=None):
        return resolve_precision(
            self.description,
            self.precision_policy_target,
            self.precision_policy_default,
            default_policy,
        )


class _FakeQuantModule:
    def __init__(self, scheme, attr):
        setattr(self, attr, scheme)


class _FakeCalibrator:
    def __init__(self, context):
        self._context = context

    def reset_cache(self):
        self._context.calibrator_reset_calls += 1


class _FakeCachedContext:
    def __init__(self, num_steps):
        self.clear_buffers_calls = 0
        self.calibrator_reset_calls = 0
        self.executed_steps = 0
        self.cache_config = SimpleNamespace(steps_computation_mask=[0] * num_steps)

    def clear_buffers(self):
        self.clear_buffers_calls += 1

    def get_calibrators(self):
        return (_FakeCalibrator(self),)

    def get_cfg_calibrators(self):
        return (_FakeCalibrator(self),)


class _FakeTransformer:
    def __init__(self, modules, contexts):
        self._modules_list = list(modules)
        self._contexts = contexts
        self._context_manager = SimpleNamespace(
            get_context=lambda name: contexts[int(name)]
        )
        self._context_names = [str(i) for i in range(len(contexts))]
        self.forward_calls = 0
        self.observed_masks = []

    def modules(self):
        return iter(self._modules_list)

    def forward(self):
        self.forward_calls += 1
        self.observed_masks.append(
            [list(c.cache_config.steps_computation_mask) for c in self._contexts]
        )
        return "out"


def _run_steps(transformer, batch, steps):
    for step in steps:
        with set_forward_context(step, None, forward_batch=batch):
            transformer.forward()


class TestResolvePrecision(CustomTestCase):
    def _at_step(self, step, target, default, description, default_policy):
        with set_forward_context(step, None, forward_batch=SimpleNamespace()):
            return resolve_precision(description, target, default, default_policy)

    def test_linear_boundary_schedule(self):
        schedule = default_mxfp4_policy(NUM_STEPS)
        self.assertEqual(
            sorted(schedule), ["0", "1", "6", "7"], "W4A8 at the boundaries only"
        )
        for step, expected in ((0, "W4A8"), (1, "W4A8"), (2, "W4A4"), (5, "W4A4")):
            self.assertEqual(
                self._at_step(
                    step, "w4a4_linear", "W4A4", {}, default_mxfp4_policy(NUM_STEPS)
                ),
                expected,
            )

    def test_attention_ignores_linear_schedule(self):
        """Regression: the linear W4A8 fallback must not hit the fa family.

        The guard resolves every scheme against the shared linear boundary
        schedule; attention accepts only FLOAT/MXFP4 and previously raised
        ``ValueError: Unsupported fa precision: W4A8`` at boundary steps.
        """
        for step in (0, 1, 6, 7):
            self.assertEqual(
                self._at_step(step, "fa", "MXFP4", {}, default_mxfp4_policy(NUM_STEPS)),
                "MXFP4",
            )

    def test_attention_scheme_resolves_boundary_steps(self):
        scheme = AttentionQuantScheme("blocks.0.attn", "MXFP4", {}, True, True)
        with set_forward_context(0, None, forward_batch=SimpleNamespace()):
            self.assertEqual(scheme.resolve_precision(default_mxfp4_policy(8)), "MXFP4")

    def test_checkpoint_policy_overrides_schedule(self):
        description = {
            "timestep_policy": {
                "fa": {"default": "FLOAT"},
                "w4a4_linear": {"2": "W4A8"},
            }
        }
        self.assertEqual(
            self._at_step(2, "fa", "MXFP4", description, default_mxfp4_policy(8)),
            "FLOAT",
        )
        self.assertEqual(
            self._at_step(
                2, "w4a4_linear", "W4A4", description, default_mxfp4_policy(8)
            ),
            "W4A8",
        )

    def test_unquantized_attention_has_no_policy(self):
        scheme = AttentionQuantScheme("blocks.0.attn", None, {}, False, False)
        with set_forward_context(3, None, forward_batch=SimpleNamespace()):
            self.assertEqual(scheme.resolve_precision(default_mxfp4_policy(8)), "FLOAT")

    def test_invalid_precision_still_rejected(self):
        description = {"timestep_policy": {"fa": {"default": "W4A8"}}}
        with (
            set_forward_context(0, None, forward_batch=SimpleNamespace()),
            self.assertRaises(ValueError),
        ):
            resolve_precision(description, "fa", "MXFP4", None)


class TestRegisterQKRotationBuffers(CustomTestCase):
    """Scheme-owned q_rot/k_rot registration, shared by Wan and Flux paths."""

    def _scheme(self, description):
        return AttentionQuantScheme.from_quant_config(
            SimpleNamespace(quant_description=description), "blocks.0.attn1"
        )

    def test_registers_buffers_and_binds(self):
        scheme = self._scheme(
            {
                "blocks.0.attn1.quant_type": "MXFP4_DYNAMIC",
                "blocks.0.attn1.to_q_rot": "FLOAT",
                "blocks.0.attn1.to_k_rot": "FLOAT",
            }
        )
        registered = []
        module = type(
            "Module",
            (),
            {
                "register_buffer": staticmethod(
                    lambda name, tensor, persistent=True: registered.append(
                        (name, tuple(tensor.shape), tensor.dtype, persistent)
                    )
                ),
                "q_rot": None,
                "k_rot": None,
            },
        )()
        self.assertTrue(scheme.register_qk_rotation_buffers(module, head_dim=128))
        self.assertEqual(
            registered,
            [
                ("q_rot", (128, 128), torch.bfloat16, True),
                ("k_rot", (128, 128), torch.bfloat16, True),
            ],
        )
        # apply_qk_rotation can now reach the module through the weakref bind.
        module.q_rot = torch.eye(128)
        module.k_rot = torch.eye(128)
        query = torch.randn(2, 4, 128)
        rotated_q, rotated_k = scheme.apply_qk_rotation(query, query)
        torch.testing.assert_close(rotated_q, query)
        torch.testing.assert_close(rotated_k, query)

    def test_no_registration_without_exported_rotations(self):
        scheme = self._scheme({"blocks.0.attn1.quant_type": "MXFP4_DYNAMIC"})
        module = type("Module", (), {"register_buffer": lambda *a, **k: None})()
        self.assertFalse(scheme.register_qk_rotation_buffers(module, head_dim=128))

    def test_legacy_q_rot_spelling_also_registers(self):
        scheme = self._scheme(
            {
                "blocks.0.attn1.q_rot": "FLOAT",
                "blocks.0.attn1.k_rot": "FLOAT",
            }
        )
        module = type("Module", (), {"register_buffer": lambda *a, **k: None})()
        self.assertTrue(scheme.use_offline_qk_rotation)
        self.assertFalse(scheme.is_quant, "Flux-style legacy path stays unquantized")
        # Registration works from the legacy descriptor spelling too.
        self.assertTrue(scheme.register_qk_rotation_buffers(module, head_dim=128))


class TestPrecisionGuard(CustomTestCase):
    def _transformer(self):
        contexts = [_FakeCachedContext(NUM_STEPS) for _ in range(2)]
        modules = [
            _FakeQuantModule(
                AttentionQuantScheme("blocks.0.attn", "MXFP4", {}, True, True),
                "attention_quant_scheme",
            ),
            _FakeQuantModule(_FakeLinearScheme({}), "scheme"),
        ]
        return _FakeTransformer(modules, contexts)

    def test_invalidates_when_linear_precision_flips(self):
        transformer = self._transformer()
        install_quant_precision_guard(transformer)
        batch = SimpleNamespace(num_inference_steps=NUM_STEPS)

        _run_steps(transformer, batch, [0, 1, 2, 3])
        context = transformer._contexts[0]
        # Step 0 initializes the histories; step 2 flips W4A8 -> W4A4.
        self.assertEqual(context.clear_buffers_calls, 2)
        self.assertEqual(context.calibrator_reset_calls, 4)  # 2 calls x 2 steps
        # Steps 1 and 3 keep the signature stable and must not invalidate.
        self.assertEqual(transformer.forward_calls, 4)
        # Boundary steps (0, 1) and the precision-change step (2) compute;
        # the stable step 3 keeps the caller's mask.
        self.assertEqual(
            [all(all(m) for m in masks) for masks in transformer.observed_masks],
            [True, True, True, False],
        )
        for context in transformer._contexts:
            self.assertEqual(
                context.cache_config.steps_computation_mask, [0] * NUM_STEPS
            )

    def test_new_batch_resets_histories(self):
        transformer = self._transformer()
        install_quant_precision_guard(transformer)
        _run_steps(transformer, SimpleNamespace(num_inference_steps=NUM_STEPS), [0, 2])
        context = transformer._contexts[0]
        self.assertEqual(context.clear_buffers_calls, 2)
        _run_steps(transformer, SimpleNamespace(num_inference_steps=NUM_STEPS), [0])
        self.assertEqual(context.clear_buffers_calls, 3)

    def test_install_is_idempotent(self):
        transformer = self._transformer()
        install_quant_precision_guard(transformer)
        guarded = transformer.forward
        self.assertTrue(getattr(guarded, "_sglang_precision_guard", False))
        install_quant_precision_guard(transformer)
        self.assertIs(transformer.forward, guarded)

    def test_no_schemes_no_wrap(self):
        transformer = self._transformer()
        transformer._modules_list = []
        install_quant_precision_guard(transformer)
        # No policy-bearing schemes: forward is left exactly as it was.
        self.assertNotIn("forward", vars(transformer))


if __name__ == "__main__":
    unittest.main()
