"""Scalar refresh contracts: prepare is private, captured outputs stay in place."""

import ast
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_root = Path(__file__).resolve().parents[4] / "python/sglang/srt"
_spec = importlib.util.spec_from_file_location(
    "gpu_delta_derived_under_test", _root / "weight_sync/gpu_delta/derived.py"
)
derived = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = derived
_spec.loader.exec_module(derived)


def _production_helpers(relative_path, names):
    # Execute the actual pure helpers without loading CUDA inference backends.
    # This follows the checkpoint-admission tests in this same test directory.
    path = _root / relative_path
    tree = ast.parse(path.read_text())
    scope = {
        "torch": torch,
        "Optional": Optional,
        "logger": logging.getLogger(__name__),
        "_cutedsl_logged_scalarize": set(),
        "log_info_on_rank0": lambda *args: None,
        "print_warning_once": lambda *args: None,
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(  # noqa: S102 - exact checked-in source, no CUDA-only module imports.
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope
    )
    return SimpleNamespace(**{name: scope[name] for name in names})


def _layer(backend, w4a16=False):
    cute = backend == "flashinfer_cutedsl"
    mega = backend == "flashinfer_megamoe"
    vector_input = cute
    return SimpleNamespace(
        num_local_experts=2,
        num_experts=4,
        moe_ep_rank=1,
        quant_config=SimpleNamespace(use_per_token_activation=False),
        moe_runner_config=SimpleNamespace(
            is_gated=True,
            activation="silu",
            gemm1_clamp_limit=24.0,
            swiglu_limit=None,
            gemm1_alpha=1.0,
        ),
        w13_weight_scale_2=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        w2_weight_scale_2=torch.tensor([6.0, 7.0]),
        w13_input_scale=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        w2_input_scale=torch.tensor([2.0, 4.0, 6.0, 8.0]),
        g1_alphas=torch.zeros(2),
        g1_alphas_up=torch.zeros(2),
        g2_alphas=torch.zeros(2),
        g1_scale_c=torch.zeros(2),
        w13_input_scale_quant=torch.ones(2) if vector_input else torch.tensor(1.0),
        w2_input_scale_quant=torch.ones(2) if cute or mega else torch.tensor(1.0),
        gemm1_clamp_limit=torch.zeros(2) if not (cute or mega) else None,
        gemm1_beta=torch.zeros(2) if not (cute or mega) else None,
        _cutedsl_wrapper=(
            SimpleNamespace(quant_mode="w4a16" if w4a16 else "w4a4") if cute else None
        ),
        _cutedsl_scales=(torch.zeros(2), torch.ones(1), torch.zeros(2)),
        _cutedsl_input_scale=torch.ones(1),
        _flashinfer_megamoe_input_norm_const=0.25,
        _flashinfer_megamoe_layer=SimpleNamespace(
            _megakernel_config=SimpleNamespace(input_norm_const=0.25)
        ),
    )


def _value(copies, destination):
    return next(copy.source for copy in copies if copy.destination is destination)


def _prepare_updates(updates, raw_copies):
    targets = [target for group, _ in raw_copies.values() for target in group]
    return derived.prepare_updates(derived.plan_updates(updates, targets), raw_copies)


class TestDerivedNvfp4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normal = _production_helpers(
            "layers/quantization/modelopt_quant.py",
            {"_compute_gemm1_alphas", "_input_scale_to_local_experts"},
        )
        cls.cute = _production_helpers(
            "layers/moe/moe_runner/flashinfer_cutedsl.py",
            {"cutedsl_quant_scale_to_scalar", "resolve_cutedsl_standard_scales"},
        )
        cls.trt = _production_helpers(
            "layers/moe/moe_runner/flashinfer_trtllm.py", {"_compute_g1_scale_c"}
        )

    def setUp(self):
        self.helpers = patch.object(
            derived,
            "_normal_scale_helpers",
            return_value=(
                self.normal._compute_gemm1_alphas,
                self.normal._input_scale_to_local_experts,
            ),
        )
        self.helpers.start()
        self.addCleanup(self.helpers.stop)

        self.cute_patch = patch.object(
            derived,
            "_cutedsl_scale_helper",
            return_value=self.cute.resolve_cutedsl_standard_scales,
        )
        self.cute_patch.start()
        self.addCleanup(self.cute_patch.stop)
        self.trt_patch = patch.object(
            derived,
            "_trtllm_scale_helper",
            return_value=self.trt._compute_g1_scale_c,
        )
        self.trt_patch.start()
        self.addCleanup(self.trt_patch.stop)

    def test_activation_only_update_prepares_complete_values_without_mutation(self):
        for backend in ("flashinfer_trtllm", "flashinfer_trtllm_routed"):
            with self.subTest(backend=backend):
                layer = _layer(backend)
                update = derived.nvfp4_derived_update(layer, backend)
                before = [
                    (tensor, tensor.clone(), tensor.data_ptr())
                    for tensor in (*update.sources, *update.destinations)
                ]
                incoming = torch.tensor([1.0, 2.0, 3.0, 8.0])
                copies = _prepare_updates(
                    (update,),
                    {torch.float32: ([layer.w13_input_scale[3:4]], [incoming[3:4]])},
                )
                torch.testing.assert_close(
                    _value(copies, layer.g1_alphas), torch.tensor([16.0, 32.0])
                )
                torch.testing.assert_close(
                    _value(copies, layer.g1_alphas_up), torch.tensor([24.0, 40.0])
                )
                torch.testing.assert_close(
                    _value(copies, layer.g2_alphas), torch.tensor([48.0, 56.0])
                )
                torch.testing.assert_close(
                    _value(copies, layer.g1_scale_c), torch.tensor([3.0, 5.0])
                )
                torch.testing.assert_close(
                    _value(copies, layer.gemm1_clamp_limit), torch.tensor([1.5, 0.75])
                )
                torch.testing.assert_close(
                    _value(copies, layer.gemm1_beta), torch.tensor([1 / 16, 1 / 32])
                )
                for tensor, old, pointer in before:
                    torch.testing.assert_close(tensor, old)
                    self.assertEqual(tensor.data_ptr(), pointer)
                incoming.fill_(100)
                torch.testing.assert_close(
                    _value(copies, layer.g1_alphas), torch.tensor([16.0, 32.0])
                )

    def test_cutedsl_local_experts_and_shared_fc2_quantization(self):
        layer = _layer("flashinfer_cutedsl")
        update = derived.nvfp4_derived_update(layer, "flashinfer_cutedsl")
        copies = update.prepare(
            {id(layer.w2_input_scale): torch.tensor([100.0, 200.0, 10.0, 20.0])}
        )
        torch.testing.assert_close(
            _value(copies, layer.g2_alphas), torch.tensor([60.0, 140.0])
        )
        # Standard CuTe scalarizes only this rank's selected experts for FC2.
        torch.testing.assert_close(
            _value(copies, layer._cutedsl_scales[1]), torch.tensor([0.05])
        )
        torch.testing.assert_close(
            _value(copies, layer._cutedsl_scales[2]), torch.tensor([120.0, 140.0])
        )
        torch.testing.assert_close(
            _value(copies, layer._cutedsl_input_scale), torch.tensor([0.25])
        )

    def test_w4a16_retains_copy_only_semantics(self):
        layer = _layer("flashinfer_cutedsl", w4a16=True)
        update = derived.nvfp4_derived_update(layer, "flashinfer_cutedsl")
        self.assertEqual(len(update.sources), 2)
        weight = torch.tensor([[11.0, 13.0], [17.0, 19.0]])
        with patch.object(
            derived,
            "_normal_scale_helpers",
            side_effect=AssertionError("W4A16 needs no arithmetic helper"),
        ):
            copies = update.prepare({id(layer.w13_weight_scale_2): weight})
        torch.testing.assert_close(
            _value(copies, layer.g1_alphas), torch.tensor([11.0, 17.0])
        )
        torch.testing.assert_close(
            _value(copies, layer.g1_alphas_up), torch.tensor([13.0, 19.0])
        )
        torch.testing.assert_close(
            _value(copies, layer._cutedsl_scales[0]), torch.tensor([11.0, 17.0])
        )
        self.assertTrue(
            all(copy.destination is not layer._cutedsl_input_scale for copy in copies)
        )

    def test_megamoe_rejects_captured_scalar_change_and_gate_up_mismatch(self):
        layer = _layer("flashinfer_megamoe")
        layer.w13_weight_scale_2[:, 1].copy_(layer.w13_weight_scale_2[:, 0])
        update = derived.nvfp4_derived_update(layer, "flashinfer_megamoe")
        copies = update.prepare(
            {id(layer.w2_input_scale): torch.tensor([1.0, 1.0, 3.0, 5.0])}
        )
        torch.testing.assert_close(
            _value(copies, layer.g2_alphas), torch.tensor([18.0, 35.0])
        )
        with self.assertRaisesRegex(ValueError, "input normalization changed"):
            update.prepare(
                {id(layer.w13_input_scale): torch.tensor([1.0, 2.0, 3.0, 5.0])}
            )
        with self.assertRaisesRegex(ValueError, "matching gate/up"):
            update.prepare(
                {id(layer.w13_weight_scale_2): torch.tensor([[2.0, 3.0], [4.0, 4.0]])}
            )
        torch.testing.assert_close(layer.g1_alphas, torch.zeros(2))

        layer = _layer("flashinfer_megamoe")
        layer.w13_weight_scale_2[:, 1].copy_(layer.w13_weight_scale_2[:, 0])
        update = derived.nvfp4_derived_update(layer, "megamoe")
        near = layer.w13_weight_scale_2.clone()
        near[:, 1] += 1e-6
        update.prepare({id(layer.w13_weight_scale_2): near})
        near[0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "matching gate/up"):
            update.prepare({id(layer.w13_weight_scale_2): near})
        layer._flashinfer_megamoe_input_norm_const = 0.1
        layer._flashinfer_megamoe_layer._megakernel_config.input_norm_const = 0.1
        update = derived.nvfp4_derived_update(layer, "megamoe")
        with self.assertRaisesRegex(ValueError, "input normalization changed"):
            update.prepare({id(layer.w13_input_scale): torch.full((4,), 10.0)})

    def test_megamoe_batch_validates_once_before_returning_any_copies(self):
        layers = [_layer("flashinfer_megamoe") for _ in range(2)]
        for layer in layers:
            layer.w13_weight_scale_2[:, 1].copy_(layer.w13_weight_scale_2[:, 0])
        updates = [derived.nvfp4_derived_update(layer, "megamoe") for layer in layers]
        targets = [layer.w2_weight_scale_2[:1] for layer in layers]
        plans = derived.plan_updates(updates, targets)
        raw = {torch.float32: (targets, [torch.tensor([11.0]), torch.tensor([13.0])])}
        cpu = torch.Tensor.cpu
        transfers = []

        def transfer(tensor, *args, **kwargs):
            transfers.append(tensor.numel())
            return cpu(tensor, *args, **kwargs)

        with (
            patch.object(torch.Tensor, "cpu", transfer),
            patch.object(
                torch,
                "allclose",
                side_effect=AssertionError("must use device predicates"),
            ),
        ):
            copies = derived.prepare_updates(plans, raw)
        self.assertEqual(transfers, [4])
        torch.testing.assert_close(
            _value(copies, layers[1].g2_alphas), torch.tensor([78.0, 56.0])
        )
        for layer in layers:
            torch.testing.assert_close(layer.g2_alphas, torch.zeros(2))
            torch.testing.assert_close(
                layer.w2_weight_scale_2, torch.tensor([6.0, 7.0])
            )

        # Failure in a later layer must not expose or write the earlier copies.
        layers[1].w13_input_scale[-1] = 5.0
        transfers.clear()
        with (
            patch.object(torch.Tensor, "cpu", transfer),
            self.assertRaisesRegex(ValueError, "input normalization changed"),
        ):
            derived.prepare_updates(plans, raw)
        self.assertEqual(transfers, [4])
        for layer in layers:
            torch.testing.assert_close(layer.g1_alphas, torch.zeros(2))

    def test_helpers_are_resolved_once_and_per_token_constants_are_reused(self):
        for backend in ("flashinfer_trtllm", "flashinfer_cutedsl"):
            with self.subTest(backend=backend):
                layer = _layer(backend)
                layer.quant_config.use_per_token_activation = True
                update = derived.nvfp4_derived_update(layer, backend)
                weights = torch.tensor([[11.0, 13.0], [17.0, 19.0]])
                with (
                    patch.object(
                        derived, "_normal_scale_helpers", side_effect=AssertionError
                    ),
                    patch.object(
                        derived, "_cutedsl_scale_helper", side_effect=AssertionError
                    ),
                    patch.object(
                        derived, "_trtllm_scale_helper", side_effect=AssertionError
                    ),
                    patch.object(
                        torch,
                        "ones_like",
                        side_effect=AssertionError("neutral tensors already exist"),
                    ),
                ):
                    copies = update.prepare({id(layer.w13_weight_scale_2): weights})
                torch.testing.assert_close(
                    _value(copies, layer.g1_alphas), weights[:, 0]
                )
                torch.testing.assert_close(
                    _value(copies, layer.g1_alphas_up), weights[:, 1]
                )
                torch.testing.assert_close(
                    _value(copies, layer.w13_input_scale_quant),
                    layer.w13_input_scale_quant,
                )
                if backend == "flashinfer_cutedsl":
                    torch.testing.assert_close(
                        _value(copies, layer._cutedsl_input_scale),
                        layer._cutedsl_input_scale,
                    )

    def test_prepare_updates_overlays_partial_gate_up_scalars(self):
        layer = _layer("flashinfer_trtllm")
        update = derived.nvfp4_derived_update(layer, "trtllm")
        other = _layer("flashinfer_trtllm")
        unchanged = derived.nvfp4_derived_update(other, "trtllm")
        sources_before = [source.clone() for source in update.sources]
        copies = _prepare_updates(
            (update, unchanged),
            {
                torch.float32: (
                    [
                        layer.w13_weight_scale_2[0, 1:2],
                        layer.w13_weight_scale_2[1, 0:1],
                    ],
                    [torch.tensor([11.0]), torch.tensor([17.0])],
                )
            },
        )
        torch.testing.assert_close(
            _value(copies, layer.g1_alphas), torch.tensor([8.0, 68.0])
        )
        torch.testing.assert_close(
            _value(copies, layer.g1_alphas_up), torch.tensor([44.0, 20.0])
        )
        self.assertEqual(len(copies), len(update.destinations))
        self.assertEqual(_prepare_updates((update,), {}), ())
        for source, before in zip(update.sources, sources_before):
            torch.testing.assert_close(source, before)

    def test_derived_source_destination_and_patch_geometry_fail_before_writes(self):
        layer = _layer("flashinfer_trtllm")
        update = derived.nvfp4_derived_update(layer, "flashinfer_trtllm")
        with self.assertRaisesRegex(ValueError, "source geometry changed"):
            update.prepare({id(layer.w2_weight_scale_2): torch.ones(3)})
        layer.g1_alphas.resize_(3)
        with self.assertRaisesRegex(ValueError, "destination geometry changed"):
            update.prepare({})
        torch.testing.assert_close(layer.g2_alphas, torch.zeros(2))

        allocation = torch.arange(4, dtype=torch.float32)
        source = allocation[:2]
        update = derived.DerivedUpdate(
            (source,), (torch.zeros(2),), lambda values: values
        )
        with self.assertRaisesRegex(ValueError, "outside its source"):
            _prepare_updates(
                (update,), {torch.float32: ([allocation[3:4]], [torch.tensor([9.0])])}
            )
        with self.assertRaisesRegex(ValueError, "updates overlap"):
            _prepare_updates(
                (update,),
                {
                    torch.float32: (
                        [source[:1], source[:1]],
                        [torch.tensor([9.0]), torch.tensor([9.0])],
                    )
                },
            )
        torch.testing.assert_close(allocation, torch.arange(4, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
