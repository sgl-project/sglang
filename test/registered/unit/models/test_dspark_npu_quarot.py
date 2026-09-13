"""CPU tests for the opt-in GLM DSpark loading contract, not NPU accuracy."""

import asyncio
import copy
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from sglang.srt.hardware_backend.npu import dspark_quarot as qr
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestGlmDSparkQuaRot(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.q_path = self.root / "optional" / "quarot.safetensors"
        self.q_path.parent.mkdir()
        # An asymmetric permutation distinguishes Q from Q.T; its scale is
        # deliberately not one so automatic norm compensation fails the test.
        self.q = torch.tensor([[0, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0]])
        self.save_q(self.q)
        self.hf = SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], hidden_size=3)
        self.model_config = SimpleNamespace(
            hf_text_config=self.hf, model_path=str(self.root)
        )
        self.description = {
            "is_rot_used": True,
            "metadata": {},
            "optional": {
                "quarot": {
                    "rotation_map": {"global_rotation": "optional/quarot.safetensors"}
                }
            },
        }
        self.quant = SimpleNamespace(
            get_name=lambda: "modelslim", quant_description=self.description
        )
        self.model = SimpleNamespace(quant_config=self.quant)
        self.config = qr.GlmDSparkQuaRotConfig(str(self.q_path), 3, str(self.root))

    def save_q(self, value, key="global_rotation"):
        save_file({key: value.contiguous()}, str(self.q_path))

    def build(self, **overrides):
        arguments = {
            "device": "npu:0",
            "mode": "original",
            "target_model_config": self.model_config,
            "target_model": self.model,
        }
        arguments.update(overrides)
        return qr.build_glm_dspark_quarot_config(**arguments)

    def test_target_metadata_is_read_only_and_config_is_frozen(self):
        before = copy.deepcopy(self.description)
        self.assertEqual(self.build(), self.config)
        self.assertEqual(self.description, before)
        with self.assertRaises(FrozenInstanceError):
            self.config.hidden_size = 4

    def test_inactive_or_unrelated_targets_do_not_open_q(self):
        cases = (
            {"mode": None},
            {"mode": ""},
            {"device": "cuda:0"},
            {"device": "cpu"},
            {"target_model_config": SimpleNamespace()},
            {
                "target_model_config": SimpleNamespace(
                    hf_text_config=SimpleNamespace(
                        architectures=["DeepseekV4ForCausalLM"]
                    )
                )
            },
            {"target_model": SimpleNamespace()},
            {
                "target_model": SimpleNamespace(
                    quant_config=SimpleNamespace(get_name=lambda: "fp8")
                )
            },
        )
        with patch.object(
            qr, "safe_open", side_effect=AssertionError("unexpected Q read")
        ):
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    self.assertIsNone(self.build(**arguments))

    def test_enabled_target_rejects_unknown_mode_and_missing_quarot(self):
        with self.assertRaisesRegex(ValueError, "must be 'original'"):
            self.build(mode="auto")
        self.description["is_rot_used"] = False
        with self.assertRaisesRegex(ValueError, "QuaRot target"):
            self.build()
        self.description["is_rot_used"] = True
        del self.description["optional"]
        with self.assertRaisesRegex(ValueError, "global_rotation path"):
            self.build()

    def test_enabled_target_rejects_missing_file_bad_shape_and_bad_key(self):
        self.q_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()
        self.save_q(torch.eye(2))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.build()
        self.save_q(self.q, key="rot.weight")
        with self.assertRaisesRegex(ValueError, "global_rotation tensor"):
            self.build()

    def test_relative_path_can_resolve_to_shared_q_symlink(self):
        rotation_map = self.description["optional"]["quarot"]["rotation_map"]
        rotation_map["global_rotation"] = str(self.q_path)
        with self.assertRaisesRegex(ValueError, "relative"):
            self.build()
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "other.safetensors"
            save_file({"global_rotation": self.q}, str(target))
            link = self.root / "outside.safetensors"
            link.symlink_to(target)
            rotation_map["global_rotation"] = link.name
            config = self.build()
            self.assertEqual(config.rotation_path, str(target.resolve()))
            self.assertEqual(config.target_model_path, str(self.root))
            weight = torch.arange(18, dtype=torch.float32).reshape(3, 6)
            expected = torch.cat([block @ self.q for block in weight.split(3, 1)], 1)
            torch.testing.assert_close(qr.fold_glm_dspark_fc(weight, config), expected)

    def test_target_hidden_width_must_be_positive_integer(self):
        for value in (None, 0, -1, True, 3.5):
            with self.subTest(value=value):
                self.hf.hidden_size = value
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.build()

    def test_scope_nesting_none_and_exception_restore(self):
        other = qr.GlmDSparkQuaRotConfig("other", 3, "target")
        self.assertIsNone(qr.get_glm_dspark_quarot_config())
        with qr.glm_dspark_quarot_scope(self.config):
            self.assertIs(qr.get_glm_dspark_quarot_config(), self.config)
            with qr.glm_dspark_quarot_scope(None):
                self.assertIsNone(qr.get_glm_dspark_quarot_config())
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with qr.glm_dspark_quarot_scope(other):
                    self.assertIs(qr.get_glm_dspark_quarot_config(), other)
                    raise RuntimeError("nested")
            self.assertIs(qr.get_glm_dspark_quarot_config(), self.config)
        self.assertIsNone(qr.get_glm_dspark_quarot_config())

    def test_async_constructions_do_not_share_scope_updates(self):
        async def visit(config):
            with qr.glm_dspark_quarot_scope(config):
                await asyncio.sleep(0)
                self.assertIs(qr.get_glm_dspark_quarot_config(), config)
            self.assertIsNone(qr.get_glm_dspark_quarot_config())

        async def run():
            await asyncio.gather(visit(self.config), visit(None))

        asyncio.run(run())

    def test_asymmetric_scaled_q_folds_each_block_without_correction(self):
        original = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 8 - 1
        # Independent dense block-diagonal oracle also checks feature order.
        rotation = torch.block_diag(self.q.double(), self.q.double())
        expected = original.double() @ rotation
        with patch.object(
            torch, "set_num_threads", side_effect=AssertionError("threads")
        ):
            result = qr.fold_glm_dspark_fc(original, self.config)
        torch.testing.assert_close(result.double(), expected, rtol=0, atol=0)
        self.assertFalse(
            torch.equal(result, (original.double() @ rotation * 4).float())
        )

    def test_dtype_source_and_reload_are_preserved(self):
        file_before = self.q_path.read_bytes()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = torch.arange(18, dtype=dtype).reshape(3, 6).requires_grad_()
                before = source.detach().clone()
                first = qr.fold_glm_dspark_fc(source, self.config)
                again = qr.fold_glm_dspark_fc(source, self.config)
                self.assertEqual(first.dtype, dtype)
                self.assertEqual(first.device.type, "cpu")
                self.assertFalse(first.requires_grad)
                self.assertNotEqual(first.data_ptr(), source.data_ptr())
                torch.testing.assert_close(first, again, rtol=0, atol=0)
                torch.testing.assert_close(source, before, rtol=0, atol=0)
        self.assertEqual(self.q_path.read_bytes(), file_before)

    def test_fp32_product_rounds_once_to_bf16_storage(self):
        q = torch.tensor(
            [[0.12, 0.93, -0.27], [1.21, -0.48, 0.37], [-0.82, 0.45, 0.68]]
        )
        self.save_q(q)
        source = (torch.arange(18).reshape(3, 6) / 7 - 1).bfloat16()
        expected = (
            source.double() @ torch.block_diag(q.double(), q.double())
        ).bfloat16()
        result = qr.fold_glm_dspark_fc(source, self.config)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_row_tiles_include_tail_and_second_feature_block(self):
        width = 129
        q = torch.roll(torch.eye(width), 7, 1) * 0.5
        self.save_q(q)
        config = qr.GlmDSparkQuaRotConfig(str(self.q_path), width, str(self.root))
        source = torch.arange(width * width * 2, dtype=torch.float32).reshape(width, -1)
        result = qr.fold_glm_dspark_fc(source, config)
        expected = source.double() @ torch.block_diag(q.double(), q.double())
        torch.testing.assert_close(result.double(), expected, rtol=0, atol=0)

    def test_bad_fc_shapes_and_dtypes_are_rejected_before_q_read(self):
        with patch.object(
            qr, "safe_open", side_effect=AssertionError("unexpected Q read")
        ):
            for value in (
                torch.ones(2, 6),
                torch.ones(3, 5),
                torch.ones(3, 0),
                torch.ones(3),
                torch.ones(3, 6, dtype=torch.int8),
            ):
                with self.subTest(shape=value.shape, dtype=value.dtype):
                    with self.assertRaisesRegex(ValueError, "FC must be floating"):
                        qr.fold_glm_dspark_fc(value, self.config)

    def test_changed_or_nonfinite_q_and_fc_fail_without_source_changes(self):
        source = torch.arange(18, dtype=torch.float32).reshape(3, 6)
        before = source.clone()
        for q in (
            torch.eye(2),
            torch.eye(3, dtype=torch.int32),
            torch.full((3, 3), float("nan")),
            torch.full((3, 3), float("inf")),
        ):
            with self.subTest(dtype=q.dtype, shape=q.shape):
                self.save_q(q)
                with self.assertRaises(ValueError):
                    qr.fold_glm_dspark_fc(source, self.config)
                torch.testing.assert_close(source, before, rtol=0, atol=0)
        self.save_q(self.q)
        source[1, 1] = float("inf")
        with self.assertRaisesRegex(ValueError, "FC contains non-finite"):
            qr.fold_glm_dspark_fc(source, self.config)

    def test_overflow_on_bf16_storage_is_rejected(self):
        self.save_q(torch.eye(3) * 1e4)
        source = torch.full((3, 6), 1e35, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "converted FC contains non-finite"):
            qr.fold_glm_dspark_fc(source, self.config)


if __name__ == "__main__":
    unittest.main()
