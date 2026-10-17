# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for bounded ROCm prefill policy and DSA integration.

These tests exercise dispatch/configuration, not GPU kernel numerics or speed.
"""

import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.rocm_prefill_policy import (
    ROCM_DSA_PREFILL_POLICIES,
    DSAPrefillShape,
    RocmDSAPrefillPolicy,
)
from sglang.srt.runtime_context import get_context, get_exec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def shape(rows, longest=None, batch=1, kv=None):
    return DSAPrefillShape(
        total_q_tokens=rows,
        max_q_len=rows if longest is None else longest,
        batch_size=batch,
        max_kv_len=rows if kv is None else kv,
    )


class TestRocmDSAPrefillPolicy(CustomTestCase):
    def test_bf16_crossover_boundaries(self):
        policy = RocmDSAPrefillPolicy("glm53-mi350x-tp4-bf16-v1")
        for rows, kv, expected in (
            (2048, 8192, "tilelang"),
            (2049, 8192, "aiter"),
            (1536, 32767, "tilelang"),
            (1535, 32768, "tilelang"),
            (1536, 32768, "aiter"),
            (1024, 65535, "tilelang"),
            (1023, 65536, "tilelang"),
            (1024, 65536, "aiter"),
        ):
            with self.subTest(rows=rows, kv=kv):
                self.assertEqual(policy.select(shape(rows, kv=kv)), expected)

    def test_fp8_uses_longest_fresh_row_not_aggregate_rows(self):
        policy = RocmDSAPrefillPolicy("glm53-mi350x-tp4-fp8-v1")
        for item, expected in (
            (shape(3071), "tilelang"),
            (shape(3072), "aiter"),
            (shape(3072, longest=1536, batch=2, kv=1536), "tilelang"),
            (shape(6144, longest=1536, batch=4, kv=1536), "tilelang"),
            (shape(6144, longest=3072, batch=2, kv=3072), "aiter"),
        ):
            with self.subTest(shape=item):
                self.assertEqual(policy.select(item), expected)

    def test_fp8_cached_context_boundaries(self):
        policy = RocmDSAPrefillPolicy("glm53-mi350x-tp4-fp8-v1")
        for rows, longest, batch, kv, expected in (
            (4096, 1024, 4, 8191, "tilelang"),
            (4095, 1024, 4, 8192, "tilelang"),
            (4096, 1024, 4, 8192, "aiter"),
            (2048, 2048, 1, 16383, "tilelang"),
            (2048, 2047, 2, 16384, "tilelang"),
            (2048, 2048, 1, 16384, "aiter"),
            (1024, 256, 4, 65535, "tilelang"),
            (1023, 256, 4, 65536, "tilelang"),
            (1024, 256, 4, 65536, "aiter"),
            (512, 128, 4, 131072, "tilelang"),
        ):
            with self.subTest(rows=rows, longest=longest, batch=batch, kv=kv):
                self.assertEqual(
                    policy.select(shape(rows, longest, batch, kv)), expected
                )

    def test_workspace_and_unmeasured_batch_guards_take_precedence(self):
        for policy_id in ROCM_DSA_PREFILL_POLICIES:
            policy = RocmDSAPrefillPolicy(policy_id)
            for item, expected in (
                (shape(32768, longest=4096, batch=8, kv=131072), "aiter"),
                (shape(32769, longest=4096, batch=8, kv=131072), "tilelang"),
                (shape(27648, longest=3072, batch=9, kv=131072), "tilelang"),
                (shape(131072, kv=131072), "tilelang"),
                (shape(0, batch=0, kv=0), "tilelang"),
            ):
                with self.subTest(policy=policy_id, shape=item):
                    self.assertEqual(policy.select(item), expected)

    def test_unknown_policy_is_not_silently_auto_selected(self):
        for policy_id in ("auto", "force-aiter", "glm53-mi350x-tp8-fp8-v1"):
            with (
                self.subTest(policy=policy_id),
                self.assertRaisesRegex(ValueError, "Unknown ROCm DSA prefill policy"),
            ):
                RocmDSAPrefillPolicy(policy_id)

    def test_negative_metadata_is_rejected(self):
        valid = dict(total_q_tokens=1, max_q_len=1, batch_size=1, max_kv_len=1)
        for name in valid:
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, name):
                DSAPrefillShape(**{**valid, name: -1})


class TestRocmDSAPrefillRuntimeValidation(CustomTestCase):
    @staticmethod
    def validate(policy_id="glm53-mi350x-tp4-bf16-v1", **overrides):
        policy = RocmDSAPrefillPolicy(policy_id)
        config = dict(
            architecture="gfx950",
            attn_tp_size=4,
            attn_cp_size=1,
            dcp_size=1,
            kv_dtype=policy.kv_dtype,
            prefill_backend="tilelang",
            prefill_graph_enabled=False,
            deterministic=False,
            index_topk=2048,
            index_kpool=1,
            num_q_heads=16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        policy.validate_runtime(**{**config, **overrides})

    def test_measured_profiles_and_architecture_qualifiers_are_accepted(self):
        for policy_id in ROCM_DSA_PREFILL_POLICIES:
            for arch in ("gfx950", "gfx950:sramecc+:xnack-"):
                with self.subTest(policy=policy_id, architecture=arch):
                    self.assertIsNone(self.validate(policy_id, architecture=arch))

    def test_untuned_runtime_is_rejected_before_forward(self):
        for overrides, error in (
            ({"architecture": "gfx942"}, "gfx950"),
            ({"attn_tp_size": 8}, "TP4"),
            ({"attn_cp_size": 2}, "CP or DCP"),
            ({"dcp_size": 2}, "CP or DCP"),
            ({"prefill_backend": "aiter"}, "fallback"),
            ({"prefill_graph_enabled": True}, "eager prefill"),
            ({"deterministic": True}, "deterministic"),
            ({"index_topk": 1024}, "index_topk"),
            ({"index_kpool": 2}, "index_kpool"),
            ({"num_q_heads": 32}, "16 local query heads"),
            ({"kv_lora_rank": 256}, "kv_lora_rank"),
            ({"qk_rope_head_dim": 32}, "qk_rope_head_dim"),
        ):
            with (
                self.subTest(config=overrides),
                self.assertRaisesRegex(ValueError, error),
            ):
                self.validate(**overrides)

    def test_dtype_profiles_cannot_be_interchanged(self):
        for policy_id, wrong_dtype in (
            ("glm53-mi350x-tp4-bf16-v1", "float8_e4m3fn"),
            ("glm53-mi350x-tp4-fp8-v1", "bfloat16"),
            ("glm53-mi350x-tp4-fp8-v1", "float8_e4m3fnuz"),
        ):
            with (
                self.subTest(policy=policy_id),
                self.assertRaisesRegex(ValueError, "requires KV dtype"),
            ):
                self.validate(policy_id, kv_dtype=wrong_dtype)


class TestRocmDSAPrefillIntegration(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        maybe_stub_sgl_kernel()
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        cls.backend_class = DeepseekSparseAttnBackend
        cls.forward_mode = ForwardMode

    def backend(self, policy_id="glm53-mi350x-tp4-fp8-v1", use_mha=False):
        backend = self.backend_class.__new__(self.backend_class)
        backend.rocm_prefill_policy = (
            None if policy_id is None else RocmDSAPrefillPolicy(policy_id)
        )
        backend.use_mha = use_mha
        backend.dsa_prefill_impl = "tilelang"
        backend.dsa_decode_impl = "tilelang"
        return backend

    def select(self, backend, lengths, kv, mode=None):
        batch = SimpleNamespace(
            forward_mode=self.forward_mode.EXTEND if mode is None else mode
        )
        return backend._select_rocm_prefill_impl(batch, lengths, kv)

    def test_disabled_policy_and_dense_prefill_keep_existing_dispatch(self):
        self.assertIsNone(self.select(self.backend(None), [8192], 8192))
        self.assertIsNone(self.select(self.backend(use_mha=True), [8192], 8192))

    def test_every_non_prefill_mode_keeps_existing_dispatch(self):
        backend = self.backend()
        for mode in self.forward_mode:
            if mode != self.forward_mode.EXTEND:
                with self.subTest(mode=mode):
                    self.assertIsNone(self.select(backend, [8192], 131072, mode))

    def test_cached_prefix_is_not_counted_as_fresh_rows(self):
        backend = self.backend()
        self.assertEqual(self.select(backend, [128] * 4, 131072), "tilelang")
        self.assertEqual(self.select(backend, [256] * 4, 131072), "aiter")

    def test_backend_choice_does_not_leak_into_the_next_batch(self):
        backend = self.backend()
        self.assertEqual(self.select(backend, [3072], 8192), "aiter")
        self.assertEqual(self.select(backend, [128], 8192), "tilelang")
        self.assertIsNone(self.select(backend, [1], 8192, self.forward_mode.DECODE))
        self.assertEqual(backend.dsa_prefill_impl, "tilelang")
        self.assertEqual(backend.dsa_decode_impl, "tilelang")

    def test_layers_consume_the_stored_decision_without_reselecting(self):
        backend = self.backend()
        batch = SimpleNamespace(forward_mode=self.forward_mode.EXTEND)
        backend.forward_metadata = SimpleNamespace(selected_prefill_impl="aiter")
        # A different layer/query shape cannot overwrite this batch's snapshot.
        backend.rocm_prefill_policy = RocmDSAPrefillPolicy("glm53-mi350x-tp4-bf16-v1")
        self.assertEqual(backend._get_extend_impl(batch), "aiter")
        self.assertEqual(backend._get_extend_impl(batch), "aiter")
        backend.forward_metadata = SimpleNamespace(selected_prefill_impl="tilelang")
        self.assertEqual(backend._get_extend_impl(batch), "tilelang")

    def test_unselected_metadata_preserves_configured_speculative_backend(self):
        backend = self.backend()
        backend.dsa_decode_impl = "triton"
        for metadata in (
            SimpleNamespace(selected_prefill_impl=None),
            SimpleNamespace(),
        ):
            backend.forward_metadata = metadata
            for mode, expected in (
                (self.forward_mode.EXTEND, "tilelang"),
                (self.forward_mode.TARGET_VERIFY, "triton"),
                (self.forward_mode.DRAFT_EXTEND_V2, "triton"),
            ):
                with self.subTest(mode=mode):
                    batch = SimpleNamespace(forward_mode=mode)
                    self.assertEqual(backend._get_extend_impl(batch), expected)

    def test_stored_prefill_selection_cannot_override_other_phases(self):
        backend = self.backend()
        backend.dsa_decode_impl = "triton"
        backend.forward_metadata = SimpleNamespace(selected_prefill_impl="aiter")
        for mode in self.forward_mode:
            if mode == self.forward_mode.EXTEND:
                continue
            expected = (
                "triton"
                if mode.is_target_verify() or mode.is_draft_extend_v2()
                else "tilelang"
            )
            with self.subTest(mode=mode):
                self.assertEqual(
                    backend._get_extend_impl(SimpleNamespace(forward_mode=mode)),
                    expected,
                )

    def test_policy_rejects_non_rocm_before_probing_hardware(self):
        backend = self.backend()
        with (
            get_context().override_server_args(
                dsa_prefill_policy="glm53-mi350x-tp4-fp8-v1"
            ),
            patch("sglang.srt.layers.attention.dsa_backend._is_hip", False),
            patch("torch.cuda.get_device_properties", side_effect=AssertionError),
            self.assertRaisesRegex(ValueError, "only on ROCm"),
        ):
            backend._init_rocm_prefill_policy()

    def test_disabled_policy_does_not_probe_a_gpu(self):
        backend = self.backend(None)
        with (
            get_context().override_server_args(dsa_prefill_policy=None),
            patch("torch.cuda.get_device_properties", side_effect=AssertionError),
        ):
            self.assertIsNone(backend._init_rocm_prefill_policy())

    def test_missing_aiter_is_rejected_before_probing_hardware(self):
        backend = self.backend()
        with (
            get_context().override_server_args(
                dsa_prefill_policy="glm53-mi350x-tp4-bf16-v1"
            ),
            patch("sglang.srt.layers.attention.dsa_backend._is_hip", True),
            patch(
                "sglang.srt.layers.attention.dsa_backend.mla_decode_fwd",
                None,
                create=True,
            ),
            patch("torch.cuda.get_device_properties", side_effect=AssertionError),
            self.assertRaisesRegex(ImportError, "requires AITER"),
        ):
            backend._init_rocm_prefill_policy()

    def test_fp8_startup_warns_about_external_numerical_dependency(self):
        backend = self.backend()
        backend.device = "cpu"
        backend.token_to_kv_pool = SimpleNamespace(dtype=torch.float8_e4m3fn)
        backend.dsa_index_topk = 2048
        backend.dsa_index_kpool = 1
        backend.num_q_heads = 16
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        with (
            get_context().override_server_args(
                dsa_prefill_policy="glm53-mi350x-tp4-fp8-v1", tp_size=4
            ),
            patch("sglang.srt.layers.attention.dsa_backend._is_hip", True),
            patch(
                "sglang.srt.layers.attention.dsa_backend.mla_decode_fwd",
                lambda *args, **kwargs: None,
                create=True,
            ),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(gcnArchName="gfx950"),
            ),
            self.assertLogs(
                "sglang.srt.layers.attention.dsa_backend", "WARNING"
            ) as logs,
        ):
            policy = backend._init_rocm_prefill_policy()
        self.assertEqual(policy.policy_id, "glm53-mi350x-tp4-fp8-v1")
        self.assertIn("ROCm/aiter/pull/5535", "\n".join(logs.output))


class TestRocmDSAPrefillCLI(CustomTestCase):
    @staticmethod
    def parse(*args):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        namespace = parser.parse_args(["--model-path", "dummy", *args])
        return ServerArgs.from_cli_args(namespace)

    def test_cli_is_opt_in_and_accepts_only_versioned_profiles(self):
        self.assertIsNone(self.parse().dsa_prefill_policy)
        for policy_id in ROCM_DSA_PREFILL_POLICIES:
            with self.subTest(policy=policy_id):
                self.assertEqual(
                    self.parse("--dsa-prefill-policy", policy_id).dsa_prefill_policy,
                    policy_id,
                )
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                self.parse("--dsa-prefill-policy", "auto")
        self.assertIn("invalid choice", stderr.getvalue())

    def test_policy_is_projected_into_the_execution_namespace(self):
        for policy_id in (None, *ROCM_DSA_PREFILL_POLICIES):
            with (
                self.subTest(policy=policy_id),
                get_context().override_server_args(dsa_prefill_policy=policy_id),
            ):
                self.assertEqual(get_exec().kernel.dsa_prefill_policy, policy_id)


if __name__ == "__main__":
    unittest.main()
