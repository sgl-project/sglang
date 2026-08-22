import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear import utils as linear_attn_utils
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNKernelDispatcher,
    is_hip_gdn_decode_supported,
)
from sglang.srt.layers.attention.linear.kernels import gdn_hip
from sglang.srt.layers.attention.linear.kernels.gdn_hip import HipGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    resolve_linear_attn_backends,
)
from sglang.srt.server_args import (
    LINEAR_ATTN_DECODE_BACKEND_CHOICES,
    LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
    ServerArgs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_exact_inputs(batch: int = 2):
    return {
        "mixed_qkv": torch.empty(batch, 6144, dtype=torch.bfloat16),
        "a": torch.empty(batch, 32, dtype=torch.bfloat16),
        "b": torch.empty(batch, 32, dtype=torch.bfloat16),
        "A_log": torch.empty(32, dtype=torch.float32),
        "dt_bias": torch.empty(32, dtype=torch.bfloat16),
        "scale": 128**-0.5,
        "ssm_states": torch.empty(3, 32, 128, 128, dtype=torch.bfloat16),
        "cache_indices": torch.tensor([1, -1], dtype=torch.int32),
        "num_v_heads": 32,
        "head_v_dim": 128,
    }


class TestHipGDNKernel(unittest.TestCase):
    def test_exact_contract_uses_hip_without_parent_fallback(self):
        inputs = make_exact_inputs()

        def run_hip(**kwargs):
            kwargs["out"].zero_()

        hip_op = MagicMock(side_effect=run_hip)
        with (
            patch.object(gdn_hip, "is_hip", return_value=True),
            patch.object(gdn_hip, "is_gfx95_supported", return_value=True),
            patch.object(gdn_hip, "gdr_decode_packed_bf16", hip_op),
            patch.object(TritonGDNKernel, "packed_decode") as parent,
        ):
            out = HipGDNKernel().packed_decode(**inputs)

        self.assertEqual(out.shape, (1, 2, 32, 128))
        hip_op.assert_called_once()
        parent.assert_not_called()
        call = hip_op.call_args.kwargs
        self.assertIs(call["state"], inputs["ssm_states"])
        self.assertIs(call["indices"], inputs["cache_indices"])
        self.assertEqual(call["indices"].tolist(), [1, -1])

    def test_replayssm_uses_parent_triton_path(self):
        inputs = make_exact_inputs()
        inputs.update(
            replayssm_d=sentinel.d,
            replayssm_k=sentinel.k,
            replayssm_g=sentinel.g,
            replayssm_write_pos=sentinel.write_pos,
        )
        with patch.object(
            TritonGDNKernel, "packed_decode", return_value=sentinel.output
        ) as parent:
            out = HipGDNKernel().packed_decode(**inputs)

        self.assertIs(out, sentinel.output)
        parent.assert_called_once()

    def test_non_exact_contract_uses_parent_triton_path(self):
        inputs = make_exact_inputs()
        inputs["cache_indices"] = inputs["cache_indices"].to(torch.int64)
        with (
            patch.object(gdn_hip, "is_hip", return_value=True),
            patch.object(gdn_hip, "is_gfx95_supported", return_value=True),
            patch.object(
                TritonGDNKernel, "packed_decode", return_value=sentinel.output
            ) as parent,
        ):
            out = HipGDNKernel().packed_decode(**inputs)

        self.assertIs(out, sentinel.output)
        parent.assert_called_once()

    def test_non_gfx95_uses_parent_triton_path(self):
        inputs = make_exact_inputs()
        with (
            patch.object(gdn_hip, "is_hip", return_value=True),
            patch.object(gdn_hip, "is_gfx95_supported", return_value=False),
            patch.object(
                TritonGDNKernel, "packed_decode", return_value=sentinel.output
            ) as parent,
        ):
            out = HipGDNKernel().packed_decode(**inputs)

        self.assertIs(out, sentinel.output)
        parent.assert_called_once()

    def test_dispatcher_keeps_prefill_and_verify_on_triton(self):
        with (
            patch.object(gdn_backend, "is_hip", return_value=True),
            patch.object(gdn_backend, "is_gfx95_supported", return_value=True),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.HIP,
                LinearAttnKernelBackend.TRITON,
            )

        self.assertIsInstance(dispatcher.decode_kernel, HipGDNKernel)
        self.assertIsInstance(dispatcher.extend_kernel, TritonGDNKernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)
        self.assertIsInstance(dispatcher.tree_verify_kernel, TritonGDNKernel)

    def test_dispatcher_falls_back_to_triton_on_non_gfx95_rocm(self):
        with (
            patch.object(gdn_backend, "is_hip", return_value=True),
            patch.object(gdn_backend, "is_gfx95_supported", return_value=False),
            patch.object(gdn_backend, "rank0_log") as log,
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.HIP,
                LinearAttnKernelBackend.TRITON,
            )

        self.assertIsInstance(dispatcher.decode_kernel, TritonGDNKernel)
        log.assert_any_call(
            "GDN HIP decode backend requires ROCm gfx95; "
            "falling back to Triton decode."
        )


class TestHipBackendScope(unittest.TestCase):
    @staticmethod
    def make_mamba(*, base="triton", decode=None, prefill=None, verify=None):
        return SimpleNamespace(
            linear_attn_backend=base,
            linear_attn_decode_backend=decode,
            linear_attn_prefill_backend=prefill,
            linear_attn_verify_backend=verify,
        )

    def resolve(
        self,
        *,
        base="triton",
        decode=None,
        prefill=None,
        verify=None,
        is_gdn=False,
        hip_decode_supported=False,
    ):
        mamba = self.make_mamba(
            base=base, decode=decode, prefill=prefill, verify=verify
        )
        with patch.object(
            linear_attn_utils,
            "get_exec",
            return_value=SimpleNamespace(mamba=mamba),
        ):
            return resolve_linear_attn_backends(
                is_gdn=is_gdn,
                hip_decode_supported=hip_decode_supported,
            )

    def test_hip_is_exposed_only_as_decode_override(self):
        self.assertNotIn("hip", LINEAR_ATTN_KERNEL_BACKEND_CHOICES)
        self.assertIn("hip", LINEAR_ATTN_DECODE_BACKEND_CHOICES)

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        base_args = ["--model-path", "dummy"]
        parsed = parser.parse_args([*base_args, "--linear-attn-decode-backend", "hip"])
        self.assertEqual(parsed.linear_attn_decode_backend, "hip")

        for flag in ("--linear-attn-backend", "--linear-attn-prefill-backend"):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([*base_args, flag, "hip"])

    def test_rejects_hip_as_shared_backend(self):
        with self.assertRaisesRegex(ValueError, "GDN decode-only"):
            self.resolve(base="hip", is_gdn=True, hip_decode_supported=True)

    def test_rejects_hip_as_prefill_backend(self):
        with self.assertRaisesRegex(ValueError, "GDN decode-only"):
            self.resolve(
                decode="hip",
                prefill="hip",
                is_gdn=True,
                hip_decode_supported=True,
            )

    def test_rejects_hip_decode_for_non_gdn_model(self):
        with self.assertRaisesRegex(ValueError, "only for GDN models"):
            self.resolve(decode="hip")

    def test_accepts_hip_decode_only_for_supported_scope(self):
        backends = self.resolve(
            decode="hip",
            prefill="triton",
            is_gdn=True,
            hip_decode_supported=True,
        )

        self.assertEqual(backends.decode, LinearAttnKernelBackend.HIP)
        self.assertEqual(backends.prefill, LinearAttnKernelBackend.TRITON)
        self.assertEqual(backends.verify, LinearAttnKernelBackend.TRITON)

    def test_unsupported_gdn_scope_falls_back_to_triton(self):
        with patch.object(linear_attn_utils, "rank0_log") as log:
            backends = self.resolve(
                decode="hip",
                is_gdn=True,
                hip_decode_supported=False,
            )

        self.assertEqual(backends.decode, LinearAttnKernelBackend.TRITON)
        self.assertEqual(backends.prefill, LinearAttnKernelBackend.TRITON)
        self.assertEqual(backends.verify, LinearAttnKernelBackend.TRITON)
        log.assert_any_call(
            "GDN HIP decode currently supports only non-speculative Qwen3.5 "
            "on ROCm gfx950; falling back to Triton decode."
        )

    @staticmethod
    def make_runner(*, speculative_algorithm=None, is_draft_worker=False):
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_config=sentinel.hf_config),
            server_args=SimpleNamespace(
                speculative_algorithm=speculative_algorithm,
            ),
            is_draft_worker=is_draft_worker,
        )

    def test_capability_accepts_qwen35_non_speculative_gfx950(self):
        runner = self.make_runner()
        with (
            patch.object(gdn_backend, "is_hip", return_value=True),
            patch.object(gdn_backend, "is_gfx95_supported", return_value=True),
            patch.object(gdn_backend, "is_qwen3_5", return_value=True),
        ):
            self.assertTrue(is_hip_gdn_decode_supported(runner))

    def test_capability_rejects_mtp_draft_and_other_models(self):
        with (
            patch.object(gdn_backend, "is_hip", return_value=True),
            patch.object(gdn_backend, "is_gfx95_supported", return_value=True),
            patch.object(gdn_backend, "is_qwen3_5", return_value=True),
        ):
            self.assertFalse(
                is_hip_gdn_decode_supported(
                    self.make_runner(speculative_algorithm="EAGLE")
                )
            )
            self.assertFalse(
                is_hip_gdn_decode_supported(self.make_runner(is_draft_worker=True))
            )

        with (
            patch.object(gdn_backend, "is_hip", return_value=True),
            patch.object(gdn_backend, "is_gfx95_supported", return_value=True),
            patch.object(gdn_backend, "is_qwen3_5", return_value=False),
        ):
            self.assertFalse(is_hip_gdn_decode_supported(self.make_runner()))

    def test_capability_rejects_non_gfx950(self):
        with (
            patch.object(gdn_backend, "is_hip", return_value=True),
            patch.object(gdn_backend, "is_gfx95_supported", return_value=False),
            patch.object(gdn_backend, "is_qwen3_5", return_value=True),
        ):
            self.assertFalse(is_hip_gdn_decode_supported(self.make_runner()))


if __name__ == "__main__":
    unittest.main()
