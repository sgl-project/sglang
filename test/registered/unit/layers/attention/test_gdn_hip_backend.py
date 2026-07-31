import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear import utils as linear_attn_utils
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.attention.linear.kernels import gdn_hip
from sglang.srt.layers.attention.linear.kernels.gdn_hip import HipGDNKernel
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    initialize_linear_attn_config,
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
            patch.object(gdn_hip, "gdn_decode_packed_bf16_hip", hip_op),
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
    def setUp(self):
        self.decode_backend = linear_attn_utils.LINEAR_ATTN_DECODE_BACKEND
        self.prefill_backend = linear_attn_utils.LINEAR_ATTN_PREFILL_BACKEND

    def tearDown(self):
        linear_attn_utils.LINEAR_ATTN_DECODE_BACKEND = self.decode_backend
        linear_attn_utils.LINEAR_ATTN_PREFILL_BACKEND = self.prefill_backend

    @staticmethod
    def make_args(*, base="triton", decode=None, prefill=None):
        return SimpleNamespace(
            linear_attn_backend=base,
            linear_attn_decode_backend=decode,
            linear_attn_prefill_backend=prefill,
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
            initialize_linear_attn_config(self.make_args(base="hip"), is_gdn=True)

    def test_rejects_hip_as_prefill_backend(self):
        with self.assertRaisesRegex(ValueError, "GDN decode-only"):
            initialize_linear_attn_config(
                self.make_args(decode="hip", prefill="hip"), is_gdn=True
            )

    def test_rejects_hip_decode_for_non_gdn_model(self):
        with self.assertRaisesRegex(ValueError, "only for GDN models"):
            initialize_linear_attn_config(self.make_args(decode="hip"))

    def test_accepts_hip_decode_for_gdn_model(self):
        initialize_linear_attn_config(
            self.make_args(decode="hip", prefill="triton"), is_gdn=True
        )

        self.assertEqual(
            linear_attn_utils.LINEAR_ATTN_DECODE_BACKEND,
            LinearAttnKernelBackend.HIP,
        )
        self.assertEqual(
            linear_attn_utils.LINEAR_ATTN_PREFILL_BACKEND,
            LinearAttnKernelBackend.TRITON,
        )


if __name__ == "__main__":
    unittest.main()
