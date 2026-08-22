"""CPU checks for the trtllm_mla-family DCP decode hooks.

Covers two traps that only show up at the family boundary:

1. CuteDslMLABackend._run_decode_kernel used to drop ``return_lse`` when
   deferring to the base at ``cp_world <= 1``. Inherited DCP decode calls
   that hook with ``return_lse=True`` and ``cp_world`` defaulting to 1.
2. TRTLLMMLABackend.forward_decode used to miss the autotune dummy-run
   skip that TokenspeedMLABackend already had, so a cold FlashInfer MoE
   autotune pass under DCP overflowed the trtllm-gen workspace.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCuteDslDecodeKernelForwardsReturnLse(CustomTestCase):
    def test_cp_world_one_forwards_return_lse_to_base(self):
        backend = object.__new__(CuteDslMLABackend)
        captured = {}

        def _fake_base_hook(self, *args, **kwargs):
            captured.update(kwargs)
            return torch.empty(0)

        with patch.object(TRTLLMMLABackend, "_run_decode_kernel", _fake_base_hook):
            backend._run_decode_kernel(
                query=None,
                kv_cache=None,
                block_tables=None,
                seq_lens=None,
                max_seq_len=0,
                layer=None,
                return_lse=True,
            )

        self.assertTrue(captured.get("return_lse"))
        self.assertEqual(captured.get("cp_world"), 1)


class TestDcpDecodeAutotuneDummySkip(CustomTestCase):
    def _assert_dummy_skip(self, backend_cls):
        backend = object.__new__(backend_cls)
        backend.q_data_type = torch.bfloat16
        q = torch.zeros(3, 4, 128)
        layer = SimpleNamespace(tp_q_head_num=4, v_head_dim=128)
        parallel = SimpleNamespace(dcp_enabled=True)
        sentinel = object()

        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mla_backend.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.attention.tokenspeed_mla_backend.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.attention.cutedsl_mla_backend.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mla_backend.get_in_autotune_dummy_run",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.tokenspeed_mla_backend.get_in_autotune_dummy_run",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.cutedsl_mla_backend.get_in_autotune_dummy_run",
                return_value=True,
            ),
            patch.object(
                TRTLLMMLABackend,
                "_forward_decode_dcp",
                side_effect=AssertionError("dummy run must not reach the kernel"),
            ),
            patch.object(
                TRTLLMMLABackend,
                "_run_decode_kernel",
                side_effect=AssertionError("dummy run must not reach the kernel"),
            ),
        ):
            out, lse = backend.forward_decode(
                q, sentinel, sentinel, layer, sentinel
            )

        self.assertEqual(tuple(out.shape), (3, 4 * 128))
        self.assertEqual(tuple(lse.shape), (3, 4))
        self.assertTrue(torch.all(out == 0))
        self.assertTrue(torch.all(lse == 0))

    def test_trtllm_mla_skips_kernel_in_autotune_dummy_run(self):
        self._assert_dummy_skip(TRTLLMMLABackend)

    def test_cutedsl_mla_skips_kernel_in_autotune_dummy_run(self):
        self._assert_dummy_skip(CuteDslMLABackend)

    def test_tokenspeed_mla_skips_kernel_in_autotune_dummy_run(self):
        self._assert_dummy_skip(TokenspeedMLABackend)
