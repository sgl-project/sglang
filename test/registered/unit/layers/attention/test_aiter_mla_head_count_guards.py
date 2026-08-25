"""CPU unit tests for the two aiter MLA head-count guards.

Both fire on local head counts aiter has no kernel instantiation for, and both
used to reject ``--prefill-attention-backend aiter --decode-attention-backend
triton`` on Kimi-K3, which has 96/8 = 12 heads per rank at tp8.

1. ``mla_decode_fwd`` takes 4, 8, or a multiple of 16 in [16, 128], and
   ``AiterAttnBackend.__init__`` asserts it up front. That is the *decode*
   kernel's limit: plain EXTEND runs ``flash_attn_varlen_func`` /
   ``mla_prefill_fwd``, which take any count, so a prefill-only aiter was being
   rejected for a kernel it never calls.
   ``_mla_decode_kernel_reachable`` decides that, and only holds while it agrees
   with ``HybridAttnBackend._select_backend`` in another file -- pinned below.

2. ``mla_fp8_prefill_attn`` (on by default on gfx95) reduces through
   ``mla_reduce_v1``, whose dispatch table admits a fixed set of
   (num_head, head_dim) pairs. An unlisted pair is a host-side abort --
   "kn_mla_reduce_v1 doesn't support the specified settings" -- with no Python
   frame, so the backend has to choose the fallback before calling it.

Usage:
    python -m pytest test_aiter_mla_head_count_guards.py -v
    python test_aiter_mla_head_count_guards.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.aiter_backend import (
    _MLA_REDUCE_V1_HEADS,
    _decode_head_pad_plan,
    _fp8_prefill_num_head,
    _mla_decode_kernel_reachable,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    _ROCM_FORWARD_METHODS,
    handle_attention_aiter,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMlaDecodeKernelReachable(CustomTestCase):
    def test_aiter_decode_backend_is_reachable(self):
        """decode=aiter keeps the guard, with or without spec decoding."""
        for algorithm in (None, "DSPARK"):
            for mode in ("prefill", "decode"):
                with self.subTest(algorithm=algorithm, mode=mode):
                    self.assertTrue(
                        _mla_decode_kernel_reachable(
                            decode_attention_backend="aiter",
                            speculative_algorithm=algorithm,
                            speculative_attention_mode=mode,
                        )
                    )

    def test_prefill_only_aiter_is_unreachable(self):
        """The config this guard used to reject: aiter prefill, triton decode."""
        self.assertFalse(
            _mla_decode_kernel_reachable(
                decode_attention_backend="triton",
                speculative_algorithm=None,
                speculative_attention_mode="prefill",
            )
        )

    def test_spec_verify_routed_to_prefill_is_reachable(self):
        """Verify lands on the prefill backend at mode=prefill, so aiter runs
        the decode kernel there even though it never serves DECODE."""
        self.assertTrue(
            _mla_decode_kernel_reachable(
                decode_attention_backend="triton",
                speculative_algorithm="DSPARK",
                speculative_attention_mode="prefill",
            )
        )

    def test_spec_verify_routed_to_decode_is_unreachable(self):
        self.assertFalse(
            _mla_decode_kernel_reachable(
                decode_attention_backend="triton",
                speculative_algorithm="DSPARK",
                speculative_attention_mode="decode",
            )
        )

    def test_matches_hybrid_backend_routing(self):
        """The predicate's premise: which modes actually reach the prefill
        backend. Pins it against _select_backend rather than restating it."""
        prefill, decode = object(), object()
        for spec_attn_is_decode in (False, True):
            backend = HybridAttnBackend.__new__(HybridAttnBackend)
            backend.prefill_backend = prefill
            backend.decode_backend = decode
            backend.spec_attn_is_decode = spec_attn_is_decode

            with self.subTest(spec_attn_is_decode=spec_attn_is_decode):
                # EXTEND is the prefill backend's alone, always.
                self.assertIs(backend._select_backend(ForwardMode.EXTEND), prefill)
                # DECODE never reaches it.
                self.assertIs(backend._select_backend(ForwardMode.DECODE), decode)
                # Verify follows speculative_attention_mode -- the only way a
                # decode-kernel mode lands on the prefill backend.
                self.assertIs(
                    backend._select_backend(ForwardMode.TARGET_VERIFY),
                    decode if spec_attn_is_decode else prefill,
                )


class TestMlaReduceV1Heads(CustomTestCase):
    def test_kimi_k3_tp8_shape_is_excluded(self):
        """K3's (12 heads, 128 v_head_dim) has no mla_reduce_v1 instantiation.

        The table drives the padding, so listing 12 would send the fp8 asm
        prefill straight at a kernel that aborts the process rather than
        raising -- no Python frame, nothing to catch.
        """
        self.assertNotIn(12, _MLA_REDUCE_V1_HEADS[128])
        # The neighbours the table does carry, so a wholesale deletion is red too.
        self.assertIn(16, _MLA_REDUCE_V1_HEADS[128])
        self.assertIn(128, _MLA_REDUCE_V1_HEADS[128])

    def test_kimi_k3_pads_to_16(self):
        """12 has no instantiation, so it runs at the next count that does.
        Returning 12 unchanged would abort the process inside the kernel."""
        self.assertEqual(
            _fp8_prefill_num_head(num_head=12, num_kv_head=12, v_head_dim=128), 16
        )

    def test_supported_count_is_never_padded(self):
        """A head count the table already carries runs as-is -- padding it would
        only add heads whose output is discarded."""
        self.assertEqual(
            _fp8_prefill_num_head(num_head=16, num_kv_head=16, v_head_dim=128), 16
        )

    def test_pad_declines_outside_its_validated_shape(self):
        """Padding holds q and k/v at the same count, so it is only offered at
        GQA ratio 1, and only when the table has something larger to reach.
        None sends the caller to flash_attn_varlen_func instead."""
        # ratio 4: padding q to 16 would move the ratio the metadata is built for.
        self.assertIsNone(
            _fp8_prefill_num_head(num_head=12, num_kv_head=3, v_head_dim=128)
        )
        # nothing above 128 in the table.
        self.assertIsNone(
            _fp8_prefill_num_head(num_head=130, num_kv_head=130, v_head_dim=128)
        )
        # head_dim with no table at all.
        self.assertIsNone(
            _fp8_prefill_num_head(num_head=12, num_kv_head=12, v_head_dim=192)
        )


class TestDecodeHeadPadPlan(CustomTestCase):
    def test_kimi_k3_tp8_needs_zero_padding(self):
        """12 does not divide 16, so repetition cannot reach the kernel's shape.

        Measured directly: 12 heads reaches aiter as
        "get_heuristic_kernel_mla: cannot get heuristic kernel ... gqa:12" and
        aborts the process. Collapsing this back to the old
        ``16 // n if n < 16 else 1`` gives repeat_factor 1 and no pad, which is
        exactly that abort.
        """
        self.assertEqual(_decode_head_pad_plan(12), (1, 4))

    def test_divisors_of_16_still_repeat(self):
        """4 and 8 keep the established repeat_interleave path -- unlike zero
        heads, its extra columns carry real values."""
        self.assertEqual(_decode_head_pad_plan(4), (4, 0))
        self.assertEqual(_decode_head_pad_plan(8), (2, 0))

    def test_at_or_above_16_is_untouched(self):
        """Includes the DCP case: K3 gathers to 96, which the kernel serves."""
        for n in (16, 32, 96, 128):
            with self.subTest(num_head=n):
                self.assertEqual(_decode_head_pad_plan(n), (1, 0))


class TestAiterDcpPrefillRouting(CustomTestCase):
    """What handle_attention_aiter picks for a DCP extend.

    The DCP prefill runs on flash_attn, which needs per-head k/v. Absorb cannot
    use flash_attn at all -- aiter's FA caps head_dim at 256 and absorb is 576 --
    so MHA is the only route to it.
    """

    HANDLER = "sglang.srt.models.deepseek_common.attention_backend_handler"

    def _dispatch(self, *, dcp_enabled):
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
        )
        with patch(
            f"{self.HANDLER}.is_in_tc_piecewise_cuda_graph", return_value=False
        ), patch(
            f"{self.HANDLER}.is_in_breakable_cuda_graph", return_value=False
        ), patch(
            f"{self.HANDLER}.get_parallel",
            return_value=SimpleNamespace(dcp_enabled=dcp_enabled),
        ):
            return handle_attention_aiter(object(), forward_batch)

    def test_dcp_picks_one_shot_not_plain_mha(self):
        """MHA_ONE_SHOT, never plain MHA.

        forward_normal_rocm_prepare assembles the sharded prefix
        (all_gather_kv_cache_for_mha_extend) only when forward_batch.mha_one_shot
        is set, and only the one-shot prepare sets it. Returning plain MHA skips
        the assembly silently: k/v then hold the extend tokens alone while
        cu_seqlens_k still spans prefix + extend, so flash_attn reads past the
        end of k and never attends the prefix. That scored gsm8k 0.895 at
        parallel 128 while looking clean at 32, because low-concurrency batches
        are mostly zero-prefix, where extend-only k/v happens to be correct.
        """
        self.assertIs(self._dispatch(dcp_enabled=True), AttnForwardMethod.MHA_ONE_SHOT)

    def test_non_dcp_stays_on_plain_mha(self):
        """Without DCP there is no assembled buffer to hand the backend, so the
        one-shot prepare would only add work the plain path already does."""
        self.assertIs(self._dispatch(dcp_enabled=False), AttnForwardMethod.MHA)

    def test_one_shot_resolves_to_its_rocm_variant(self):
        """The fix only reaches the assembly through this mapping; without it
        MHA_ONE_SHOT would fall through to the non-ROCm prepare."""
        self.assertIs(
            _ROCM_FORWARD_METHODS[AttnForwardMethod.MHA_ONE_SHOT],
            AttnForwardMethod.MHA_ONE_SHOT_ROCM,
        )


if __name__ == "__main__":
    unittest.main()
