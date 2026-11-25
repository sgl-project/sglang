"""CP router-row alignment for the MoE token ids.

Regression test for https://github.com/sgl-project/sglang/issues/52215.

The a2a-none CP fast path expands hidden rows into the rank-major router view
(`attn_cp_interleave_gather`, rows *= attn_cp_size), and `moe_fused_gate`
indexes the per-token vision bias by those same rows. Two id geometries reach
the transport in practice, both observed on 8xH20 DeepSeek-V4.1-Flash CP8:

* already aligned (`global == local * cp_size`): `cp_shard_model_inputs`
  attached the full view before the model body ran - 18849 of 27074 sampled
  batches in the v19 trace;
* rank-local (`global == local`): DSpark prefill never enters that context
  manager, so the model-level `getattr` falls back to the local ids - the
  remaining 25944 batches, and the shape that crashed the fused gate.

`_align_cp_router_ids` must reuse the first and rebuild the second from the
local operand. Rebuilding both produces `local * cp_size * cp_size` router rows
(the 65536-row regression); rebuilding neither leaves the crash in place, which
is why the old gate fell back to `input_ids=None` and silently disabled the
per-token bias.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_common import moe_transport
from sglang.srt.models.deepseek_v2 import _check_router_row_ids
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CP = 8
# Distinct per-rank marker so a wrong rank order cannot pass silently.
RANK_STRIDE = 100_000


def _rank_major_view(x, cp=CP):
    """Stand-in for attn_cp_interleave_gather: rank-major, row preserving."""
    return torch.cat([x + rank * RANK_STRIDE for rank in range(cp)])


def _fill_rank_major(out, inp, cp=CP):
    out.copy_(_rank_major_view(inp, cp))


class TestAlignCpRouterIds(CustomTestCase):
    def _align(self, input_ids, input_ids_global, local_rows, router_rows, cp=CP):
        with patch.object(
            moe_transport,
            "attn_cp_all_gather_into_tensor",
            side_effect=_fill_rank_major,
        ):
            return moe_transport._align_cp_router_ids(
                input_ids, input_ids_global, local_rows, router_rows
            )

    def test_aligned_view_is_reused_without_a_collective(self):
        # The dominant v19 geometry: 8192 router rows, 1024 rank-local rows.
        local_ids = torch.arange(1024)
        attached = torch.arange(8192)
        calls = []
        with patch.object(
            moe_transport,
            "attn_cp_all_gather_into_tensor",
            side_effect=lambda out, inp: calls.append(inp.shape),
        ):
            aligned = moe_transport._align_cp_router_ids(
                local_ids, attached, 1024, 8192
            )
        self.assertIs(aligned, attached)
        self.assertEqual(calls, [])

    def test_rank_local_view_is_rebuilt_from_the_local_operand(self):
        # The crash geometry: 256 router rows against 32 rank-local rows.
        local_ids = torch.arange(32)
        collapsed = local_ids.clone()
        aligned = self._align(local_ids, collapsed, 32, 32 * CP)
        self.assertEqual(tuple(aligned.shape), (32 * CP,))
        torch.testing.assert_close(aligned, _rank_major_view(local_ids))

    def test_missing_global_rebuilds_from_local(self):
        local_ids = torch.arange(16)
        aligned = self._align(local_ids, None, 16, 16 * CP)
        torch.testing.assert_close(aligned, _rank_major_view(local_ids))

    def test_no_ids_path_stays_none(self):
        # DSpark next-token layers are not hash-routed.
        self.assertIsNone(self._align(None, None, 16, 16 * CP))

    def test_rows_already_matching_router_are_not_regathered(self):
        # Decode under prefill-only CP: no expansion, nothing to align.
        ids = torch.arange(1)
        self.assertIs(self._align(ids, ids, 1, 1), ids)

    def test_stale_global_view_is_replaced_not_trusted(self):
        # A view that matches neither row count belongs to some other geometry
        # (different padding or CP split): rebuild from the local operand
        # instead of using it. V4 CP rejects DP attention outright
        # (`attn_dp_size == 1`), so this is not a DP-crossover case.
        local_ids = torch.arange(16)
        aligned = self._align(local_ids, torch.arange(40), 16, 16 * CP)
        torch.testing.assert_close(aligned, _rank_major_view(local_ids))

    def test_non_local_source_fails_loudly(self):
        # YaoleWu's precondition: whatever we gather from must be rank-local.
        with self.assertRaisesRegex(AssertionError, r"neither the rank-local"):
            self._align(torch.arange(16 * CP), None, 16, 16 * CP)


class TestCheckRouterRowIds(CustomTestCase):
    def test_matching_rows_pass(self):
        _check_router_row_ids(torch.arange(128), 128)

    def test_missing_ids_pass(self):
        _check_router_row_ids(None, 128)

    def test_mismatch_raises(self):
        with self.assertRaisesRegex(AssertionError, r"do not describe the router rows"):
            _check_router_row_ids(torch.arange(16), 128)

    def test_mismatch_message_does_not_read_runtime_state(self):
        # The diagnostic must be buildable from its arguments alone. Reading a
        # config namespace here raises whenever that namespace is not published
        # (CPU runs, for one), which replaces the AssertionError with a
        # ValueError and hides the mismatch this helper exists to surface.
        with patch.object(
            deepseek_v2, "get_parallel", side_effect=RuntimeError("not published")
        ):
            with self.assertRaisesRegex(AssertionError, r"ids=\(16,\) rows=128"):
                _check_router_row_ids(torch.arange(16), 128)


class TestMoeRowTransportCpIds(CustomTestCase):
    """End-to-end through MoeRowTransport.__init__ with mocked collectives."""

    def _build(self, local_rows, ids, ids_global, cp=CP):
        hidden = torch.zeros(local_rows, 4)
        gather_calls = []

        def gather_hidden(x):
            gather_calls.append(("hidden", tuple(x.shape)))
            return _rank_major_view(x, cp)

        def gather_ids(out, inp):
            gather_calls.append(("ids", tuple(inp.shape)))
            _fill_rank_major(out, inp, cp)

        batch = SimpleNamespace(dp_padding_mode=None)
        parallel = SimpleNamespace(
            attn_cp_size=cp,
            attn_cp_rank=0,
            attn_tp_size=1,
            attn_dp_size=1,
            tp_size=cp,
        )
        with (
            patch.object(moe_transport, "is_cp_active", return_value=True),
            patch.object(
                moe_transport,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch.object(moe_transport, "get_parallel", return_value=parallel),
            patch.object(
                moe_transport, "attn_cp_interleave_gather", side_effect=gather_hidden
            ),
            patch.object(
                moe_transport, "attn_cp_all_gather_into_tensor", side_effect=gather_ids
            ),
        ):
            transport = moe_transport.MoeRowTransport(
                SimpleNamespace(), hidden, batch, ids, ids_global
            )
        return transport, gather_calls

    def test_aligned_batch_moves_ids_once_with_hidden(self):
        ids = torch.arange(1024)
        attached = torch.arange(8192)
        transport, calls = self._build(1024, ids, attached)
        self.assertIs(transport.input_ids_global, attached)
        self.assertEqual(calls, [("hidden", (1024, 4))])
        self.assertEqual(tuple(transport.input.shape), (8192, 4))

    def test_rank_local_batch_gets_the_router_row_view(self):
        ids = torch.arange(32)
        transport, calls = self._build(32, ids, ids.clone())
        self.assertEqual(tuple(transport.input.shape), (32 * CP, 4))
        self.assertEqual(tuple(transport.input_ids_global.shape), (32 * CP,))
        torch.testing.assert_close(transport.input_ids_global, _rank_major_view(ids))
        self.assertEqual([c[0] for c in calls], ["hidden", "ids"])
        # Engram reads the rank-local ids before the transport.
        self.assertIs(transport.input_ids, ids)

    def test_ids_none_keeps_the_plain_routing_path(self):
        transport, calls = self._build(16, None, None)
        self.assertIsNone(transport.input_ids_global)
        self.assertEqual([c[0] for c in calls], ["hidden"])


if __name__ == "__main__":
    unittest.main()
