"""Regression test for the R-Fork weight-transfer DP precondition.

Two NCCL-only HTTP endpoints used by R-Fork weight transfer
(`init_weights_send_group_for_remote_instance` and
`send_weights_to_remote_instance` in `tokenizer_control_mixin.py`)
previously hard-asserted `dp_size == 1`, blocking the working
DP-attention path. This test guards the relaxed precondition:

    assert dp_size == 1 or enable_dp_attention

which matches the pattern already used by sibling weight-update
methods in the same file. Plain DP (no dp_attention) is intentionally
still rejected because the seed-side per-tp_rank port allocation is
not safe for it.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_obj(dp_size: int, enable_dp_attention: bool):
    """Minimal stand-in for TokenizerManager — only the attrs the methods read."""
    obj = MagicMock()
    obj.server_args = MagicMock()
    obj.server_args.dp_size = dp_size
    obj.server_args.enable_dp_attention = enable_dp_attention
    obj.auto_create_handle_loop = MagicMock()

    fake_result = MagicMock()
    fake_result.success = True
    fake_result.message = "ok"
    comm = AsyncMock(return_value=[fake_result])
    obj.init_weights_send_group_for_remote_instance_communicator = comm
    obj.send_weights_to_remote_instance_communicator = comm
    return obj


class TestRForkDPAttentionAssertions(unittest.IsolatedAsyncioTestCase):
    """Verify the relaxed `dp_size == 1 or enable_dp_attention` precondition."""

    # ---- init_weights_send_group_for_remote_instance ----

    async def test_init_dp1_accepts(self):
        """dp_size=1 keeps working — no regression for the common case."""
        obj = _make_obj(dp_size=1, enable_dp_attention=False)
        ok, _ = await TokenizerControlMixin.init_weights_send_group_for_remote_instance(
            obj, MagicMock()
        )
        self.assertTrue(ok)
        obj.auto_create_handle_loop.assert_called_once()

    async def test_init_dp_attention_accepts(self):
        """dp_size>1 with enable_dp_attention=True is now accepted."""
        obj = _make_obj(dp_size=8, enable_dp_attention=True)
        ok, _ = await TokenizerControlMixin.init_weights_send_group_for_remote_instance(
            obj, MagicMock()
        )
        self.assertTrue(ok)

    async def test_init_plain_dp_still_rejected(self):
        """Plain DP (no dp_attention) is intentionally still rejected."""
        obj = _make_obj(dp_size=8, enable_dp_attention=False)
        with self.assertRaises(AssertionError):
            await TokenizerControlMixin.init_weights_send_group_for_remote_instance(
                obj, MagicMock()
            )

    # ---- send_weights_to_remote_instance ----

    async def test_send_dp1_accepts(self):
        obj = _make_obj(dp_size=1, enable_dp_attention=False)
        ok, _ = await TokenizerControlMixin.send_weights_to_remote_instance(
            obj, MagicMock()
        )
        self.assertTrue(ok)

    async def test_send_dp_attention_accepts(self):
        obj = _make_obj(dp_size=8, enable_dp_attention=True)
        ok, _ = await TokenizerControlMixin.send_weights_to_remote_instance(
            obj, MagicMock()
        )
        self.assertTrue(ok)

    async def test_send_plain_dp_still_rejected(self):
        obj = _make_obj(dp_size=8, enable_dp_attention=False)
        with self.assertRaises(AssertionError):
            await TokenizerControlMixin.send_weights_to_remote_instance(
                obj, MagicMock()
            )


if __name__ == "__main__":
    unittest.main()
