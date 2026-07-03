"""CPU unit test for the decode-context-parallel (DCP) strategy.

Pins the standalone ``DecodeContextParallelStrategy`` (``layers/dcp/strategy.py``):

* it reports the DCP-specific kind ``DecodeContextParallelStrategyKind.DECODE`` (NOT
  the prefill ``INTERLEAVE``) and is configured by ``dcp_size``;
* its delegating methods are behavior-preserving wrappers over the ``layers/dcp``
  primitives (``local_decode_kv_lens`` == ``get_dcp_lens``; ``merge_decode_attention``
  dispatches mha/mla and short-circuits on a world_size==1 group);
* ``can_apply`` fires on decode only.

Usage:
    python -m pytest test_dcp_strategy_unit.py -v
    python test_dcp_strategy_unit.py
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dcp.base import DecodeContextParallelStrategyKind
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.layers.dcp.strategy import DecodeContextParallelStrategy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 8]
LENS = list(range(0, 41))


def _fake_batch(is_decode: bool):
    mode = SimpleNamespace(
        is_decode=lambda: is_decode,
        is_extend=lambda: not is_decode,
    )
    return SimpleNamespace(forward_mode=mode)


class _FakeGroup:
    """A single-rank group: the merge/gather ops short-circuit on world_size==1."""

    world_size = 1
    rank_in_group = 0


class TestDecodeStrategy(unittest.TestCase):
    def test_identity(self):
        strategy = DecodeContextParallelStrategy(dcp_size=4)
        self.assertEqual(strategy.kind, DecodeContextParallelStrategyKind.DECODE)
        self.assertEqual(strategy.name, "decode_context_parallel")
        self.assertEqual(strategy.dcp_size, 4)

    def test_local_decode_kv_lens_matches_free_function(self):
        strategy = DecodeContextParallelStrategy(dcp_size=2)
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int32)
                got = strategy.local_decode_kv_lens(lens, n, rank)
                ref = get_dcp_lens(lens, n, rank)
                self.assertTrue(
                    torch.equal(got, ref),
                    f"delegation mismatch at n={n}, rank={rank}",
                )

    def test_can_apply_fires_on_decode_only(self):
        strategy = DecodeContextParallelStrategy(dcp_size=2)
        self.assertTrue(strategy.can_apply(1, _fake_batch(is_decode=True)))
        self.assertFalse(strategy.can_apply(1, _fake_batch(is_decode=False)))
        # dcp_size <= 1 disables DCP regardless of mode.
        self.assertFalse(
            DecodeContextParallelStrategy(dcp_size=1).can_apply(
                1, _fake_batch(is_decode=True)
            )
        )

    def test_merge_decode_attention_dispatch(self):
        strategy = DecodeContextParallelStrategy(dcp_size=2)
        out = torch.randn(3, 2, 4)
        lse = torch.randn(3, 2)
        group = _FakeGroup()
        # world_size==1 short-circuit: both backends return the input unchanged.
        mha = strategy.merge_decode_attention(out, lse, group, backend="mha")
        self.assertTrue(torch.equal(mha, out))
        mla = strategy.merge_decode_attention(out, lse, group, backend="mla")
        self.assertTrue(torch.equal(mla, out))
        # return_lse path (mha) yields the (out, lse) tuple on the single-rank group.
        mha_o, mha_lse = strategy.merge_decode_attention(
            out, lse, group, backend="mha", return_lse=True
        )
        self.assertTrue(torch.equal(mha_o, out) and torch.equal(mha_lse, lse))
        with self.assertRaises(ValueError):
            strategy.merge_decode_attention(out, lse, group, backend="bogus")


if __name__ == "__main__":
    unittest.main()
