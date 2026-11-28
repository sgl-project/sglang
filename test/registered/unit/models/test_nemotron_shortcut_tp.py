"""Nemotron Shortcut's tensor-parallel GDP sharding, decay clamp and MXFP8
linear scale names."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.nemotron_shortcut import (  # noqa: E402
    NemotronShortcutGDPMixer,
    _fp8_linear_scale_name,
    gdp_in_proj_blocks,
    shard_gdp_rows,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HEADS, HOUSEHOLDER = 4, 3
CONFIG = SimpleNamespace(
    gdp_num_heads=HEADS,
    gdp_num_groups=HEADS,
    gdp_head_dim=2,
    gdp_key_head_dim=3,
    gdp_num_householder=HOUSEHOLDER,
)


def _row_heads(blocks):
    """The GDP head each in_proj row belongs to: every householder slot of a
    block is head-major."""
    heads = []
    for rows, householder in blocks:
        per_slot = rows // householder
        heads += [
            i // (per_slot // HEADS)
            for _ in range(householder)
            for i in range(per_slot)
        ]
    return torch.tensor(heads)


class TestGDPHeadSharding(CustomTestCase):
    def test_each_rank_gets_its_heads_from_every_slot(self):
        blocks = gdp_in_proj_blocks(CONFIG)
        # [z, V0..V2, K0..K2, Q, b0..b2, a]
        self.assertEqual(
            [rows for rows, _ in blocks], [8, 3 * 8, 3 * 12, 12, 3 * HEADS, HEADS]
        )
        total = sum(rows for rows, _ in blocks)
        row_ids = torch.arange(total)
        heads = _row_heads(blocks)
        for tp in (1, 2, 4):
            seen = []
            for rank in range(tp):
                shard = shard_gdp_rows(row_ids, blocks, rank=rank, size=tp)
                self.assertEqual(shard.numel(), total // tp)
                owned = set(range(rank * HEADS // tp, (rank + 1) * HEADS // tp))
                self.assertTrue(set(heads[shard].tolist()) <= owned)
                seen += shard.tolist()
            self.assertEqual(sorted(seen), row_ids.tolist())

    def test_conv_rows_shard_like_their_in_proj_rows(self):
        # conv1d holds in_proj's [V0..V2, K0..K2, Q] rows in the same order.
        blocks = gdp_in_proj_blocks(CONFIG)
        conv_blocks = blocks[1:4]
        offset = blocks[0][0]
        conv_rows = sum(rows for rows, _ in conv_blocks)
        in_proj_rows = torch.arange(sum(rows for rows, _ in blocks))
        for rank in range(2):
            from_in_proj = shard_gdp_rows(in_proj_rows, blocks, rank=rank, size=2)
            in_conv = from_in_proj[
                (from_in_proj >= offset) & (from_in_proj < offset + conv_rows)
            ]
            from_conv = shard_gdp_rows(
                torch.arange(conv_rows) + offset, conv_blocks, rank=rank, size=2
            )
            self.assertEqual(in_conv.tolist(), from_conv.tolist())


class TestDecayClamp(CustomTestCase):
    def test_log_decay_is_bounded_and_otherwise_unchanged(self):
        clamp = 20.0
        mixer = SimpleNamespace(
            g_clamp_value=clamp,
            A_log=torch.tensor([-2.0, 0.0, 1.0, 2.7]),
            dt_bias=torch.tensor([-3.0, 0.5, 0.0, 1.0]),
        )
        a = torch.linspace(-30, 30, 241).repeat(4, 1).T.contiguous()

        def log_decay(x):
            # The GDP kernels' g = -exp(A_log) * softplus(a + dt_bias).
            return -mixer.A_log.exp() * F.softplus(x.float() + mixer.dt_bias)

        clamped = NemotronShortcutGDPMixer._clamp_decay_input(mixer, a)
        g_before, g_after = log_decay(a), log_decay(clamped)
        self.assertTrue(bool((g_after >= -clamp - 1e-3).all()))
        inside = g_before >= -clamp
        torch.testing.assert_close(clamped[inside], a[inside])
        self.assertTrue(bool((~inside).any()))
        torch.testing.assert_close(
            g_after[~inside],
            torch.full_like(g_after[~inside], -clamp),
            atol=1e-3,
            rtol=0,
        )


class TestFp8LinearScaleName(CustomTestCase):
    def test_mxfp8_scale_loads_into_weight_scale_inv(self):
        params = {"qkv_proj.weight_scale_inv": 0, "o_proj.weight_scale": 0}
        self.assertEqual(
            _fp8_linear_scale_name(params, "qkv_proj.weight_scale"),
            "qkv_proj.weight_scale_inv",
        )
        # A parameter that really is called weight_scale (NVFP4) keeps its name.
        self.assertEqual(
            _fp8_linear_scale_name(params, "o_proj.weight_scale"),
            "o_proj.weight_scale",
        )
        self.assertEqual(
            _fp8_linear_scale_name(params, "o_proj.input_scale"), "o_proj.input_scale"
        )


if __name__ == "__main__":
    unittest.main()
