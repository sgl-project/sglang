"""The gfx950 fused DSA prologue must leave the stock path's inputs unchanged: the merged
front GEMM re-points the four projections at row views of one buffer (same values), its
split hands each consumer the columns its own GEMM would have produced, every hand-over
is consumed or cleared within one layer forward, and the switch is off by default."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_prologue_hip as prologue
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 64
# Kimi-K3.1 at TP8 is (1536 + 576, 128, 32, 12 * 128) over K = 7168
Q_LORA, KV_A, KEY, HEADS, GATE = 24, 9, 8, 4, 12


def _linear(rows, gen, dtype=torch.bfloat16):
    weight = torch.nn.Parameter(
        torch.randn(rows, HIDDEN, generator=gen).to(dtype), requires_grad=False
    )
    return SimpleNamespace(weight=weight, quant_method=UnquantizedLinearMethod())


def _attn(seed=0, gate_dtype=torch.bfloat16, merged_k_gate=True):
    g = torch.Generator().manual_seed(seed)
    wk, wp = _linear(KEY, g), _linear(HEADS, g)
    nope = SimpleNamespace(k_gate_weight=None, front_key_gate=None)
    if merged_k_gate:
        merged = torch.cat([wk.weight.data, wp.weight.data])
        wk.weight.data, wp.weight.data = merged[:KEY], merged[KEY:]
        nope.k_gate_weight = merged
    indexer = SimpleNamespace(
        wk=wk, weights_proj=wp, head_dim=KEY, n_heads=HEADS, rocm_nope_prologue=nope
    )
    return SimpleNamespace(
        fused_qkv_a_proj_with_mqa=_linear(Q_LORA + KV_A, g),
        g_proj=_linear(GATE, g, gate_dtype),
        indexer=indexer,
        use_output_gate=True,
        dsa_front=None,
        front_output_gate=None,
    )


def _parts(value):
    prologue.enabled_parts.cache_clear()
    env = {"SGLANG_ENABLE_ROCM_K3_DSA_FUSED_PROLOGUE": "1"}
    if value is not None:
        env["SGLANG_ROCM_K3_DSA_FUSED_PROLOGUE_PARTS"] = value
    with (
        mock.patch.object(prologue, "_is_gfx95", True),
        mock.patch.dict("os.environ", env),
    ):
        try:
            return prologue.enabled_parts()
        finally:
            prologue.enabled_parts.cache_clear()


class TestSwitch(CustomTestCase):
    def test_off_by_default(self):
        prologue.enabled_parts.cache_clear()
        with (
            mock.patch.object(prologue, "_is_gfx95", True),
            mock.patch.dict("os.environ", {}, clear=False) as env,
        ):
            env.pop("SGLANG_ENABLE_ROCM_K3_DSA_FUSED_PROLOGUE", None)
            self.assertEqual(prologue.enabled_parts(), frozenset())
        prologue.enabled_parts.cache_clear()

    def test_parts(self):
        self.assertEqual(_parts(None), frozenset(prologue.PARTS))
        self.assertEqual(_parts("index, kv"), frozenset({"index", "kv"}))
        self.assertEqual(_parts(""), frozenset())
        with self.assertRaises(ValueError):
            _parts("gemm,rope")

    def test_needs_gfx950(self):
        prologue.enabled_parts.cache_clear()
        with (
            mock.patch.object(prologue, "_is_gfx95", False),
            mock.patch.dict(
                "os.environ", {"SGLANG_ENABLE_ROCM_K3_DSA_FUSED_PROLOGUE": "1"}
            ),
        ):
            self.assertEqual(prologue.enabled_parts(), frozenset())
        prologue.enabled_parts.cache_clear()


class TestMergedFront(CustomTestCase):
    def setUp(self):
        patch = mock.patch.object(
            prologue, "part_enabled", side_effect=lambda part: part == "gemm"
        )
        patch.start()
        self.addCleanup(patch.stop)

    def _merge(self, attn):
        front = prologue.merge_front_weights(attn)
        attn.dsa_front = front
        return front

    def test_weights_become_views_with_same_values(self):
        attn = _attn()
        mods = [
            attn.fused_qkv_a_proj_with_mqa,
            attn.indexer.wk,
            attn.indexer.weights_proj,
            attn.g_proj,
        ]
        before = [m.weight.data.clone() for m in mods]
        front = self._merge(attn)
        self.assertIsNotNone(front)
        self.assertEqual(front.split, (Q_LORA + KV_A, KEY, HEADS, GATE))
        self.assertEqual(
            front.weight.shape, (Q_LORA + KV_A + KEY + HEADS + GATE, HIDDEN)
        )
        for m, w in zip(mods, before):
            self.assertTrue(torch.equal(m.weight.data, w))
            self.assertEqual(
                m.weight.data.untyped_storage().data_ptr(),
                front.weight.untyped_storage().data_ptr(),
            )
        # the stock [wk | weights_proj] GEMM keeps reading the same rows
        k_gate = attn.indexer.rocm_nope_prologue.k_gate_weight
        self.assertTrue(torch.equal(k_gate, torch.cat(before[1:3])))
        self.assertTrue(k_gate.is_contiguous())

    def test_reload_reuses_storage(self):
        attn = _attn()
        first = self._merge(attn).weight
        ptr = first.data_ptr()
        attn.fused_qkv_a_proj_with_mqa.weight.data = torch.ones(
            Q_LORA + KV_A, HIDDEN, dtype=torch.bfloat16
        )
        second = self._merge(attn).weight
        self.assertEqual(second.data_ptr(), ptr)
        self.assertTrue(torch.all(second[: Q_LORA + KV_A] == 1))

    def test_declines(self):
        self.assertIsNone(self._merge(_attn(gate_dtype=torch.float16)))
        attn = _attn()
        attn.use_output_gate = False
        self.assertIsNone(self._merge(attn))
        attn = _attn()
        attn.indexer.rocm_nope_prologue = None
        self.assertIsNone(self._merge(attn))
        attn = _attn()
        attn.indexer.wk.quant_method = object()
        self.assertIsNone(self._merge(attn))

    def test_split_matches_separate_gemms(self):
        attn = _attn(seed=3, merged_k_gate=False)
        ref_w = [
            attn.fused_qkv_a_proj_with_mqa.weight.data.float(),
            attn.indexer.wk.weight.data.float(),
            attn.indexer.weights_proj.weight.data.float(),
            attn.g_proj.weight.data.float(),
        ]
        self._merge(attn)
        x = torch.randn(5, HIDDEN).to(torch.bfloat16)
        qkv = prologue.run_front(attn, x)
        key, gate = attn.indexer.rocm_nope_prologue.front_key_gate
        out_gate = attn.front_output_gate
        for got, w in zip((qkv, key, gate, out_gate), ref_w):
            torch.testing.assert_close(
                got.float(), x.float() @ w.t(), rtol=2e-2, atol=2e-2
            )
        self.assertEqual(key.stride(-1), 1)
        self.assertEqual(gate.stride(-1), 1)
        prologue.clear_front(attn)
        self.assertIsNone(attn.front_output_gate)
        self.assertIsNone(attn.indexer.rocm_nope_prologue.front_key_gate)

    def test_front_applies_only_to_decode_and_verify(self):
        attn = _attn()
        self._merge(attn)
        x = torch.zeros(3, HIDDEN, dtype=torch.bfloat16)

        def batch(decode=False, verify=False):
            mode = SimpleNamespace(
                is_decode_or_idle=lambda: decode, is_target_verify=lambda: verify
            )
            return SimpleNamespace(forward_mode=mode)

        with mock.patch("sglang.srt.layers.cp.utils.is_cp_active", return_value=False):
            self.assertTrue(prologue.front_applies(attn, x, batch(decode=True)))
            self.assertTrue(prologue.front_applies(attn, x, batch(verify=True)))
            self.assertFalse(prologue.front_applies(attn, x, batch()))
            self.assertFalse(
                prologue.front_applies(attn, (x, x, x), batch(decode=True))
            )
            attn._mono_pre = {}
            self.assertFalse(prologue.front_applies(attn, x, batch(decode=True)))


class TestKvStoreDeclines(CustomTestCase):
    def test_declines_when_part_off_or_shape_unsupported(self):
        pool = SimpleNamespace(
            write_loc_is_dcp_resolved=False, dcp_replicated=False, kv_scale_ue8m0=True
        )
        buf = torch.zeros(16, 1, 656, dtype=torch.uint8)
        loc = torch.arange(4)
        nope = torch.zeros(4, 1, 512, dtype=torch.bfloat16)
        rope = torch.zeros(4, 1, 64, dtype=torch.bfloat16)
        with mock.patch.object(prologue, "part_enabled", return_value=False):
            self.assertFalse(prologue.fp8_ds_mla_store(pool, buf, loc, nope, rope))
        with mock.patch.object(prologue, "part_enabled", return_value=True):
            self.assertFalse(
                prologue.fp8_ds_mla_store(pool, buf, loc, nope.float(), rope)
            )
            self.assertFalse(
                prologue.fp8_ds_mla_store(pool, buf, loc, nope, rope[..., :32])
            )


if __name__ == "__main__":
    unittest.main()
