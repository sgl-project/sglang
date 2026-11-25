"""Unit tests for the K3 die-local routed MoE (SGLANG_K3_ROUTED_MOE_DEEPGEMM)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import json
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_ar_fusion
from sglang.srt.layers.moe import k3_deepgemm_routed as routed
from sglang.srt.layers.moe import k3_die_local_moe as policy
from sglang.srt.models import kimi_k3
from sglang.test.test_utils import CustomTestCase

H, LATENT = 64, 32
FORK, JOIN = "alt waits main", "main waits alt"
STOCK_ORDER = ["front_gemm", FORK, "shared_gemm", "front_gather", FORK, "shared_ar"]
STOCK_ORDER += ["routed", JOIN, "latent_ar", "tail"]


@contextlib.contextmanager
def _env(ar_fusion=False):
    with (
        envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM.override(True),
        envs.SGLANG_K3_AR_FUSION.override(ar_fusion),
        envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM_VERIFY_WIDTHS.override(""),
    ):
        policy.reset_caches()
        try:
            yield
        finally:
            policy.reset_caches()


def _taken(verify):
    token = policy._verify.set((verify, False))
    try:
        return [rows for rows in range(1, 300) if policy.takes(rows)]
    finally:
        policy._verify.reset(token)


def _released_deep_gemm():
    def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, weight_layout=None):
        pass

    class SymmBuffer:
        def __init__(self, group, num_experts, mma_type=None):
            pass

    return types.SimpleNamespace(
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        mega=types.SimpleNamespace(SymmBuffer=SymmBuffer),
        __version__="0.2.9",
    )


class _Stream:
    def __init__(self, trace, name):
        self.trace, self.name = trace, name

    def wait_stream(self, other):
        self.trace.order.append(f"{self.name} waits {other.name}")


class _Trace:
    def __init__(self):
        self.order, self.on_alt = [], []
        self.main, self.alt = _Stream(self, "main"), _Stream(self, "alt")
        self.current = self.main

    def op(self, name, result=None):
        self.order.append(name)
        if self.current is self.alt:
            self.on_alt.append(name)
        return result

    @contextlib.contextmanager
    def stream(self, s):
        prev, self.current = self.current, s
        try:
            yield
        finally:
            self.current = prev


def _forward_fused(*, mega, sharded=True, rows=80, in_table=True):
    tr, got, buffers = _Trace(), {}, []

    def symm_buffer(name, n, width, dtype):
        buffers.append(torch.zeros(n, width, dtype=dtype))
        return buffers[-1]

    def all_reduce_norm(x, *norm_params, num_tokens=None):
        got["norm"] = (tuple(x.shape), num_tokens)
        tr.op("fused_ar" if x.numel() == buffers[-1].numel() else "latent_ar")

    def deferred(hs, logits, x):
        got["deferred"] = True
        return tr.op(
            "routed",
            types.SimpleNamespace(
                gemm2_out=None, expanded_idx_to_permuted_idx=None, expert_weights=None
            ),
        )

    fusion = types.SimpleNamespace(
        MOE_LATENT_SHARED="moe_latent_shared",
        NORM_DIM=LATENT,
        enabled=lambda: True,
        finalize_push_fits=lambda n: n <= 73,
        gemm_ag_up_fits=lambda n: True,
        symm_buffer=symm_buffer,
        all_reduce_low_sm=lambda x, **kw: tr.op("shared_ar"),
        all_reduce_norm=all_reduce_norm,
        finalize_all_reduce_push_norm=lambda *a: tr.op("latent_ar"),
        gemm_ag_up_proj=lambda latent, w, b, c: tr.op("tail", torch.zeros(rows, H)),
    )
    me = types.SimpleNamespace(
        moe_hidden_size=LATENT,
        alt_stream=tr.alt,
        experts=types.SimpleNamespace(**(dict(mega_l1_weights=(1, 1)) if mega else {})),
        _defer_moe_finalize=True,
        fuse_ar_norm=True,
        _gemm_ag_up_eligible=True,
        routed_expert_up_proj=types.SimpleNamespace(weight=torch.zeros(H, LATENT)),
        _sharded_front_covers=lambda n: sharded,
        _sharded_front_gemm=lambda hs: tr.op("front_gemm", ("gate_up", "block")),
        _fused_front=lambda hs: tr.op("front_gemm", ("gate_up", "logits", "x")),
        _sharded_front_gather=lambda b, hs: tr.op("front_gather", ("logits", "x")),
        _forward_shared=lambda gate_up, out: tr.op("shared_gemm"),
        _forward_routed=lambda hs, logits, x, latent: tr.op("routed"),
        _forward_routed_deferred=deferred,
        _get_fused_norm_params=lambda: (torch.ones(LATENT), 1e-6),
    )
    with (
        mock.patch.object(policy, "takes", lambda rows: in_table),
        mock.patch.object(kimi_k3, "k3_ar_fusion", fusion),
        mock.patch.object(torch.cuda, "current_stream", lambda: tr.current),
        mock.patch.object(torch.cuda, "stream", tr.stream),
    ):
        kimi_k3.KimiK3MoE._forward_fused(
            me, torch.zeros(rows, H), prefix_sum=None, scatters=False
        )
    return tr, got


class TestK3DieLocalMoE(CustomTestCase):
    def test_deep_gemm_without_the_api_keeps_the_default_path(self):
        config = types.SimpleNamespace(
            activation="situ",
            gemm1_alpha=4.0,
            gemm1_clamp_limit=25.0,
            gate_up_interleaved=False,
        )
        layer = types.SimpleNamespace(moe_runner_config=config)
        warnings = []
        routed.missing_capability.cache_clear()
        self.addCleanup(routed.missing_capability.cache_clear)
        with (
            _env(),
            mock.patch.object(routed, "_deep_gemm", _released_deep_gemm),
            mock.patch.object(policy, "print_warning_once", warnings.append),
        ):
            self.assertFalse(policy.wanted(layer, False, "default"))
        self.assertEqual(len(warnings), 1)
        self.assertIn("SGLANG_K3_ROUTED_MOE_DEEPGEMM is set but not used", warnings[0])
        self.assertIn("deep_gemm 0.2.9 lacks the one-rank MegaMoE API", warnings[0])

    def test_in_tree_table_dispatch(self):
        table = json.loads(policy._DEFAULT_TABLE.read_text())
        fold = table["ar_fusion_fold_max_rows"]
        self.assertEqual(
            table["widths_ar_fusion"], [rows for rows in table["widths"] if rows > fold]
        )
        # the fold limit of the table is the one k3_ar_fusion applies
        state = types.SimpleNamespace(push_slot_bytes=1 << 30)
        with mock.patch.object(k3_ar_fusion, "_get_state", return_value=state):
            self.assertTrue(k3_ar_fusion.finalize_push_fits(fold))
            self.assertFalse(k3_ar_fusion.finalize_push_fits(fold + 1))
        for ar_fusion, key in ((False, "widths"), (True, "widths_ar_fusion")):
            with self.subTest(ar_fusion=ar_fusion), _env(ar_fusion):
                self.assertEqual(_taken(verify=True), table[key])
                self.assertEqual(_taken(verify=False), [])

    def test_page_homes(self):
        page = 2 << 20
        self.assertEqual(policy.page_homes((3, 4, page), 1, page), [0, 0, 1, 1] * 3)
        # a page split evenly between the domains alternates by page index
        self.assertEqual(policy.page_homes((4, 2, page // 2), 1, page), [0, 1, 0, 1])

    def test_kernel_call_is_one_pull_behind_the_kernel(self):
        for sharded, rows in ((True, 80), (False, 80), (True, 40)):
            with self.subTest(sharded=sharded, rows=rows):
                tr, got = _forward_fused(mega=True, sharded=sharded, rows=rows)
                gather = ["front_gather"] if sharded else []
                self.assertEqual(
                    tr.order,
                    ["front_gemm", FORK, "shared_gemm", *gather, JOIN]
                    + ["routed", JOIN, "fused_ar", "tail"],
                )
                self.assertEqual(tr.on_alt, ["shared_gemm"])
                self.assertEqual(got["norm"], ((3 * rows, LATENT), rows))
                self.assertNotIn("deferred", got)

    def test_other_calls_keep_the_stock_order(self):
        for mega, rows in ((False, 80), (False, 40), (True, 80), (True, 40)):
            with self.subTest(mega=mega, rows=rows):
                tr, got = _forward_fused(mega=mega, rows=rows, in_table=False)
                self.assertEqual(tr.order, STOCK_ORDER)
                self.assertEqual(tr.on_alt, ["shared_gemm", "shared_ar"])
                self.assertEqual("deferred" in got, rows <= 73)


if __name__ == "__main__":
    unittest.main()
