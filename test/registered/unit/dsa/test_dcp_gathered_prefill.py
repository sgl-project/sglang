"""The ROCm DSA DCP prefill must weigh the gathered route against the owned one by wire bytes, and gather only page-whole prefixes below that crossing."""

import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_dcp import (
    dcp_gathered_prefill_crossover,
    dcp_use_gathered_prefill,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dcp.prefill_route import (
    DcpOwnedRoute,
    init_dcp_prefill_crossover,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# GLM-5.2 MLA geometry with the bf16 partials and the fp8 KV cache.
_GLM52 = dict(
    v_head_dim=256,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    partial_dtype=torch.bfloat16,
    kv_dtype=torch.float8_e4m3fn,
)


class TestDCPGatheredPrefillCrossover(CustomTestCase):
    def test_glm52_routes(self):
        for project_before_merge, q_all_gather, want in (
            (True, False, 0.896),
            (True, True, 1.896),
            (False, False, 1.785),
            (False, True, 2.785),
        ):
            with self.subTest(
                project_before_merge=project_before_merge, q_all_gather=q_all_gather
            ):
                got = dcp_gathered_prefill_crossover(
                    **_GLM52,
                    project_before_merge=project_before_merge,
                    q_all_gather=q_all_gather,
                )
                self.assertAlmostEqual(got, want, places=3)


class TestDCPUseGatheredPrefill(CustomTestCase):
    def test_prefix_not_page_whole(self):
        self.assertFalse(dcp_use_gathered_prefill([6], 100, 16, 100.0, dcp_size=4))

    def test_one_prefix_not_page_whole(self):
        self.assertFalse(dcp_use_gathered_prefill([8, 6], 100, 16, 100.0, dcp_size=4))

    def test_zero_crossover_never_gathers(self):
        self.assertFalse(dcp_use_gathered_prefill([0], 100, 16, 0.0, dcp_size=4))
        self.assertFalse(dcp_use_gathered_prefill([8], 100, 16, 0.0, dcp_size=4))

    def test_zero_prefix_gathers(self):
        self.assertTrue(dcp_use_gathered_prefill([0], 100, 16, 0.896, dcp_size=4))

    def test_crossing_is_exclusive(self):
        # 16 heads * 100 extend tokens * crossover 1.0 = 1600 prefix tokens.
        self.assertTrue(dcp_use_gathered_prefill([1596], 100, 16, 1.0, dcp_size=4))
        self.assertFalse(dcp_use_gathered_prefill([1600], 100, 16, 1.0, dcp_size=4))


_ROUTE = DcpOwnedRoute(v_head_dim=256, project_before_merge=True, q_all_gather=False)


class _Leaf:
    """A leaf backend; ``prices`` gives it the gathered-prefill hook."""

    token_to_kv_pool = req_to_token_pool = runner_ids = None

    def __init__(self, prices=True):
        self.calls = []
        if prices:
            self.init_dcp_prefill_crossover = lambda **kw: self.calls.append(kw)

    def window_capacity_children(self):
        return []


class _Attn(torch.nn.Module):
    def __init__(self, route):
        super().__init__()
        self.route = route

    def dcp_owned_route(self):
        return self.route


def _model(*routes):
    return torch.nn.Sequential(torch.nn.Linear(1, 1), *map(_Attn, routes))


def _hybrid(prefill, decode):
    hybrid = object.__new__(HybridAttnBackend)
    hybrid.prefill_backend, hybrid.decode_backend = prefill, decode
    # What TboAttnBackend reads off its primary.
    hybrid.token_to_kv_pool = hybrid.req_to_token_pool = hybrid.runner_ids = None
    return hybrid


class TestInitDCPPrefillCrossover(CustomTestCase):
    def _init(self, model, backend):
        init_dcp_prefill_crossover(
            model=model, attn_backend=backend, partial_dtype=torch.bfloat16
        )

    def test_leaf_gets_first_mla_route(self):
        leaf = _Leaf()
        other = DcpOwnedRoute(
            v_head_dim=128, project_before_merge=False, q_all_gather=True
        )
        self._init(_model(_ROUTE, other), leaf)
        self.assertEqual(leaf.calls, [dict(route=_ROUTE, partial_dtype=torch.bfloat16)])

    def test_tbo_prices_primary_and_children_once_each(self):
        primary, children = _Leaf(), [_Leaf(), _Leaf()]
        self._init(_model(_ROUTE), TboAttnBackend(primary, children))
        for leaf in (primary, *children):
            self.assertEqual(len(leaf.calls), 1)

    def test_hybrid_under_tbo_skips_non_dsa_leaf(self):
        dsa, other = _Leaf(), _Leaf(prices=False)
        tbo = TboAttnBackend(_hybrid(dsa, other), [_hybrid(_Leaf(), other)])
        self._init(_model(_ROUTE), tbo)
        self.assertEqual(len(dsa.calls), 1)
        self.assertEqual(len(tbo.children[0].prefill_backend.calls), 1)

    def test_model_without_mla_route(self):
        leaf = _Leaf()
        self._init(_model(), leaf)
        self.assertEqual(leaf.calls, [])

    def test_no_pricing_backend_skips_the_model(self):
        class _Unscannable(torch.nn.Module):
            def modules(self):
                raise AssertionError("scanned the model with no DSA backend")

        self._init(_Unscannable(), _hybrid(_Leaf(prices=False), _Leaf(prices=False)))


if __name__ == "__main__":
    unittest.main()
