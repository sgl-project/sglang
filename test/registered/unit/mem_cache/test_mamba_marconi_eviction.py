"""CPU unit tests for --radix-eviction-policy marconi.

Marconi (arXiv:2411.19379) evicts the recurrent-state checkpoint with the
lowest norm(recency) + alpha * norm(prefill FLOPs saved per state byte), keeps
branch points (two or more children) for last, and measures the saving against
the nearest ancestor that still holds a state. The cost-model tests pin the
per-token FLOPs and the per-checkpoint bytes (conv + SSM state) of each layer
type the MAMBA component serves: Mamba2/SSD, GDN and KDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.arg_groups.kv_cache_hook import validate_marconi_eviction_policy
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.prefill_cost_model import (
    TOKEN_COUNT_COST_MODEL,
    PrefillCostModel,
    build_prefill_cost_model,
    recurrence_kind,
    recurrent_flops_per_token,
)
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    _rust_fallback_reason,
)
from sglang.srt.mem_cache.utils import get_eviction_strategy
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA
STATE_DTYPE = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)


# ---------------------------------------------------------------- validation


def _view(**overrides):
    fields = dict(
        radix_eviction_policy="marconi",
        radix_eviction_policy_config=None,
        enable_session_radix_cache=False,
        enable_hierarchical_cache=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestMarconiValidation(unittest.TestCase):
    def test_default_alpha_is_valid(self):
        validate_marconi_eviction_policy(_view())
        self.assertEqual(get_eviction_strategy("marconi").alpha, 1.0)

    def test_alpha_is_passed_to_the_strategy(self):
        validate_marconi_eviction_policy(
            _view(radix_eviction_policy_config={"alpha": 0.5})
        )
        self.assertEqual(get_eviction_strategy("marconi", {"alpha": 0.5}).alpha, 0.5)

    def test_rejects_bad_alpha(self):
        for alpha in (-1.0, math.inf, math.nan, "1", True):
            with (
                self.subTest(alpha=alpha),
                self.assertRaisesRegex(ValueError, "alpha"),
            ):
                validate_marconi_eviction_policy(
                    _view(radix_eviction_policy_config={"alpha": alpha})
                )

    def test_rejects_session_radix_cache_and_hicache(self):
        with self.assertRaisesRegex(ValueError, "session-radix-cache"):
            validate_marconi_eviction_policy(_view(enable_session_radix_cache=True))
        with self.assertRaisesRegex(ValueError, "hierarchical-cache"):
            validate_marconi_eviction_policy(_view(enable_hierarchical_cache=True))

    def test_other_policies_are_untouched(self):
        validate_marconi_eviction_policy(
            _view(radix_eviction_policy="lru", enable_session_radix_cache=True)
        )

    def test_runs_on_the_python_tree_core(self):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            eviction_policy="marconi",
        )
        self.assertIn("marconi", _rust_fallback_reason(params))


# ---------------------------------------------------------------- cost model


def _qwen35_9b_gdn() -> Mamba2CacheParams:
    # 32 layers, every 4th full attention; 16 key heads, 32 value heads, dim 128.
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
        conv_shard_groups=[16 * 128, 16 * 128, 32 * 128],
    )
    layers = [i for i in range(32) if i % 4 != 3]
    return Mamba2CacheParams(shape=shape, layers=layers, dtype=STATE_DTYPE)


NEMOTRON_H_4B_PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"


def _nemotron_h_4b_mamba2() -> Mamba2CacheParams:
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=112 * 64,
        n_groups=8,
        num_heads=112,
        head_dim=64,
        state_size=128,
        conv_kernel=4,
    )
    layers = [i for i, c in enumerate(NEMOTRON_H_4B_PATTERN) if c == "M"]
    return Mamba2CacheParams(shape=shape, layers=layers, dtype=STATE_DTYPE)


def _kimi_linear_48b_kda() -> KimiLinearCacheParams:
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=32, head_dim=128, conv_kernel_size=4
    )
    return KimiLinearCacheParams(shape=shape, layers=list(range(20)), dtype=STATE_DTYPE)


class TestRecurrentCost(unittest.TestCase):
    """Hand-derived FLOPs and bytes for one model of each recurrent family."""

    def test_gdn(self):
        params = _qwen35_9b_gdn()
        # conv over cat(q, k, v): 2 * 16 * 128 + 32 * 128 = 8192 channels, 3 taps
        # kept; delta-rule state 32 x 128 x 128.
        conv, state = 8192 * 3, 32 * 128 * 128
        self.assertEqual(params.mamba_cache_per_req, 24 * (2 * conv + 4 * state))
        self.assertEqual(
            recurrent_flops_per_token("gdn", params),
            24 * (7 * state + 2 * 8192 * 4),
        )

    def test_mamba2(self):
        params = _nemotron_h_4b_mamba2()
        self.assertEqual(len(params.layers), 24)
        # conv over x, B, C: 112 * 64 + 2 * 8 * 128 = 9216 channels; SSD state
        # 112 heads x 64 x 128.
        conv, state = 9216 * 3, 112 * 64 * 128
        self.assertEqual(params.mamba_cache_per_req, 24 * (2 * conv + 4 * state))
        self.assertEqual(
            recurrent_flops_per_token("mamba2", params),
            24 * (5 * state + 2 * 9216 * 4),
        )

    def test_kda(self):
        params = _kimi_linear_48b_kda()
        self.assertTrue(params.is_kda)
        # conv state is stored (taps, channels) for q, k and v: 3 x 12288.
        conv, state = 3 * 12288, 32 * 128 * 128
        self.assertEqual(params.mamba_cache_per_req, 20 * (2 * conv + 4 * state))
        self.assertEqual(
            recurrent_flops_per_token("kda", params),
            20 * (7 * state + 2 * 12288 * 4),
        )

    def test_unknown_kind_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "recurrence kind"):
            recurrent_flops_per_token("rwkv", _qwen35_9b_gdn())

    def test_recurrence_kind_follows_the_model_family(self):
        from sglang.srt.configs.falcon_h1 import FalconH1Config
        from sglang.srt.configs.kimi_linear import KimiLinearConfig
        from sglang.srt.configs.nemotron_h import NemotronHConfig
        from sglang.srt.configs.qwen3_next import Qwen3NextConfig

        for config, kind in (
            (Qwen3NextConfig(), "gdn"),
            (KimiLinearConfig(), "kda"),
            (NemotronHConfig(), "mamba2"),
            (FalconH1Config(), "mamba2"),
        ):
            model_config = SimpleNamespace(
                hf_config=config, is_draft_model=False, linear_attn_registry_result=None
            )
            with self.subTest(config=type(config).__name__):
                self.assertEqual(recurrence_kind(model_config), kind)


class TestPrefillCostModel(unittest.TestCase):
    def test_extension_is_additive_and_quadratic(self):
        cost = PrefillCostModel(
            linear_flops_per_token=10.0,
            attention_flops_per_token_pair=2.0,
            state_bytes=4,
        )
        self.assertEqual(cost.prefill_flops(8), 10 * 8 + 64)
        self.assertEqual(
            cost.extend_flops(0, 24), cost.extend_flops(0, 8) + cost.extend_flops(8, 24)
        )
        # The same 8 tokens cost more deeper in the context.
        self.assertGreater(cost.extend_flops(64, 72), cost.extend_flops(0, 8))
        self.assertEqual(cost.extend_flops(8, 8), 0.0)
        self.assertEqual(cost.flops_per_state_byte(0, 8), (10 * 8 + 64) / 4)

    def test_rejects_degenerate_models(self):
        with self.assertRaises(ValueError):
            PrefillCostModel(0.0, 0.0, 1)
        with self.assertRaises(ValueError):
            PrefillCostModel(1.0, 0.0, 0)

    def test_build_from_modules(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        class _Table(VocabParallelEmbedding):
            def __init__(self):
                nn.Module.__init__(self)
                self.weight = nn.Parameter(torch.zeros(100, 8))

        class _Experts(FusedMoE):
            def __init__(self):
                nn.Module.__init__(self)
                self.top_k, self.num_experts = 2, 8
                self.w13_weight = nn.Parameter(torch.zeros(8, 16, 8))

        class _Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = _Table()
                self.proj = nn.Linear(8, 8, bias=False)
                self.experts = _Experts()
                self.attn = RadixAttention(
                    num_heads=4, head_dim=8, scaling=1.0, num_kv_heads=2, layer_id=0
                )
                self.local_attn = RadixAttention(
                    num_heads=4,
                    head_dim=8,
                    scaling=1.0,
                    num_kv_heads=2,
                    layer_id=1,
                    sliding_window_size=16,
                )
                self.visual = nn.Linear(8, 8, bias=False)
                self.lm_head = _Table()

        params = _qwen35_9b_gdn()
        cost = build_prefill_cost_model(SimpleNamespace(), _Tiny(), params, kind="gdn")
        pair = 2 * 4 * (8 + 8)
        # 64 dense + 1024 * 2/8 expert parameters; embeddings, lm_head and the
        # vision tower are not multiplied through by a prompt token.
        self.assertEqual(
            cost.linear_flops_per_token,
            2 * (64 + 256) + recurrent_flops_per_token("gdn", params) + pair * 16,
        )
        self.assertEqual(cost.attention_flops_per_token_pair, pair)
        self.assertEqual(cost.state_bytes, params.mamba_cache_per_req)


# ---------------------------------------------------------------- eviction


class TestMarconiEviction(UnifiedRadixCacheFixtures, CustomTestCase):
    """Eviction order on a FULL + MAMBA tree (page 1, no extra buffer).

    Utilities below use the token-count cost model unless a test sets another:
    the saving of a checkpoint is the number of tokens between it and its
    nearest stateful ancestor. last_access_time is set by hand so that the
    normalized recencies are exact.
    """

    cfg = CacheConfig(
        page_size=1,
        components=(FULL, MAMBA),
        eviction_policy="marconi",
        kv_size=512,
        max_num_reqs=12,
    )

    def _build(self, alpha=1.0, cost=TOKEN_COUNT_COST_MODEL):
        cache, allocator, pool = build_fixture(self.cfg)
        self.assertTrue(cache.components[MAMBA].marconi_eviction)
        cache.tree_core.eviction_strategy.alpha = alpha
        cache.components[MAMBA].prefill_cost_model = cost
        return cache, allocator, pool

    def _put(self, cache, allocator, pool, tokens, t):
        """Insert ``tokens`` with a state at their end; access time ``t``."""
        self._insert(cache, allocator, pool, tokens)
        node = self._node(cache, tokens)
        self.assertEqual(self._depth(cache, tokens), len(tokens))
        node.last_access_time = t
        return node

    def _node(self, cache, tokens):
        node_id = match_prefix(cache, tokens).last_device_node
        return cache.tree_core._node_arena.get(node_id)

    def _depth(self, cache, tokens):
        return match_prefix(cache, tokens).device_prefix_len

    def _has_state(self, node):
        return node.component_data[MAMBA].value is not None

    def _evict(self, cache, pool, n):
        free_before = pool.mamba_allocator.available_size()
        result = cache.evict(EvictParams(mamba_num=n))
        self.assertEqual(result.mamba_num_evicted, n)
        self.assertEqual(pool.mamba_allocator.available_size() - free_before, n)
        cache.sanity_check()

    def test_alpha_zero_is_recency_order(self):
        cache, allocator, pool = self._build(alpha=0.0)
        a = list(range(1, 65))
        b = list(range(100, 108))
        self._put(cache, allocator, pool, a, t=0)
        node_b = self._put(cache, allocator, pool, b, t=1)
        self._evict(cache, pool, 1)
        self.assertEqual(self._depth(cache, a), 0)
        self.assertTrue(self._has_state(node_b))

    def test_saving_outweighs_recency(self):
        # a: 64 tokens, oldest; b and c: 8 tokens. Utilities (alpha 1):
        # a = 0 + 1, b = 0.5 + 0, c = 1 + 0, so b goes although a is older.
        cache, allocator, pool = self._build()
        a = list(range(1, 65))
        b = list(range(100, 108))
        c = list(range(200, 208))
        node_a = self._put(cache, allocator, pool, a, t=0)
        self._put(cache, allocator, pool, b, t=1)
        node_c = self._put(cache, allocator, pool, c, t=2)
        self._evict(cache, pool, 1)
        self.assertEqual(self._depth(cache, b), 0)
        self.assertTrue(self._has_state(node_a) and self._has_state(node_c))

    def test_attention_term_favours_deep_checkpoints(self):
        # p: 56 tokens (t 10) -> b: +8 (t 0); c: 8 standalone tokens (t 1).
        # b and c both hold 8 tokens. Counting tokens, b saves 8 = c and is
        # older, so b goes. With F(L) = L + L^2 / 2 b's 8 tokens attend to 56
        # more and save F(64) - F(56) = 488 against c's F(8) = 40: savings
        # normalize to p 1, b 0.27, c 0, recencies to p 1, b 0, c 0.1, and c
        # goes.
        quadratic = PrefillCostModel(
            linear_flops_per_token=1.0,
            attention_flops_per_token_pair=1.0,
            state_bytes=1,
        )
        for cost, victim in ((TOKEN_COUNT_COST_MODEL, "b"), (quadratic, "c")):
            with self.subTest(victim=victim):
                cache, allocator, pool = self._build(alpha=1.0, cost=cost)
                p = list(range(1, 57))
                c = list(range(200, 208))
                node_p = self._put(cache, allocator, pool, p, t=10)
                nodes = {
                    "b": self._put(
                        cache, allocator, pool, p + list(range(300, 308)), t=0
                    ),
                    "c": self._put(cache, allocator, pool, c, t=1),
                }
                node_p.last_access_time = 10
                self._evict(cache, pool, 1)
                for name, node in nodes.items():
                    self.assertEqual(self._has_state(node), name != victim, name)

    def _parent_and_child(self, cache, allocator, pool, *, t_parent, t_child):
        """A checkpoint at 16 tokens whose only child adds 16 more."""
        parent = list(range(1, 17))
        child = parent + list(range(500, 516))
        node_p = self._put(cache, allocator, pool, parent, t=t_parent)
        node_c = self._put(cache, allocator, pool, child, t=t_child)
        node_p.last_access_time = t_parent
        self.assertIs(node_c.parent, node_p)
        return node_p, node_c

    def test_saving_is_marginal_to_the_nearest_live_ancestor(self):
        # p (16, t 10) -> b (+16, t 1); c: 16 standalone tokens (t 2). Every
        # checkpoint saves 16, so only recency counts: p 1, b 0, c 0.11.
        cache, allocator, pool = self._build(alpha=2.0)
        node_p, node_b = self._parent_and_child(
            cache, allocator, pool, t_parent=10, t_child=1
        )
        node_c = self._put(cache, allocator, pool, list(range(200, 216)), t=2)
        self._evict(cache, pool, 1)
        self.assertFalse(self._has_state(node_b))
        self.assertTrue(self._has_state(node_p) and self._has_state(node_c))

    def test_tombstoned_parent_raises_the_child_saving(self):
        # As above, but p's state is gone: b now saves 32 against c's 16, and
        # utility b = 0 + 2 * 1 > c = 1 + 0, so c goes instead. Measuring b
        # against its parent's depth would keep b's saving at 16 and take b.
        cache, allocator, pool = self._build(alpha=2.0)
        node_p, node_b = self._parent_and_child(
            cache, allocator, pool, t_parent=0, t_child=1
        )
        c = list(range(200, 216))
        self._put(cache, allocator, pool, c, t=2)
        self._evict_device_component(cache, node_p.id, MAMBA)
        self._evict(cache, pool, 1)
        self.assertEqual(self._depth(cache, c), 0)
        self.assertTrue(self._has_state(node_b))

    def test_tombstone_during_the_walk_reorders_the_heap(self):
        # p (16, t 0) -> b (+16, t 1); c: 20 standalone tokens (t 1).
        # Savings at the start: p 16, b 16, c 20 -> normalized 0, 0, 1;
        # utilities p 0, b 1, c 2. p is tombstoned first; b then saves 32 ->
        # 4 and rises to 5, above c, so the second victim is c, not b.
        cache, allocator, pool = self._build(alpha=1.0)
        node_p, node_b = self._parent_and_child(
            cache, allocator, pool, t_parent=0, t_child=1
        )
        c = list(range(200, 220))
        self._put(cache, allocator, pool, c, t=1)
        self._evict(cache, pool, 2)
        self.assertFalse(self._has_state(node_p))
        self.assertTrue(self._has_state(node_b))
        self.assertEqual(self._depth(cache, c), 0)

    def test_branch_point_goes_last_until_it_loses_a_child(self):
        # p (8, t 0) with leaves b1 (t 1) and b2 (t 3); c standalone (t 2).
        # Every checkpoint saves 8. p is the oldest but has two children, so
        # b1 goes first; p then has one child and is next, ahead of c.
        cache, allocator, pool = self._build(alpha=1.0)
        p = list(range(1, 9))
        node_p = self._put(cache, allocator, pool, p, t=0)
        node_b1 = self._put(cache, allocator, pool, p + list(range(600, 608)), t=1)
        node_b2 = self._put(cache, allocator, pool, p + list(range(700, 708)), t=3)
        node_c = self._put(cache, allocator, pool, list(range(200, 208)), t=2)
        node_p.last_access_time = 0
        self.assertEqual(len(node_p.children), 2)
        self._evict(cache, pool, 1)
        self.assertNotIn(node_b1, node_p.children.values())
        self.assertTrue(self._has_state(node_p) and self._has_state(node_b2))
        self._evict(cache, pool, 1)
        self.assertFalse(self._has_state(node_p))
        self.assertTrue(self._has_state(node_c) and self._has_state(node_b2))

    def test_branch_point_refreshed_within_one_walk(self):
        # Same tree; one walk of two victims takes b1 and then p.
        cache, allocator, pool = self._build(alpha=1.0)
        p = list(range(1, 9))
        node_p = self._put(cache, allocator, pool, p, t=0)
        self._put(cache, allocator, pool, p + list(range(600, 608)), t=1)
        node_b2 = self._put(cache, allocator, pool, p + list(range(700, 708)), t=3)
        node_c = self._put(cache, allocator, pool, list(range(200, 208)), t=2)
        node_p.last_access_time = 0
        self._evict(cache, pool, 2)
        self.assertFalse(self._has_state(node_p))
        self.assertTrue(self._has_state(node_c) and self._has_state(node_b2))

    def test_locked_checkpoints_are_skipped(self):
        cache, allocator, pool = self._build(alpha=0.0)
        a = list(range(1, 9))
        b = list(range(100, 108))
        node_a = self._put(cache, allocator, pool, a, t=0)
        node_b = self._put(cache, allocator, pool, b, t=1)
        lock = cache.lock(node_a.id)
        self._evict(cache, pool, 1)
        self.assertTrue(self._has_state(node_a))
        self.assertFalse(self._has_state(node_b))
        cache.unlock(lock)

    def test_evicting_everything_balances(self):
        cache, allocator, pool = self._build(alpha=1.0)
        p = list(range(1, 9))
        self._put(cache, allocator, pool, p, t=0)
        self._put(cache, allocator, pool, p + list(range(600, 608)), t=1)
        self._put(cache, allocator, pool, p + list(range(700, 708)), t=3)
        self._put(cache, allocator, pool, list(range(200, 220)), t=2)
        self._evict(cache, pool, 4)
        result = cache.evict(EvictParams(mamba_num=1))
        self.assertEqual(result.mamba_num_evicted, 0)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
