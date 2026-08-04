"""Unit tests for DeepEP low_latency capacity planning — CPU-only, fully mocked."""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ENV = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK

# The clamp does a call-time local import, so patch at the source module.
_MIXIN_MODULE = "sglang.srt.distributed.parallel_state"

try:
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplBase

    _HAS_DEEPEP_MODULE = True
except Exception:
    _HAS_DEEPEP_MODULE = False

try:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.model_executor.deepep_capacity import (
        _MAX_RESERVE_FRACTION,
        DeepEPCapacityPlan,
        deepep_tokens_per_req,
        plan_deepep_capacity,
    )
    from sglang.srt.model_executor.model_runner import ModelRunner

    _HAS_MODEL_RUNNER = True
except Exception:
    _HAS_MODEL_RUNNER = False


@contextlib.contextmanager
def _env_unset():
    was_set = _ENV.is_set()
    old = _ENV.get() if was_set else None
    _ENV.clear()
    try:
        yield
    finally:
        if was_set:
            _ENV.set(old)
        else:
            _ENV.clear()


def _plan(ceiling=1024, tokens_per_req=1, auto_sized=True):
    return DeepEPCapacityPlan(
        ceiling=ceiling, tokens_per_req=tokens_per_req, auto_sized=auto_sized
    )


def _runner(plan, pool_size=4096, is_draft_worker=False):
    return SimpleNamespace(
        deepep_capacity_plan=plan,
        is_draft_worker=is_draft_worker,
        req_to_token_pool=SimpleNamespace(size=pool_size),
        tp_rank=0,
    )


@unittest.skipUnless(_HAS_DEEPEP_MODULE, "deepep token dispatcher not importable")
class TestDeepEPNumMaxDispatchTokensProperty(unittest.TestCase):
    """The dispatcher field resolves from the env lazily, then caches."""

    def _stub(self):
        # Skip __init__ (needs DeepEP + a process group); only the lazy field matters.
        stub = _DeepEPDispatcherImplBase.__new__(_DeepEPDispatcherImplBase)
        stub._num_max_dispatch_tokens_per_rank = None
        return stub

    def test_resolves_default_when_unset(self):
        with _env_unset():
            self.assertEqual(self._stub().num_max_dispatch_tokens_per_rank, 128)

    def test_resolves_value_set_after_construction(self):
        stub = self._stub()
        with _ENV.override(512):
            self.assertEqual(stub.num_max_dispatch_tokens_per_rank, 512)

    def test_caches_first_read(self):
        stub = self._stub()
        with _ENV.override(256):
            self.assertEqual(stub.num_max_dispatch_tokens_per_rank, 256)
        # Env moved after the first read; the cached value must not follow it.
        with _ENV.override(512):
            self.assertEqual(stub.num_max_dispatch_tokens_per_rank, 256)

    def test_rejects_above_hard_cap(self):
        stub = self._stub()
        with _ENV.override(2048):
            with self.assertRaises(AssertionError):
                _ = stub.num_max_dispatch_tokens_per_rank


@unittest.skipUnless(_HAS_MODEL_RUNNER, "model_runner not importable")
class TestDeepEPTokensPerReq(unittest.TestCase):
    def _args(self, num_draft=None, max_draft=None):
        return SimpleNamespace(
            speculative_num_draft_tokens=num_draft,
            max_speculative_num_draft_tokens=max_draft,
        )

    def test_defaults_to_one(self):
        self.assertEqual(deepep_tokens_per_req(self._args()), 1)

    def test_uses_startup_draft_tokens(self):
        self.assertEqual(deepep_tokens_per_req(self._args(num_draft=4)), 4)

    def test_adaptive_max_wins_over_startup(self):
        # Adaptive spec can grow draft tokens to the max at runtime, so capacity
        # must track the max (8), not the startup value (2).
        self.assertEqual(deepep_tokens_per_req(self._args(num_draft=2, max_draft=8)), 8)


@unittest.skipUnless(_HAS_MODEL_RUNNER, "model_runner not importable")
class TestDeepEPCapacityPlanning(unittest.TestCase):
    """plan_deepep_capacity gates, tiers, and sizes the reservation."""

    def _plan_for(
        self,
        *,
        gpu_gib=185,
        slack_gib=36.0,
        hidden=6144,
        num_experts=256,
        moe_intermediate=2048,
        max_running=None,
        dp_size=8,
        backend="deepep",
        mode="auto",
        num_draft=None,
        max_draft=None,
    ):
        server_args = SimpleNamespace(
            moe_a2a_backend=backend,
            deepep_mode=mode,
            max_speculative_num_draft_tokens=max_draft,
            speculative_num_draft_tokens=num_draft,
            max_running_requests=max_running,
            dp_size=dp_size,
            auto_mem_deepep_slack_mib=(
                slack_gib * 1024 if slack_gib is not None else None
            ),
        )
        model_config = SimpleNamespace(
            hidden_size=hidden,
            hf_config=SimpleNamespace(
                n_routed_experts=num_experts,
                moe_intermediate_size=moe_intermediate,
            ),
        )
        return plan_deepep_capacity(
            server_args,
            model_config,
            gpu_total_mib=gpu_gib * 1024 if gpu_gib is not None else None,
            moe_ep_size=dp_size,
        )

    def test_none_for_non_deepep_backend(self):
        with _env_unset():
            self.assertIsNone(self._plan_for(backend="none"))
            self.assertIsNone(self._plan_for(backend="mooncake"))

    def test_none_for_normal_mode(self):
        with _env_unset():
            self.assertIsNone(self._plan_for(mode="normal"))

    def test_tight_card_tiers_ceiling_down(self):
        # V3.2-class: 140 GiB H200, short-context slack ~10.5 GiB, 256x2048x7168.
        # The 1024 ceiling's ~14 GiB buffer + capture exceeds the 12% cap, so the
        # plan tiers down to 512 (its buffer fits un-clamped).
        with _env_unset():
            plan = self._plan_for(
                gpu_gib=139.8,
                slack_gib=10.5,
                hidden=7168,
                num_experts=256,
                moe_intermediate=2048,
            )
            self.assertEqual(plan.ceiling, 512)
            self.assertTrue(plan.auto_sized)
            self.assertGreater(plan.reserve_mib, 0)

    def test_wide_card_keeps_ceiling(self):
        # GLM-5.2-class on GB300: large chunked slack credits buffer + capture,
        # so the 1024 ceiling fits and is kept (no decode-concurrency clamp).
        with _env_unset():
            plan = self._plan_for()
            self.assertEqual(plan.ceiling, 1024)
            self.assertTrue(plan.auto_sized)

    def test_slack_covering_reservation_reserves_nothing(self):
        # GLM-5.2 balanced: the chunked over-reservation covers buffer + capture
        # entirely, so the KV budget must be byte-identical to main's.
        with _env_unset():
            plan = self._plan_for(slack_gib=40.0)
            self.assertEqual(plan.reserve_mib, 0.0)

    def test_user_max_running_caps_ceiling(self):
        # An explicit --max-running-requests bounds the ceiling below 1024 even
        # on a wide card (per-rank concurrency is the real dispatch bound).
        with _env_unset():
            plan = self._plan_for(max_running=2048, dp_size=8)
            self.assertEqual(plan.ceiling, 256)

    def test_small_user_cap_aligns_ceiling_up(self):
        # A tiny per-rank cap (48 reqs / dp 4 x 2 draft tokens = 24) still gets
        # a 128-aligned ceiling: deep_ep's fp8 recv-scale layout corrupts
        # silently on non-multiples, and 128 is the static default footprint.
        with _env_unset():
            plan = self._plan_for(max_running=48, dp_size=4, num_draft=2)
            self.assertEqual(plan.ceiling, 128)

    def test_user_mem_fraction_returns_static_plan(self):
        # slack None means mem_fraction_static was user-set: the auto formula
        # budgeted nothing, so the plan must not size a larger buffer.
        with _env_unset():
            plan = self._plan_for(slack_gib=None)
            self.assertEqual(plan.ceiling, 128)
            self.assertFalse(plan.auto_sized)
            self.assertEqual(plan.reserve_mib, 0.0)

    def test_user_env_pins_ceiling_and_reserves_for_it(self):
        # A user-set env skips the tier-down but still reserves for the pinned
        # buffer size.
        with _ENV.override(512):
            plan = self._plan_for(
                gpu_gib=139.8, slack_gib=10.5, hidden=7168, moe_intermediate=2048
            )
            self.assertEqual(plan.ceiling, 512)
            self.assertTrue(plan.auto_sized)
            self.assertGreater(plan.reserve_mib, 0)

    def test_missing_experts_returns_static_plan(self):
        with _env_unset():
            server_args = SimpleNamespace(
                moe_a2a_backend="deepep",
                deepep_mode="auto",
                max_speculative_num_draft_tokens=None,
                speculative_num_draft_tokens=None,
                max_running_requests=None,
                dp_size=8,
                auto_mem_deepep_slack_mib=10.5 * 1024,
            )
            model_config = SimpleNamespace(
                hidden_size=7168, hf_config=SimpleNamespace()
            )
            plan = plan_deepep_capacity(
                server_args, model_config, gpu_total_mib=139.8 * 1024, moe_ep_size=8
            )
            self.assertEqual(plan.ceiling, 128)
            self.assertFalse(plan.auto_sized)

    def test_reservation_capped_at_max_fraction(self):
        # Even when every tier overflows the cap (tiny slack, huge MoE), the
        # final reservation is clamped to the cap instead of eating the KV pool.
        with _env_unset():
            plan = self._plan_for(
                gpu_gib=139.8,
                slack_gib=0.0,
                hidden=7168,
                num_experts=384,
                moe_intermediate=3072,
            )
            cap_mib = _MAX_RESERVE_FRACTION * 139.8 * 1024
            self.assertLessEqual(plan.reserve_mib, cap_mib)


@unittest.skipUnless(_HAS_MODEL_RUNNER, "model_runner not importable")
class TestDeepEPResolveNumMax(unittest.TestCase):
    """ModelRunner._maybe_auto_tune_deepep_num_max_dispatch_tokens behavior."""

    def _run(self, runner):
        ModelRunner._maybe_auto_tune_deepep_num_max_dispatch_tokens(runner)

    def test_user_env_always_wins(self):
        with _ENV.override(700):
            runner = _runner(_plan(ceiling=1024))
            self._run(runner)
            self.assertEqual(_ENV.get(), 700)
            self.assertEqual(runner.deepep_capacity_plan.num_max, 700)

    def test_no_op_without_plan(self):
        # Non-deepep backends / normal mode have no plan (see
        # TestDeepEPCapacityPlanning): nothing to resolve, env untouched.
        with _env_unset():
            self._run(_runner(None))
            self.assertFalse(_ENV.is_set())

    def test_no_op_for_draft_worker(self):
        # The target resolves and exports the shared bound; a draft runner must
        # not re-resolve from its own (differently sized) pool.
        with _env_unset():
            self._run(_runner(_plan(ceiling=1024), pool_size=64, is_draft_worker=True))
            self.assertFalse(_ENV.is_set())

    def test_sizes_to_decode_concurrency(self):
        # With a non-binding ceiling, num_max tracks decode concurrency.
        with _env_unset():
            self._run(_runner(_plan(ceiling=1024), pool_size=512))
            self.assertEqual(_ENV.get(), 512)

    def test_caps_at_finished_sum_tag(self):
        # req pool above the 1024 FINISHED_SUM_TAG ceiling clamps to 1024.
        with _env_unset():
            self._run(_runner(_plan(ceiling=1024), pool_size=4096))
            self.assertEqual(_ENV.get(), 1024)

    def test_static_plan_keeps_default(self):
        # No reservation (user-set mem_fraction): the buffer for a larger
        # num_max was never budgeted, so the bound must NOT be raised — stay
        # at the static default instead of OOMing at capture.
        with _env_unset():
            self._run(_runner(_plan(ceiling=128, auto_sized=False), pool_size=4096))
            self.assertFalse(_ENV.is_set())
            self.assertEqual(_ENV.get(), 128)

    def test_ceiling_caps_num_max(self):
        # The reservation sized the buffer for the ceiling; the runtime must not
        # resolve above it even when decode concurrency is far higher, or the
        # larger buffer would OOM at capture.
        with _env_unset():
            self._run(_runner(_plan(ceiling=128), pool_size=4096))
            self.assertEqual(_ENV.get(), 128)

    def test_low_concurrency_aligns_up_to_128(self):
        # deep_ep's fp8 recv-scale layout needs num_max * num_ranks % 128 == 0;
        # a 24-token bound (12 reqs x 2 draft tokens) silently corrupts scales,
        # so the resolved bound rounds up to the 128 alignment (= the static
        # default buffer size, so no extra footprint vs main).
        with _env_unset():
            self._run(_runner(_plan(ceiling=1024, tokens_per_req=2), pool_size=12))
            self.assertEqual(_ENV.get(), 128)

    def test_spec_scales_num_max_by_draft_tokens(self):
        # Spec verify dispatches tokens_per_req per request, so num_max tracks
        # concurrency * tokens_per_req (128-aligned), clamped to the ceiling.
        with _env_unset():
            self._run(_runner(_plan(ceiling=1024, tokens_per_req=4), pool_size=100))
            self.assertEqual(_ENV.get(), 512)
        with _env_unset():
            self._run(_runner(_plan(ceiling=1024, tokens_per_req=4), pool_size=512))
            self.assertEqual(_ENV.get(), 1024)


@unittest.skipUnless(_HAS_MODEL_RUNNER, "model_runner not importable")
class TestDeepEPKvBudgetReserve(unittest.TestCase):
    """_reserve_deepep_capacity subtracts the plan from the KV budget in GiB."""

    def _reserve(self, plan, rest_gib):
        runner = SimpleNamespace(deepep_capacity_plan=plan)
        return KVCacheConfigurator._reserve_deepep_capacity(runner, rest_gib)

    def test_passthrough_without_plan(self):
        self.assertEqual(self._reserve(None, 40.0), 40.0)

    def test_passthrough_for_static_plan(self):
        plan = _plan(ceiling=128, auto_sized=False)
        self.assertEqual(self._reserve(plan, 40.0), 40.0)

    def test_zero_reserve_keeps_budget(self):
        plan = _plan(ceiling=1024)
        plan.reserve_mib = 0.0
        self.assertEqual(self._reserve(plan, 40.0), 40.0)

    def test_reserve_subtracted(self):
        plan = _plan(ceiling=1024)
        plan.reserve_mib = 8 * 1024
        self.assertAlmostEqual(self._reserve(plan, 40.0), 32.0)

    def test_oversized_reserve_degrades_to_kv_floor(self):
        # The capture estimate is conservative; when it exceeds the budget the
        # reserve degrades to what is affordable (keeping the KV floor) instead
        # of refusing to serve a config that would in fact capture fine.
        plan = _plan(ceiling=128)
        plan.reserve_mib = 18 * 1024
        self.assertAlmostEqual(self._reserve(plan, 16.5), 2.0)

    def test_exhausted_budget_raises(self):
        plan = _plan(ceiling=128)
        plan.reserve_mib = 18 * 1024
        with self.assertRaisesRegex(ValueError, "KV budget"):
            self._reserve(plan, 1.8)


@unittest.skipUnless(_HAS_MODEL_RUNNER, "model_runner not importable")
class TestDeepEPConcurrencyClamp(unittest.TestCase):
    """_clamp_deepep_low_latency_concurrency caps + EP-group-syncs concurrency."""

    def _clamp(self, plan, max_num_reqs):
        runner = SimpleNamespace(deepep_capacity_plan=plan)
        return KVCacheConfigurator._clamp_deepep_low_latency_concurrency(
            runner, max_num_reqs
        )

    def test_passthrough_without_plan(self):
        # Non-deepep backends / normal mode have no plan: no clamp.
        self.assertEqual(self._clamp(None, 2048), 2048)

    def test_caps_to_ceiling_single_rank(self):
        with patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(self._clamp(_plan(ceiling=1024), 2048), 1024)

    def test_static_plan_caps_to_default_buffer(self):
        # No reservation (user mem_fraction): the buffer is the static
        # default, so concurrency caps to it — NOT the loose FINISHED_SUM_TAG
        # bound, which would let the decode batch overrun the small default
        # buffer.
        with patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(_plan(ceiling=128, auto_sized=False), 2048), 128
            )

    def test_below_cap_unchanged_single_rank(self):
        with patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(self._clamp(_plan(ceiling=1024), 512), 512)

    def test_spec_divides_cap_by_draft_tokens(self):
        # tokens_per_req=4 means batch * 4 tokens dispatched, so the ceiling
        # (1024) is divided: 1024 // 4 = 256.
        with patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(_plan(ceiling=1024, tokens_per_req=4), 2048), 256
            )

    def test_adaptive_spec_divides_cap_by_max_draft_tokens(self):
        # tokens_per_req carries the adaptive max (8, not the startup value):
        # 1024 // 8 = 128.
        with patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=None),
        ):
            self.assertEqual(
                self._clamp(_plan(ceiling=1024, tokens_per_req=8), 2048), 128
            )

    def test_takes_ep_group_minimum(self):
        # Simulate a peer rank contributing a smaller concurrency: cap is 800,
        # the group MIN drives it to 600.
        def fake_all_reduce(tensor, op=None, group=None):
            tensor.fill_(600)

        with patch(
            f"{_MIXIN_MODULE}.get_moe_ep_group",
            return_value=SimpleNamespace(world_size=2, cpu_group=object()),
        ), patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            self.assertEqual(self._clamp(_plan(ceiling=1024), 800), 600)


if __name__ == "__main__":
    unittest.main()
