"""Unit tests for pool_configurator.py -- CPU only, no GPU required.

Tests the end-to-end computation: available_bytes -> MemoryPoolConfig,
verifying tokens are correct, constraints are respected, and memory
invariants hold (tokens * per_token_cost <= available_bytes).
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextlib.contextmanager
def mock_cpu_env(kv_size=2, tp_size=1, swa_eviction_interval=4):
    """Mock GPU-dependent functions for CPU-only testing.

    swa_eviction_interval pins SGLANG_SWA_EVICTION_INTERVAL (decode batches between
    SWA evictions) to a small value so the chunk-cap formula stays hand-computable;
    only SWAChunkCapPoolConfigurator reads it.
    """
    from sglang.srt.environ import envs

    with (
        patch("torch._utils._element_size", return_value=kv_size),
        get_parallel().override(attn_tp_size=tp_size),
        envs.SGLANG_SWA_EVICTION_INTERVAL.override(swa_eviction_interval),
    ):
        yield


def _make_model_runner(
    *,
    num_kv_heads=4,
    head_dim=64,
    v_head_dim=64,
    num_layers=32,
    use_mla_backend=False,
    is_hybrid_swa=False,
    full_attention_layer_ids=None,
    swa_attention_layer_ids=None,
    swa_num_kv_heads=None,
    swa_head_dim=None,
    swa_v_head_dim=None,
    swa_full_tokens_ratio=0.5,
    page_size=1,
    mambaish_config=None,
    disable_radix_cache=False,
    chunked_prefill_size=None,
    disable_overlap_schedule=False,
    sliding_window_size=None,
    speculative_num_draft_tokens=None,
    max_speculative_num_draft_tokens=None,
    speculative_algorithm=None,
    speculative_num_steps=None,
    speculative_eagle_topk=None,
    disaggregation_mode="null",
    max_running_requests=None,
    disaggregation_decode_extra_slots=0,
    hidden_size=0,
):
    """Create a mock ModelRunner with the fields configurators need."""
    mr = MagicMock()

    mr.use_mla_backend = use_mla_backend
    mr.is_draft_worker = False
    mr.num_effective_layers = num_layers
    mr.start_layer = 0
    mr.end_layer = num_layers
    mr.dp_size = 1
    mr.page_size = page_size
    mr.mambaish_config = mambaish_config
    mr.is_hybrid_swa = is_hybrid_swa
    mr.sliding_window_size = sliding_window_size

    mc = SimpleNamespace()
    mc.head_dim = head_dim
    mc.v_head_dim = v_head_dim
    mc.is_hybrid_swa = is_hybrid_swa
    mc.full_attention_layer_ids = (
        full_attention_layer_ids
        if full_attention_layer_ids is not None
        else list(range(num_layers))
    )
    mc.swa_attention_layer_ids = (
        swa_attention_layer_ids if swa_attention_layer_ids is not None else []
    )
    mc.swa_head_dim = swa_head_dim or head_dim
    mc.swa_v_head_dim = swa_v_head_dim or v_head_dim
    mc.get_num_kv_heads = lambda tp_size: num_kv_heads
    mc.get_swa_num_kv_heads = lambda tp_size: swa_num_kv_heads or num_kv_heads
    mc.hidden_size = hidden_size
    mc.dtype = "fake_bf16"  # torch._utils._element_size is patched in mock_cpu_env
    mc.hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    mc.hf_config.get_text_config = lambda: mc.hf_config
    mc.linear_attn_registry_result = None
    mr.model_config = mc

    mr.kv_cache_dtype = "fake_bf16"

    sa = SimpleNamespace()
    sa.kv_cache_dtype = "fake_bf16"
    sa.swa_full_tokens_ratio = swa_full_tokens_ratio
    sa.page_size = page_size
    sa.disable_radix_cache = disable_radix_cache
    sa.chunked_prefill_size = chunked_prefill_size
    sa.max_prefill_tokens = 16384
    sa.disable_overlap_schedule = disable_overlap_schedule
    sa.speculative_num_draft_tokens = speculative_num_draft_tokens
    sa.max_speculative_num_draft_tokens = (
        max_speculative_num_draft_tokens or speculative_num_draft_tokens
    )
    sa.speculative_algorithm = speculative_algorithm
    sa.speculative_num_steps = speculative_num_steps
    sa.speculative_eagle_topk = speculative_eagle_topk
    sa.disaggregation_mode = disaggregation_mode
    sa.max_running_requests = max_running_requests
    sa.disaggregation_decode_extra_slots = disaggregation_decode_extra_slots
    sa.enable_dsa_cache_layer_split = False
    sa.kv_cache_dtype = "auto"
    mr.server_args = sa

    spec = MagicMock()
    spec.is_eagle.return_value = False
    spec.is_standalone.return_value = False
    spec.is_dflash.return_value = False
    spec.is_dflash_family.return_value = False
    spec.is_none.return_value = True
    mr.spec_algorithm = spec

    mr.layer_info = SimpleNamespace(
        start_layer=0, end_layer=num_layers, num_effective_layers=num_layers
    )
    mr.ps = ParallelState.trivial()
    mr.pp_group = SimpleNamespace(rank_in_group=0)
    mr.spec_aux_config = SimpleNamespace(
        eagle_draft_num_layers=None,
        dflash_draft_num_layers=None,
        dflash_draft_total_num_kv_heads=None,
        dflash_draft_head_dim=None,
        dflash_draft_v_head_dim=None,
        dflash_draft_kv_element_size=None,
    )

    return mr


KV_SIZE = 2  # bf16


def _full_per_token(mr):
    mc = mr.model_config
    return mc.get_num_kv_heads(1) * (mc.head_dim + mc.v_head_dim) * KV_SIZE


def _swa_per_token(mr):
    mc = mr.model_config
    return mc.get_swa_num_kv_heads(1) * (mc.swa_head_dim + mc.swa_v_head_dim) * KV_SIZE


def _actual_memory_used(mr, config):
    """Compute actual memory consumed by the pool sizes in config."""
    mc = mr.model_config
    full_pt = _full_per_token(mr)
    swa_pt = _swa_per_token(mr)
    nf = len(mc.full_attention_layer_ids)
    ns = len(mc.swa_attention_layer_ids)

    if mr.is_hybrid_swa:
        full = config.full_max_total_num_tokens or 0
        swa = config.swa_max_total_num_tokens or 0
        return full * full_pt * nf + swa * swa_pt * ns
    else:
        return config.max_total_num_tokens * full_pt * (nf + ns)


class TestDefaultConfigurator(unittest.TestCase):
    """Default (MHA): available_bytes -> tokens, memory invariant holds."""

    def _run(self, available_bytes, page_size=1, **kwargs):
        mr = _make_model_runner(page_size=page_size, **kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, cfg, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_page_alignment(self):
        available = 10_000_000
        _, _, config = self._run(available, page_size=128)
        self.assertEqual(config.max_total_num_tokens % 128, 0)

    def test_constraint_respected(self):
        """calculate_pool_sizes_from_max_tokens respects the limit."""
        mr, cfg, config = self._run(10_000_000)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(100, page_size=1)
        self.assertEqual(constrained.max_total_num_tokens, 100)

    def test_constraint_page_aligned(self):
        mr, cfg, _ = self._run(10_000_000, page_size=128)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(1000, page_size=128)
        self.assertEqual(constrained.max_total_num_tokens, 896)  # 1000 // 128 * 128

    def test_no_swa_fields(self):
        _, _, config = self._run(10_000_000)
        self.assertIsNone(config.full_max_total_num_tokens)
        self.assertIsNone(config.swa_max_total_num_tokens)


class TestPrefillRuntimeWorkspaceReserve(unittest.TestCase):
    """KVCacheConfigurator._profile_available_bytes reserves transient prefill
    workspace for the target worker (chunk x hidden x dtype x 20), never for
    the draft worker. Built via __new__ to isolate the slack arithmetic from
    the dataclass's unrelated construction machinery.
    """

    def _profile(
        self,
        *,
        hidden_size,
        is_draft_worker,
        chunked_prefill_size=8192,
        kv_cache_dtype="auto",
    ):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        mr = _make_model_runner(
            hidden_size=hidden_size, chunked_prefill_size=chunked_prefill_size
        )
        mr.server_args.mem_fraction_static = 0.875
        mr.server_args.kv_cache_dtype = kv_cache_dtype
        cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
        cfg.device = "cpu"
        cfg.gpu_id = 0
        cfg.server_args = mr.server_args
        cfg.model_config = mr.model_config
        cfg.is_draft_worker = is_draft_worker
        cfg.mambaish_config = None
        with (
            mock_cpu_env(),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_available_gpu_memory",
                return_value=20.0,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_world_group",
                return_value=SimpleNamespace(world_size=1, cpu_group=None),
            ),
        ):
            return cfg._profile_available_bytes(pre_model_load_memory=24.0)

    def test_reserve_scales_with_chunk_and_hidden(self):
        # slack = 24 * (1 - 0.875) = 3.0; reserve = 8192*4096*2*20 / 2^30 = 1.25
        # (_profile_available_bytes returns bytes: GiB x 2^30)
        self.assertEqual(self._profile(hidden_size=0, is_draft_worker=False), 17 << 30)
        self.assertEqual(
            self._profile(hidden_size=4096, is_draft_worker=False),
            int(15.75 * (1 << 30)),
        )

    def test_chunk_fallback_to_max_prefill_tokens(self):
        # chunked_prefill_size=-1 -> max_prefill_tokens (16384): reserve = 2.5
        self.assertEqual(
            self._profile(
                hidden_size=4096, is_draft_worker=False, chunked_prefill_size=-1
            ),
            int(14.5 * (1 << 30)),
        )

    def test_draft_worker_not_reserved(self):
        self.assertEqual(
            self._profile(hidden_size=4096, is_draft_worker=True), 17 << 30
        )

    def test_kvarn_uses_reduced_multiplier(self):
        # KVarN: multiplier=3 -> reserve = 8192*4096*2*3 / 2^30 = 0.1875
        # slack=3.0, reserve=0.1875 -> available = 24 - 3.1875 = 20.8125
        self.assertEqual(
            self._profile(
                hidden_size=4096,
                is_draft_worker=False,
                kv_cache_dtype="kvarn_k4v2_g128",
            ),
            int(20.8125 * (1 << 30)),
        )


class TestHybridSWAConfigurator(unittest.TestCase):
    """Hybrid SWA: full/swa split, ratio, memory invariant."""

    def _make_swa_runner(self, full_layers=16, swa_layers=16, ratio=0.5, page_size=1):
        return _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
        )

    def _run(self, available_bytes, **kwargs):
        mr = self._make_swa_runner(**kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_ratio_respected(self):
        """swa_tokens ~= full_tokens * ratio (within page alignment)"""
        available = 10_000_000
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            mr, _, config = self._run(available, ratio=ratio, page_size=1)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(swa, int(full * ratio), f"ratio={ratio}")

    def test_ratio_with_page_alignment(self):
        """With page alignment, swa_tokens = align(full_tokens * ratio)"""
        available = 10_000_000
        mr, _, config = self._run(available, ratio=0.5, page_size=128)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_max_total_equals_full(self):
        """For hybrid, max_total_num_tokens = full_max_total_num_tokens"""
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_constraint_respected(self):
        """full_tokens = constrained value after re-run"""
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(200, page_size=1)
        self.assertEqual(config.full_max_total_num_tokens, 200)
        self.assertEqual(config.swa_max_total_num_tokens, 100)

    def test_constraint_memory_within_budget(self):
        """After constraint, memory <= original budget (but less than profiled due to constraint)."""
        available = 10_000_000
        mr, cfg, original = self._run(available, page_size=1)
        user_limit = original.full_max_total_num_tokens // 2
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(
                user_limit, mr.server_args.page_size
            )
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        # constrained should use roughly half the memory
        original_used = _actual_memory_used(mr, original)
        self.assertAlmostEqual(used / original_used, 0.5, delta=0.01)

    def test_different_layer_counts(self):
        """Asymmetric full/swa layer counts"""
        available = 10_000_000
        mr, _, config = self._run(available, full_layers=24, swa_layers=8, ratio=0.5)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.5),
        )

    def test_chunk_cache_cap_accounts_for_spec_topk_page_rounding(self):
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            disable_overlap_schedule=True,  # spec-v1: no double allocation
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # spec-v1 (overlap off): decode_alloc = max(ceil_align(3+4,4)*2,
        # ceil_align(5,4)) = 16. trailing = 8 + 20 + page(4) = 32; per req =
        # 32 + 16 = 48. Global prefill = 1*chunk(4) + page(4) = 8.
        # cap = 48 * 2 + 8 = 104 -> ceil_align(104, 4) = 104.
        self.assertEqual(config.swa_max_total_num_tokens, 104)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_doubles_decode_alloc_for_spec_v2_overlap(self):
        # Overlap on -> spec-v2: decode_alloc = 2 * get_alloc_len_per_decode =
        # 2 * max(steps*topk, max_draft) = 2 * max(6, 5) = 12 (page=1, since the
        # v2 allocator does not support page>1 & topk>1). trailing = 8 + 20 +
        # page(1) = 29; per req = 29 + 12 = 41. Global prefill =
        # 2*chunk(4) + page(1) = 9; cap = 41 * 2 + 9 = 91.
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=1,
            max_running_requests=2,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            disable_overlap_schedule=False,  # spec-v2: 2 * get_alloc_len_per_decode
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        self.assertEqual(config.swa_max_total_num_tokens, 91)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_drops_prefill_for_disagg_decode(self):
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=1000,
            sliding_window_size=4,
            page_size=1,
            max_running_requests=10,
            disaggregation_mode="decode",
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        # disagg decode drops the prefill term: per req = 4 + 1 + 4 + 1 = 10 (as above).
        self.assertEqual(config.swa_max_total_num_tokens, 100)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_prefill_holds_window_plus_chunk(self):
        # Non-decode (prefill) engine: each request keeps its decode footprint, while
        # in-flight chunked-prefill tokens are a global batch budget -- two chunks
        # under overlap.
        available = 1_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=16,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            disaggregation_mode="prefill",
            disable_overlap_schedule=False,  # overlap -> 2 chunks in flight
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # per req = trailing(window(8) + eviction(4) + page(4)) + decode_alloc(4)
        # = 20. Global prefill = 2*chunk(16) + page(4) = 36.
        # cap = 20 * max_running_requests(2) + 36 = 76.
        self.assertEqual(config.swa_max_total_num_tokens, 76)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_disagg_decode_pre_alloc(self):
        # decode adds disaggregation_decode_extra_slots in-transfer slots to the
        # request count (num_reserved_decode_tokens is a full-pool concern, not SWA).
        available = 2_000_000
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=1000,
            sliding_window_size=4,
            page_size=1,
            max_running_requests=10,
            disaggregation_mode="decode",
            disaggregation_decode_extra_slots=2,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        # active per req = 4 + 1 + 4 + 1 = 10 for the 10 running requests; the 2
        # in-transfer extra slots hold only window + page = 4 + 1 = 5 each.
        # cap = 10 * 10 + 5 * 2 = 110.
        self.assertEqual(config.swa_max_total_num_tokens, 110)
        self.assertLessEqual(_actual_memory_used(mr, config), available)


class TestAllSWAConfigurator(unittest.TestCase):
    """All-SWA (full_layers=0): special case."""

    def _run(self, available_bytes, ratio=0.5, page_size=1, **kwargs):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[],
            swa_attention_layer_ids=list(range(32)),
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=ratio,
            page_size=page_size,
            **kwargs,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_full_max_is_zero(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.full_max_total_num_tokens, 0)

    def test_max_total_equals_swa(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.swa_max_total_num_tokens)

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        swa_pt = _swa_per_token(mr)
        ns = len(mr.model_config.swa_attention_layer_ids)
        used = config.swa_max_total_num_tokens * swa_pt * ns
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_constraint_respected(self):
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(500, page_size=1)
        self.assertEqual(config.max_total_num_tokens, 500)
        self.assertEqual(config.swa_max_total_num_tokens, 500)


class TestEagleConfigurator(unittest.TestCase):
    """EAGLE: draft KV cache must be accounted for so total allocation fits in budget."""

    def test_eagle_does_not_exceed_budget(self):
        """Total memory (target + draft KV cache) must not exceed available."""
        available = 10_000_000
        num_layers = 32
        eagle_draft_num_layers = 4

        mr = _make_model_runner(num_layers=num_layers)
        mr.spec_algorithm.is_eagle.return_value = True
        mr.spec_algorithm.is_standalone.return_value = False
        mr.spec_algorithm.is_none.return_value = False
        mr.spec_aux_config.eagle_draft_num_layers = eagle_draft_num_layers

        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, 1)

        full_pt = _full_per_token(mr)
        total_layers = num_layers + eagle_draft_num_layers
        used = config.max_total_num_tokens * full_pt * total_layers
        self.assertLessEqual(used, available)


class TestDflashConfigurator(unittest.TestCase):
    """DFLASH: draft KV pool spans the full shared token-index space, so the
    target's token budget must cover target pool + draft pool together."""

    def _run(self, mr, available):
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, 1)
        return config

    def _make_dflash_runner(self, num_layers, draft_num_layers):
        mr = _make_model_runner(num_layers=num_layers)
        mr.spec_algorithm.is_none.return_value = False
        mr.spec_algorithm.is_dflash_family.return_value = True
        mr.spec_aux_config.dflash_draft_num_layers = draft_num_layers
        return mr

    def test_explicit_draft_cell_size_respects_budget(self):
        """Dense bf16 draft behind a cheap (e.g. compressed) target: reserving
        only the layer-ratio share of the target cell under-reserves the draft
        pool and OOMs. The explicit per-token sum must hold target+draft within
        budget."""
        available = 10_000_000
        num_layers = 32
        draft_num_layers = 6

        mr = self._make_dflash_runner(num_layers, draft_num_layers)
        # Draft pool costs ~3x the target pool per token (e.g. dense bf16
        # draft behind a KVarN-compressed target). Raw shape fields feed the
        # lazy cell-size computation in the configurator.
        mr.spec_aux_config.dflash_draft_total_num_kv_heads = 16
        mr.spec_aux_config.dflash_draft_head_dim = 128
        mr.spec_aux_config.dflash_draft_v_head_dim = 128
        mr.spec_aux_config.dflash_draft_kv_element_size = 2  # bf16

        config = self._run(mr, available)

        target_cell = _full_per_token(mr) * num_layers
        draft_cell = 16 * (128 + 128) * draft_num_layers * 2
        used = config.max_total_num_tokens * (target_cell + draft_cell)
        self.assertLessEqual(used, available)
        # Should still utilize most of the budget.
        self.assertGreater(used, available * 0.99)

    def test_layer_ratio_fallback_when_cell_size_unknown(self):
        """Without the raw draft KV shape, keep the layer-ratio heuristic
        (correct when draft and target share per-layer KV cost)."""
        available = 10_000_000
        num_layers = 32
        draft_num_layers = 4

        mr = self._make_dflash_runner(num_layers, draft_num_layers)
        assert mr.spec_aux_config.dflash_draft_total_num_kv_heads is None

        config = self._run(mr, available)

        full_pt = _full_per_token(mr)
        used = config.max_total_num_tokens * full_pt * (num_layers + draft_num_layers)
        self.assertLessEqual(used, available)


class TestDflashDraftCellSize(unittest.TestCase):
    """Draft KV cell-size helpers: dtype mirroring and the tp-aware formula."""

    def test_resolve_element_size_kvarn_target(self):
        # KVarN targets fall back to the draft model dtype (bf16 = 2 bytes)
        # when no explicit draft KV dtype is requested.
        import torch

        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_draft_kv_element_size,
        )

        self.assertEqual(
            resolve_dflash_draft_kv_element_size(
                draft_model_dtype=torch.bfloat16,
                server_args_kv_cache_dtype="kvarn_k4v2_g128",
                speculative_draft_attention_backend=None,
            ),
            2,
        )

    def test_resolve_element_size_explicit_draft_dtype_wins(self):
        # --speculative-draft-kv-cache-dtype fp8_e4m3 overrides the KVarN
        # fallback: 1 byte/element.
        import torch

        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_draft_kv_element_size,
        )

        self.assertEqual(
            resolve_dflash_draft_kv_element_size(
                draft_model_dtype=torch.bfloat16,
                server_args_kv_cache_dtype="kvarn_k4v2_g128",
                speculative_draft_attention_backend=None,
                speculative_draft_kv_cache_dtype="fp8_e4m3",
            ),
            1,
        )

    def test_resolve_element_size_fp8(self):
        import torch

        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_draft_kv_element_size,
        )

        self.assertEqual(
            resolve_dflash_draft_kv_element_size(
                draft_model_dtype=torch.bfloat16,
                server_args_kv_cache_dtype="fp8_e4m3",
                speculative_draft_attention_backend=None,
            ),
            1,
        )

    def test_resolve_element_size_fa4_forces_model_dtype(self):
        # fa4 draft attention needs K.dtype == Q.dtype; fp8 request ignored.
        import torch

        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_draft_kv_element_size,
        )

        self.assertEqual(
            resolve_dflash_draft_kv_element_size(
                draft_model_dtype=torch.bfloat16,
                server_args_kv_cache_dtype="fp8_e4m3",
                speculative_draft_attention_backend="fa4",
            ),
            2,
        )

    def test_compute_cell_size(self):
        # 8 kv heads (tp=1) * (128+128) * 6 layers * 2 bytes = 24576
        from sglang.srt.speculative.dflash_utils import (
            compute_dflash_draft_kv_cell_size_per_token,
        )

        self.assertEqual(
            compute_dflash_draft_kv_cell_size_per_token(
                draft_total_num_kv_heads=8,
                draft_head_dim=128,
                draft_v_head_dim=128,
                draft_num_layers=6,
                draft_kv_element_size=2,
                attn_tp_size=1,
            ),
            24576,
        )

    def test_compute_cell_size_tp_division(self):
        # 16 total heads, tp=2 -> 8 per rank.
        from sglang.srt.speculative.dflash_utils import (
            compute_dflash_draft_kv_cell_size_per_token,
        )

        self.assertEqual(
            compute_dflash_draft_kv_cell_size_per_token(
                draft_total_num_kv_heads=16,
                draft_head_dim=128,
                draft_v_head_dim=128,
                draft_num_layers=6,
                draft_kv_element_size=2,
                attn_tp_size=2,
            ),
            24576,
        )


class TestFactory(unittest.TestCase):
    def test_default_for_non_swa(self):
        mr = _make_model_runner(is_hybrid_swa=False)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, DefaultPoolConfigurator)

    def test_swa_for_hybrid(self):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(16)),
            swa_attention_layer_ids=list(range(16, 32)),
            swa_num_kv_heads=4,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                HybridSWAPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, HybridSWAPoolConfigurator)

    def test_chunk_cap_configurator_selection(self):
        # SWAChunkCapPoolConfigurator is selected only when max_running_requests is set.
        def _cfg(max_running_requests):
            mr = _make_model_runner(
                is_hybrid_swa=True,
                full_attention_layer_ids=[0],
                swa_attention_layer_ids=[1],
                swa_num_kv_heads=4,
                disable_radix_cache=True,
                chunked_prefill_size=4,
                sliding_window_size=8,
                max_running_requests=max_running_requests,
            )
            with mock_cpu_env():
                from sglang.srt.model_executor.pool_configurator import (
                    create_memory_pool_configurator,
                )

                return create_memory_pool_configurator(mr)

        from sglang.srt.model_executor.pool_configurator import (
            SWAChunkCapPoolConfigurator,
        )

        self.assertIsInstance(_cfg(2), SWAChunkCapPoolConfigurator)
        self.assertNotIsInstance(_cfg(None), SWAChunkCapPoolConfigurator)


if __name__ == "__main__":
    unittest.main()
