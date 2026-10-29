"""Declared payloads preserve identity through reset, relocation and checkpointing."""

import unittest
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    DerivedPartDecl,
    MHAGeometry,
    RequestPartDecl,
    StateGeometry,
    StateReplay,
    TensorPartDecl,
    kv_layer_decls,
    with_state_ple,
    with_state_replay,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _RequestKV:
    def __init__(self):
        self.req_pool_idx = None
        self.kv_allocated_len = 1

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None


def _build(
    *,
    extra=(),
    requests=(),
    derived=(),
    state=False,
    gamma=None,
    replay=None,
    ple=False,
    fused_draft=None,
    window_with_state=False,
):
    setup_allocator_context()
    dense = MHAGeometry(
        head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
    )
    geometry = StateGeometry(
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(1, 4, 8),
        temporal_dtype=torch.float32,
        extra_parts=extra,
    )
    if replay is not None:
        geometry = with_state_replay(
            geometry, replay, layer_ids=(1, 2), speculative_num_draft_tokens=gamma
        )
    if ple:
        geometry = with_state_ple(
            geometry,
            short_conv_layer_ids=(1, 2),
            short_conv_state_shape=(3, 4),
            ngram_context_len=3,
            ngram_eos_token_id=17,
            speculative_num_draft_tokens=gamma,
        )
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=dense,
        window_layer_ids=()
        if state and not window_with_state
        else (3 if state else 1,),
        window_geometry=dense,
        window=16,
        state_layer_ids=(1, 2) if state else (),
        state_geometry=geometry if state else None,
    )
    plan = plan_kv_storage(
        decls,
        page_size=4,
        speculative_num_draft_tokens=gamma,
        request_parts=requests,
        derived_parts=derived,
        fused_draft=fused_draft,
    )
    inputs = KVSizingInputs(
        full_tokens=256,
        window_tokens=128,
        state_slots=16,
        max_running_requests=3,
        context_len=64,
        alignment=1,
    )
    sized = size_plan(plan, inputs)
    ctx = UnifiedBuildContext(
        device="cpu",
        start_layer=0,
        end_layer=(4 if window_with_state else 3) if state else 2,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=3,
        model_context_len=64,
        state_cache_params=SimpleNamespace(layers=(1, 2)),
        speculative_num_draft_tokens=gamma,
    )
    return build_kv_storage(plan, sized, ctx), plan, sized


class TestDeclaredParts(CustomTestCase):
    def test_unified_sizing_rejects_an_unbudgeted_private_draft(self):
        from unittest.mock import patch

        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.runtime_context import get_context

        setup_allocator_context()
        cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
        cfg._unified_plan = None
        cfg._unified_sized = None
        cfg.is_draft_worker = False
        cfg.spec_aux_config = SimpleNamespace(draft_kv_num_layers=1)
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        cfg.spec_algorithm = SpeculativeAlgorithm.DSPARK
        cfg.page_size = 4
        cfg.attn_dp_size = 1
        cfg.model_config = SimpleNamespace(context_len=2048, sliding_window_size=16)
        dense = MHAGeometry(
            head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
        )
        declarations = kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=dense,
            window_layer_ids=(1,),
            window_geometry=dense,
            window=16,
        )
        with (
            get_context().override_server_args(max_running_requests=4),
            patch.object(
                KVCacheConfigurator,
                "_unified_window_and_state",
                return_value=(True, False),
            ),
            patch.object(
                KVCacheConfigurator, "_fused_draft_decision", return_value=None
            ),
            patch.object(
                KVCacheConfigurator, "_fused_draft_for_pool_factory", return_value=None
            ),
            patch.object(
                KVCacheConfigurator, "_kv_layer_decls", return_value=declarations
            ),
            patch.object(
                KVCacheConfigurator, "_unified_request_rows", return_value=(0, 0)
            ),
            patch.object(
                KVCacheConfigurator, "_apply_token_constraints", side_effect=lambda n: n
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "private draft pool is not budgeted"
            ):
                cfg._unified_config_from_budget(1 << 20)
            self.assertIsNone(cfg._unified_plan)
            # No-KV speculation must not be rejected merely for lacking placement.
            cfg.spec_aux_config.draft_kv_num_layers = 0
            self.assertGreater(
                cfg._unified_config_from_budget(1 << 20).max_total_num_tokens, 0
            )

    def test_kda_spec_replay_pd_is_rejected_before_state_allocation(self):
        from unittest.mock import patch

        from sglang.srt.configs.mamba_utils import (
            KimiLinearCacheParams,
            KimiLinearStateShape,
            Mamba2StateDType,
        )
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.runtime_context import get_context

        setup_allocator_context()
        params = KimiLinearCacheParams(
            shape=KimiLinearStateShape.create(
                tp_world_size=1, num_heads=2, head_dim=4, num_k_heads=2, head_k_dim=4
            ),
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=[1],
        )
        cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
        cfg.mambaish_config = SimpleNamespace(
            mamba2_cache_params=params, full_attention_layer_ids=[0]
        )
        cfg.is_hybrid_swa = False
        cfg.use_mla_backend = True
        cfg.model_config = SimpleNamespace(kv_lora_rank=8, qk_rope_head_dim=8)
        cfg.kv_cache_dtype = torch.bfloat16
        cfg.layer_info = SimpleNamespace(start_layer=0, end_layer=2)
        cfg.hybrid_gdn_config = None
        cfg.hybrid_kda_config = cfg.mambaish_config
        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.mamba_cache_chunk_size",
                return_value=4,
            ),
            get_context().override_server_args(
                enable_linear_replayssm_spec=True,
                disaggregation_mode="decode",
                speculative_algorithm="DFLASH",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "different state entry layouts"):
                cfg._kv_layer_decls()

        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.mamba_cache_chunk_size",
                return_value=4,
            ),
            get_context().override_server_args(
                enable_linear_replayssm_spec=True,
                disaggregation_mode="null",
                speculative_algorithm="DFLASH",
            ),
        ):
            self.assertEqual(len(cfg._kv_layer_decls()), 2)

    def test_window_shares_do_not_limit_shared_arena_borrowing(self):
        """A long FULL history may borrow bytes released by its short window."""
        _, plan, _ = _build()
        for cap in (None, 1300):
            with self.subTest(cap=cap):
                sized = size_plan(
                    plan,
                    KVSizingInputs(
                        full_tokens=1020,
                        window_tokens=1020,
                        max_total_tokens=cap,
                        max_running_requests=1,
                        context_len=2048,
                        alignment=1,
                    ),
                )
                storage = build_kv_storage(
                    plan,
                    sized,
                    UnifiedBuildContext(
                        device="cpu",
                        start_layer=0,
                        end_layer=2,
                        enable_memory_saver=False,
                        need_sort=False,
                        max_num_reqs=1,
                        model_context_len=2048,
                    ),
                )
                a = storage.allocator
                if cap is not None:
                    self.assertLessEqual(a.size_full, cap)
                    self.assertFalse(a.can_reserve(1500, 20))
                    continue
                self.assertGreaterEqual(a.size_full, 1500)
                self.assertTrue(a.can_reserve(1500, 20))
                chunks = []
                for i in range(375):
                    if i >= 5:
                        a.free_swa_segment(chunks[i - 5], start_pos=(i - 5) * 4)
                    chunk = a.alloc(4)
                    self.assertIsNotNone(chunk)
                    chunks.append(chunk)
                self.assertEqual(a.full_attn_allocator.allocated_count(), 1500)
                self.assertEqual(a.swa_attn_allocator.allocated_count(), 20)
                a.free(torch.cat(chunks))
                self.assertEqual(storage.accounting_violations(), [])

    def test_transfer_registration_excludes_unbacked_reservation(self):
        """Every view of one arena must register only its accessible prefix."""
        from unittest.mock import patch

        from sglang.srt.mem_cache.kv.decl import MLAGeometry
        from sglang.srt.mem_cache.kv.layout.mla import MLAView

        storage, _, _ = _build(state=True, window_with_state=True)
        plan = plan_kv_storage(
            kv_layer_decls(
                full_layer_ids=(0,),
                full_geometry=MLAGeometry(
                    kv_lora_rank=8, qk_rope_head_dim=8, kv_cache_dtype=torch.bfloat16
                ),
                state_layer_ids=(1,),
                state_geometry=StateGeometry(
                    conv_state_shapes=((3, 8),),
                    conv_dtype=torch.bfloat16,
                    temporal_state_shape=(1, 4, 8),
                    temporal_dtype=torch.float32,
                ),
            ),
            page_size=4,
        )
        spec = plan.owner().spec
        mla = MLAView(
            arena=storage.allocator.arena,
            spec=spec,
            kv_cache_dtype=torch.bfloat16,
            page_size=4,
        )
        arena = storage.allocator.arena
        prefix = arena.raw.numel() // 2
        with patch.object(arena, "accessible_bytes", prefix):
            regions = set()
            for view in (*storage.allocator.views.values(), mla):
                pointers, lengths, _ = view.get_contiguous_buf_infos()
                regions.update(zip(pointers, lengths))
            self.assertEqual(regions, {(arena.raw.data_ptr(), prefix)})

    def test_each_request_part_resets_only_on_fresh_acquisition(self):
        declarations = tuple(
            RequestPartDecl(part=p)
            for p in (
                TensorPartDecl(name="cursor", shape=(), dtype=torch.int32),
                TensorPartDecl(
                    name="ring", shape=(4, 8), dtype=torch.float32, layer_ids=(1, 3)
                ),
                TensorPartDecl(
                    name="context", shape=(3,), dtype=torch.int64, reset_value=7
                ),
            )
        )
        storage, _, sized = _build(requests=declarations)
        memory = storage._request_memory
        self.assertEqual(
            memory.nbytes,
            sum(t.bytes for t in sized.fixed if t.name != "request_table"),
        )
        req = SimpleNamespace(kv=_RequestKV())
        row = storage.acquire([req])[0]
        addresses = {n: t.data_ptr() for n, t in memory.tensors.items()}
        for part in memory.parts:
            tensor = memory.tensors[part.name].select(part.row_axis, row)
            self.assertTrue(torch.all(tensor == part.reset_value))
            tensor.fill_(19)
        self.assertEqual(storage.acquire([req]), [row])
        for part in memory.parts:
            self.assertTrue(
                torch.all(memory.tensors[part.name].select(part.row_axis, row) == 19)
            )
        storage.release_row(req)
        other = SimpleNamespace(kv=_RequestKV())
        self.assertEqual(storage.acquire([other]), [row])
        for part in memory.parts:
            self.assertTrue(
                torch.all(
                    memory.tensors[part.name].select(part.row_axis, row)
                    == part.reset_value
                )
            )
        self.assertEqual(
            addresses, {n: t.data_ptr() for n, t in memory.tensors.items()}
        )

    def test_total_budget_charges_fixed_storage_once(self):
        storage, plan, _ = _build(state=True, gamma=3)
        inp = KVSizingInputs(
            budget_bytes=1 << 20,
            context_len=64,
            max_running_requests=3,
            state_share=0.2,
            alignment=256,
        )
        sized = size_plan(plan, inp)
        self.assertLessEqual(sized.total_bytes + sized.fixed_bytes, inp.budget_bytes)
        self.assertLess(inp.budget_bytes - sized.total_bytes - sized.fixed_bytes, 256)
        expected_table = sized.request_rows * (64 * 4 + 4)
        self.assertEqual(sized.fixed[0].bytes, expected_table)
        self.assertEqual(
            sized.units("mamba"),
            int((sized.total_bytes - plan.sink_bytes()) * 0.2)
            // plan.sub_pool("mamba").spec.entry_bytes(),
        )
        smaller = size_plan(
            plan, msgspec.structs.replace(inp, budget_bytes=700000), frozen=sized
        )
        self.assertEqual(smaller.fixed, sized.fixed)
        self.assertEqual(smaller.request_rows, sized.request_rows)
        self.assertLess(smaller.total_bytes, sized.total_bytes)
        self.assertLessEqual(smaller.full_tokens, sized.full_tokens)
        self.assertLessEqual(smaller.max_running_requests, sized.max_running_requests)

    def test_derived_extent_is_in_the_total_budget(self):
        _, plan, _ = _build(
            derived=(
                DerivedPartDecl(
                    sub_pool="full",
                    part=TensorPartDecl(
                        name="compressed", shape=(32, 8), dtype=torch.float32
                    ),
                ),
            )
        )
        inp = KVSizingInputs(
            budget_bytes=1 << 20, context_len=64, max_running_requests=3, alignment=256
        )
        sized = size_plan(plan, inp)
        self.assertLessEqual(sized.total_bytes + sized.fixed_bytes, inp.budget_bytes)
        page_bytes = plan.owner().unit * plan.owner().spec.entry_bytes()
        blob = plan.derived_parts[0].part.row_bytes
        self.assertEqual(sized.fixed[-1].bytes, sized.total_bytes // page_bytes * blob)
        next_cost = (
            sized.total_bytes
            + 256
            + sized.fixed[0].bytes
            + (sized.total_bytes + 256) // page_bytes * blob
        )
        self.assertGreater(next_cost, inp.budget_bytes)

    def test_state_target_does_not_cap_the_state_band(self):
        storage, _, sized = _build(state=True)
        count = sized.units("mamba") + 1
        slots = storage.allocator.alloc_state_slots(count)
        self.assertIsNotNone(slots)
        self.assertEqual(slots.numel(), count)

    def test_full_cap_reclaims_cached_tokens_before_allocation(self):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        for state in (False, True):
            with self.subTest(state=state):
                storage, _, _ = _build(state=state, window_with_state=state)
                a = storage.allocator
                a.full_attn_allocator.max_live_tokens = 16
                cache = UnifiedRadixCache(
                    CacheInitParams(
                        disable=False,
                        enable_mamba_extra_buffer=state,
                        req_to_token_pool=storage.requests,
                        token_to_kv_pool_allocator=a,
                        page_size=4,
                        sliding_window_size=16,
                        tree_slots=tuple(
                            msgspec.structs.replace(slot, checkpoint_interval=4)
                            if slot.family == "state"
                            else slot
                            for slot in storage.tree_slots()
                        ),
                        tree_core_backend="python",
                    )
                )
                ids = a.alloc(16)
                self.assertIsNotNone(ids)
                cache.insert(
                    InsertParams(
                        key=RadixKey(list(range(16))),
                        value=ids,
                        mamba_value=a.alloc_state_slots(1) if state else None,
                    )
                )
                if not state:
                    self.assertTrue(
                        a.can_reserve(
                            4,
                            4,
                            full_evictable_tokens=cache.full_evictable_size(),
                            swa_evictable_tokens=cache.swa_evictable_size(),
                        )
                    )
                self.assertFalse(a.can_reserve(20, 4, empty_pool=True))
                self.assertFalse(a.ensure_capacity(1, 1))
                self.assertTrue(a.evict_to_free_tokens(cache, 4))
                self.assertLessEqual(a.full_attn_allocator.allocated_count(), 12)
                self.assertIsNotNone(a.alloc(4))
                self.assertEqual(storage.accounting_violations(), [])

    def test_replay_commits_translate_state_destinations_only(self):
        from unittest.mock import patch

        from sglang.srt.configs.qwen3_next import Qwen3NextConfig
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        for mode in ("gdn_circular", "gdn_fold", "kda_fold"):
            with self.subTest(mode=mode):
                storage, _, _ = _build(
                    state=True,
                    gamma=4,
                    replay=StateReplay(
                        decode=False,
                        speculative=True,
                        is_kda=mode == "kda_fold",
                        cache_len=8,
                        key_heads=1,
                    ),
                )
                a, req = storage.allocator, storage.requests
                pool = a.get_kvcache()
                a.state.bind_routes(pool.layer_router)
                a.state.pool.replayssm_spec_fold = mode != "gdn_circular"
                slots = a.alloc_state_slots(8)
                a.state_slots.free(slots[:4])
                active, track = slots[4:5], slots[7:8]
                physical_active = a.mamba_allocator.ids.translate(active)
                physical_track = a.mamba_allocator.ids.translate(track)
                self.assertFalse(torch.equal(active, physical_active))
                req.req_index_to_mamba_index_mapping[1] = active.int()[0]
                batch = SimpleNamespace(
                    req_pool_indices=torch.tensor([1]),
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    mamba_track_indices=track,
                    seq_lens=torch.tensor([15]),
                    tree_cache=SimpleNamespace(page_size=4),
                )
                runner = SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=Qwen3NextConfig(full_attention_interval=4),
                        linear_attn_registry_result=None,
                    ),
                    req_to_token_pool=req,
                    token_to_kv_pool=pool,
                    runner_ids=RunnerIds(
                        req_to_token=req.req_to_token,
                        token_to_kv_pool_allocator=a,
                        token_to_kv_pool=pool,
                        page_size=4,
                        device="cpu",
                    ),
                )
                outputs, conv_destinations = [], []

                def commit(**kw):
                    outputs.append(
                        (kw["state_batch_indices"], kw["mamba_track_indices"])
                    )
                    if "replay_indices" in kw:
                        torch.testing.assert_close(
                            kw["replay_indices"], batch.req_pool_indices
                        )

                def scatter(*args):
                    conv_destinations.append(args[2])

                target = {
                    "gdn_circular": "gdn_replayssm_spec_decode.commit_gdn_replayssm_circular",
                    "gdn_fold": "gdn_replayssm_spec_fold.commit_gdn_replayssm_fold_after_verify",
                    "kda_fold": "kda_replayssm_spec_decode.commit_kda_replayssm_after_verify",
                }[mode]
                with (
                    patch(
                        "sglang.kernels.ops.attention.linear." + target,
                        side_effect=commit,
                    ),
                    patch(
                        "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode.commit_gdn_replayssm_spec"
                    ),
                    patch(
                        "sglang.kernels.ops.mamba.mamba_state_scatter_triton.fused_conv_window_scatter_with_mask",
                        side_effect=scatter,
                    ),
                ):
                    commit_mamba_states_after_verify(
                        SimpleNamespace(model_runner=runner),
                        batch,
                        torch.tensor([2]),
                        torch.tensor([[0, 1, 2, 3]]),
                        4,
                        prepared_step_indices=(torch.tensor([1]), torch.tensor([0])),
                    )
                self.assertEqual(len(outputs), 1)
                torch.testing.assert_close(
                    outputs[0][0], physical_active, check_dtype=False
                )
                torch.testing.assert_close(
                    outputs[0][1], physical_track, check_dtype=False
                )
                if mode == "gdn_circular":
                    torch.testing.assert_close(
                        conv_destinations[0], physical_active, check_dtype=False
                    )
                    torch.testing.assert_close(
                        conv_destinations[1], physical_track, check_dtype=False
                    )
                self.assertIs(batch.mamba_track_indices, track)
                torch.testing.assert_close(
                    req.get_mamba_indices(batch.req_pool_indices),
                    active,
                    check_dtype=False,
                )

    def test_replay_algorithm_guard_covers_unified_and_static_construction(self):
        from sglang.srt.configs.qwen3_next import Qwen3NextConfig
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.runtime_context import get_exec, get_spec

        setup_allocator_context()
        cfg = KVCacheConfigurator.__new__(KVCacheConfigurator)
        cfg.hybrid_kda_config = None
        hf = Qwen3NextConfig(full_attention_interval=2, num_hidden_layers=2)
        cfg.mambaish_config = cfg.hybrid_gdn_config = hf
        cfg.is_hybrid_swa = False
        cfg.use_mla_backend = True
        cfg.model_config = SimpleNamespace(kv_lora_rank=8, qk_rope_head_dim=4)
        cfg.kv_cache_dtype = torch.bfloat16
        cfg.layer_info = SimpleNamespace(start_layer=0, end_layer=2)
        for algorithm in ("DFLASH", "DSPARK"):
            with (
                self.subTest(algorithm=algorithm),
                get_exec().mamba.override(enable_linear_replayssm_spec=True),
                get_spec().override(
                    speculative_algorithm=algorithm, speculative_num_draft_tokens=4
                ),
            ):
                with self.assertRaisesRegex(ValueError, "requires a KDA"):
                    cfg._kv_layer_decls()
                with self.assertRaisesRegex(ValueError, "requires a KDA"):
                    cfg._build_hybrid_req_pool(max_num_reqs=3, extra_max_context_len=4)
                cfg.hybrid_kda_config = object()
                cfg._validate_replay_spec_algorithm()
                cfg.hybrid_kda_config = None
        with (
            get_exec().mamba.override(enable_linear_replayssm_spec=True),
            get_spec().override(
                speculative_algorithm="EAGLE", speculative_num_draft_tokens=4
            ),
        ):
            self.assertEqual(len(cfg._kv_layer_decls()), 2)

    def test_full_cap_is_enforced_on_live_allocations(self):
        storage, _, _ = _build()
        owner = storage.allocator.full_attn_allocator
        owner.max_live_tokens = 16
        first = storage.allocator.alloc(12)
        self.assertIsNotNone(first)
        self.assertIsNone(storage.allocator.alloc(8))
        self.assertEqual(owner.allocated_count(), 12)
        self.assertIsNotNone(storage.allocator.alloc(4))
        self.assertIsNone(storage.allocator.alloc(4))

    def test_ple_views_reset_and_checkpoint_the_same_entry(self):
        storage, plan, _ = _build(state=True, gamma=3, ple=True)
        state = storage.allocator.state
        conv, ngram = state.short_conv_pool, state.ngram_pool
        self.assertTrue(conv.enabled and ngram.enabled)
        self.assertIs(conv.conv_state, state.pool.extra_views["short_conv"])
        self.assertIs(ngram.context, state.pool.extra_views["ngram"])
        ids = torch.tensor([3, 5])
        state.pool.clear_slots(ids)
        self.assertTrue(torch.all(ngram.get_context(ids) == 17))
        conv.layer_cache(2)[3].fill_(29)
        ngram.set_context(torch.tensor([3]), torch.tensor([[1, 2, 3]]))
        state.pool.copy_from(torch.tensor([3]), torch.tensor([7]))
        saved = state.pool.get_cpu_copy(torch.tensor([7]))
        state.pool.clear_slots(torch.tensor([7]))
        state.pool.load_cpu_copy(saved, torch.tensor([7]))
        self.assertTrue(torch.all(conv.layer_cache(2)[7] == 29))
        torch.testing.assert_close(
            ngram.get_context(torch.tensor([7])), torch.tensor([[1, 2, 3]])
        )
        self.assertEqual(conv.layer_intermediate_cache(2).shape, (4, 3, 3, 4))
        self.assertTrue(torch.all(ngram.intermediate_context == 17))
        self.assertEqual(len(plan.scratch), 4)

    def test_gdn_replay_request_parts_reset_on_row_reuse(self):
        storage, plan, _ = _build(
            state=True,
            gamma=3,
            replay=StateReplay(
                decode=False,
                speculative=True,
                is_kda=False,
                cache_len=8,
                key_heads=1,
            ),
        )
        view = storage.view("mamba")
        self.assertIsNone(view.mamba_cache.intermediate_ssm)
        self.assertEqual(len(plan.request_parts), 8)
        row = storage.take_rows(1)[0]
        for part in storage._request_memory.parts:
            with self.subTest(part=part.name):
                storage._request_memory.tensors[part.name].select(
                    part.row_axis, row
                ).fill_(23)
        storage.release_rows([row])
        self.assertEqual(storage.take_rows(1), [row])
        for part in storage._request_memory.parts:
            with self.subTest(part=part.name):
                self.assertTrue(
                    torch.all(
                        storage._request_memory.tensors[part.name].select(
                            part.row_axis, row
                        )
                        == part.reset_value
                    )
                )
        self.assertIs(
            view.replayssm_spec_write_pos,
            storage._request_memory.tensors["mamba.replayssm_spec_write_pos"],
        )
        self.assertEqual(view._checkpoint_extra_buffers, ())

    def test_replay_state_records_and_cursor_move_together(self):
        for speculative in (False, True):
            with self.subTest(speculative=speculative):
                storage, plan, _ = _build(
                    state=True,
                    gamma=3 if speculative else None,
                    replay=StateReplay(
                        decode=True,
                        speculative=speculative,
                        is_kda=speculative,
                        cache_len=8,
                        key_heads=1,
                    ),
                )
                view = storage.view("mamba")
                self.assertFalse(plan.request_parts)
                self.assertEqual(len(view.extra_parts), 7 if speculative else 4)
                for part in view.extra_parts:
                    view.extra_views[part.name].select(part.row_axis, 3).fill_(5)
                view.copy_from(torch.tensor([3]), torch.tensor([7]))
                saved = view.get_cpu_copy(torch.tensor([7]))
                view.clear_slots(torch.tensor([7]))
                view.load_cpu_copy(saved, torch.tensor([7]))
                for part in view.extra_parts:
                    self.assertTrue(
                        torch.all(
                            view.extra_views[part.name].select(part.row_axis, 7) == 5
                        )
                    )
                self.assertIs(
                    view.replayssm_write_pos, view.extra_views["replayssm_write_pos"]
                )

    def test_window_resume_keeps_the_handoff_boundary(self):
        storage, _, _ = _build()
        req = SimpleNamespace(kv=_RequestKV())
        row = storage.acquire([req])[0]
        tokens = storage.allocator.alloc(24)
        storage.requests.req_to_token[row, :24] = tokens.to(torch.int32)
        expected = (
            storage.allocator.translate_swa_indices_for_transfer(tokens[4:23])[::4] // 4
        )
        resume = storage.resume_units("swa", req, token_start=0, token_boundary=23)
        self.assertEqual((resume.token_start, resume.token_boundary), (4, 23))
        torch.testing.assert_close(resume.unit_ids, expected)
        resumed = storage.resume_units("swa", req, token_start=12, token_boundary=23)
        self.assertEqual((resumed.token_start, resumed.token_boundary), (12, 23))
        torch.testing.assert_close(resumed.unit_ids, expected[2:])

    def test_scratch_is_allocated_and_priced_from_the_plan(self):
        storage, plan, sized = _build(state=True, gamma=3)
        state = storage.view("mamba")
        scratch = storage.scratch
        self.assertEqual(
            scratch.nbytes,
            sum(t.bytes for t in sized.fixed if t.name != "request_table"),
        )
        self.assertEqual(set(scratch.tensors), {d.part.name for d in plan.scratch})
        self.assertIs(
            state.mamba_cache.intermediate_ssm,
            scratch.tensors["mamba.intermediate_ssm"],
        )
        self.assertEqual(state.mamba_cache.intermediate_ssm.shape, (2, 4, 3, 1, 4, 8))
        self.assertIs(
            state.mamba_cache.intermediate_conv_window[0],
            scratch.tensors["mamba.intermediate_conv_window.0"],
        )
        self.assertEqual(
            state.get_device_pool_infos()[0].buffer_info.extra_checkpoint_buffers, ()
        )

    def test_owner_only_plan_allocates_reuses_and_checkpoints_pages(self):
        setup_allocator_context()
        plan = plan_kv_storage(
            kv_layer_decls(
                full_layer_ids=(0, 1),
                full_geometry=MHAGeometry(
                    head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
                ),
            ),
            page_size=4,
        )
        sized = size_plan(
            plan,
            KVSizingInputs(
                budget_bytes=8192, max_running_requests=2, context_len=32, alignment=1
            ),
        )
        storage = build_kv_storage(
            plan,
            sized,
            UnifiedBuildContext(
                device="cpu",
                start_layer=0,
                end_layer=2,
                enable_memory_saver=False,
                need_sort=False,
                max_num_reqs=2,
                model_context_len=32,
            ),
        )
        allocator = storage.allocator
        available = allocator.available_size()
        ids = allocator.alloc(available)
        self.assertIsNotNone(ids)
        self.assertIsNone(allocator.alloc(4))
        physical = allocator.full_attn_allocator.ids.translate(ids)
        view = storage.view("full")
        view.k_buffer[0][physical] = 7
        saved = allocator.get_cpu_copy(ids)
        view.k_buffer[0][physical] = 0
        allocator.load_cpu_copy(saved, ids)
        self.assertTrue(torch.all(view.k_buffer[0][physical] == 7))
        allocator.free(ids)
        self.assertEqual(allocator.available_size(), available)
        self.assertEqual(allocator.verify_byte_accounting(), [])
        self.assertEqual(
            storage.plan.tree_slots()[0].component,
            storage.plan.owner().external.tree_component,
        )

    def test_full_cap_uses_widened_dcp_units(self):
        _, plan, _ = _build(state=True)
        sized = size_plan(
            plan,
            KVSizingInputs(
                budget_bytes=65536,
                state_slots=16,
                max_running_requests=3,
                context_len=64,
                max_total_tokens=32,
                alignment=1,
            ),
        )
        for dcp in (2, 4):
            with self.subTest(dcp=dcp), get_parallel().override(attn_dcp_size=dcp):
                storage = build_kv_storage(
                    plan,
                    sized,
                    UnifiedBuildContext(
                        device="cpu",
                        start_layer=0,
                        end_layer=3,
                        enable_memory_saver=False,
                        need_sort=False,
                        max_num_reqs=3,
                        model_context_len=64,
                        state_cache_params=SimpleNamespace(layers=(1, 2)),
                    ),
                )
                alloc = storage.allocator
                self.assertEqual(alloc.available_size(), 32 * dcp)
                first = alloc.alloc(16 * dcp)
                self.assertIsNotNone(first)
                self.assertEqual(alloc.available_size(), 16 * dcp)
                self.assertFalse(alloc.can_reserve(20 * dcp, 0))
                second = alloc.alloc(16 * dcp)
                self.assertIsNotNone(second)
                self.assertIsNone(alloc.alloc(4 * dcp))
                alloc.free(first)
                self.assertEqual(alloc.available_size(), 16 * dcp)
                alloc.free(second)
                self.assertEqual(alloc.verify_byte_accounting(), [])

    def test_entry_parts_follow_copy_move_and_nonzero_reset(self):
        parts = (
            TensorPartDecl(name="cursor", shape=(), dtype=torch.int32),
            TensorPartDecl(
                name="ring", shape=(4, 8), dtype=torch.bfloat16, layer_ids=(1, 2)
            ),
            TensorPartDecl(name="ngram", shape=(3,), dtype=torch.int64, reset_value=17),
        )
        storage, plan, _ = _build(state=True, extra=parts)
        view = storage.view("mamba")
        for p in parts:
            view.extra_views[p.name].select(p.row_axis, 3).fill_(23)
        view.copy_from(torch.tensor([3]), torch.tensor([5]))
        view._arena.move_units("mamba", torch.tensor([7]), torch.tensor([5]))
        for p in parts:
            self.assertTrue(
                torch.all(view.extra_views[p.name].select(p.row_axis, 7) == 23)
            )
        saved = view.get_cpu_copy(torch.tensor([7]))
        view.clear_slots(torch.tensor([7]))
        for p in parts:
            self.assertTrue(
                torch.all(
                    view.extra_views[p.name].select(p.row_axis, 7) == p.reset_value
                )
            )
        view.load_cpu_copy(saved, torch.tensor([7]))
        for p in parts:
            self.assertTrue(
                torch.all(view.extra_views[p.name].select(p.row_axis, 7) == 23)
            )
        info = view.get_device_pool_infos()[0].buffer_info
        self.assertEqual(len(info.extra_checkpoint_buffers), len(parts))
        info.validate(layer_ids=tuple(view.mamba_layer_ids))
        for part, tensor in zip(parts, info.extra_checkpoint_buffers):
            self.assertEqual(
                tensor.stride(0) * tensor.itemsize,
                plan.sub_pool("mamba").spec.entry_bytes(),
            )

    def test_runtime_budget_adapter_returns_the_declared_plan_sizes(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        _, plan, _ = _build(state=True)
        cfg = SimpleNamespace(
            _unified_plan=plan,
            _unified_sized=None,
            _unified_window_and_state=lambda: (False, True),
            _unified_request_rows=lambda **kwargs: (4, 0),
            _apply_token_constraints=lambda tokens: tokens,
            model_config=SimpleNamespace(context_len=64, sliding_window_size=None),
            attn_dp_size=1,
        )
        from sglang.srt.runtime_context import get_schedule

        with get_schedule().override(max_running_requests=3):
            config = KVCacheConfigurator._unified_config_from_budget(cfg, 1 << 20)
        self.assertEqual(config.max_total_num_tokens, cfg._unified_sized.full_tokens)
        self.assertEqual(config.unified_total_bytes, cfg._unified_sized.total_bytes)
        self.assertGreater(config.max_running_requests, 0)

    def test_state_slot_strategy_sizes_floor_and_acquisition_together(self):
        from sglang.srt.mem_cache.kv.state_policy import StateSlotStrategy

        _, plan, _ = _build(state=True)
        cases = (
            (StateSlotStrategy(), 3, 0),
            (StateSlotStrategy(extra_buffer=True, overlap=True), 5, 2),
            (
                StateSlotStrategy(
                    extra_buffer=True, overlap=True, lazy=True, skip_decode_lock=True
                ),
                4,
                1,
            ),
            (StateSlotStrategy(extra_buffer=True, decode_only=True), 2, 1),
            (StateSlotStrategy(radix_cache=False), 1, 0),
        )
        for strategy, slots, initial in cases:
            with self.subTest(strategy=strategy):
                sized = size_plan(
                    plan,
                    KVSizingInputs(
                        full_tokens=256,
                        state_slots=10,
                        context_len=64,
                        max_running_requests=8,
                        alignment=1,
                        state_strategy=strategy,
                        radix_cache=strategy.radix_cache,
                    ),
                )
                self.assertEqual(sized.max_running_requests, min(8, 10 // slots))
                self.assertEqual(
                    dict(sized.bs1_floor_terms)["mamba_bs1_state_slots"],
                    slots * plan.sub_pool("mamba").spec.entry_bytes(),
                )
                storage = build_kv_storage(
                    plan,
                    sized,
                    UnifiedBuildContext(
                        device="cpu",
                        start_layer=0,
                        end_layer=3,
                        enable_memory_saver=False,
                        need_sort=False,
                        max_num_reqs=8,
                        model_context_len=64,
                        state_cache_params=SimpleNamespace(layers=(1, 2)),
                    ),
                )
                self.assertEqual(storage.requests.mamba_initial_tracking_slots, initial)
                self.assertEqual(
                    storage.requests.mamba_ping_pong_track_buffer_size,
                    strategy.tracking_slots,
                )
                self.assertEqual(storage.requests.size, sized.max_running_requests)

    def test_retraction_carries_whole_entries_derived_pages_and_request_parts(self):
        from unittest.mock import patch

        from torch.utils._python_dispatch import TorchDispatchMode

        class BoundedGather(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if func == torch.ops.aten.index_select.default:
                    assert result.numel() * result.element_size() <= 256
                return result

        from sglang.srt.mem_cache.layout.fused_draft import (
            DenseDraftRegion,
            DraftStateGeometry,
            DraftStateRegion,
            FusedDraftPlacement,
        )

        for with_state in (False, True):
            with self.subTest(state=with_state):
                regions = {
                    "full": DenseDraftRegion(
                        lane_num=2,
                        head_num=1,
                        head_dim=3,
                        v_head_dim=5,
                        store_dtype=torch.bfloat16,
                    )
                }
                if with_state:
                    regions["mamba"] = DraftStateRegion(
                        lane_num=1,
                        state=DraftStateGeometry(
                            conv_state_shapes=((1, 3),),
                            conv_dtype=torch.bfloat16,
                            temporal_state_shape=(3,),
                            temporal_dtype=torch.float32,
                        ),
                    )
                placement = FusedDraftPlacement.from_counts(
                    counts={
                        name: [region.lane_num] for name, region in regions.items()
                    },
                    regions=regions,
                )
                storage, _, _ = _build(
                    state=with_state,
                    fused_draft=placement,
                    derived=(
                        DerivedPartDecl(
                            sub_pool="full",
                            part=TensorPartDecl(
                                name="index",
                                shape=(528,),
                                dtype=torch.uint8,
                                layer_ids=(0,),
                            ),
                        ),
                    ),
                    requests=(
                        RequestPartDecl(
                            part=TensorPartDecl(
                                name="pending", shape=(3,), dtype=torch.int32
                            )
                        ),
                    ),
                )
                allocator = storage.allocator
                source = allocator.alloc(16)
                state_ids = allocator.alloc_state_slots(1) if with_state else None
                (request_row,) = storage.take_rows(1)
                storage.request_memory.tensors["pending"][request_row].fill_(97)
                original = {}
                for n, band in enumerate(allocator.members):
                    ids = (
                        state_ids
                        if band.sub_pool_name == allocator.roles.state
                        else source
                    )
                    physical = band.ids.translate(ids.to(torch.int64))
                    value = n + 17
                    allocator.arena.entries(band.sub_pool_name)[physical] = value
                    original[band.sub_pool_name] = value
                blob = allocator.arena.page_parts["full"].tensors["index"]
                src_pages = allocator.full_attn_allocator.ids.virtual_to_physical[
                    source[::4] // 4
                ]
                blob[:, src_pages] = 83
                with (
                    patch(
                        "sglang.srt.mem_cache.kv.memory.snapshot._COPY_CHUNK_BYTES",
                        256,
                        create=True,
                    ),
                    BoundedGather(),
                ):
                    saved = allocator.get_cpu_copy(
                        source, mamba_indices=state_ids, req_pool_index=request_row
                    )
                # A distinct allocation changes both virtual and physical rows.
                destination = allocator.alloc(16)
                new_state = allocator.alloc_state_slots(1) if with_state else None
                (new_row,) = storage.take_rows(1)
                if not with_state:
                    allocator.free_swa_segment(destination[:4], start_pos=0)
                with (
                    patch(
                        "sglang.srt.mem_cache.kv.memory.snapshot._COPY_CHUNK_BYTES",
                        256,
                        create=True,
                    ),
                    BoundedGather(),
                ):
                    allocator.load_cpu_copy(
                        saved,
                        destination,
                        mamba_indices=new_state,
                        req_pool_index=new_row,
                    )
                for band in allocator.members:
                    ids = (
                        new_state
                        if band.sub_pool_name == allocator.roles.state
                        else destination
                    )
                    physical = band.ids.translate(ids.to(torch.int64))
                    physical = physical[physical > 0]
                    self.assertTrue(
                        torch.all(
                            allocator.arena.entries(band.sub_pool_name)[physical]
                            == original[band.sub_pool_name]
                        )
                    )
                pages = allocator.full_attn_allocator.ids.virtual_to_physical[
                    destination[::4] // 4
                ]
                self.assertTrue(torch.all(blob[:, pages] == 83))
                self.assertTrue(
                    torch.all(storage.request_memory.tensors["pending"][new_row] == 97)
                )

    def test_sparse_page_regions_and_request_ring_reuse(self):
        # Packed DSA pages put all 128-byte keys before the 4-byte scales;
        # QSA pages contain two compressed K rows. Neither is token-interleaved.
        dsa = TensorPartDecl(
            name="index", shape=(4 * (128 + 4),), dtype=torch.uint8, layer_ids=(0, 4)
        )
        qsa = TensorPartDecl(
            name="compressed",
            shape=(2, 2, 8),
            dtype=torch.bfloat16,
            layer_ids=(0, 4),
            slice_axis=1,
        )
        tails = tuple(
            RequestPartDecl(
                part=TensorPartDecl(
                    name=name, shape=(7, 128), dtype=torch.bfloat16, layer_ids=(0, 4)
                ),
                wire="dsa_tail",
            )
            for name in ("tail_k", "tail_score")
        )
        pending = RequestPartDecl(
            part=TensorPartDecl(
                name="pending", shape=(2, 2, 8), dtype=torch.bfloat16, layer_ids=(0, 4)
            ),
            wire="qsa_pending",
        )
        rope = RequestPartDecl(
            part=TensorPartDecl(name="rope", shape=(2, 3), dtype=torch.int64),
            wire="qsa_pending",
        )
        storage, plan, sized = _build(
            derived=(
                DerivedPartDecl(sub_pool="full", part=dsa, wire="dsa"),
                DerivedPartDecl(sub_pool="full", part=qsa, wire="qsa_compressed"),
            ),
            requests=(*tails, pending, rope),
        )
        wires = plan.wire_entries()
        self.assertEqual(
            [e.state_type for e in wires],
            ["swa", "dsa", "dsa_tail", "qsa_pending", "qsa_compressed"],
        )
        self.assertEqual(storage.regions("full", part="dsa").item_lens, (528, 528))
        region = storage.regions("full", part="qsa_compressed")
        self.assertEqual(region.item_lens, (64, 64))
        self.assertEqual(region.layer_ids, (0, 4))
        self.assertEqual(region.dim_per_tensor, (2, 2))
        self.assertEqual(region.slice_outer_counts, (2, 2))
        self.assertEqual(
            storage.regions("full", part="dsa_tail").item_lens, (1792,) * 4
        )
        self.assertEqual(
            storage.regions("full", part="qsa_pending").item_lens, (64, 64, 48)
        )
        arena = storage.allocator.arena
        for tensor in arena.page_parts["full"].tensors.values():
            tensor[:, 3].fill_(7)
        arena.move_units("full", torch.tensor([9]), torch.tensor([3]))
        for tensor in arena.page_parts["full"].tensors.values():
            self.assertTrue(torch.all(tensor[:, 9] == 7))
        for declared in plan.request_parts:
            storage._request_memory.tensors[declared.part.name].fill_(9)
        row = storage.take_rows(1)[0]
        for declared in plan.request_parts:
            part = declared.part
            tensor = storage._request_memory.tensors[part.name]
            self.assertTrue(
                torch.all(tensor.select(part.row_axis, row) == part.reset_value)
            )
        storage.release_rows([row])
        for declared in plan.request_parts:
            part = declared.part
            storage._request_memory.tensors[part.name].select(part.row_axis, row).fill_(
                11
            )
        self.assertEqual(storage.take_rows(1), [row])
        for declared in plan.request_parts:
            part = declared.part
            self.assertTrue(
                torch.all(
                    storage._request_memory.tensors[part.name].select(
                        part.row_axis, row
                    )
                    == part.reset_value
                )
            )

    def test_derived_page_blobs_move_with_physical_pages(self):
        part = TensorPartDecl(
            name="compressed_k", shape=(2, 8), dtype=torch.float32, layer_ids=(0,)
        )
        storage, plan, sized = _build(
            derived=(DerivedPartDecl(sub_pool="full", part=part),)
        )
        arena = storage.allocator.arena
        memory = arena.page_parts["full"]
        blob = memory.tensors[part.name]
        self.assertEqual(blob[0, 0].stride(), (8, 1))
        self.assertEqual(memory.rows, arena.extent_entries("full") // plan.page_size)
        self.assertEqual(
            memory.nbytes,
            sum(t.bytes for t in sized.fixed if t.name != "request_table"),
        )
        blob[:, 3].fill_(29)
        arena.move_units("full", torch.tensor([7]), torch.tensor([3]))
        self.assertTrue(torch.all(blob[:, 7] == 29))
        arena.zero_units("full", torch.tensor([7]))
        self.assertTrue(torch.all(blob[:, 7] == 0))
        self.assertTrue(torch.all(blob[:, 3] == 29))


if __name__ == "__main__":
    unittest.main()
