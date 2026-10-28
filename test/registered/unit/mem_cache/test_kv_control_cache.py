# SPDX-License-Identifier: Apache-2.0
"""Real CPU KV pools: copy priorities, eviction, splits and hybrid state."""

import unittest
from array import array
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.managers.kv_control import KVControl
from sglang.srt.managers.kv_control.contracts import (
    ContentTarget,
    ControlError,
    Limits,
    Location,
    OperationRef,
    SetPriority,
    Status,
    TokensTarget,
)
from sglang.srt.managers.kv_control.controller import ExecutionRequest
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.priority_claims import CacheControlConfig
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


def target(tokens=(1, 2, 3, 4), **kwargs):
    return TokensTarget(namespace="ns", token_ids=tuple(tokens), **kwargs)


def insert(cache, tokens, *, salt=None, extra_key=None):
    allocator = cache.token_to_kv_pool_allocator
    slots = allocator.alloc(len(tokens))
    assert slots is not None
    mamba = (
        cache.req_to_token_pool.mamba_allocator.alloc(1)
        if cache.is_mamba_enabled
        else None
    )
    result = cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens), cache_salt=salt, extra_key=extra_key),
            value=slots.to(torch.int64),
            mamba_value=mamba,
        )
    )
    if mamba is not None and result.mamba_exist:
        cache.req_to_token_pool.mamba_allocator.free(mamba)
    return cache.tree_core.node_by_id(result.last_device_node)


def match_len(cache, tokens, *, salt=None):
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens), cache_salt=salt))
    ).device_prefix_len


def register_session(cache, tokens, session):
    generation = cache.ensure_session_generation(session)
    leaf = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)))
    ).last_device_node
    cache.session_refs.register_session_ref(
        SimpleNamespace(
            session_id=session,
            session_generation=generation,
            session=None,
            origin_input_ids=array("q", tokens),
            output_ids=array("q"),
            extra_key=None,
        ),
        leaf=leaf,
    )


class KVCacheTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        scope = get_context().override_server_args(
            _mamba_cache_chunk_size=64, mamba_max_states_per_path=-1
        )
        scope.install()
        self.addCleanup(scope.restore)
        self.clock = Clock()
        self.message = 0

    def make(
        self,
        components=(ComponentType.FULL,),
        *,
        enabled=True,
        page_size=1,
        session=True,
        backend="python",
        config=None,
        limits=Limits(),
    ):
        if ComponentType.MAMBA in components:
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=16,
                n_groups=1,
                num_heads=2,
                head_dim=8,
                state_size=4,
                conv_kernel=4,
            )
            req = HybridReqToTokenPool(
                size=8,
                mamba_size=32,
                mamba_spec_state_size=8,
                max_context_len=128,
                device="cpu",
                enable_memory_saver=False,
                cache_params=Mamba2CacheParams(shape=shape, layers=[1]),
                mamba_layer_ids=[1],
                enable_mamba_extra_buffer=False,
            )
        else:
            req = ReqToTokenPool(
                size=8, max_context_len=128, device="cpu", enable_memory_saver=False
            )
        if ComponentType.SWA in components:
            pool = SWAKVPool(
                size=64,
                size_swa=64,
                page_size=page_size,
                dtype=torch.float16,
                head_num=1,
                head_dim=8,
                swa_attention_layer_ids=[1],
                full_attention_layer_ids=[0],
                device="cpu",
            )
            allocator = SWATokenToKVPoolAllocator(
                size=64,
                size_swa=64,
                page_size=page_size,
                dtype=torch.float16,
                device="cpu",
                kvcache=pool,
                need_sort=False,
            )
        else:
            pool = MHATokenToKVPool(
                size=64,
                page_size=page_size,
                dtype=torch.float16,
                head_num=1,
                head_dim=8,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
            )
            allocator_type = (
                PagedTokenToKVPoolAllocator if page_size > 1 else TokenToKVPoolAllocator
            )
            kwargs = {"page_size": page_size} if page_size > 1 else {}
            allocator = allocator_type(
                size=64,
                dtype=torch.float16,
                device="cpu",
                kvcache=pool,
                need_sort=False,
                **kwargs,
            )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=req,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            sliding_window_size=4,
            tree_components=components,
            enable_session_radix_cache=session,
            tree_core_backend=backend,
            kv_control_config=config or CacheControlConfig(namespace="ns")
            if enabled
            else None,
        )
        cache = UnifiedRadixCache(params)
        control = KVControl(
            "engine", "ns", enabled=True, clock=self.clock, limits=limits
        )
        if enabled:
            cache.tree_core.priority_claims.clock = self.clock
            control.executor = cache.create_kv_control_executor(
                engine_id="engine", engine_epoch=control.registry.engine_epoch
            )
        return (cache, control)

    def submit(self, control, kind, payload, *, caller="router", message=None):
        self.message += 1
        env = KvHintsEnvelope(
            protocol_version="0.1",
            message_id=message or str(self.message),
            actions=[
                KvHintAction(
                    action_id="a",
                    action_type=f"sglang.{kind}",
                    action_version="1.0",
                    payload=payload,
                )
            ],
        )
        return control.submit(env, control.context(caller)).results[0]

    def priority(
        self,
        control,
        selection=None,
        *,
        value=90,
        ttl=10,
        revision=1,
        claim="c",
        caller="router",
        message=None,
    ):
        payload = {
            "target": msgspec.json.decode(msgspec.json.encode(selection or target())),
            "location": {"tier": "gpu"},
            "claim_id": claim,
            "revision": revision,
            "value": value,
            "ttl_seconds": ttl if value is not None else None,
        }
        return self.submit(
            control, "set_priority", payload, caller=caller, message=message
        )

    def inspect(self, control, selection=None):
        return control.executor.residency(control.index.resolve(selection or target()))

    def local_policy_priority(
        self, control, *, value=90, ttl=20, revision=1, claim="c"
    ):
        payload = SetPriority(
            target=target(),
            location=Location(tier="gpu"),
            claim_id=claim,
            revision=revision,
            value=value,
            ttl_seconds=ttl if value is not None else None,
        )
        return control.executor.start(
            ExecutionRequest(
                ref=OperationRef(
                    context=control.context("router"),
                    message_id="local-policy",
                    action_id=claim,
                    operation_id="local-policy-" + claim,
                ),
                payload=payload,
                target=control.index.resolve(target()),
                origin="policy",
            )
        )


class TestKVControlCache(KVCacheTestCase):
    def test_explicit_priority_precedes_policy_and_has_separate_claim_capacity(self):
        cache, control = self.make(
            config=CacheControlConfig(
                namespace="ns", max_claims_per_copy=1, max_policy_claims_per_copy=1
            )
        )
        insert(cache, [1, 2, 3, 4])
        self.assertEqual(
            self.local_policy_priority(control, revision=10).status, Status.COMPLETED
        )
        self.assertEqual(
            self.priority(control, value=-15, ttl=1).status, Status.COMPLETED
        )
        self.assertEqual(
            [x.effective_priority for x in self.inspect(control).items], [-15]
        )
        self.assertEqual(
            self.local_policy_priority(control, claim="other").status, Status.REJECTED
        )
        self.clock.now += 2
        self.assertEqual(
            [x.effective_priority for x in self.inspect(control).items], [90]
        )
        self.local_policy_priority(control, value=None, revision=11)
        self.priority(control, value=None, revision=2)
        self.priority(control, value=100, revision=1)
        self.assertEqual(
            [x.effective_priority for x in self.inspect(control).items], [0]
        )

    def test_disabled_has_no_claim_state_and_rust_fallback_is_explicit(self):
        disabled, _ = self.make(enabled=False)
        self.assertIsNone(disabled.tree_core.priority_claims)
        with self.assertRaisesRegex(ControlError, "disabled"):
            disabled.create_kv_control_executor(engine_id="engine", engine_epoch="e")
        cache, control = self.make(backend="rust", session=False)
        self.assertEqual(cache._tree_core_backend, "python")
        self.assertEqual(
            control.executor.capabilities()["actions"], ("sglang.set_priority",)
        )
        result = self.submit(
            control,
            "copy_to",
            {
                "target": msgspec.json.decode(msgspec.json.encode(target())),
                "src": {"tier": "gpu"},
                "dst": {"tier": "host"},
            },
        )
        self.assertEqual(result.status, Status.UNSUPPORTED)

    def test_explicit_priority_beats_session_bias_and_actual_pool_frees(self):
        for components in (
            (ComponentType.FULL,),
            (ComponentType.FULL, ComponentType.SWA),
            (ComponentType.FULL, ComponentType.MAMBA),
            (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
        ):
            with self.subTest(components=components):
                cache, control = self.make(components)
                keep = insert(cache, [1, 2, 3, 4])
                victim = insert(cache, [7, 8, 9, 10])
                register_session(cache, [7, 8, 9, 10], "legacy-preferred")
                before = cache.token_to_kv_pool_allocator.available_size()
                mamba_before = (
                    cache.req_to_token_pool.mamba_allocator.available_size()
                    if cache.is_mamba_enabled
                    else None
                )
                result = self.priority(control)
                self.assertEqual(result.status, Status.COMPLETED, result.update)
                self.assertEqual(
                    cache.token_to_kv_pool_allocator.available_size(), before
                )
                self.assertTrue(all((cd.lock_ref == 0 for cd in keep.component_data)))
                if ComponentType.MAMBA in components:
                    params = EvictParams(mamba_num=1)
                elif ComponentType.SWA in components:
                    params = EvictParams(swa_num_tokens=4)
                else:
                    params = EvictParams(num_tokens=4)
                cache.evict(params)
                self.assertEqual(match_len(cache, [7, 8, 9, 10]), 0)
                self.assertEqual(match_len(cache, [1, 2, 3, 4]), 4)
                self.assertNotIn(victim.id, cache.tree_core._node_arena)
                self.assertEqual(
                    cache.token_to_kv_pool_allocator.available_size(), before + 4
                )
                if mamba_before is not None:
                    self.assertEqual(
                        cache.req_to_token_pool.mamba_allocator.available_size(),
                        mamba_before + 1,
                    )
                cache.sanity_check()

    def test_independent_expiry_negative_priority_and_lru_ties(self):
        cache, control = self.make()
        insert(cache, [1, 2, 3, 4])
        self.priority(control, value=90, ttl=10, caller="A")
        self.priority(control, value=20, ttl=60, caller="B")
        self.assertEqual(self.inspect(control).items[0].effective_priority, 90)
        self.clock.now += 11
        match_len(cache, [1, 2, 3, 4])
        self.assertEqual(self.inspect(control).items[0].effective_priority, 20)
        self.clock.now += 50
        self.assertEqual(self.inspect(control).items[0].effective_priority, 0)
        older = insert(cache, [9, 8, 7, 6])
        self.priority(control, value=-5, revision=2)
        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(match_len(cache, [1, 2, 3, 4]), 0)
        self.assertIn(older.id, cache.tree_core._node_arena)
        insert(cache, [1, 2, 3, 4])
        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(match_len(cache, [9, 8, 7, 6]), 0)
        cache.sanity_check()

    def test_equal_stale_clear_and_conflict_are_atomic_across_copies(self):
        cache, control = self.make()
        insert(cache, [1, 2, 3, 4])
        self.priority(control, target(end_token=2), value=10)
        self.clock.now += 4
        self.assertEqual(self.priority(control, value=20).status, Status.REJECTED)
        self.assertEqual(
            [x.effective_priority for x in self.inspect(control).items], [10, 0]
        )
        self.priority(control, target(end_token=2), value=10)
        self.clock.now += 7
        self.assertEqual(
            [x.effective_priority for x in self.inspect(control).items], [0, 0]
        )
        self.priority(control, value=None, revision=3)
        self.priority(control, value=99, revision=2)
        self.assertEqual(
            [x.effective_priority for x in self.inspect(control).items], [0, 0]
        )
        self.assertEqual(
            self.priority(control, value=99, revision=3).status, Status.REJECTED
        )
        cache.sanity_check()

    def test_split_preserves_coverage_revision_and_original_expiry(self):
        cache, control = self.make()
        insert(cache, list(range(8)))
        self.priority(control, target(range(8), start_token=2, end_token=6))
        self.clock.now += 4
        insert(cache, [0, 1, 2, 99])
        rows = self.inspect(control, target(range(8))).items
        self.assertEqual(
            [
                (x.descriptor.start_token, x.descriptor.end_token, x.effective_priority)
                for x in rows
            ],
            [(0, 2, 0), (2, 3, 90), (3, 6, 90), (6, 8, 0)],
        )
        self.clock.now += 7
        self.assertTrue(
            all(
                (
                    x.effective_priority == 0
                    for x in self.inspect(control, target(range(8))).items
                )
            )
        )
        self.assertEqual(
            cache.tree_core.priority_claims.capacity()["reserved_copy_units"], 4
        )
        cache.sanity_check()

    def test_eviction_recompute_slot_reuse_and_reset_end_claim_lifetimes(self):
        for components in (
            (ComponentType.FULL,),
            (ComponentType.FULL, ComponentType.SWA),
            (ComponentType.FULL, ComponentType.MAMBA),
        ):
            with self.subTest(components=components):
                cache, control = self.make(components)
                node = insert(cache, [1, 2, 3, 4])
                self.priority(control, message="original")
                receipt = cache.inc_lock_ref(node.id)
                self.assertEqual(
                    cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted, 0
                )
                cache.dec_lock_ref(node.id, receipt.to_dec_params())
                cache.evict(EvictParams(num_tokens=4))
                self.assertEqual(
                    cache.tree_core.priority_claims.capacity()["copies"], 0
                )
                cache.token_to_kv_pool_allocator.clear()
                new = insert(cache, [1, 2, 3, 4])
                self.assertNotEqual(new.id, node.id)
                self.priority(control, message="original")
                self.assertTrue(
                    all(
                        (x.effective_priority == 0 for x in self.inspect(control).items)
                    )
                )
                self.priority(control, revision=0)
                self.assertTrue(
                    all(
                        (
                            x.effective_priority == 90
                            for x in self.inspect(control).items
                        )
                    )
                )
                cache.reset()
                self.assertEqual(
                    cache.tree_core.priority_claims.capacity()["copies"], 0
                )
                self.assertEqual(
                    self.priority(control, revision=2).update.code, "stale_epoch"
                )

    def test_real_auxiliary_geometry_and_descriptor_survive_tree_split(self):
        cache, control = self.make(
            (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
        )
        insert(cache, list(range(8)))
        result = self.priority(control, target(range(8)))
        self.assertEqual(result.status, Status.COMPLETED, result.update)
        mamba = [
            o for o in result.update.target_outcomes if o.coverage.component == "mamba"
        ]
        self.assertEqual(
            [
                (o.coverage_kind, o.coverage.start_token, o.coverage.end_token)
                for o in mamba
            ],
            [("checkpoint", 8, 8)],
        )
        swa = [
            o for o in result.update.target_outcomes if o.coverage.component == "swa"
        ]
        self.assertTrue(
            all(
                (
                    o.coverage_kind == "window"
                    and o.window_size == 4
                    and (o.coverage.start_token >= 4)
                    for o in swa
                )
            )
        )
        snapshot = self.inspect(control, target(range(8)))
        windows = tuple(
            (
                row.descriptor
                for row in snapshot.items
                if row.descriptor.component == "swa"
            )
        )
        window_target = ContentTarget(
            namespace="ns", token_count=8, components=("swa",), spans=windows
        )
        self.assertEqual(
            self.priority(control, window_target, revision=2).status, Status.COMPLETED
        )
        checkpoints = tuple(
            (
                row.descriptor
                for row in snapshot.items
                if row.descriptor.component == "mamba"
            )
        )
        descriptor = ContentTarget(
            namespace="ns", token_count=8, components=("mamba",), spans=checkpoints
        )
        insert(cache, [0, 1, 2, 3, 4, 77])
        self.assertEqual(
            self.priority(control, descriptor, revision=2).status, Status.COMPLETED
        )
        self.assertEqual(len(self.inspect(control, descriptor).items), 1)
        cache.sanity_check()

    def test_cross_component_cascade_respects_explicit_mamba_value(self):
        cache, control = self.make(
            (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
        )
        insert(cache, [1, 2, 3, 4])
        insert(cache, [9, 8, 7, 6])
        checkpoint = next(
            (
                r.descriptor
                for r in self.inspect(control).items
                if r.descriptor.component == "mamba"
            )
        )
        selection = ContentTarget(
            namespace="ns", token_count=4, components=("mamba",), spans=(checkpoint,)
        )
        self.assertEqual(self.priority(control, selection).status, Status.COMPLETED)
        cache.evict(EvictParams(swa_num_tokens=4))
        self.assertEqual(match_len(cache, [1, 2, 3, 4]), 4)
        self.assertEqual(match_len(cache, [9, 8, 7, 6]), 0)
        insert(cache, [9, 8, 7, 6])
        self.assertEqual(
            self.priority(control, selection, value=-20, revision=2).status,
            Status.COMPLETED,
        )
        cache.evict(EvictParams(mamba_num=1))
        self.assertEqual(match_len(cache, [1, 2, 3, 4]), 0)
        self.assertEqual(match_len(cache, [9, 8, 7, 6]), 4)
        cache.sanity_check()

    def test_swa_partial_history_is_reported_and_missing_checkpoint_is_not_success(
        self,
    ):
        cache, control = self.make(
            (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
        )
        insert(cache, list(range(8)))
        result = self.priority(control, target(range(8), end_token=4))
        self.assertEqual(result.status, Status.PARTIAL, result.update)
        missing = [
            o for o in result.update.target_outcomes if o.status == Status.MISSING
        ]
        self.assertEqual(
            [
                (
                    o.coverage.component,
                    o.coverage_kind,
                    o.coverage.start_token,
                    o.coverage.end_token,
                )
                for o in missing
            ],
            [("mamba", "checkpoint", 4, 4)],
        )
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
