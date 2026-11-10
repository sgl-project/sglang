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
    resolve_prefix,
    ControlError,
    Limits,
    Status,
    TokensTarget,
)
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
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
from sglang.srt.mem_cache.unified_cache.control_state import CacheControlConfig
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of

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
        allocator.state = req.state
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=req,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            sliding_window_size=4,
            tree_slots=tree_slots_of(components, window=4, checkpoint_interval=64),
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
            cache.tree_core.control_state.clock = self.clock
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

    def inspect(self, control, selection=None):
        return control.executor.residency(
            resolve_prefix(
                selection or target(), control.context("test"), control.limits
            )
        )


class TestKVControlCache(KVCacheTestCase):
    def test_disabled_has_no_claim_state_and_rust_fallback_is_explicit(self):
        disabled, _ = self.make(enabled=False)
        self.assertIsNone(disabled.tree_core.control_state)
        with self.assertRaisesRegex(ControlError, "disabled"):
            disabled.create_kv_control_executor(engine_id="engine", engine_epoch="e")
        cache, control = self.make(backend="rust", session=False)
        self.assertEqual(cache._tree_core_backend, "python")
        self.assertEqual(control.executor.capabilities()["actions"], ())
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


if __name__ == "__main__":
    unittest.main()
