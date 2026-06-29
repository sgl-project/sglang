"""Unit tests for MLX attention discovery and generic cache handling."""

from __future__ import annotations

import importlib.util
import unittest
from collections import deque
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    from mlx_lm.models.cache import ArraysCache

    import sglang.srt.hardware_backend.mlx.aot as mlx_aot
    from sglang.srt.hardware_backend.mlx.aot import (
        MlxAOTKernelSet,
        MlxAOTRoPEKernel,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        BatchedDecodeContext,
        ContiguousAttentionKVCache,
        MlxAttentionKVPool,
        MLXAttentionWrapper,
        MlxAuxiliaryStateComponent,
        MlxAuxiliaryStatePool,
        MlxAuxiliaryStateReqToTokenPool,
        MlxModelCacheLayout,
        find_attention_layers,
        is_attention_module,
        patch_model_attention,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import (
        MlxModelRunner,
        MlxPendingDecode,
    )
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
        MlxPendingJob,
        SchedulerMlxOverlapMixin,
    )
    from sglang.srt.managers.scheduler_components import (
        batch_result_processor as batch_result_processor_module,
    )
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _set_runner_cache_layout(
    runner,
    *,
    num_layers: int,
    attention_layer_indices: list[int],
    attention_modules: dict[int, object] | None = None,
) -> None:
    attention_modules = attention_modules or {}
    attention_set = set(attention_layer_indices)
    layers = []
    attrs = []
    for layer_idx in range(num_layers):
        if layer_idx in attention_set:
            attrs.append("self_attn")
            layers.append(
                SimpleNamespace(self_attn=attention_modules.get(layer_idx, object()))
            )
        else:
            attrs.append(None)
            layers.append(SimpleNamespace())
    runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(layers, attrs)


def _set_runner_decode_context_defaults(runner) -> None:
    runner._aot_kernels = MlxAOTKernelSet()
    runner._attention_kv_pool = None
    runner._req_pool_idx = {}
    runner._req_to_token_pool = None


def _set_dummy_server_args_for_auxiliary_state_tests() -> None:
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = 64
    set_global_server_args_for_scheduler(server_args)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxAttentionPatching(unittest.TestCase):
    def test_standard_attention_is_patched_once(self):
        model = FakeModel(
            [
                FakeLayer("self_attn", FakeAttention()),
                FakeLayer("self_attn", FakeAttention()),
            ]
        )

        layers, attrs = find_attention_layers(model)

        self.assertEqual(len(layers), 2)
        self.assertEqual(attrs, ["self_attn", "self_attn"])
        self.assertEqual(patch_model_attention(model), 2)
        self.assertIsInstance(model.layers[0].self_attn, MLXAttentionWrapper)
        self.assertIsInstance(model.layers[1].self_attn, MLXAttentionWrapper)
        self.assertEqual(patch_model_attention(model), 0)

    def test_alias_head_names_are_supported(self):
        model = FakeModel([FakeLayer("attention", FakeAttention(use_aliases=True))])

        _, attrs = find_attention_layers(model)

        self.assertEqual(attrs, ["attention"])
        self.assertEqual(patch_model_attention(model), 1)
        self.assertIsInstance(model.layers[0].attention, MLXAttentionWrapper)

    def test_aot_rope_kernel_build_uses_head_aliases(self):
        attn = FakeAttention(use_aliases=True)
        attn.rope = SimpleNamespace(dims=2, traditional=False, base=10000.0)
        original_loader = mlx_aot._load_metal_rope_pool_fused
        mlx_aot._load_metal_rope_pool_fused = lambda: object()
        try:
            kernel = mlx_aot._build_rope_kernel(
                mlx_aot.MlxAOTKernelBuildInputs(
                    sample_attn=attn,
                    n_kv_heads=1,
                    head_dim=2,
                )
            )
        finally:
            mlx_aot._load_metal_rope_pool_fused = original_loader

        self.assertTrue(kernel.enabled)
        self.assertEqual(kernel.config["num_qo_heads"], 2)

    def test_auxiliary_state_model_returns_per_layer_attention_attrs(self):
        model = FakeModel(
            [
                FakeLayer("linear_attn", ProjectionOnlyMixer()),
                FakeLayer("self_attn", FakeAttention()),
                FakeLayer("linear_attn", ProjectionOnlyMixer()),
            ]
        )

        _, attrs = find_attention_layers(model)

        self.assertEqual(attrs, [None, "self_attn", None])
        self.assertEqual(patch_model_attention(model), 1)
        self.assertFalse(isinstance(model.layers[0].linear_attn, MLXAttentionWrapper))
        self.assertIsInstance(model.layers[1].self_attn, MLXAttentionWrapper)

    def test_projection_only_mixer_is_not_attention(self):
        self.assertFalse(is_attention_module(ProjectionOnlyMixer()))

    def test_cache_layout_separates_attention_and_auxiliary_layers(self):
        layout = MlxModelCacheLayout.from_attention_discovery(
            [object(), object(), object(), object()],
            [None, "self_attn", None, "self_attn"],
        )

        self.assertEqual(layout.num_layers, 4)
        self.assertEqual(layout.attention_layer_indices, (1, 3))
        self.assertEqual(layout.auxiliary_layer_indices, (0, 2))
        self.assertEqual(layout.attention_pool_index(1), 0)
        self.assertEqual(layout.attention_pool_index(3), 1)
        self.assertTrue(layout.has_auxiliary_state)

    def test_gated_query_projection_keeps_attention_width(self):
        inner = FakeGatedAttention()
        wrapper = MLXAttentionWrapper(inner, layer_idx=0)
        cache = ContiguousAttentionKVCache(
            n_kv_heads=1, head_dim=2, max_seq_len=4, dtype=mx.float32
        )
        ctx = BatchedDecodeContext(
            batch_size=1,
            seq_lens=[0],
            attention_layer_caches=[[cache]],
        )

        out = wrapper._batched_decode(mx.zeros((1, 1, 4), dtype=mx.float32), ctx)
        mx.eval(out)

        self.assertEqual(out.shape, (1, 1, 4))
        self.assertEqual(inner.o_proj.last_input_shape, (1, 1, 4))

    def test_attn_config_uses_float_dtype_for_quantized_projection(self):
        runner = object.__new__(MlxModelRunner)
        attn = FakeAttention()
        attn.k_proj.weight = mx.zeros((2, 4), dtype=mx.uint32)
        _set_runner_cache_layout(
            runner,
            num_layers=1,
            attention_layer_indices=[0],
            attention_modules={0: attn},
        )

        n_kv_heads, head_dim, dtype = MlxModelRunner._get_attn_config(runner)

        self.assertEqual(n_kv_heads, 1)
        self.assertEqual(head_dim, 2)
        self.assertEqual(dtype, mx.float32)

    def test_attn_config_rejects_heterogeneous_kv_shapes(self):
        runner = object.__new__(MlxModelRunner)
        first = FakeAttention()
        second = FakeAttention()
        second.n_kv_heads = 2
        _set_runner_cache_layout(
            runner,
            num_layers=2,
            attention_layer_indices=[0, 1],
            attention_modules={0: first, 1: second},
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "uniform softmax-attention KV shape",
        ):
            MlxModelRunner._get_attn_config(runner)

    def test_attn_config_rejects_sliding_window_attention(self):
        runner = object.__new__(MlxModelRunner)
        _set_runner_cache_layout(
            runner,
            num_layers=1,
            attention_layer_indices=[0],
            attention_modules={0: FakeAttention()},
        )
        runner._cache_layout.layers[0].use_sliding = True

        with self.assertRaisesRegex(
            NotImplementedError,
            "sliding-window attention",
        ):
            MlxModelRunner._get_attn_config(runner)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxAuxiliaryStateRunnerCache(unittest.TestCase):
    def test_dense_prefill_keeps_pool_backed_radix_path(self):
        runner = object.__new__(MlxModelRunner)
        runner.model = FakeDenseModel(num_layers=2)
        _set_runner_cache_layout(
            runner,
            num_layers=2,
            attention_layer_indices=[0, 1],
        )
        runner._max_seq_len = 8
        runner._cache_pool = []
        runner.disable_radix_cache = False
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=8,
            num_layers=2,
            n_kv_heads=1,
            head_dim=2,
            dtype=mx.float32,
        )
        runner._req_to_token_pool = None
        runner._req_caches = {}
        runner._req_token_ids = {}
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        prefix_slots = mx.array([2, 3], dtype=mx.int32)
        k_prefix = mx.stack(
            [
                mx.ones((2, 1, 2), dtype=mx.float32) * 10,
                mx.ones((2, 1, 2), dtype=mx.float32) * 20,
            ]
        )
        runner._attention_kv_pool.set_kv_all_layers(
            prefix_slots, k_prefix, k_prefix * 2
        )
        mx.eval(*runner._attention_kv_pool.all_buffers())

        pending = runner.prefill_start(
            req_id="r0",
            new_token_ids=[13],
            full_token_ids=[11, 12, 13],
            prefix_slot_ids=[2, 3],
            new_slot_ids=[4],
            req_pool_idx=0,
        )
        MlxModelRunner._eval_with_cache(pending.lazy_token, pending.cache)
        mx.eval(*runner._attention_kv_pool.all_buffers())
        runner.prefill_finalize(pending)

        self.assertEqual(runner.model.seen_inputs, [[[13]]])
        self.assertEqual(runner.model.seen_offsets, [[2, 2]])
        self.assertEqual(pending.synced_offset, 3)
        self.assertTrue(
            all(isinstance(c, ContiguousAttentionKVCache) for c in pending.cache)
        )
        layer0_k, layer0_v = runner._attention_kv_pool.get_kv(
            0, mx.array([4], dtype=mx.int32)
        )
        layer1_k, layer1_v = runner._attention_kv_pool.get_kv(
            1, mx.array([4], dtype=mx.int32)
        )
        mx.eval(layer0_k, layer0_v, layer1_k, layer1_v)
        self.assertEqual(layer0_k.tolist(), [[[1.0, 1.0]]])
        self.assertEqual(layer0_v.tolist(), [[[2.0, 2.0]]])
        self.assertEqual(layer1_k.tolist(), [[[2.0, 2.0]]])
        self.assertEqual(layer1_v.tolist(), [[[4.0, 4.0]]])

    def test_dense_decode_uses_batched_attention_for_single_and_multi_request(self):
        for req_ids in (["r0"], ["r0", "r1"]):
            with self.subTest(req_ids=req_ids):
                runner = object.__new__(MlxModelRunner)
                _set_runner_cache_layout(
                    runner,
                    num_layers=1,
                    attention_layer_indices=[0],
                )
                runner._req_caches = {rid: [object()] for rid in req_ids}
                runner._req_token_ids = {
                    rid: [idx + 10] for idx, rid in enumerate(req_ids)
                }
                calls = []

                def fake_batched(caches, batched_input, helper_req_ids):
                    calls.append(
                        (len(caches), batched_input.tolist(), list(helper_req_ids))
                    )
                    return mx.array(list(range(len(caches))), dtype=mx.int32)

                def fail_native(*args, **kwargs):
                    raise AssertionError("dense decode should use batched attention")

                runner._decode_with_batched_attention = fake_batched
                runner._decode_with_native_cache = fail_native

                pending = runner.decode_batch_start(req_ids)

                self.assertEqual(
                    calls,
                    [
                        (
                            len(req_ids),
                            [[idx + 10] for idx in range(len(req_ids))],
                            req_ids,
                        )
                    ],
                )
                self.assertEqual(
                    pending.lazy_tokens.tolist(),
                    list(range(len(req_ids))),
                )

    def test_dense_chained_decode_uses_batched_attention_for_single_request(self):
        runner = object.__new__(MlxModelRunner)
        _set_runner_cache_layout(
            runner,
            num_layers=1,
            attention_layer_indices=[0],
        )
        calls = []

        def fake_batched(caches, batched_input, helper_req_ids):
            calls.append((len(caches), batched_input.tolist(), list(helper_req_ids)))
            return mx.array([8], dtype=mx.int32)

        def fail_native(*args, **kwargs):
            raise AssertionError("dense chained decode should use batched attention")

        runner._decode_with_batched_attention = fake_batched
        runner._decode_with_native_cache = fail_native
        prev = MlxPendingDecode(
            lazy_tokens=mx.array([7], dtype=mx.int32),
            req_ids=["r0"],
            caches=[[object()]],
        )

        pending = runner.decode_batch_start_chained(prev)

        self.assertEqual(calls, [(1, [[7]], ["r0"])])
        self.assertEqual(pending.lazy_tokens.tolist(), [8])

    def test_mlx_scheduler_init_overlap_keeps_future_map_relay(self):
        from sglang.srt.managers import scheduler as scheduler_module
        from sglang.srt.managers.overlap_utils import RelayPayload
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        scheduler = object.__new__(Scheduler)
        scheduler.device = "cpu"
        scheduler.draft_worker = None
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=None)
        )
        scheduler.server_args = SimpleNamespace(
            enable_two_batch_overlap=False,
            cuda_graph_config=None,
            speculative_algorithm=None,
        )
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.req_to_token_pool = ReqToTokenPool(
            size=4,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
        )
        scheduler.enable_overlap = False

        original_use_mlx = scheduler_module.use_mlx
        scheduler_module.use_mlx = lambda: True
        try:
            Scheduler.init_overlap(scheduler)
        finally:
            scheduler_module.use_mlx = original_use_mlx

        self.assertIsNotNone(scheduler.future_map)
        indices = torch.tensor([1], dtype=torch.int64)
        scheduler.future_map.stash(
            indices, RelayPayload(bonus_tokens=torch.tensor([7], dtype=torch.int64))
        )
        self.assertEqual(int(scheduler.future_map.output_tokens_buf[1].item()), 7)

    def test_decode_finalize_does_not_snapshot_auxiliary_state(self):
        runner = object.__new__(MlxModelRunner)
        runner._req_token_ids = {"r0": [8]}
        runner._decode_step_ct = 0
        runner._clear_steps = 0
        calls = []
        runner._store_auxiliary_state = lambda req_pool_idx, cache: calls.append(
            (req_pool_idx, cache)
        )
        pending = MlxPendingDecode(
            lazy_tokens=mx.array([9], dtype=mx.int32),
            req_ids=["r0"],
            caches=[[object()]],
        )

        next_tokens = runner.decode_batch_finalize(pending)

        self.assertEqual(next_tokens, [9])
        self.assertEqual(runner._req_token_ids["r0"], [8, 9])
        self.assertEqual(calls, [])

    def test_store_auxiliary_state_for_request_snapshots_on_demand(self):
        runner = object.__new__(MlxModelRunner)
        cache = [object()]
        runner._req_pool_idx = {"r0": 3}
        runner._req_caches = {"r0": cache}
        calls = []
        runner._store_auxiliary_state = lambda req_pool_idx, cache_arg: calls.append(
            (req_pool_idx, cache_arg)
        )

        runner.store_auxiliary_state_for_request("r0")
        runner.store_auxiliary_state_for_request("missing")

        self.assertEqual(calls, [(3, cache)])

    def test_dense_batched_attention_helper_supports_single_request(self):
        runner = object.__new__(MlxModelRunner)
        model = FakeWrappedAttentionModel()
        runner.model = model
        _set_runner_cache_layout(
            runner,
            num_layers=1,
            attention_layer_indices=[0],
        )
        _set_runner_decode_context_defaults(runner)
        cache = [
            [
                ContiguousAttentionKVCache(
                    n_kv_heads=1,
                    head_dim=2,
                    max_seq_len=4,
                    dtype=mx.float32,
                )
            ]
        ]

        lazy_tokens = runner._decode_with_batched_attention(
            cache,
            mx.array([[7]], dtype=mx.int32),
            ["r0"],
        )
        mx.eval(lazy_tokens, *MlxModelRunner._cache_state_arrays(cache))

        self.assertEqual(lazy_tokens.tolist(), [0])
        self.assertEqual(cache[0][0].offset, 1)
        self.assertEqual(model.seen_inputs, [[[7]]])
        self.assertEqual(model.seen_cache_types, [["AttentionOffsetCache"]])

    def test_batched_decode_context_resolves_aot_rope_slots_from_request_ids(self):
        cache0 = ContiguousAttentionKVCache(
            n_kv_heads=1,
            head_dim=2,
            max_seq_len=4,
            dtype=mx.float32,
        )
        cache1 = ContiguousAttentionKVCache(
            n_kv_heads=1,
            head_dim=2,
            max_seq_len=4,
            dtype=mx.float32,
        )
        cache0.offset = 1
        cache1.offset = 2
        kernel_set = MlxAOTKernelSet(
            rope=MlxAOTRoPEKernel(
                base=10000.0,
                config={
                    "head_dim": 2,
                    "num_qo_heads": 1,
                    "num_kv_heads": 1,
                },
                rope_pool_fused=object(),
            )
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor(
                [
                    [0, 41, 42],
                    [0, 51, 52],
                ],
                dtype=torch.int64,
            )
        )

        ctx = BatchedDecodeContext.from_decode(
            caches=[[cache0], [cache1]],
            req_ids=["r0", "r1"],
            aot_kernels=kernel_set,
            kv_pool=object(),
            req_pool_idx={"r0": 0, "r1": 1},
            req_to_token_pool=req_to_token_pool,
            attention_layer_indices=[0],
        )

        self.assertEqual(ctx.seq_lens, [1, 2])
        self.assertIsNotNone(ctx.aot.rope)
        self.assertEqual(ctx.aot.rope.new_token_slots.tolist(), [41, 52])

    def test_auxiliary_decode_uses_hybrid_batching_for_multi_request(self):
        runner = object.__new__(MlxModelRunner)
        _set_runner_cache_layout(
            runner,
            num_layers=2,
            attention_layer_indices=[1],
        )
        req_ids = ["r0", "r1"]
        runner._req_caches = {rid: [object(), object()] for rid in req_ids}
        runner._req_token_ids = {rid: [idx + 20] for idx, rid in enumerate(req_ids)}
        calls = []

        def fake_hybrid(caches, batched_input, helper_req_ids):
            calls.append((len(caches), batched_input.tolist(), list(helper_req_ids)))
            return mx.array([4, 5], dtype=mx.int32)

        def fail_batched(*args, **kwargs):
            raise AssertionError(
                "auxiliary decode should use hybrid batching, not full batched"
            )

        runner._decode_with_hybrid_batching = fake_hybrid
        runner._decode_with_batched_attention = fail_batched

        pending = runner.decode_batch_start(req_ids)

        self.assertEqual(calls, [(2, [[20], [21]], req_ids)])
        self.assertEqual(pending.lazy_tokens.tolist(), [4, 5])

    def test_auxiliary_layer_batches_mergeable_native_cache(self):
        runner = object.__new__(MlxModelRunner)
        layer = FakeBatchableAuxiliaryLayer()
        cache0 = ArraysCache(size=1)
        cache1 = ArraysCache(size=1)

        out = runner._decode_auxiliary_layer(
            layer,
            mx.zeros((2, 1, 4), dtype=mx.float32),
            [cache0, cache1],
        )
        mx.eval(out, cache0[0], cache1[0])

        self.assertEqual(layer.input_layernorm.seen_shapes, [(2, 1, 4)])
        self.assertEqual(layer.linear_attn.seen_shapes, [(2, 1, 4)])
        self.assertEqual(layer.post_attention_layernorm.seen_shapes, [(2, 1, 4)])
        self.assertEqual(layer.mlp.seen_shapes, [(2, 1, 4)])
        self.assertEqual(layer.linear_attn.cache_type, "ArraysCache")
        self.assertEqual(
            out.tolist(),
            [[[2.0, 2.0, 2.0, 2.0]], [[2.0, 2.0, 2.0, 2.0]]],
        )
        self.assertEqual(cache0[0].tolist(), [[0.0]])
        self.assertEqual(cache1[0].tolist(), [[1.0]])

    def test_arrays_cache_auxiliary_batching_uses_fast_merge(self):
        runner = object.__new__(MlxModelRunner)
        layer = FakeBatchableAuxiliaryLayer()
        cache0 = ArraysCache(size=1)
        cache1 = ArraysCache(size=1)
        original_merge = ArraysCache.merge

        def fail_merge(cls, caches):
            raise AssertionError("ArraysCache fast path should not call merge()")

        ArraysCache.merge = classmethod(fail_merge)
        try:
            out = runner._decode_auxiliary_layer(
                layer,
                mx.zeros((2, 1, 4), dtype=mx.float32),
                [cache0, cache1],
            )
            mx.eval(out, cache0[0], cache1[0])
        finally:
            ArraysCache.merge = original_merge

        self.assertEqual(layer.linear_attn.cache_type, "ArraysCache")
        self.assertEqual(
            out.tolist(),
            [[[2.0, 2.0, 2.0, 2.0]], [[2.0, 2.0, 2.0, 2.0]]],
        )
        self.assertEqual(cache0[0].tolist(), [[0.0]])
        self.assertEqual(cache1[0].tolist(), [[1.0]])

    def test_auxiliary_layer_split_back_copies_cache_metadata(self):
        runner = object.__new__(MlxModelRunner)
        layer = FakeBatchableAuxiliaryLayer()
        cache0 = FakeMergeableAuxiliaryCache(tag="old0")
        cache1 = FakeMergeableAuxiliaryCache(tag="old1")

        out = runner._decode_auxiliary_layer(
            layer,
            mx.zeros((2, 1, 4), dtype=mx.float32),
            [cache0, cache1],
        )
        mx.eval(out, cache0[0], cache1[0])

        self.assertEqual(layer.linear_attn.cache_type, "FakeMergeableAuxiliaryCache")
        self.assertEqual(cache0.tag, "split-0")
        self.assertEqual(cache1.tag, "split-1")
        self.assertEqual(cache0.extra_metadata, {"idx": 0})
        self.assertEqual(cache1.extra_metadata, {"idx": 1})
        self.assertEqual(cache0[0].tolist(), [[0.0]])
        self.assertEqual(cache1[0].tolist(), [[1.0]])

    def test_auxiliary_state_prefill_restores_prefix_state(self):
        runner = object.__new__(MlxModelRunner)
        runner.model = FakeAuxiliaryStateModel()
        _set_runner_cache_layout(
            runner,
            num_layers=2,
            attention_layer_indices=[1],
        )
        runner._max_seq_len = 8
        runner._cache_pool = []
        runner.disable_radix_cache = False
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=8,
            num_layers=1,
            n_kv_heads=1,
            head_dim=2,
            dtype=mx.float32,
        )
        runner._req_to_token_pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        runner._req_caches = {}
        runner._req_token_ids = {}
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        req = FakeRequest()
        runner._req_to_token_pool.alloc([req])
        runner._req_to_token_pool.auxiliary_state_pool.store_cache(
            req.mamba_pool_idx,
            [FakeNativeCache(mx.array([42.0], dtype=mx.float32)), None],
            [0],
        )

        pending = runner.prefill_start(
            req_id="r0",
            new_token_ids=[13],
            full_token_ids=[11, 12, 13],
            prefix_slot_ids=[2, 3],
            new_slot_ids=[4],
            req_pool_idx=req.req_pool_idx,
        )
        MlxModelRunner._eval_with_cache(pending.lazy_token, pending.cache)
        runner.prefill_finalize(pending)

        self.assertEqual(runner.model.seen_inputs, [[[13]]])
        self.assertEqual(runner.model.seen_auxiliary_states, [[42.0]])
        self.assertEqual(pending.synced_offset, 3)
        self.assertIsInstance(pending.cache[0], FakeNativeCache)
        self.assertIsInstance(pending.cache[1], ContiguousAttentionKVCache)
        restored = [FakeNativeCache(), None]
        runner._req_to_token_pool.auxiliary_state_pool.restore_cache(
            req.mamba_pool_idx, restored, [0]
        )
        self.assertEqual(restored[0].state[0].tolist(), [1.0])

    def test_auxiliary_state_prefill_tracks_chunk_aligned_auxiliary_state(self):
        _set_dummy_server_args_for_auxiliary_state_tests()
        runner = object.__new__(MlxModelRunner)
        runner.model = FakeAuxiliaryStateModel()
        _set_runner_cache_layout(
            runner,
            num_layers=2,
            attention_layer_indices=[1],
        )
        runner._max_seq_len = 128
        runner._cache_pool = []
        runner.disable_radix_cache = False
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=96,
            num_layers=1,
            n_kv_heads=1,
            head_dim=2,
            dtype=mx.float32,
        )
        runner._req_to_token_pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=128,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        runner._req_caches = {}
        runner._req_token_ids = {}
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        req = FakeRequest()
        runner._req_to_token_pool.alloc([req])
        token_ids = list(range(70))

        pending = runner.prefill_start(
            req_id="r0",
            new_token_ids=token_ids,
            full_token_ids=token_ids,
            prefix_slot_ids=[],
            new_slot_ids=list(range(1, 71)),
            req_pool_idx=req.req_pool_idx,
            req=req,
        )
        MlxModelRunner._eval_with_cache(pending.lazy_token, pending.cache)
        runner.prefill_finalize(pending)
        tracked = [FakeNativeCache(), None]
        runner._req_to_token_pool.auxiliary_state_pool.restore_cache(
            req.mamba_ping_pong_track_buffer[0], tracked, [0]
        )

        self.assertEqual([len(x[0]) for x in runner.model.seen_inputs], [64, 6])
        self.assertEqual(req.mamba_last_track_seqlen, 64)
        self.assertEqual(tracked[0].state[0].tolist(), [64.0])
        self.assertEqual(pending.synced_offset, 70)

    def test_auxiliary_state_prefill_advances_tracked_boundary_after_cached_prefix(
        self,
    ):
        _set_dummy_server_args_for_auxiliary_state_tests()
        runner = object.__new__(MlxModelRunner)
        runner.model = FakeAuxiliaryStateModel()
        _set_runner_cache_layout(
            runner,
            num_layers=2,
            attention_layer_indices=[1],
        )
        runner._max_seq_len = 512
        runner._cache_pool = []
        runner.disable_radix_cache = False
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=320,
            num_layers=1,
            n_kv_heads=1,
            head_dim=2,
            dtype=mx.float32,
        )
        runner._req_to_token_pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=512,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        runner._req_caches = {}
        runner._req_token_ids = {}
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        req = FakeRequest()
        runner._req_to_token_pool.alloc([req])
        runner._req_to_token_pool.auxiliary_state_pool.store_cache(
            req.mamba_pool_idx,
            [FakeNativeCache(mx.array([64.0], dtype=mx.float32)), None],
            [0],
        )
        token_ids = list(range(257))

        pending = runner.prefill_start(
            req_id="r0",
            new_token_ids=token_ids[64:],
            full_token_ids=token_ids,
            prefix_slot_ids=list(range(1, 65)),
            new_slot_ids=list(range(65, 258)),
            req_pool_idx=req.req_pool_idx,
            req=req,
        )
        MlxModelRunner._eval_with_cache(pending.lazy_token, pending.cache)
        runner.prefill_finalize(pending)
        tracked = [FakeNativeCache(), None]
        runner._req_to_token_pool.auxiliary_state_pool.restore_cache(
            req.mamba_ping_pong_track_buffer[0], tracked, [0]
        )

        self.assertEqual([len(x[0]) for x in runner.model.seen_inputs], [192, 1])
        self.assertEqual(runner.model.seen_auxiliary_states, [[64.0], [192.0]])
        self.assertEqual(req.mamba_last_track_seqlen, 256)
        self.assertEqual(tracked[0].state[0].tolist(), [192.0])
        self.assertEqual(pending.synced_offset, 257)

    def test_cache_arrays_flattens_native_array_cache_state(self):
        cache = FakeNestedStateCache()

        arrays = MlxModelRunner._cache_arrays(cache)

        self.assertEqual(len(arrays), 2)
        self.assertTrue(all(isinstance(arr, mx.array) for arr in arrays))

    def test_auxiliary_state_pool_tracks_scheduler_slots_and_snapshots(self):
        pool = MlxAuxiliaryStatePool(size=4, device="cpu")

        first = pool.alloc(2)
        cache = [FakeNativeCache(mx.array([1.0], dtype=mx.float32))]
        pool.store_cache(first[0], cache, [0])
        cache[0].state[0][0] = 9.0
        forked = pool.fork_from(first[0].unsqueeze(0))
        restored = [FakeNativeCache()]
        pool.restore_cache(forked[0], restored, [0])
        pool.free(first)

        self.assertEqual(first.tolist(), [1, 2])
        self.assertEqual(forked.tolist(), [3])
        self.assertEqual(restored[0].state[0].tolist(), [1.0])
        self.assertEqual(pool.available_size(), 3)

    def test_auxiliary_state_pool_restores_instance_meta_state(self):
        pool = MlxAuxiliaryStatePool(size=2, device="cpu")
        slot = pool.alloc(1)
        cache = [
            FakeNativeCache(
                mx.array([1.0], dtype=mx.float32),
                meta_state={"seen": mx.array([3.0], dtype=mx.float32)},
            )
        ]
        pool.store_cache(slot[0], cache, [0])
        cache[0].meta_state["seen"][0] = 9.0

        restored = [FakeNativeCache(meta_state={})]
        pool.restore_cache(slot[0], restored, [0])

        self.assertEqual(restored[0].meta_state["seen"].tolist(), [3.0])

    def test_auxiliary_state_req_pool_maps_request_indices(self):
        pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        req = FakeRequest()

        req_indices = pool.alloc([req])
        auxiliary_state_idx = pool.get_auxiliary_state_indices(req.req_pool_idx)
        pool.free(req)

        self.assertEqual(req_indices, [1])
        self.assertIsNotNone(auxiliary_state_idx)
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertEqual(pool.auxiliary_state_pool.available_size(), 3)
        pool.free_auxiliary_state_cache(req)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(pool.available_size(), 2)
        self.assertEqual(pool.auxiliary_state_pool.available_size(), 4)

    def test_auxiliary_state_req_pool_can_keep_tracked_auxiliary_slot(self):
        pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        req = FakeRequest()
        pool.alloc([req])
        req.mamba_ping_pong_track_buffer = pool.auxiliary_state_pool.alloc(1)
        req.mamba_next_track_idx = 0

        pool.free_auxiliary_state_cache(req, track_buffer_to_keep=0)

        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertEqual(pool.auxiliary_state_pool.available_size(), 3)

    def test_auxiliary_state_component_inserts_tracked_slot_and_frees_live_slot(self):
        pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        req = FakeRequest()
        pool.alloc([req])
        req.mamba_ping_pong_track_buffer = pool.auxiliary_state_pool.alloc(1)
        req.mamba_next_track_idx = 0
        req.mamba_last_track_seqlen = 64
        component = MlxAuxiliaryStateComponent(
            SimpleNamespace(req_to_token_pool=pool),
            SimpleNamespace(enable_mamba_extra_buffer=False),
        )
        insert_params = InsertParams()

        cache_len = component.prepare_for_caching_req(
            req=req,
            insert_params=insert_params,
            token_ids_len=70,
            is_finished=True,
        )
        component.cleanup_after_caching_req(
            req=req,
            is_finished=True,
            insert_result=InsertResult(prefix_len=0, mamba_exist=False),
            insert_params=insert_params,
        )

        self.assertEqual(cache_len, 64)
        self.assertTrue(getattr(insert_params, "mlx_auxiliary_state_uses_track_slot"))
        self.assertEqual(insert_params.mamba_value.tolist(), [2])
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertIsNone(req.mamba_last_track_seqlen)
        self.assertEqual(pool.auxiliary_state_pool.available_size(), 3)

    def test_auxiliary_state_component_unfinished_frees_tracked_source_slot(self):
        pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        req = FakeRequest()
        pool.alloc([req])
        req.mamba_ping_pong_track_buffer = pool.auxiliary_state_pool.alloc(1)
        req.mamba_next_track_idx = 0
        req.mamba_last_track_seqlen = 64
        component = MlxAuxiliaryStateComponent(
            SimpleNamespace(req_to_token_pool=pool),
            SimpleNamespace(enable_mamba_extra_buffer=False),
        )
        insert_params = InsertParams()

        cache_len = component.prepare_for_caching_req(
            req=req,
            insert_params=insert_params,
            token_ids_len=70,
            is_finished=False,
        )
        component.cleanup_after_caching_req(
            req=req,
            is_finished=False,
            insert_result=InsertResult(prefix_len=0, mamba_exist=False),
            insert_params=insert_params,
        )

        self.assertEqual(cache_len, 64)
        self.assertEqual(insert_params.mamba_value.tolist(), [3])
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertIsNone(req.mamba_last_track_seqlen)
        self.assertEqual(pool.auxiliary_state_pool.available_size(), 2)

    def test_auxiliary_state_component_keeps_new_live_slot_owned_by_radix(self):
        pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        req = FakeRequest()
        pool.alloc([req])
        component = MlxAuxiliaryStateComponent(
            SimpleNamespace(req_to_token_pool=pool),
            SimpleNamespace(enable_mamba_extra_buffer=False),
        )
        insert_params = InsertParams()

        cache_len = component.prepare_for_caching_req(
            req=req,
            insert_params=insert_params,
            token_ids_len=7,
            is_finished=True,
        )
        component.cleanup_after_caching_req(
            req=req,
            is_finished=True,
            insert_result=InsertResult(prefix_len=0, mamba_exist=False),
            insert_params=insert_params,
        )

        self.assertEqual(cache_len, 7)
        self.assertFalse(getattr(insert_params, "mlx_auxiliary_state_uses_track_slot"))
        self.assertEqual(insert_params.mamba_value.tolist(), [1])
        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(pool.auxiliary_state_pool.available_size(), 3)

    def test_auxiliary_state_component_frees_stale_track_slot_when_live_slot_inserted(
        self,
    ):
        pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        req = FakeRequest()
        pool.alloc([req])
        req.mamba_ping_pong_track_buffer = pool.auxiliary_state_pool.alloc(1)
        req.mamba_next_track_idx = 0
        component = MlxAuxiliaryStateComponent(
            SimpleNamespace(req_to_token_pool=pool),
            SimpleNamespace(enable_mamba_extra_buffer=False),
        )
        insert_params = InsertParams()

        cache_len = component.prepare_for_caching_req(
            req=req,
            insert_params=insert_params,
            token_ids_len=7,
            is_finished=True,
        )
        component.cleanup_after_caching_req(
            req=req,
            is_finished=True,
            insert_result=InsertResult(prefix_len=0, mamba_exist=False),
            insert_params=insert_params,
        )

        self.assertEqual(cache_len, 7)
        self.assertFalse(getattr(insert_params, "mlx_auxiliary_state_uses_track_slot"))
        self.assertEqual(insert_params.mamba_value.tolist(), [1])
        self.assertIsNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)
        self.assertIsNone(req.mamba_next_track_idx)
        self.assertEqual(pool.auxiliary_state_pool.available_size(), 3)

    def test_auxiliary_state_component_frees_duplicate_live_slot(self):
        pool = MlxAuxiliaryStateReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        req = FakeRequest()
        pool.alloc([req])
        component = MlxAuxiliaryStateComponent(
            SimpleNamespace(req_to_token_pool=pool),
            SimpleNamespace(enable_mamba_extra_buffer=False),
        )
        insert_params = InsertParams()

        component.prepare_for_caching_req(
            req=req,
            insert_params=insert_params,
            token_ids_len=7,
            is_finished=True,
        )
        component.cleanup_after_caching_req(
            req=req,
            is_finished=True,
            insert_result=InsertResult(prefix_len=7, mamba_exist=True),
            insert_params=insert_params,
        )

        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(pool.auxiliary_state_pool.available_size(), 4)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxOverlapScheduler(unittest.TestCase):
    def test_finalize_pending_job_updates_scheduler_last_batch(self):
        token_ids = torch.tensor([7], dtype=torch.long)
        scheduler = FakeOverlapScheduler(token_ids)
        stale_batch = SimpleNamespace(input_ids=None)
        batch_copy = SimpleNamespace(input_ids=None)
        schedule_batch = SimpleNamespace(input_ids=None)
        scheduler.last_batch = stale_batch

        pending = MlxPendingJob(
            lazy_tokens=None,
            prefills=["prefill"],
            extends=[],
            decode=None,
            mode="extend",
            batch_copy=batch_copy,
            schedule_batch=schedule_batch,
            reqs=[SimpleNamespace(rid="r0")],
        )

        scheduler._finalize_mlx_pending_job(pending)

        self.assertIs(scheduler.last_batch, schedule_batch)
        self.assertTrue(torch.equal(batch_copy.input_ids, token_ids))
        self.assertTrue(torch.equal(schedule_batch.input_ids, token_ids))
        self.assertIs(scheduler.processed_batch, batch_copy)
        self.assertIs(scheduler.processed_result, scheduler.tp_worker.result)

    def test_overlap_loop_materializes_prefill_input_ids(self):
        # Regression: the MLX overlap loop must materialize batch.input_ids
        # (deferred input materialization) before launching the forward.
        # Without resolve_forward_inputs in _launch_fresh, input_ids stays
        # None and async_forward_batch_generation_mlx dereferences a None.
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        class _StopLoop(Exception):
            pass

        captured = {}

        def fake_forward(batch):
            captured["input_ids"] = batch.input_ids
            raise _StopLoop

        scheduler = SchedulerMlxOverlapMixin.__new__(SchedulerMlxOverlapMixin)
        scheduler.request_receiver = SimpleNamespace(recv_requests=lambda: [])
        scheduler.process_input_requests = lambda recv_reqs: None
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.result_queue = deque()
        scheduler.future_map = SimpleNamespace()
        scheduler.cur_batch = None
        scheduler.last_batch = None
        scheduler.tp_worker = SimpleNamespace(
            async_forward_batch_generation_mlx=fake_forward
        )

        batch = SimpleNamespace(
            prefill_input_ids_cpu=torch.tensor([1, 2, 3], dtype=torch.int64),
            input_ids=None,
            mix_running_indices=None,
            enable_overlap=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            device="cpu",
        )
        scheduler.get_next_batch_to_run = lambda: batch

        with self.assertRaises(_StopLoop):
            scheduler.event_loop_overlap_mlx()

        self.assertIsNotNone(captured["input_ids"])
        self.assertTrue(torch.equal(captured["input_ids"], torch.tensor([1, 2, 3])))

    def test_finished_request_snapshots_before_release(self):
        events = []
        tree_cache = object()
        processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            enable_overlap=False,
            enable_overlap_mlx=False,
            server_args=SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False,
                enable_hisparse=False,
            ),
            model_config=None,
            token_to_kv_pool_allocator=None,
            tree_cache=tree_cache,
            hisparse_coordinator=SimpleNamespace(request_finished=lambda req: None),
            req_to_token_pool=None,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=None,
            draft_worker=None,
            model_worker=SimpleNamespace(
                prepare_for_kv_cache_release=lambda req: events.append(
                    ("prepare", req.rid)
                )
            ),
            logprob_result_processor=None,
            output_streamer=None,
            abort_request=lambda req: None,
        )
        # Stub out the methods _handle_finish_state_updated_req calls that
        # are not relevant to this test.  SchedulerBatchResultProcessor is
        # @dataclass(slots=True, frozen=True), so patches go on the class.
        noop_stubs = {
            "_mamba_prefix_cache_update": lambda *a, **k: None,
            "_maybe_collect_routed_experts": lambda *a, **k: None,
            "_maybe_collect_indexer_topk": lambda *a, **k: None,
            "_maybe_collect_customized_info": lambda *a, **k: None,
        }
        saved = {
            name: getattr(SchedulerBatchResultProcessor, name) for name in noop_stubs
        }
        for name, value in noop_stubs.items():
            setattr(SchedulerBatchResultProcessor, name, value)
        req = SimpleNamespace(
            rid="r0",
            finished=lambda: True,
            multimodal_inputs=None,
            session=None,
            return_routed_experts=False,
            mamba_lazy_is_insert=True,
            time_stats=SimpleNamespace(
                set_completion_time=lambda: events.append(("completion", "r0"))
            ),
        )
        batch = SimpleNamespace()
        result = SimpleNamespace()
        i = 0
        logits_output = SimpleNamespace(customized_info=None)
        original_release = batch_result_processor_module.release_kv_cache
        original_get_indexer = batch_result_processor_module.get_global_indexer_capturer
        original_get_server_args = batch_result_processor_module.get_global_server_args

        def fake_release_kv_cache(release_req, tree_cache, is_insert=False):
            events.append(("release", release_req.rid))
            self.assertIs(tree_cache, processor.tree_cache)

        batch_result_processor_module.release_kv_cache = fake_release_kv_cache
        batch_result_processor_module.get_global_indexer_capturer = lambda: None
        batch_result_processor_module.get_global_server_args = lambda: SimpleNamespace(
            enable_mamba_extra_buffer_lazy=lambda: False
        )
        try:
            SchedulerBatchResultProcessor._handle_finish_state_updated_req(
                processor, req, batch, result, i, logits_output
            )
        finally:
            for name, original in saved.items():
                setattr(SchedulerBatchResultProcessor, name, original)
            batch_result_processor_module.release_kv_cache = original_release
            batch_result_processor_module.get_global_indexer_capturer = (
                original_get_indexer
            )
            batch_result_processor_module.get_global_server_args = (
                original_get_server_args
            )

        self.assertEqual(
            events,
            [
                ("prepare", "r0"),
                ("release", "r0"),
                ("completion", "r0"),
            ],
        )


if _HAS_MLX:

    class FakeProjection(nn.Module):
        def __init__(self, out_dim: int = 4):
            super().__init__()
            self.weight = mx.zeros((out_dim, 4), dtype=mx.float32)

        def __call__(self, x):
            shape = (*x.shape[:-1], self.weight.shape[0])
            return mx.zeros(shape, dtype=x.dtype)

    class FakeAttention(nn.Module):
        def __init__(self, use_aliases: bool = False):
            super().__init__()
            if use_aliases:
                self.num_attention_heads = 2
                self.num_key_value_heads = 1
            else:
                self.n_heads = 2
                self.n_kv_heads = 1
            self.head_dim = 2
            self.scale = self.head_dim**-0.5
            self.q_proj = FakeProjection(4)
            self.k_proj = FakeProjection(2)
            self.v_proj = FakeProjection(2)
            self.o_proj = FakeProjection(4)
            self.rope = lambda x, offset=None: x

    class ProjectionOnlyMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = 2
            self.n_kv_heads = 1
            self.q_proj = FakeProjection(4)
            self.k_proj = FakeProjection(2)
            self.v_proj = FakeProjection(2)
            self.o_proj = FakeProjection(4)

    class FakeLayer(nn.Module):
        def __init__(self, attr_name: str, module: nn.Module):
            super().__init__()
            setattr(self, attr_name, module)

    class FakeModel(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

    class IdentityNorm(nn.Module):
        def __call__(self, x):
            return x

    class IdentityRope:
        def __call__(self, x, offset=None):
            return x

    class CapturingOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input_shape = None

        def __call__(self, x):
            self.last_input_shape = x.shape
            return x

    class FakeGatedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_attention_heads = 2
            self.num_key_value_heads = 1
            self.head_dim = 2
            self.scale = self.head_dim**-0.5
            self.q_proj = FakeProjection(8)
            self.k_proj = FakeProjection(2)
            self.v_proj = FakeProjection(2)
            self.o_proj = CapturingOutput()
            self.q_norm = IdentityNorm()
            self.k_norm = IdentityNorm()
            self.rope = IdentityRope()

    class RecordingIdentity(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_shapes = []

        def __call__(self, x):
            self.seen_shapes.append(x.shape)
            return x

    class FakeMergeableLinearAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_shapes = []
            self.cache_type = None

        def __call__(self, x, mask=None, cache=None):
            self.seen_shapes.append(x.shape)
            self.cache_type = type(cache).__name__
            cache[0] = mx.arange(x.shape[0], dtype=mx.float32).reshape(x.shape[0], 1)
            return x + 1

    class FakeBatchableAuxiliaryLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_linear = True
            self.input_layernorm = RecordingIdentity()
            self.linear_attn = FakeMergeableLinearAttention()
            self.post_attention_layernorm = RecordingIdentity()
            self.mlp = RecordingIdentity()

    class FakeMergeableAuxiliaryCache:
        def __init__(self, state=None, tag="init", extra_metadata=None):
            self.cache = [state]
            self.tag = tag
            self.extra_metadata = extra_metadata or {}

        def __getitem__(self, idx):
            return self.cache[idx]

        def __setitem__(self, idx, value):
            self.cache[idx] = value

        @classmethod
        def merge(cls, caches):
            merged = cls(tag="merged")
            values = [cache[0] for cache in caches]
            if all(value is None for value in values):
                return merged
            merged[0] = mx.concatenate(
                [
                    (
                        value
                        if value is not None
                        else mx.zeros_like(next(v for v in values if v is not None))
                    )
                    for value in values
                ],
                axis=0,
            )
            return merged

        def extract(self, idx):
            return type(self)(
                self.cache[0][idx : idx + 1],
                tag=f"split-{idx}",
                extra_metadata={"idx": idx},
            )

    class FakeNativeCache:
        def __init__(self, value=None, meta_state=None):
            self._state = [
                value if value is not None else mx.array([0.0], dtype=mx.float32)
            ]
            if meta_state is not None:
                self.meta_state = meta_state
            self.lengths = None
            self.left_padding = None

        @property
        def state(self):
            return self._state

        @state.setter
        def state(self, value):
            self._state = value

    class FakeAuxiliaryStateModel:
        def __init__(self):
            self.seen_inputs = []
            self.seen_auxiliary_states = []

        def make_cache(self):
            return [FakeNativeCache(), FakeNativeCache()]

        def __call__(self, inputs, cache=None):
            self.seen_inputs.append(inputs.tolist())
            if cache is not None:
                self.seen_auxiliary_states.append(cache[0].state[0].tolist())
                cache[0].state = [mx.array([float(inputs.shape[1])], dtype=mx.float32)]
                keys = mx.ones((1, 1, inputs.shape[1], 2), dtype=mx.float32)
                values = keys * 2
                cache[1].update_and_fetch(keys, values)
            return mx.zeros((1, inputs.shape[1], 4), dtype=mx.float32)

    class FakeDenseModel:
        def __init__(self, num_layers):
            self.num_layers = num_layers
            self.seen_inputs = []
            self.seen_offsets = []

        def __call__(self, inputs, cache=None):
            self.seen_inputs.append(inputs.tolist())
            if cache is not None:
                offsets = []
                for layer_idx in range(self.num_layers):
                    offsets.append(cache[layer_idx].offset)
                    scale = float(layer_idx + 1)
                    keys = mx.ones((1, 1, inputs.shape[1], 2), dtype=mx.float32) * scale
                    cache[layer_idx].update_and_fetch(keys, keys * 2)
                self.seen_offsets.append(offsets)
            return mx.zeros((1, inputs.shape[1], 4), dtype=mx.float32)

    class FakeWrappedAttentionModel:
        def __init__(self):
            self.attn = MLXAttentionWrapper(FakeAttention(), layer_idx=0)
            self.seen_inputs = []
            self.seen_cache_types = []

        def __call__(self, inputs, cache=None):
            self.seen_inputs.append(inputs.tolist())
            self.seen_cache_types.append([type(c).__name__ for c in cache])
            hidden = mx.zeros((*inputs.shape, 4), dtype=mx.float32)
            self.attn(hidden, cache=cache[0])
            return mx.zeros((*inputs.shape, 8), dtype=mx.float32)

    class FakeNestedStateCache:
        @property
        def state(self):
            return [
                mx.array([1.0], dtype=mx.float32),
                None,
                {"nested": (mx.array([2.0], dtype=mx.float32),)},
            ]

    class FakeRequest:
        def __init__(self):
            self.req_pool_idx = None
            self.mamba_pool_idx = None
            self.inflight_middle_chunks = 0
            self.kv_committed_len = 0

    class FakeTpWorker:
        def __init__(self, next_token_ids):
            self.result = GenerationBatchResult(next_token_ids=next_token_ids)
            self.calls = []

        def finalize_mlx_result(self, *args):
            self.calls.append(args)
            return self.result

    class FakeOverlapScheduler(SchedulerMlxOverlapMixin):
        def __init__(self, next_token_ids):
            self.tp_worker = FakeTpWorker(next_token_ids)
            self.last_batch = None
            self.processed_batch = None
            self.processed_result = None

        def process_batch_result(self, batch, result):
            self.processed_batch = batch
            self.processed_result = result


if __name__ == "__main__":
    unittest.main()
