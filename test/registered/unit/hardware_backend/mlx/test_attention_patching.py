"""Unit tests for MLX attention discovery and hybrid cache handling."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    import torch

    import mlx.core as mx
    import mlx.nn as nn
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        BatchedDecodeContext,
        ContiguousKVCache,
        MLXAttentionWrapper,
        find_attention_layers,
        is_attention_module,
        patch_model_attention,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from sglang.srt.hardware_backend.mlx.model_runner_stub import (
        _DummyMambaPool,
        _DummyMambaReqToTokenPool,
    )
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
        MlxPendingJob,
        SchedulerMlxOverlapMixin,
    )
    from sglang.srt.managers.utils import GenerationBatchResult


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

    def test_hybrid_model_returns_per_layer_attention_attrs(self):
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

    def test_qwen_next_gated_query_projection_keeps_attention_width(self):
        inner = FakeGatedAttention()
        wrapper = MLXAttentionWrapper(inner, layer_idx=0)
        cache = ContiguousKVCache(
            n_kv_heads=1, head_dim=2, max_seq_len=4, dtype=mx.float32
        )
        ctx = BatchedDecodeContext(
            batch_size=1,
            seq_lens=[0],
            layer_caches=[[cache]],
        )

        out = wrapper._batched_decode(mx.zeros((1, 1, 4), dtype=mx.float32), ctx)
        mx.eval(out)

        self.assertEqual(out.shape, (1, 1, 4))
        self.assertEqual(inner.o_proj.last_input_shape, (1, 1, 4))


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxHybridRunnerCache(unittest.TestCase):
    def test_hybrid_prefill_uses_full_prompt_and_native_cache(self):
        runner = object.__new__(MlxModelRunner)
        runner.model = FakeHybridModel()
        runner._has_non_attention_layers = True
        runner._num_layers = 2
        runner._attention_layer_indices = [1]
        runner._max_seq_len = 8
        runner._cache_pool = []

        pending = runner.prefill_start(
            req_id="r0",
            new_token_ids=[],
            full_token_ids=[11, 12, 13],
            prefix_slot_ids=[2, 3, 4],
            new_slot_ids=[],
            req_pool_idx=0,
        )

        self.assertEqual(runner.model.seen_inputs, [[[11, 12, 13]]])
        self.assertEqual(pending.synced_offset, 0)
        self.assertIsInstance(pending.cache[0], FakeNativeCache)
        self.assertIsInstance(pending.cache[1], ContiguousKVCache)

    def test_cache_arrays_flattens_native_array_cache_state(self):
        cache = FakeNestedStateCache()

        arrays = MlxModelRunner._cache_arrays(cache)

        self.assertEqual(len(arrays), 2)
        self.assertTrue(all(isinstance(arr, mx.array) for arr in arrays))

    def test_dummy_mamba_pool_tracks_scheduler_slots_without_state(self):
        pool = _DummyMambaPool(size=4, device="cpu")

        first = pool.alloc(2)
        forked = pool.fork_from(torch.zeros((1, 2), dtype=torch.int64))
        pool.free(first)

        self.assertEqual(first.tolist(), [1, 2])
        self.assertEqual(forked.tolist(), [3])
        self.assertEqual(pool.available_size(), 3)
        self.assertIsNone(pool.mamba_cache)

    def test_dummy_mamba_req_pool_maps_request_indices(self):
        pool = _DummyMambaReqToTokenPool(
            size=2,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
            mamba_size=4,
        )
        req = FakeRequest()

        req_indices = pool.alloc([req])
        mamba_idx = pool.get_mamba_indices(req.req_pool_idx)
        pool.free(req)

        self.assertEqual(req_indices, [1])
        self.assertIsNotNone(mamba_idx)
        self.assertIsNone(req.req_pool_idx)
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertEqual(pool.mamba_pool.available_size(), 3)
        pool.free_mamba_cache(req)
        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(pool.available_size(), 2)
        self.assertEqual(pool.mamba_pool.available_size(), 4)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxOverlapScheduler(unittest.TestCase):
    def test_finalize_pending_job_updates_scheduler_last_batch(self):
        token_ids = torch.tensor([7], dtype=torch.long)
        scheduler = FakeOverlapScheduler(token_ids)
        stale_batch = SimpleNamespace(output_ids=None)
        batch_copy = SimpleNamespace(output_ids=None)
        schedule_batch = SimpleNamespace(output_ids=None)
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
        self.assertTrue(torch.equal(batch_copy.output_ids, token_ids))
        self.assertTrue(torch.equal(schedule_batch.output_ids, token_ids))
        self.assertIs(scheduler.processed_batch, batch_copy)
        self.assertIs(scheduler.processed_result, scheduler.tp_worker.result)


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

    class FakeNativeCache:
        @property
        def state(self):
            return [mx.array([1.0], dtype=mx.float32)]

    class FakeHybridModel:
        def __init__(self):
            self.seen_inputs = []

        def make_cache(self):
            return [FakeNativeCache(), FakeNativeCache()]

        def __call__(self, inputs, cache=None):
            self.seen_inputs.append(inputs.tolist())
            return mx.zeros((1, inputs.shape[1], 4), dtype=mx.float32)

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
