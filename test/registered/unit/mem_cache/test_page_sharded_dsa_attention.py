# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""DSA attention and indexer integration for page-interleaved KV caches.

The multi-GPU test reuses the byte-exact pool fixture and additionally exercises
real DSA metadata construction, the real indexer read, and each supported sparse
attention backend. Full and shared indexer layers must produce the same output
as an unsharded cache, including packed FlashMLA FP8 and raw TRT-LLM FP8 caches.
"""

import os
import tempfile
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveDSATokenToKVPool,
    PageInterleaveKVPoolMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-b200")


class TestShardedDSAIndexerWrite(unittest.TestCase):
    def test_indexer_writers_store_through_sharded_pool(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer

        if not dsa_indexer._is_cuda:
            self.skipTest("requires CUDA indexer writers")

        from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant
        from sglang.srt.mem_cache.page_interleave import (
            PageInterleavePlacement,
            PageShardSpec,
        )

        device = "cuda"
        generator = torch.Generator(device=device).manual_seed(20261013)
        num_tokens, num_pages, page_size = 2305, 80, 64
        # Nonconsecutive logical locations span both owners and partial pages.
        locations = (
            torch.randperm(
                (num_pages - 2) * page_size, device=device, generator=generator
            )[:num_tokens]
            + 2 * page_size
        )
        key = torch.randn(
            (num_tokens, 128), device=device, dtype=torch.bfloat16, generator=generator
        )
        angles = torch.randn((4096, 32), device=device, generator=generator)
        positions = torch.randint(
            0, angles.shape[0], (num_tokens,), device=device, generator=generator
        )
        indexer = SimpleNamespace(
            block_size=128,
            scale_fmt="ue8m0",
            k_norm=SimpleNamespace(
                weight=torch.randn((128,), device=device, generator=generator),
                bias=torch.randn((128,), device=device, generator=generator),
                variance_epsilon=1e-6,
            ),
            _indexer_cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
        )
        indexer._store_index_k_cache = MethodType(
            dsa_indexer.Indexer._store_index_k_cache, indexer
        )
        batch = SimpleNamespace(out_cache_loc=locations)
        pool_args = dict(
            page_size=page_size,
            kv_lora_rank=512,
            dtype=torch.bfloat16,
            qk_rope_head_dim=64,
            layer_num=1,
            device=device,
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
        )
        reference = DSATokenToKVPool(num_pages * page_size, **pool_args)
        sharded = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
        # Rank 0 of 2 with an active extend; a write-only batch needs no gather.
        DSATokenToKVPool.__init__(sharded, num_pages * page_size // 2, **pool_args)
        sharded.shard_spec = PageShardSpec(
            shard_rank=0,
            shard_size=2,
            page_size=page_size,
            max_prefix_tokens=0,
            chunk_tokens=num_pages * page_size,
        )
        sharded.shard_rank = 0
        sharded.placement = PageInterleavePlacement(sharded.shard_spec)
        sharded._shard_extend_active = True
        reference_buffer = reference.index_key_cache.get_local_buffer(0)
        scratch = torch.zeros_like(reference_buffer)
        sharded._slots = [SimpleNamespace(tensors={"index_k": scratch})]
        sharded._page_pos = torch.arange(
            reference_buffer.shape[0], device=device, dtype=torch.int32
        ).roll(7)
        local_buffer = sharded.index_key_cache.get_local_buffer(0)

        for fused_prepare in (False, True):
            with self.subTest(fused_prepare=fused_prepare):
                for buffer in (reference_buffer, local_buffer, scratch):
                    buffer.zero_()
                stored_key = key
                if fused_prepare:
                    stored_key = dsa_indexer.fused_k_indexer_norm_rope(
                        key,
                        indexer.k_norm.weight,
                        indexer.k_norm.bias,
                        indexer.k_norm.variance_epsilon,
                        indexer._indexer_cos_sin_cache,
                        positions,
                    )
                reference.set_index_k_scale_buffer(
                    0, locations, *act_quant(stored_key, 128, "ue8m0")
                )
                with patch.object(
                    dsa_indexer, "get_token_to_kv_pool", return_value=sharded
                ):
                    if fused_prepare:
                        dsa_indexer.Indexer._fused_k_prepare_and_store(
                            indexer, key, positions, batch, 0, act_quant
                        )
                    else:
                        indexer._store_index_k_cache(batch, 0, key, act_quant=act_quant)
                # Every key is staged at its scratch row; rank 0 keeps even pages.
                torch.testing.assert_close(
                    scratch[sharded._page_pos.long()], reference_buffer, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    local_buffer, reference_buffer[::2], rtol=0, atol=0
                )
        torch.cuda.synchronize()


def _attention_backend(pool, req_to_token, device, prefill_impl="flashmla_sparse"):
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.layers.attention.kv_shard_hooks import get_kv_shard_pool

    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.device = device
    backend.token_to_kv_pool = pool
    backend.req_to_token = req_to_token
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend._kv_shard_pool = get_kv_shard_pool(pool)
    backend.dsa_index_kpool = 1
    backend.dsa_index_topk = 2048
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.physical_page_size = 64
    backend.use_mha = False
    backend.use_fused_topk = False
    backend.num_q_heads = 8
    backend.kv_lora_rank = 512
    backend.qk_nope_head_dim = 128
    backend.qk_rope_head_dim = 64
    backend.kv_cache_dim = getattr(pool, "kv_cache_dim", 576)
    backend.kv_cache_dtype = pool.dtype
    backend.dsa_kv_cache_store_fp8 = pool.dsa_kv_cache_store_fp8
    backend.dsa_prefill_impl = prefill_impl
    backend.dsa_decode_impl = "flashmla_kv"
    backend.flashmla_kv_num_q_heads = 64
    backend.flashmla_kv_format = "V32"
    backend.hisparse_coordinator = None
    backend.device_sm_major = (
        torch.cuda.get_device_capability()[0]
        if torch.device(device).type == "cuda"
        else 0
    )
    backend._arange_buf = torch.arange(1024, dtype=torch.int32, device=device)
    if prefill_impl == "trtllm":
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )

        backend.workspace_buffer = torch.empty(
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
            dtype=torch.uint8,
            device=device,
        )
        backend._multi_ctas_kv_counter_buffer = (
            make_persistent_multi_ctas_kv_counter_buffer(
                torch.device(device), backend.num_q_heads, max_batch_size=128
            )
        )
    # Fix the backend choice and avoid optional top-k-v2 planning; the metadata
    # construction, cache accesses, index transforms, and attention are real.
    backend.set_dsa_prefill_impl = lambda _: None
    backend._build_topk_v2_plan = lambda _: None
    return backend


class _MetadataShardPool(PageInterleaveKVPoolMixin):
    def __init__(self):
        self.dsa_kv_cache_store_fp8 = True
        self.dtype = torch.float8_e4m3fn
        self.size = 64
        self.page_size = 64
        self.begin_args = None

    def begin_shard_extend(self, *args):
        self.begin_args = args

    def translate_loc_to_scratch(self, loc):
        # Nonidentity, page-aligned address translation. The real collective
        # and translation kernels are covered by the multi-GPU test below.
        return loc % 512


class TestShardedDSAMetadata(unittest.TestCase):
    def test_cp_request_filtering_keeps_both_address_spaces_aligned(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        lengths = torch.tensor([129, 65, 193])
        req_ids = torch.tensor([2, 0, 4])
        req_to_token = torch.zeros((5, 193), dtype=torch.int64)
        for req_id, base in zip(req_ids, [11, 17, 23]):
            req_to_token[req_id] = base * 64 + torch.arange(193)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=3,
            req_pool_indices=req_ids,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
            seq_lens_sum=int(lengths.sum()),
            extend_seq_lens=torch.ones(3, dtype=torch.int32),
            extend_seq_lens_cpu=[1, 1, 1],
            extend_prefix_lens_cpu=[128, 64, 192],
            attn_cp_metadata=object(),
        )
        pool = _MetadataShardPool()
        backend = _attention_backend(pool, req_to_token, "cpu")
        # Rank one of CP2 receives the middle request's only query. The other
        # requests must disappear from both indexer page tables and lengths.
        strategy = SimpleNamespace(
            shard_local_tokens=lambda tensor: tensor[1::2],
            shard_per_request=lambda *_: (
                [1],
                torch.tensor([1], dtype=torch.int32),
                [1],
                torch.tensor([1]),
            ),
        )
        with (
            patch.object(
                dsa_backend, "can_dsa_prefill_cp_interleave", return_value=True
            ),
            patch.object(dsa_backend, "get_cp_strategy", return_value=strategy),
            patch.object(
                dsa_backend, "pad_dsa_cache_seqlens", side_effect=lambda _, x: x
            ),
        ):
            backend.init_forward_metadata(batch)
        metadata = backend.get_indexer_metadata(0, batch)
        logical = req_to_token[req_ids[1], :65]
        scratch = pool.translate_loc_to_scratch(logical)
        torch.testing.assert_close(
            metadata.get_cache_page_table_64(), torch.tensor([[17, 18]])
        )
        torch.testing.assert_close(metadata.get_page_table_64(), torch.tensor([[1, 2]]))
        torch.testing.assert_close(metadata.get_page_table_1(), scratch[None])
        torch.testing.assert_close(
            backend.forward_metadata.page_table_1_flattened, scratch
        )
        torch.testing.assert_close(
            metadata.get_indexer_seq_len_cpu(), torch.tensor([65])
        )
        self.assertIs(pool.begin_args[1], req_ids)
        self.assertEqual(pool.begin_args[2], [128, 64, 192])


def _real_indexer_topk(check, backends, batch, layer_id, query, weights):
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

    indexer = SimpleNamespace(
        index_topk=2048,
        num_init_tokens=0,
        num_local_tokens=0,
        logits_with_pp_recv=False,
        sm_count=torch.cuda.get_device_properties(check.device).multi_processor_count,
        _pad_heads_for_deep_gemm=dsa_indexer.Indexer._pad_heads_for_deep_gemm,
    )
    for name in ("_with_real_sm_count", "_mask_init_and_local_tokens"):
        setattr(indexer, name, MethodType(getattr(dsa_indexer.Indexer, name), indexer))

    class RecordingMetadata:
        def __init__(self, metadata):
            self.metadata = metadata
            self.logits = None

        def __getattr__(self, name):
            return getattr(self.metadata, name)

        def topk_transform(self, logits, *args, **kwargs):
            self.logits = logits.clone()
            return self.metadata.topk_transform(logits, *args, **kwargs)

    results, logits = [], []
    for backend in backends:
        metadata = RecordingMetadata(backend.get_indexer_metadata(layer_id, batch))
        with patch.object(
            dsa_indexer, "get_token_to_kv_pool", return_value=backend.token_to_kv_pool
        ):
            results.append(
                dsa_indexer.Indexer._get_topk_ragged(
                    indexer, False, batch, layer_id, query, weights, metadata
                )
            )
        starts, ends = metadata.get_indexer_kvcache_range()
        columns = torch.arange(metadata.logits.shape[1], device=query.device)
        valid = (columns[None] >= starts[:, None]) & (columns[None] < ends[:, None])
        logits.append(metadata.logits[valid])
    torch.testing.assert_close(*logits, rtol=0, atol=0)
    expected = results[1]
    if metadata.topk_transform_method == TopkTransformMethod.PAGED:
        expected = torch.where(
            expected >= 0,
            check.pool.translate_loc_to_scratch(expected.clamp_min(0)),
            -1,
        ).to(results[0].dtype)
    # SGL top-k emits selected values in atomic insertion order; FlashInfer's
    # deterministic mode also fixes the order.
    torch.testing.assert_close(
        results[0].sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
    )
    if backends[0].dsa_topk_backend.is_flashinfer():
        torch.testing.assert_close(results[0], expected, rtol=0, atol=0)
    # Use matching selection order for the subsequent exact attention check.
    return [expected, results[1]]


def _check_attention_batch(check, requests, *, generation, prefill_impl, topk_backend):
    import test_page_interleave_shard as fixture

    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    device = check.device
    lengths = [length for _, _, length in requests]
    prefixes = [prefix for _, prefix, _ in requests]
    extends = [length - prefix for length, prefix in zip(lengths, prefixes)]
    rows = [fixture._locs(pages, length, device) for pages, _, length in requests]
    req_ids = torch.arange(len(rows), device=device) * 2 + 1
    req_to_token = torch.zeros(
        (2 * len(rows) + 1, max(lengths)), device=device, dtype=torch.int32
    )
    for req_id, row in zip(req_ids, rows):
        req_to_token[req_id, : row.numel()] = row
    logical_table = req_to_token[req_ids].clone()
    chunk_locs = torch.cat([row[prefix:] for row, prefix in zip(rows, prefixes)])
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(rows),
        req_pool_indices=req_ids,
        seq_lens=torch.tensor(lengths, device=device, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(lengths, dtype=torch.int64),
        seq_lens_sum=sum(lengths),
        extend_seq_lens=torch.tensor(extends, device=device, dtype=torch.int32),
        extend_seq_lens_cpu=extends,
        extend_prefix_lens_cpu=prefixes,
        extend_num_tokens=sum(extends),
        out_cache_loc=chunk_locs,
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
    )
    backends = [
        _attention_backend(pool, req_to_token, device, prefill_impl)
        for pool in (check.pool, check.reference)
    ]
    for backend in backends:
        backend.use_fused_topk = True
        backend.dsa_topk_backend = DSATopKBackend(topk_backend)
        backend.init_forward_metadata(batch)
    torch.testing.assert_close(req_to_token[req_ids], logical_table, rtol=0, atol=0)
    sharded_metadata = backends[0].get_indexer_metadata(0, batch)
    torch.testing.assert_close(
        sharded_metadata.get_cache_page_table_64(), logical_table[:, ::64] // 64
    )
    torch.testing.assert_close(
        sharded_metadata.get_page_table_1(),
        check.pool.translate_loc_to_scratch(logical_table),
        check_dtype=False,
    )
    assert not torch.equal(
        sharded_metadata.get_page_table_64(), sharded_metadata.get_cache_page_table_64()
    )

    generator = torch.Generator(device=device).manual_seed(2026 + generation)
    q = (
        torch.randn(
            (sum(extends), 8, 576),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        / 8
    )
    query = torch.randn(
        (sum(extends), 32, 128),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    gates = (
        torch.rand(
            (sum(extends), 32), generator=generator, device=device, dtype=torch.bfloat16
        )
        + 0.5
    )
    angles = torch.randn((max(lengths), 32), generator=generator, device=device)
    positions = torch.cat(
        [
            torch.arange(prefix, length, device=device)
            for prefix, length in zip(prefixes, lengths)
        ]
    )
    index_q, index_weights = dsa_indexer.fused_q_indexer_rope_first_quant(
        query,
        gates,
        128**-0.5 * 32**-0.5,
        torch.cat((angles.cos(), angles.sin()), dim=-1),
        positions,
    )
    # TRT-LLM's FP8 path quantizes the split query before dispatch. FlashMLA
    # consumes the BF16 query directly, including when its KV cache is FP8.
    query_args = (
        {"q": q[..., :512].contiguous(), "q_rope": q[..., 512:].contiguous()}
        if prefill_impl == "trtllm"
        else {"q": q}
    )
    topk = None
    for local_layer, shared in enumerate(fixture._INDEXER_SKIP_TOPK):
        layer_id = fixture._INDEXER_START_LAYER + local_layer
        if shared:
            assert check.pool.index_key_cache.get_local_buffer(layer_id).numel() == 0
        else:
            keys, scales = fixture._index_values(chunk_locs, layer_id, generation)
            for pool in (check.pool, check.reference):
                pool.set_index_k_scale_buffer(layer_id, chunk_locs, keys, scales)
            topk = _real_indexer_topk(
                check, backends, batch, layer_id, index_q, index_weights
            )
        # The indexer read above acquires its slot and prefetches the next
        # layer before this layer writes its latent KV chunk.
        chunk = fixture._latent_values(chunk_locs, layer_id, generation)
        for pool in (check.pool, check.reference):
            pool.set_mla_kv_buffer(
                SimpleNamespace(layer_id=layer_id),
                chunk_locs,
                chunk[..., :512].contiguous(),
                chunk[..., 512:].contiguous(),
            )
        layer = SimpleNamespace(
            layer_id=layer_id,
            is_cross_attention=False,
            tp_q_head_num=8,
            head_dim=576,
            v_head_dim=512,
            scaling=576**-0.5,
        )
        outputs = [
            backend.forward_extend(
                **query_args,
                k=chunk[..., :512].contiguous(),
                v=chunk[..., :512].contiguous(),
                k_rope=chunk[..., 512:].contiguous(),
                layer=layer,
                forward_batch=batch,
                save_kv_cache=False,
                topk_indices=topk_indices,
            )
            for backend, topk_indices in zip(backends, topk)
        ]
        torch.testing.assert_close(
            *outputs,
            rtol=0,
            atol=0,
            msg=f"{prefill_impl}: sharded output differs at layer {layer_id}",
        )
    check.pool.end_shard_extend()
    torch.cuda.synchronize()
    torch.distributed.barrier()


def _run_attention(rank, world, init_method):
    import test_page_interleave_shard as fixture

    from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_parallel
    from sglang.test.test_utils import publish_build_topology

    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world), SGLANG_DEBUG_MEMORY_POOL="1"
    )
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=init_method,
        backend="nccl",
    )
    publish_build_topology(tp_size=world, attn_cp_size=1, world_rank=rank)
    initialize_model_parallel()
    # Consecutive shared layers exercise two-slot prefetch progression while
    # reusing the preceding full indexer's top-k, as in GLM-5.2.
    fixture._INDEXER_SKIP_TOPK[:] = [False, True, True, False, True]

    def index_values(locs, layer_id, generation):
        dims = torch.arange(128, device=locs.device)
        state = locs[:, None] * 1103515245 + dims * 12345 + layer_id * 789 + generation
        state = ((state ^ (state >> 13)) * 1274126177) & 0xFFFFFFFF
        return act_quant(((state % 65535).float() / 16384 - 2).bfloat16(), 128)

    # Avoid the short-period byte fixture's tied scores at long contexts.
    fixture._index_values = index_values
    envs.SGLANG_DSA_FUSE_TOPK.set(True)
    envs.SGLANG_OPT_USE_TOPK_V2.set(False)
    envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.set(True)
    cases = [
        ("flashmla_sparse", False),
        ("flashmla_sparse", True),
        ("flashmla_kv", True),
    ]
    if torch.cuda.get_device_capability()[0] == 10:
        cases.append(("trtllm", False))
    for prefill_impl, packed in cases:
        for topk_backend in ("sgl-kernel", "flashinfer"):
            check = fixture._IndexerCacheCheck(
                get_parallel().attn_tp_group,
                packed,
                raw_fp8=prefill_impl == "trtllm",
                prefix_contexts=16,
            )
            prefix = list(range(world, world + 32))
            check.seed(prefix, generation=1)
            # A shared prefix across two requests, then a prefix-free request.
            for generation, requests in (
                (
                    2,
                    [
                        (prefix + list(range(40, 45)), 2048, 2305),
                        (prefix + [48], 2048, 2065),
                    ],
                ),
                (3, [([20 * world + world - 1, 21 * world], 0, 73)]),
            ):
                _check_attention_batch(
                    check,
                    requests,
                    generation=generation,
                    prefill_impl=prefill_impl,
                    topk_backend=topk_backend,
                )
            if rank == 0:
                layout = "packed FP8" if packed else str(check.pool.dtype)
                print(
                    f"{prefill_impl} ({layout}, {topk_backend}): "
                    "exact sharded attention match",
                    flush=True,
                )
            del check
    destroy_model_parallel()
    destroy_distributed_environment()


class TestShardedDSAAttention(unittest.TestCase):
    def test_four_gpu_matches_unsharded_full_and_shared_layers(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("requires four CUDA GPUs")
        if torch.cuda.get_device_capability()[0] not in (9, 10):
            self.skipTest("FlashMLA sparse requires Hopper or Blackwell")
        with tempfile.TemporaryDirectory(prefix="sglang-dsa-shard-") as tmp:
            mp.spawn(
                _run_attention,
                args=(4, f"file://{tmp}/init"),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
