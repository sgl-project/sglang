"""CPU routing contracts for the replicated non-PD DSA decode batch.

Native kernels and transport have independent GPU coverage. These tests enter
the public attention/indexer methods so bypassing their owner dispatch cannot
silently fall back to broadcasting full cache buffers.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _pool(rank=None):
    kwargs = dict(
        size=192,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.float4_e2m1fn_x2,
        qk_rope_head_dim=64,
        layer_num=5,
        device="cpu",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=416,
    )
    cls = DSATokenToKVPool
    if rank is not None:
        cls = LayerSplitDSATokenToKVPool
        kwargs.update(layer_shard_rank=rank, layer_shard_size=2, unified_decode=True)
    with (
        get_parallel().override(attn_dcp_size=1),
        patch(
            "sglang.srt.mem_cache.memory_pool.maybe_init_custom_mem_pool",
            return_value=(False, None, None),
        ),
        patch(
            "sglang.srt.mem_cache.memory_pool.is_float4_e2m1fn_x2",
            side_effect=lambda dtype: dtype == torch.float4_e2m1fn_x2,
        ),
        patch.object(LayerSplitDSATokenToKVPool, "_init_layer_broadcast_comm"),
    ):
        pool = cls(**kwargs)
    for layer_id in range(5):
        pool.set_mla_kv_global_scale(layer_id, 0.5 + layer_id * 0.375)
    return pool


def _connect_result_transport(pools):
    """Sequential CPU transport; the owner enters before the receiving rank."""
    mailbox = {}

    def transport(pool, tensor, layer_id, **kwargs):
        assert kwargs["use_layer_broadcast_comm"]
        # Neither packed KV rows nor packed indexer pages may leave the owner.
        assert tensor.dtype != torch.uint8
        if pool._is_layer_owned(layer_id):
            mailbox[layer_id] = tensor.clone()
        else:
            tensor.copy_(mailbox[layer_id])
        return tensor

    for pool in pools:
        pool._broadcast_tensor_from_owner = (
            lambda tensor, layer_id, _pool=pool, **kwargs: transport(
                _pool, tensor, layer_id, **kwargs
            )
        )


def _backend(pool):
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.token_to_kv_pool = pool
    backend.forward_metadata = SimpleNamespace()
    backend.dsa_decode_impl = "flashmla_kv"
    backend.dsa_prefill_impl = "flashmla_sparse"
    backend.dsa_index_kpool = 1
    backend.dsa_kv_cache_store_fp8 = False
    backend.dsa_kv_cache_store_packed_fp4 = True
    backend.use_mha = False
    backend.hisparse_coordinator = None
    backend.use_fused_topk = True
    backend.dsa_topk_backend = SimpleNamespace(is_sgl_kernel=lambda: True)
    return backend


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("page_size", [64, 128])
def test_nvfp4_backend_constructor_and_native_decode_use_physical_page_size(
    draft, page_size
):
    """Constructed backends must validate and reshape KV with the live page field.

    An old ``real_page_size`` access previously failed before server readiness.
    Stub hardware queries and the native kernel boundary, not backend setup.
    """
    pool = _pool()
    runner = SimpleNamespace(
        device="cpu",
        page_size=page_size,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[
                    "GlmMoeDsaForCausalLMNextN" if draft else "GlmMoeDsaForCausalLM"
                ],
                index_topk=2048,
            ),
            context_len=192,
            num_attention_heads=64,
            qk_nope_head_dim=192,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        ),
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(
            size=4, req_to_token=torch.zeros(5, 192, dtype=torch.int32)
        ),
        hisparse_coordinator=None,
        is_draft_worker=draft,
        kv_cache_dtype=torch.float4_e2m1fn_x2,
        max_running_requests=4,
    )
    native = ModuleType("sgl_kernel.flash_mla")

    def native_decode(**kwargs):
        packed = kwargs["k_cache"]
        assert packed.shape == (4, 64, 1, 416)
        assert packed.data_ptr() == pool.get_key_buffer(0).data_ptr()
        return kwargs["q"][..., :512].clone(), None, None, None

    native.flash_mla_with_kvcache_nvfp4 = native_decode
    native.flash_mla_with_kvcache = None
    with (
        get_context().override_server_args(
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_kv",
            speculative_num_draft_tokens=6,
            speculative_eagle_topk=1,
        ),
        get_parallel().override(attn_tp_size=1, attn_dcp_size=1, attn_dcp_rank=0),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch.dict(sys.modules, {"sgl_kernel.flash_mla": native}),
        patch.object(
            torch.ops.sgl_kernel, "sparse_decode_fwd_nvfp4", native_decode, create=True
        ),
    ):
        if page_size != 64:
            with pytest.raises(ValueError, match="page_size=128.*expected 64"):
                DeepseekSparseAttnBackend(runner)
            return
        backend = DeepseekSparseAttnBackend(runner)
        assert backend.physical_page_size == 64
        assert backend.speculative_num_draft_tokens == 6
        query = (
            torch.arange(64 * 576, dtype=torch.float32)
            .view(1, 64, 576)
            .to(torch.bfloat16)
        )
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([64], dtype=torch.int32),
            flashmla_metadata=SimpleNamespace(flashmla_metadata=None, num_splits=None),
        )
        layer = SimpleNamespace(layer_id=0, tp_q_head_num=64, head_dim=576)
        output = backend._forward_flashmla_kv(
            q_all=query,
            kv_cache=pool.get_key_buffer(0),
            v_head_dim=512,
            sm_scale=576**-0.5,
            layer=layer,
            metadata=metadata,
            page_table_1=torch.full((1, 2048), -1, dtype=torch.int32),
        )
        torch.testing.assert_close(output, query[:, None, :, :512], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,rows", [(ForwardMode.DECODE, 1), (ForwardMode.TARGET_VERIFY, 6)]
)
@pytest.mark.parametrize("split_query", [False, True])
def test_public_attention_routes_owned_cache_and_result(mode, rows, split_query):
    reference = _pool()
    pools = [_pool(rank) for rank in range(2)]
    _connect_result_transport(pools)
    generator = torch.Generator().manual_seed(127)
    full_q = torch.randn(rows, 64, 576, generator=generator).to(torch.bfloat16)
    k = torch.randn(rows, 1, 512, generator=generator).to(torch.bfloat16)
    k_rope = torch.randn(rows, 1, 64, generator=generator).to(torch.bfloat16)
    loc = torch.arange(65, 65 + rows)
    topk = loc[:, None].to(torch.int32)
    batch = SimpleNamespace(forward_mode=mode, out_cache_loc=loc)
    q = full_q[..., :512] if split_query else full_q
    q_rope = full_q[..., 512:] if split_query else None

    for layer_id in (0, 4):  # Both ranks own a layer, with uneven ownership.
        layer = SimpleNamespace(
            layer_id=layer_id,
            head_dim=576,
            v_head_dim=512,
            tp_q_head_num=64,
            is_cross_attention=False,
            scaling=576**-0.5,
        )
        with get_parallel().override(attn_dcp_size=1):
            reference.set_mla_kv_buffer(layer, loc, k, k_rope)
        expected_bytes = reference.get_key_buffer(layer_id)
        expected = (full_q[..., :512].float() + layer_id).to(torch.bfloat16)[:, None]

        def attention_kernel(**kwargs):
            # Verify the real packed writer/getter supplied the right layer.
            torch.testing.assert_close(
                kwargs["kv_cache"][loc], expected_bytes[loc], rtol=0, atol=0
            )
            torch.testing.assert_close(kwargs["q_all"], full_q, rtol=0, atol=0)
            torch.testing.assert_close(kwargs["page_table_1"], topk, rtol=0, atol=0)
            return (kwargs["q_all"][..., :512].float() + layer_id).to(torch.bfloat16)[
                :, None
            ]

        owner = pools[0]._get_layer_owner_rank(layer_id)
        for rank in (owner, 1 - owner):
            backend = _backend(pools[rank])
            backend._forward_flashmla_kv = attention_kernel
            method = (
                backend.forward_decode if mode.is_decode() else backend.forward_extend
            )
            with (
                get_parallel().override(attn_dcp_size=1),
                patch(
                    "sglang.srt.layers.attention.dsa_backend.concat_mla_absorb_q_general",
                    side_effect=lambda nope, rope: torch.cat((nope, rope), dim=-1),
                ),
            ):
                output = method(
                    q,
                    k,
                    k,
                    layer,
                    batch,
                    q_rope=q_rope,
                    k_rope=k_rope,
                    topk_indices=topk,
                )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


class _AfterInvalidation(Exception):
    pass


@pytest.mark.parametrize("entry", ["eager", "capture", "replay", "precomputed"])
def test_metadata_entries_invalidate_remote_scratch_after_graph_replay(entry):
    pool = _pool(0)
    pool.remote_kv_layer_id = 4
    pool.index_key_cache.remote_layer_id = 4
    backend = _backend(pool)
    backend.decode_cuda_graph_metadata = {1: SimpleNamespace()}

    def stop_after_invalidation(*args, **kwargs):
        raise _AfterInvalidation

    backend.set_dsa_prefill_impl = stop_after_invalidation
    seq_lens = torch.tensor([65])
    batch = SimpleNamespace(
        batch_size=1, seq_lens=seq_lens, forward_mode=ForwardMode.DECODE
    )
    common = dict(
        bs=1,
        req_pool_indices=torch.tensor([0]),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
    )
    with patch(
        "sglang.srt.layers.attention.dsa_backend.compute_cu_seqlens",
        stop_after_invalidation,
    ), pytest.raises(_AfterInvalidation):
        if entry == "eager":
            backend.init_forward_metadata(batch)
        elif entry == "capture":
            backend._build_forward_metadata_cuda_graph(num_tokens=1, **common)
        elif entry == "replay":
            backend._apply_cuda_graph_metadata(**common)
        else:
            backend.init_forward_metadata_replay_cuda_graph_from_precomputed(
                1, None, ForwardMode.DECODE
            )
    assert pool.remote_kv_layer_id is None
    assert pool.index_key_cache.remote_layer_id is None


@pytest.mark.parametrize(
    "mode,rows", [(ForwardMode.DECODE, 1), (ForwardMode.TARGET_VERIFY, 6)]
)
def test_public_indexer_routes_paged_reads_without_rank_zero_collective(mode, rows):
    pools = [_pool(rank) for rank in range(2)]
    _connect_result_transport(pools)
    layer_id = 4  # Owner is rank 1, not attention-TP rank zero.
    seq_lens = torch.tensor([64], dtype=torch.int32)
    metadata = SimpleNamespace(
        get_page_table_64=lambda: torch.tensor([[1]], dtype=torch.int32),
        get_seqlens_int32=lambda: seq_lens,
        get_seqlens_expanded=lambda: seq_lens.repeat(rows),
        get_dsa_extend_len_cpu=lambda: [rows],
        paged_mqa_schedule_metadata=torch.zeros(1, dtype=torch.int32),
        topk_transform=lambda logits, k: logits.topk(k, dim=-1).indices.to(torch.int32),
    )
    batch = SimpleNamespace(forward_mode=mode, seq_lens=seq_lens, attn_cp_metadata=None)
    x = torch.arange(rows * 4, dtype=torch.float32).view(rows, 4)
    expected = torch.tensor([63, 62, 61], dtype=torch.int32).expand(rows, -1)

    # The projection/store and paged-MQA kernels are the CUDA-only boundaries.
    # The public indexer and its paged-cache routing remain production code.
    def project_and_store(x, _q_lora, _positions, _batch, layer_id, _quant):
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        buffer = get_token_to_kv_pool().get_owned_index_buffer(layer_id)
        buffer.fill_(17)
        return x[:, None, :], torch.ones(rows, 1, 1)

    def paged_logits(_fn, query, cache, *args, **kwargs):
        assert torch.all(cache == 17)
        return torch.arange(64, dtype=torch.float32).expand(query.shape[0], -1)

    for rank in (1, 0):
        backend = _backend(pools[rank])
        backend.get_indexer_metadata = lambda _layer, _batch: metadata
        indexer = object.__new__(Indexer)
        indexer.index_topk = 3
        indexer.alt_stream = None
        indexer.dsa_enable_prefill_cp = True
        indexer.use_gfx950_fused_indexer = False
        indexer.use_dsa_indexer_fusion = True
        indexer.sm_count = 148
        indexer.paged_mqa_logits_backend = SimpleNamespace(
            is_cutedsl=lambda: False, is_aiter=lambda: False
        )
        indexer._should_skip_logits_computation = lambda _batch: False
        indexer._aiter_fused_fp8_active = lambda _batch: False
        indexer._fused_q_prepare_and_store = project_and_store
        indexer._mask_init_and_local_tokens = lambda *_args: None
        with (
            forward_context(ForwardContext(attn_backend=backend)),
            patch.object(dsa_indexer, "_is_cuda", True),
            patch.object(
                dsa_indexer,
                "_is_in_piecewise_or_breakable_cuda_graph",
                return_value=False,
            ),
            patch.object(
                dsa_indexer, "deepgemm_paged_mqa_logits_split", side_effect=paged_logits
            ),
            patch.object(
                dsa_indexer,
                "_broadcast_indexer_topk_from_rank0",
                side_effect=AssertionError("owner entered rank-zero collective"),
            ),
        ):
            result = indexer.forward_cuda(x, x, torch.arange(rows), batch, layer_id)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
