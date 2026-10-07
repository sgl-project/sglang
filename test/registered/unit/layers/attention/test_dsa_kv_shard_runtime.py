# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for DSA/MTP sharding runtime integration.

CUDA/NCCL and complete-model validation remain separate: these exercise actual
Python dispatch, pool construction and address-space contracts with CPU tensors.
"""

import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.arg_groups import kv_shard_hook
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.mem_cache import kv_cache_configurator, page_interleave
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.page_interleave import PageShardSpec
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveDSATokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import flashinfer_autotune
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _model():
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
            index_kpool=1,
            index_kpool_compress=False,
        ),
        num_nextn_predict_layers=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        is_encoder_decoder=False,
        attention_chunk_size=None,
    )


def _cfg(**overrides):
    cfg = SimpleNamespace(
        device="cuda",
        enable_kv_cache_sharding=True,
        enable_prefill_cp=True,
        attn_cp_size=4,
        cp_strategy="interleave",
        pp_size=1,
        tp_size=4,
        dcp_size=1,
        dsa_prefill_backend="trtllm",
        dsa_decode_backend="trtllm",
        kv_cache_dtype="fp8_e4m3",
        page_size=64,
        disaggregation_transfer_backend="mooncake",
        enable_dsa_cache_layer_split=False,
        disaggregation_mode="prefill",
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        enable_multi_layer_eagle=False,
        dllm_algorithm=None,
        radix_cache_backend=None,
        disable_radix_cache=False,
        enable_unified_cache_external_linker=False,
        enable_streaming_session=False,
        enable_session_radix_cache=False,
        enable_flexkv=False,
        enable_unified_memory=False,
        enable_hierarchical_cache=False,
        enable_lmcache=False,
        enable_hisparse=False,
        enable_dynamic_chunking=False,
        enable_two_batch_overlap=False,
        chunked_prefill_size=512,
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.BREAKABLE),
            decode=PhaseConfig(backend=Backend.FULL),
        ),
    )
    vars(cfg).update(overrides)
    return cfg


@pytest.fixture
def gate():
    model = _model()
    cfg = _cfg()
    with ExitStack() as stack:
        for name in ("resolving_view", "resolved_view"):
            stack.enter_context(patch.object(kv_shard_hook, name, return_value=cfg))
        stack.enter_context(
            patch.object(kv_shard_hook, "model_config_of", return_value=model)
        )
        stack.enter_context(
            patch.object(
                kv_shard_hook, "attention_backends_of", return_value=("dsa", "dsa")
            )
        )
        stack.enter_context(
            patch.object(kv_shard_hook, "use_mla_backend", return_value=True)
        )
        stack.enter_context(
            patch.object(
                kv_shard_hook.envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND,
                "get",
                return_value="python",
            )
        )
        stack.enter_context(
            patch.object(
                kv_shard_hook.envs.SGLANG_DISAGG_STAGING_BUFFER,
                "get",
                return_value=False,
            )
        )
        declaration = stack.enter_context(
            patch.object(kv_shard_hook, "declare_resolution")
        )
        yield cfg, model, declaration


def test_glm_mtp_gate_disables_both_graph_phases(gate):
    _, _, declaration = gate
    kv_shard_hook.handle_kv_cache_sharding(object(), gpu_mem=80 * 1024)
    graph = declaration.call_args.kwargs["cuda_graph_config"]
    assert graph.prefill.backend == Backend.DISABLED
    assert graph.decode.backend == Backend.DISABLED


def test_dsa_cannot_enter_dense_backend_sharding(gate):
    with patch.object(
        kv_shard_hook, "attention_backends_of", return_value=("fa3", "fa3")
    ):
        with pytest.raises(ValueError, match="requires the dsa attention backend"):
            kv_shard_hook.handle_kv_cache_sharding(object(), gpu_mem=80 * 1024)


@pytest.mark.parametrize(
    "name,value",
    [
        ("device", "hip"),
        ("cp_strategy", "zigzag"),
        ("attn_cp_size", 1),
        ("pp_size", 2),
        ("dsa_prefill_backend", "flashmla_sparse"),
        ("dsa_decode_backend", "flashmla_sparse"),
        ("kv_cache_dtype", "bf16"),
        ("page_size", 128),
        ("disaggregation_transfer_backend", "nixl"),
        ("enable_dsa_cache_layer_split", True),
    ],
)
def test_dsa_gate_rejects_unwired_layouts(gate, name, value):
    cfg, _, _ = gate
    setattr(cfg, name, value)
    with pytest.raises(ValueError):
        kv_shard_hook.handle_kv_cache_sharding(object(), gpu_mem=80 * 1024)


@pytest.mark.parametrize(
    "name,value", [("index_kpool", 4), ("index_kpool_compress", True)]
)
def test_dsa_gate_rejects_pooled_indexer(gate, name, value):
    _, model, _ = gate
    setattr(model.hf_config, name, value)
    with pytest.raises(ValueError, match="DSA KV sharding requires"):
        kv_shard_hook.handle_kv_cache_sharding(object(), gpu_mem=80 * 1024)


@pytest.mark.parametrize(
    "overrides",
    [
        {"speculative_algorithm": "EAGLE3"},
        {"speculative_eagle_topk": 2},
        {"enable_multi_layer_eagle": True},
        {"pp_size": 2},
    ],
)
def test_mtp_gate_rejects_unvalidated_drafts(gate, overrides):
    cfg, _, _ = gate
    vars(cfg).update(overrides)
    with pytest.raises(ValueError, match="single-layer NextN"):
        kv_shard_hook.handle_kv_cache_sharding(object(), gpu_mem=80 * 1024)


def test_indexer_batched_reads_remain_logical_but_raw_reads_use_scratch():
    logical = torch.tensor([[9, 4, 17]], dtype=torch.int32)
    page_pos = torch.arange(30, dtype=torch.int32).flip(0)
    scratch = page_pos[logical.long()]
    meta = DSAIndexerMetadata(
        SimpleNamespace(real_page_table=scratch, logical_indexer_page_table=logical),
        TopkTransformMethod.PAGED,
    )
    assert meta.get_page_table_64() is scratch
    assert meta.get_batched_indexer_page_table() is logical
    assert torch.equal(page_pos[meta.get_batched_indexer_page_table().long()], scratch)
    # A second translation would silently address unrelated pages.
    assert not torch.equal(page_pos[scratch.long()], scratch)


def test_nonsharded_indexer_uses_ordinary_table():
    table = torch.tensor([[4, 5]], dtype=torch.int32)
    meta = DSAIndexerMetadata(
        SimpleNamespace(real_page_table=table, logical_indexer_page_table=None),
        TopkTransformMethod.PAGED,
    )
    assert meta.get_batched_indexer_page_table() is table


@pytest.mark.parametrize("cp_select", [False, True])
def test_extend_metadata_keeps_two_address_spaces_and_starts_one_gather_plan(cp_select):
    pool = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
    pool.page_size, pool.size = 64, 128
    pool.dtype = torch.float8_e4m3fn
    pool.shard_spec = PageShardSpec(0, 4, 64, 1024, 512)
    pool.begin_shard_extend = Mock()
    page_pos = torch.tensor([0, 6, 5, 4, 3, 2, 1, 7], dtype=torch.int64)
    pool.translate_loc_to_scratch = (
        lambda loc: page_pos[loc.long() // 64] * 64 + loc % 64
    )
    req_table = torch.stack(
        [
            torch.cat([torch.arange(p * 64, (p + 1) * 64) for p in [2, 5, 7]]),
            torch.cat([torch.arange(p * 64, (p + 1) * 64) for p in [1, 3, 6]]),
        ]
    ).to(torch.int32)
    backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
        dsa_backend.DeepseekSparseAttnBackend
    )
    backend.kv_shard_pool = backend.token_to_kv_pool = pool
    backend.req_to_token = req_table
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_table)
    backend.physical_page_size = 64
    backend.dsa_index_kpool = 1
    backend.dsa_index_topk = 2048
    backend.use_mha = False
    backend.dsa_prefill_impl = backend.dsa_decode_impl = "trtllm"
    backend.set_dsa_prefill_impl = lambda _: None
    backend.get_topk_transform_method = lambda _: TopkTransformMethod.RAGGED
    backend._cal_indexer_k_start_end = lambda *_: (None, None)
    backend._build_topk_v2_plan = lambda _: None
    backend._init_kpool_metadata = lambda metadata, *args, **kwargs: metadata
    backend._arange_buf = torch.arange(512, dtype=torch.int32)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([128, 192]),
        seq_lens_cpu=torch.tensor([128, 192]),
        seq_lens_sum=320,
        extend_prefix_lens_cpu=[64, 128],
        extend_seq_lens_cpu=[64, 64],
        extend_seq_lens=torch.tensor([64, 64]),
    )
    with (
        patch.object(
            dsa_backend, "can_dsa_prefill_cp_interleave", return_value=cp_select
        ),
        patch.object(
            dsa_backend,
            "get_cp_strategy",
            return_value=SimpleNamespace(
                shard_local_tokens=lambda x: x[64:],
                shard_per_request=lambda cpu, gpu: (
                    [cpu[1]],
                    gpu[1:],
                    [1],
                    torch.tensor([1]),
                ),
            ),
        ),
        patch.object(
            dsa_backend,
            "compute_dsa_seqlens",
            side_effect=lambda **kw: kw["original_seq_lens"].int(),
        ),
        patch.object(
            dsa_backend, "pad_dsa_cache_seqlens", side_effect=lambda _batch, x: x
        ),
        patch.object(dsa_backend, "is_cuda", return_value=False),
    ):
        backend.init_forward_metadata(batch)
    pool.begin_shard_extend.assert_called_once()
    meta = backend.forward_metadata
    if cp_select:
        req_table = req_table[1:]
    assert torch.equal(meta.logical_indexer_page_table, req_table[:, ::64] // 64)
    assert torch.equal(
        meta.real_page_table, page_pos[(req_table[:, ::64] // 64).long()]
    )
    assert meta.page_table_1_flattened.shape == (192 if cp_select else 320,)


def test_indexer_store_uses_pool_owner_filter_instead_of_direct_fused_writer():
    pool = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
    pool.page_size = 64
    pool.set_index_k_scale_buffer = Mock()
    indexer = SimpleNamespace(block_size=128, scale_fmt="ue8m0")
    loc = torch.tensor([512, 576], dtype=torch.int64)
    key = torch.randn(2, 128, dtype=torch.bfloat16)
    quantized = key.to(torch.float8_e4m3fn)
    scale = torch.ones((2, 1), dtype=torch.float32)
    quant = Mock(return_value=(quantized, scale))
    with (
        patch.object(dsa_indexer, "get_token_to_kv_pool", return_value=pool),
        patch.object(dsa_indexer, "_is_cuda", True),
        patch.object(dsa_indexer, "_is_fp8_fnuz", False),
        patch.object(dsa_indexer, "_use_aiter", False),
        patch.object(
            dsa_indexer,
            "fused_store_index_k_cache",
            side_effect=AssertionError("must not bypass pool"),
        ),
    ):
        dsa_indexer.Indexer._store_index_k_cache(
            indexer, SimpleNamespace(out_cache_loc=loc), 7, key, act_quant=quant
        )
    quant.assert_called_once_with(key, 128, "ue8m0")
    kwargs = pool.set_index_k_scale_buffer.call_args.kwargs
    assert kwargs["layer_id"] == 7
    assert kwargs["loc"] is loc
    assert kwargs["index_k"] is quantized
    assert kwargs["index_k_scale"] is scale


@pytest.mark.parametrize(
    "dtype,kv_dim",
    [(torch.bfloat16, 576), (torch.float8_e4m3fn, 576), (torch.float8_e4m3fn, 656)],
)
def test_scratch_budget_includes_indexer_for_target_and_draft(dtype, kv_dim):
    spec = PageShardSpec(0, 4, 64, 1024, 512)
    kvc = SimpleNamespace(
        model_config=_model(),
        kv_cache_dtype=dtype,
        use_mla_backend=True,
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        spec_aux_config=SimpleNamespace(eagle_draft_num_layers=1),
    )
    with (
        patch.object(page_interleave, "make_page_shard_spec", return_value=spec),
        patch.object(
            kv_cache_configurator, "calculate_mla_kv_cache_dim", return_value=kv_dim
        ),
        patch(
            "sglang.srt.runtime_context.get_spec",
            return_value=SimpleNamespace(enable_multi_layer_eagle=False),
        ),
    ):
        per_pair = 2 * spec.scratch_rows * (kv_dim * dtype.itemsize + 132)
        assert page_interleave.compute_page_shard_scratch_bytes(kvc) == per_pair
        assert (
            page_interleave.compute_page_shard_scratch_bytes(kvc, include_mtp=True)
            == 2 * per_pair
        )


def test_generic_decode_dummy_autotune_does_not_access_sharded_prefill_pool():
    with patch.object(
        flashinfer_autotune,
        "get_parallel",
        return_value=SimpleNamespace(enable_kv_cache_sharding=True),
    ):
        assert not flashinfer_autotune.should_run_flashinfer_autotune(
            SimpleNamespace(device="cuda")
        )


@pytest.mark.parametrize("start_layer", [0, 5])
def test_dsa_draft_inherits_geometry_but_keeps_its_own_indexer(start_layer):
    target = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
    target.shard_spec = PageShardSpec(1, 4, 64, 1024, 512)
    target.shard_group = object()
    target.size = 4096
    target.page_size = 64
    target.dtype = torch.float8_e4m3fn
    target.kv_lora_rank, target.qk_rope_head_dim = 512, 64
    target.kv_cache_dim, target.index_head_dim = 576, 128
    kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
    kvc.is_draft_worker = True
    kvc.spec_algorithm = SpeculativeAlgorithm.EAGLE
    kvc.model_config = _model()
    kvc.page_size = 64
    kvc.kv_cache_dtype = target.dtype
    kvc.kv_cache_dtype_str = "fp8_e4m3"
    kvc.is_hybrid_swa = False
    kvc.mambaish_config = None
    kvc.sliding_window_size = None
    kvc.post_capture_kv_active = False
    kvc.use_mla_backend = True
    kvc.device = "cuda"
    kvc.layer_info = SimpleNamespace(
        num_effective_layers=1, start_layer=start_layer, end_layer=start_layer + 1
    )
    with (
        patch.object(page_interleave, "get_shared_kv_shard_pool", return_value=target),
        patch.object(
            kv_cache_configurator,
            "get_spec",
            return_value=SimpleNamespace(enable_multi_layer_eagle=False),
        ),
        patch.object(
            kv_cache_configurator,
            "get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1),
        ),
        patch.object(
            kv_cache_configurator,
            "get_memory",
            return_value=SimpleNamespace(enable_page_major_kv_layout=False),
        ),
        patch.object(
            kv_cache_configurator,
            "get_schedule",
            return_value=SimpleNamespace(page_size=64),
        ),
        patch.object(
            kv_cache_configurator,
            "get_exec",
            return_value=SimpleNamespace(
                features=SimpleNamespace(enable_memory_saver=False)
            ),
        ),
        patch.object(
            kv_cache_configurator, "calculate_mla_kv_cache_dim", return_value=576
        ),
        patch.object(
            PageInterleaveDSATokenToKVPool, "__init__", return_value=None
        ) as init,
    ):
        draft = kvc._build_mtp_kv_shard_pool(
            sizes=SimpleNamespace(max_total_num_tokens=4096),
            is_dsa_model=True,
            is_dsv4_model=False,
        )
    assert isinstance(draft, PageInterleaveDSATokenToKVPool)
    assert draft is not target
    kwargs = init.call_args.kwargs
    assert kwargs["shard_spec"] is target.shard_spec
    assert kwargs["shard_group"] is target.shard_group
    assert kwargs["start_layer"] == start_layer
    assert kwargs["end_layer"] == start_layer + 1
    assert kwargs["kv_cache_dim"] == 576
    assert kwargs["index_head_dim"] == 128
    assert kwargs["index_kpool"] == 1
    assert "skip_topk_layers" not in kwargs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
