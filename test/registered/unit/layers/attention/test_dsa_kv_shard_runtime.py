# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for DSA/MTP sharding runtime integration.

CUDA/NCCL and complete-model validation remain separate: these exercise actual
Python dispatch, pool construction and address-space contracts with CPU tensors.
"""

import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.arg_groups import kv_shard_hook
from sglang.srt.mem_cache import kv_cache_configurator, page_interleave
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.page_interleave import PageShardSpec
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveDSATokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
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
        hicache_storage_backend=None,
        hicache_host_memory_mode="cache",
        hicache_io_backend="kernel",
        hicache_mem_layout="layer_first",
        enable_lmcache=False,
        enable_hisparse=False,
        enable_dynamic_chunking=False,
        enable_two_batch_overlap=False,
        chunked_prefill_size=512,
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
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


@pytest.mark.parametrize("layout", ["layer_first", "page_first"])
def test_glm_mtp_sharded_hicache_accepts_cpu_l2(gate, layout):
    cfg, _, declaration = gate
    cfg.enable_hierarchical_cache = True
    cfg.hicache_mem_layout = layout
    kv_shard_hook.handle_kv_cache_sharding(object(), gpu_mem=80 * 1024)
    graph = declaration.call_args.kwargs["cuda_graph_config"]
    assert graph.prefill.backend == Backend.DISABLED
    assert graph.decode.backend == Backend.DISABLED


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("attn_cp_size", 1, "DSA KV sharding requires"),
        ("hicache_storage_backend", "file", "not --hicache-storage-backend"),
        ("hicache_host_memory_mode", "buffer_only", "host memory mode cache"),
        ("hicache_io_backend", "direct", "requires kernel I/O"),
        ("hicache_mem_layout", "page_first_direct", "requires kernel I/O"),
        (
            "enable_unified_cache_external_linker",
            True,
            "does not support --enable-unified-cache-external-linker",
        ),
    ],
)
def test_sharded_hicache_rejects_unwired_paths(gate, name, value, error):
    cfg, _, _ = gate
    cfg.enable_hierarchical_cache = True
    setattr(cfg, name, value)
    with pytest.raises(ValueError, match=error):
        kv_shard_hook.handle_kv_cache_sharding(object(), gpu_mem=80 * 1024)


def test_generic_sharded_hicache_remains_enabled(gate):
    cfg, model, _ = gate
    cfg.enable_hierarchical_cache = True
    cfg.speculative_algorithm = None
    model.hf_config.architectures = ["DeepseekV3ForCausalLM"]
    del model.hf_config.index_topk
    with patch.object(
        kv_shard_hook, "attention_backends_of", return_value=("fa3", "fa3")
    ):
        kv_shard_hook.handle_kv_cache_sharding(object(), gpu_mem=80 * 1024)


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
            "sglang.srt.mem_cache.page_interleave.get_spec",
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
        patch.object(
            kv_cache_configurator, "get_shared_kv_shard_pool", return_value=target
        ),
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


def test_target_factory_selects_page_sharded_dsa_pool():
    kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
    kvc.kv_shard_rank = 1
    kvc.kv_shard_spec = PageShardSpec(1, 4, 64, 1024, 512)
    kvc.kv_shard_group = object()
    kvc.model_config = _model()
    kvc.is_draft_worker = False
    kvc.kv_cache_dtype = torch.float8_e4m3fn
    kvc.layer_info = SimpleNamespace(start_layer=0, end_layer=3, num_effective_layers=3)
    kvc.device = "cuda"
    with (
        patch(
            "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
            return_value=(None, 1),
        ),
        patch.object(
            kv_cache_configurator,
            "get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1),
        ),
        patch.object(
            kv_cache_configurator,
            "get_schedule",
            return_value=SimpleNamespace(page_size=64),
        ),
        patch.object(
            kv_cache_configurator,
            "get_memory",
            return_value=SimpleNamespace(enable_hisparse=False),
        ),
        patch.object(
            kv_cache_configurator,
            "get_exec",
            return_value=SimpleNamespace(
                features=SimpleNamespace(enable_memory_saver=False)
            ),
        ),
        patch.object(
            kv_cache_configurator, "_should_elide_dsa_index_k", return_value=False
        ),
        patch.object(
            kv_cache_configurator, "max_speculative_num_draft_tokens", return_value=4
        ),
        patch.object(
            kv_cache_configurator, "calculate_mla_kv_cache_dim", return_value=576
        ),
        patch.object(
            PageInterleaveDSATokenToKVPool, "__init__", return_value=None
        ) as init,
    ):
        pool = kvc._build_dsa_kv_pool(
            max_total_num_tokens=4096, max_running_requests=8, dsa_pool_class=object
        )
    assert isinstance(pool, PageInterleaveDSATokenToKVPool)
    assert init.call_args.args == (4096,)
    assert init.call_args.kwargs["shard_spec"] is kvc.kv_shard_spec
    assert init.call_args.kwargs["shard_group"] is kvc.kv_shard_group
    assert init.call_args.kwargs["kv_cache_dim"] == 576
    assert init.call_args.kwargs["index_head_dim"] == 128


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
