# SPDX-License-Identifier: Apache-2.0
"""Mixed-format DSA cache contracts; no model weights or GPU required."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.dsa.nvfp4_config import validate_dsa_nvfp4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _geometry(**kwargs):
    values = dict(kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_glm_geometry_and_hopper_are_accepted():
    validate_dsa_nvfp4(SimpleNamespace(), _geometry(), is_sm90=True)


@pytest.mark.parametrize(
    "name,value",
    [
        ("dcp_size", 2),
        ("attn_cp_size", 2),
        ("pp_size", 2),
        ("dtype", "float16"),
        ("enable_hierarchical_cache", True),
        ("enable_unified_memory", True),
        ("enable_dsa_cache_layer_split", True),
        ("enable_hisparse", True),
        ("enable_prefill_cp", True),
        ("enable_two_batch_overlap", True),
        ("enable_lmcache", True),
        ("speculative_algorithm", "EAGLE"),
        ("disaggregation_mode", "prefill"),
        ("page_size", 128),
        ("dsa_prefill_backend", "flashmla_sparse_q8"),
        ("dsa_decode_backend", "fa3"),
    ],
)
def test_incompatible_cache_consumers_are_rejected(name, value):
    with pytest.raises(ValueError):
        validate_dsa_nvfp4(SimpleNamespace(**{name: value}), _geometry(), is_sm90=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"kv_lora_rank": 256},
        {"qk_rope_head_dim": 128},
        {"index_topk": 1024},
        {"index_kpool": 4},
        {"learnable_sink": True},
    ],
)
def test_incompatible_model_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        validate_dsa_nvfp4(SimpleNamespace(), _geometry(**changes), is_sm90=True)


def test_non_hopper_is_rejected():
    with pytest.raises(ValueError, match="SM90"):
        validate_dsa_nvfp4(SimpleNamespace(), _geometry(), is_sm90=False)


@pytest.fixture
def pool():
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        yield DSATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.float4_e2m1fn_x2,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            kv_cache_dim=416,
            index_head_dim=128,
            skip_topk_layers=[False, True, False],
        )
    finally:
        reset_context()


def test_pool_storage_accounting_and_shared_indexers(pool):
    assert pool.get_key_buffer(0).dtype == torch.uint8
    assert pool.get_key_buffer(0).shape == (192, 1, 416)
    assert pool.get_value_buffer(0).data_ptr() == pool.get_key_buffer(0).data_ptr()
    index_buffers = pool.index_k_with_scale_buffer
    assert [tuple(x.shape) for x in index_buffers] == [
        (3, 64 * 132),
        (0, 64 * 132),
        (3, 64 * 132),
    ]
    assert all(x.dtype == torch.uint8 for x in index_buffers)
    assert pool.get_kv_size_bytes() == 3 * 192 * 416 + 2 * 192 * 132 + 3 * 4
    assert pool.mla_kv_global_scale.dtype == torch.float32


def test_pool_writer_and_gather_preserve_rope_and_layer_isolation(pool):
    layer = SimpleNamespace(layer_id=1)
    loc = torch.tensor([127, 64, 11], dtype=torch.int64)
    # All E2M1 values in this pattern are exactly representable at scale=1.
    pattern = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
        ],
        dtype=torch.bfloat16,
    )
    latent = pattern.repeat(3, 1, 32)
    rope = torch.arange(192, dtype=torch.float32).reshape(3, 1, 64).to(torch.bfloat16)
    scale_ptr = pool.get_mla_kv_global_scale(1).data_ptr()
    pool.set_mla_kv_buffer(layer, loc, latent, rope)
    got_latent, got_rope = pool.get_mla_kv_buffer(layer, loc.flip(0))
    torch.testing.assert_close(got_latent, latent.flip(0), rtol=0, atol=0)
    torch.testing.assert_close(got_rope, rope.flip(0), rtol=0, atol=0)
    assert pool.get_mla_kv_global_scale(1).data_ptr() == scale_ptr
    assert not pool.get_key_buffer(0).any()
    assert not pool.get_key_buffer(2).any()
    assert not pool.index_k_with_scale_buffer[0].any()
    assert pool.get_key_buffer(1)[0].count_nonzero() == 0


def test_pool_cache_relocation_copies_scales_and_rope(pool):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    layer = SimpleNamespace(layer_id=0)
    src = torch.tensor([65, 66])
    dst = torch.tensor([129, 130])
    latent = torch.randn(
        2, 1, 512, generator=torch.Generator().manual_seed(9)
    ).bfloat16()
    rope = torch.randn(2, 1, 64, generator=torch.Generator().manual_seed(10)).bfloat16()
    pool.set_mla_kv_buffer(layer, src, latent, rope)
    expected = pool.get_key_buffer(0)[src].clone()
    # Exercise the main-cache relocation; indexer relocation has its own page ABI.
    MLATokenToKVPool.move_kv_cache(pool, dst, src)
    assert torch.equal(pool.get_key_buffer(0)[dst], expected)
    got_latent, got_rope = pool.get_mla_kv_buffer(layer, dst)
    assert got_latent.isfinite().all()
    assert torch.equal(got_rope, rope)


def test_layer_override_uses_matching_scale_and_cache(pool):
    pool.mla_kv_global_scale[2] = 0.5
    layer = SimpleNamespace(layer_id=0)
    loc = torch.tensor([64])
    latent = torch.full((1, 1, 512), 3.0, dtype=torch.bfloat16)
    rope = torch.ones(1, 1, 64, dtype=torch.bfloat16)
    pool.set_mla_kv_buffer(layer, loc, latent, rope, layer_id_override=2)
    got, _ = pool.get_mla_kv_buffer(SimpleNamespace(layer_id=2), loc)
    assert torch.equal(got, latent)
    assert not pool.get_key_buffer(0).any()
    with pytest.raises(IndexError):
        pool.get_mla_kv_global_scale(3)


@pytest.mark.parametrize("has_prefix", [False, True])
def test_prefill_consumes_bf16_but_keeps_persistent_cache_packed(
    pool, monkeypatch, has_prefix
):
    from sglang.srt.layers.attention import dsa_backend as module
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    backend = object.__new__(module.DeepseekSparseAttnBackend)
    backend.token_to_kv_pool = pool
    backend.dsa_kv_cache_store_nvfp4 = True
    backend.dsa_kv_cache_store_fp8 = False
    backend.dsa_index_kpool = 1
    backend.dsa_prefill_impl = "flashmla_sparse"
    backend.dsa_decode_impl = "flashmla_kv"
    backend.use_mha = False
    backend.use_fused_topk = False
    backend.hisparse_coordinator = None
    backend.forward_metadata = SimpleNamespace(
        topk_indices_offset=torch.zeros(2, dtype=torch.int32),
        page_table_1_flattened=torch.tensor([91, 65, 127] if has_prefix else [65, 127]),
        dsa_cache_seqlens_int32=torch.tensor(
            [2, 3] if has_prefix else [1, 2], dtype=torch.int32
        ),
    )
    layer = SimpleNamespace(
        layer_id=0,
        is_cross_attention=False,
        tp_q_head_num=8,
        head_dim=576,
        v_head_dim=512,
        scaling=576**-0.5,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        out_cache_loc=torch.tensor([65, 127]),
        extend_prefix_lens_cpu=[1 if has_prefix else 0],
    )
    g = torch.Generator().manual_seed(2026)
    k = torch.randn(2, 1, 512, generator=g).bfloat16()
    rope = torch.randn(2, 1, 64, generator=g).bfloat16()
    if has_prefix:
        pool.set_mla_kv_buffer(layer, torch.tensor([91]), k[:1] * 2, rope[:1] * 2)
    captured = {}

    def attention(**kwargs):
        captured.update(kwargs)
        return torch.zeros(2, 8, 512, dtype=torch.bfloat16)

    backend._forward_flashmla_sparse = attention
    # Keep dispatch/writer/gather real; isolate the CUDA-only concatenation and attention.
    monkeypatch.setattr(module, "_cat", lambda xs, dim=-1: torch.cat(xs, dim=dim))
    monkeypatch.setattr(
        module, "concat_mla_absorb_q_general", lambda a, b: torch.cat([a, b], dim=-1)
    )
    topk = torch.tensor(
        [[0, 1, -1], [0, 1, 2]] if has_prefix else [[0, -1, -1], [0, 1, -1]],
        dtype=torch.int32,
    )
    backend.forward_extend(
        torch.zeros(2, 8, 576, dtype=torch.bfloat16),
        k,
        k,
        layer,
        batch,
        k_rope=rope,
        topk_indices=topk,
    )
    consumed = captured["kv_cache"]
    assert consumed.dtype == torch.bfloat16
    assert consumed.shape == (3 if has_prefix else 2, 1, 576)
    assert pool.get_key_buffer(0).shape[-1] == 416
    assert pool.get_key_buffer(0).dtype == torch.uint8
    if has_prefix:
        latent, restored_rope = pool.get_mla_kv_buffer(
            layer, backend.forward_metadata.page_table_1_flattened
        )
        assert torch.equal(consumed, torch.cat([latent, restored_rope], dim=-1))
    else:
        # The prefill attention must see the original current KV, before FP4 rounding.
        assert torch.equal(consumed, torch.cat([k, rope], dim=-1))
        latent, _ = pool.get_mla_kv_buffer(layer, batch.out_cache_loc)
        assert not torch.equal(latent, k)
    assert torch.equal(captured["page_table_1"], topk)


@pytest.mark.parametrize("glm5_next", [False, True])
def test_server_hooks_select_nvfp4_consumers(monkeypatch, glm5_next):
    from sglang.srt.arg_groups.kv_cache_hook import (
        handle_kv4_compatibility,
        handle_nvfp4_prefill_kv_dequant_dtype,
    )
    from sglang.srt.arg_groups.overrides import (
        ResolvedView,
        _dsa_kv_cache_dtype_default,
        _dsa_split_backend_resolution,
        declare_resolution,
        resolution_result,
    )
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.runtime_context import override_platform
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(model_path="dummy", kv_cache_dtype="nvfp4", dtype="bfloat16")
    args._model_config = SimpleNamespace(
        hf_config=(
            _flash_geometry()
            if glm5_next
            else _geometry(architectures=["GlmMoeDsaForCausalLM"])
        ),
        attention_arch=AttentionArch.MLA,
    )
    with override_platform(
        is_cuda=True,
        is_sm90=True,
        is_sm100=False,
        is_sm120=False,
        is_npu=False,
        is_xpu=False,
        is_hip=False,
    ):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
        assert _dsa_kv_cache_dtype_default(ResolvedView(args)) == {}
        backends = _dsa_split_backend_resolution(ResolvedView(args))
        assert backends == dict(
            dsa_prefill_backend="fa3" if glm5_next else "flashmla_sparse",
            dsa_decode_backend="fa3" if glm5_next else "flashmla_kv",
        )
        declare_resolution(args, "test_dsa", **backends)
        handle_nvfp4_prefill_kv_dequant_dtype(args)
        handle_kv4_compatibility(args)
        assert resolution_result(args, "prefill_kv_cache_dequant_dtype") == "auto"
        assert resolution_result(args, "decode_attention_backend") != "trtllm_mha"


def test_capacity_estimator_counts_packed_rows_once_and_only_full_indexers(pool):
    from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

    hf = _geometry(
        architectures=["GlmMoeDsaForCausalLM"],
        index_head_dim=128,
        indexer_types=["full", "shared", "full"],
    )
    model = SimpleNamespace(hf_config=hf, kv_lora_rank=512, qk_rope_head_dim=64)
    kvc = SimpleNamespace(
        model_config=model,
        kv_cache_dtype=torch.float4_e2m1fn_x2,
        use_mla_backend=True,
        is_draft_worker=False,
        server_args=SimpleNamespace(enable_hisparse=False),
        mambaish_config=None,
        layer_info=SimpleNamespace(start_layer=0, end_layer=3),
    )
    assert (
        calculate_mla_kv_cache_dim(
            model_config=model, kv_cache_dtype=kvc.kv_cache_dtype
        )
        == 416
    )
    planner = object.__new__(DefaultPoolConfigurator)
    per_token = planner._compute_cell_size(kvc, num_layers=3)
    assert per_token == 3 * 416 + 2 * 132
    assert per_token * (pool.size + pool.page_size) + 3 * 4 == pool.get_kv_size_bytes()


def _flash_geometry(**kwargs):
    values = dict(
        architectures=["Glm5NextForConditionalGeneration"],
        qk_rope_head_dim=0,
        index_kpool=4,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
    )
    values.update(kwargs)
    return _geometry(**values)


def test_glm5_next_fa3_contract():
    validate_dsa_nvfp4(
        SimpleNamespace(dsa_prefill_backend="fa3", dsa_decode_backend="fa3"),
        _flash_geometry(),
        is_sm90=True,
    )


@pytest.mark.parametrize("field", ["dsa_prefill_backend", "dsa_decode_backend"])
@pytest.mark.parametrize("backend", ["flashmla_sparse", "flashmla_kv", "tilelang"])
def test_glm5_next_rejects_consumers_of_other_cache_layouts(field, backend):
    with pytest.raises(ValueError, match="requires.*fa3"):
        validate_dsa_nvfp4(
            SimpleNamespace(**{field: backend}), _flash_geometry(), is_sm90=True
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"index_kpool": 1},
        {"index_kpool_compress": False},
        {"index_kpool_always_select_tail": False},
        {"qk_rope_head_dim": 64},
    ],
)
def test_glm5_next_rejects_unsupported_indexer_or_geometry(changes):
    with pytest.raises(ValueError):
        validate_dsa_nvfp4(SimpleNamespace(), _flash_geometry(**changes), is_sm90=True)


@pytest.fixture
def flash_pool():
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        yield HybridLinearKVPool(
            size=128,
            page_size=64,
            dtype=torch.float4_e2m1fn_x2,
            head_num=1,
            head_dim=512,
            full_attention_layer_ids=[3, 7],
            device="cpu",
            mamba_pool=SimpleNamespace(),
            use_mla=True,
            use_dsa=True,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            index_head_dim=128,
            kv_cache_dim=288,
            index_kpool=4,
            index_kpool_compress=True,
            max_running_requests=2,
        )
    finally:
        reset_context()


def test_hybrid_nvfp4_mapping_and_separate_indexer(flash_pool):
    pool = flash_pool
    assert pool.dsa_kv_cache_store_nvfp4
    assert not pool.dsa_kv_cache_store_fp8
    assert pool.kv_cache_dim == 288
    assert pool.get_key_buffer(7).shape == (192, 1, 288)
    pool.full_kv_pool.mla_kv_global_scale[1] = 0.5
    assert pool.get_mla_kv_global_scale(7).item() == 0.5
    layer = SimpleNamespace(layer_id=7)
    loc = torch.tensor([63, 64, 127])
    latent = torch.full((3, 1, 512), 3.0, dtype=torch.bfloat16)
    pool.set_mla_kv_buffer(layer, loc, latent, None)
    restored, rope = pool.get_mla_kv_buffer(layer, loc.flip(0))
    assert layer.layer_id == 7
    assert rope is None
    assert torch.equal(restored, latent.flip(0))
    assert not pool.get_key_buffer(3).any()
    index = pool.get_index_k_with_scale_buffer(7)
    assert index.dtype == torch.uint8 and index.shape == (3, 64 * 132)
    assert not index.any()
    for tail in pool.get_compress_tail_buffers(7):
        assert tail.dtype == torch.bfloat16
        assert not tail.any()
    with pytest.raises(ValueError, match="not in full attention layers"):
        pool.get_mla_kv_global_scale(0)


@pytest.mark.parametrize("mode", ["extend", "decode"])
@pytest.mark.parametrize("tail_tokens", [0, 1, 2, 3])
def test_glm5_next_fa3_consumes_dequantized_rows_and_preserves_tail(
    flash_pool, monkeypatch, mode, tail_tokens
):
    from sglang.srt.layers.attention import dsa_backend as module
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    backend = object.__new__(module.DeepseekSparseAttnBackend)
    backend.token_to_kv_pool = flash_pool
    backend.dsa_kv_cache_store_nvfp4 = True
    backend.dsa_kv_cache_store_fp8 = False
    backend.dsa_index_kpool = 4
    backend.dsa_prefill_impl = backend.dsa_decode_impl = "fa3"
    backend.use_mha = False
    backend.use_fused_topk = True
    backend.hisparse_coordinator = None
    # Keep writer, hybrid layer mapping, backend dispatch and gather real;
    # supply already-transformed physical indices and stub only attention.
    backend._get_fused_topk_page_table = lambda indices: indices
    backend._pad_topk_indices = lambda indices, rows: indices
    n = 4 + tail_tokens
    lengths = torch.tensor([n], dtype=torch.int32)
    backend.forward_metadata = SimpleNamespace(
        dsa_cache_seqlens_int32=lengths,
        dsa_cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        dsa_cu_seqlens_k=torch.tensor([0, n], dtype=torch.int32),
        dsa_max_seqlen_q=1,
    )
    layer = SimpleNamespace(
        layer_id=7,
        is_cross_attention=False,
        tp_q_head_num=8,
        head_dim=512,
        v_head_dim=512,
        scaling=256**-0.5,
        logit_cap=0.0,
    )
    loc = torch.arange(61, 61 + n)
    latent = torch.arange(1, n + 1).reshape(n, 1, 1).expand(n, 1, 512).bfloat16()
    flash_pool.set_mla_kv_buffer(layer, loc[:-1], latent[:-1], None)
    indices = torch.full((1, 2051), -1, dtype=torch.int32)
    indices[0, :n] = loc.to(torch.int32)
    captured = {}

    def attention(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 8, 512, dtype=torch.bfloat16)

    backend._forward_fa3 = attention
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND if mode == "extend" else ForwardMode.DECODE,
        out_cache_loc=loc[-1:],
    )
    forward = backend.forward_extend if mode == "extend" else backend.forward_decode
    forward(
        torch.zeros(1, 8, 512, dtype=torch.bfloat16),
        latent[-1:],
        latent[-1:],
        layer,
        batch,
        k_rope=None,
        topk_indices=indices,
    )
    restored, _ = flash_pool.get_mla_kv_buffer(layer, loc)
    assert captured["kv_cache"].shape == (2051, 1, 512)
    assert captured["kv_cache"].dtype == torch.bfloat16
    assert torch.equal(captured["kv_cache"][:n], restored)
    assert not captured["kv_cache"][n:].any()
    assert torch.equal(
        captured["page_table"][0, :n], torch.arange(n, dtype=torch.int32)
    )
    assert (captured["page_table"][0, n:] == -1).all()
    assert torch.equal(captured["cache_seqlens"], lengths)
    assert flash_pool.get_key_buffer(7).dtype == torch.uint8
    assert not flash_pool.get_key_buffer(3).any()


def test_flash_capacity_uses_dsa_layers_and_288_byte_rows(flash_pool):
    from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

    hf = _flash_geometry(index_head_dim=128, indexer_types=["full"] * 8)
    model = SimpleNamespace(hf_config=hf, kv_lora_rank=512, qk_rope_head_dim=0)
    kvc = SimpleNamespace(
        model_config=model,
        kv_cache_dtype=torch.float4_e2m1fn_x2,
        use_mla_backend=True,
        is_draft_worker=False,
        server_args=SimpleNamespace(enable_hisparse=False),
        mambaish_config=SimpleNamespace(full_attention_layer_ids=[3, 7]),
        layer_info=SimpleNamespace(start_layer=0, end_layer=8),
    )
    assert (
        calculate_mla_kv_cache_dim(
            model_config=model, kv_cache_dtype=kvc.kv_cache_dtype
        )
        == 288
    )
    planner = object.__new__(DefaultPoolConfigurator)
    assert planner._compute_cell_size(kvc, num_layers=2) == 2 * (288 + 132)
