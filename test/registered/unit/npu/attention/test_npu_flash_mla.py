"""CPU coverage for DSpark FlashMLA and default non-speculative NPU decode."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

for name in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.flash_mla",
    "sgl_kernel_npu.attention.sinks_attention",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(name, MagicMock())

from sglang.srt.hardware_backend.npu import memory_pool_npu as pool_mod
from sglang.srt.hardware_backend.npu.attention import ascend_backend as mod
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, KVCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode


@pytest.fixture
def backend():
    obj = mod.AscendAttnBackend.__new__(mod.AscendAttnBackend)
    obj.use_flash_mla = obj.use_mla = True
    obj.use_paged_mtp = False
    obj.needs_cpu_seq_lens = False
    obj.device = "cpu"
    obj.tp_q_head_num = 2
    obj.kv_lora_rank, obj.qk_rope_head_dim = 512, 64
    obj.page_size = 128
    obj.speculative_num_draft_tokens = 8
    obj.speculative_step_id = 0
    obj.graph_mode = obj.enable_torch_compile = False
    obj.is_hybrid_swa = obj.use_sliding_window_kv_pool = False
    obj.enable_sparsity_driven_kv_offload = False
    obj.is_dllm_model = False
    obj.q_head_num_padding = None
    obj.req_to_token = torch.arange(1024).view(2, 512)
    obj.req_to_token_pool = SimpleNamespace(req_to_token=obj.req_to_token)
    obj.graph_metadata = {"block_tables": torch.empty(2, 4, dtype=torch.int32)}
    obj.flash_mla_mask = torch.ones(16, 16, dtype=torch.int8).triu(1)
    obj.flash_mla_metadata = Mock(side_effect=lambda **kw: kw["cache_seqlens"].clone())

    def attention(query, cache, **kwargs):
        assert kwargs["layout_q"] == "TND"
        assert kwargs["layout_out"] == "TND"
        assert query.dim() == 3
        output = query[..., :512] + 1
        q_len = kwargs["cu_seqlens_q"].diff()
        inactive = (kwargs["seqused_q"] == 0).repeat_interleave(q_len)
        output[inactive] = float("nan")
        return output, torch.empty(0, dtype=torch.float32)

    obj.flash_mla = Mock(side_effect=attention)
    obj.token_to_kv_pool = SimpleNamespace(
        get_combined_kv_buffer=Mock(return_value=torch.zeros(3, 128, 1, 576)),
    )
    return obj


def make_batch(mode, lengths):
    return SimpleNamespace(
        batch_size=len(lengths),
        forward_mode=mode,
        # Match the int64 input lengths created by ScheduleBatch and graph buffers.
        seq_lens=torch.tensor(lengths, dtype=torch.int64),
        seq_lens_cpu=None,
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: False, is_dflash_family=lambda: True
        ),
        req_pool_indices=torch.arange(len(lengths)),
        spec_info=SimpleNamespace(draft_token_num=8)
        if mode.is_target_verify()
        else None,
        extend_seq_lens=None,
        extend_seq_lens_cpu=[],
        out_cache_loc=None,
    )


@pytest.mark.parametrize(
    "mode,expected",
    [(ForwardMode.TARGET_VERIFY, [133, 0]), (ForwardMode.IDLE, [0, 0])],
)
def test_eager_metadata_uses_device_lengths_without_cpu_mirror(backend, mode, expected):
    batch = make_batch(mode, [125, 0] if mode.is_target_verify() else [0, 0])
    with (
        patch.object(
            mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
    ):
        backend.init_forward_metadata(batch)
    metadata = backend.forward_metadata
    assert metadata.seq_lens.dtype == torch.int32
    assert metadata.seq_lens.data_ptr() != batch.seq_lens.data_ptr()
    assert metadata.seq_lens.tolist() == expected
    assert metadata.seqused_q.tolist() == [
        8 if mode.is_target_verify() else 0,
        0,
    ]
    assert batch.seq_lens.tolist() == ([125, 0] if mode.is_target_verify() else [0, 0])
    assert batch.seq_lens_cpu is None


@pytest.mark.parametrize("mirror_includes_block", [False, True])
def test_flash_mla_reuses_cpu_mirror_for_page_table_extent(
    backend, mirror_includes_block
):
    batch = make_batch(ForwardMode.TARGET_VERIFY, [121, 249])
    batch.seq_lens_cpu = torch.tensor(
        [129, 257] if mirror_includes_block else [121, 249]
    )
    original = batch.seq_lens_cpu.clone()
    with (
        patch.object(
            mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
    ):
        backend.init_forward_metadata(batch)
    metadata = backend.forward_metadata
    expected_pages = backend.req_to_token[:, : 3 * 128 : 128] // 128
    torch.testing.assert_close(metadata.block_tables, expected_pages)
    torch.testing.assert_close(
        metadata.seq_lens, torch.tensor([129, 257], dtype=torch.int32)
    )
    torch.testing.assert_close(
        batch.seq_lens, torch.tensor([121, 249], dtype=torch.int64)
    )
    torch.testing.assert_close(batch.seq_lens_cpu, original)
    assert metadata.seqused_q.tolist() == [8, 8]


def test_graph_metadata_preserves_prefix_and_stable_addresses(backend):
    batch = make_batch(ForwardMode.TARGET_VERIFY, [125, 0])
    with (
        patch.object(
            mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
    ):
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        metadata = backend.forward_metadata
        names = ["seq_lens", "seqused_q", "cu_seqlens_q", "flash_mla_metadata"]
        pointers = [getattr(metadata, name).data_ptr() for name in names]
        assert metadata.seq_lens.tolist() == [133, 0]
        batch.seq_lens[0] = 126
        backend.init_forward_metadata_out_graph(batch)
        assert metadata.seq_lens.tolist() == [134, 0]
        assert metadata.flash_mla_metadata.tolist() == [134, 0]
        assert metadata is backend.forward_metadata
        assert pointers == [getattr(metadata, name).data_ptr() for name in names]
    assert batch.seq_lens.tolist() == [126, 0]
    assert metadata.seq_lens.data_ptr() != batch.seq_lens.data_ptr()
    assert metadata.seq_lens_cpu_list is None


def test_target_verify_dispatches_to_flash_mla_and_zeroes_padding(backend):
    mode = ForwardMode.TARGET_VERIFY
    batch = make_batch(mode, [125, 0])
    metadata = mod.ForwardMetadata(
        seq_lens=torch.empty_like(batch.seq_lens),
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
    )
    backend._prepare_flash_mla_metadata(metadata, batch.seq_lens)
    backend.forward_metadata = metadata
    width = 8
    tokens = 2 * width + 2  # DP padding.
    q = torch.zeros(tokens, 2, 512, dtype=torch.bfloat16)
    q.copy_(torch.arange(tokens * 2).view(tokens, 2, 1))
    rope = torch.zeros(tokens, 2, 64, dtype=torch.bfloat16)
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=2, v_head_dim=512, scaling=0.1)
    with (
        patch.object(mod, "is_mla_preprocess_enabled", return_value=False),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
    ):
        output = backend.forward_extend(
            q, None, None, layer, batch, save_kv_cache=False, q_rope=rope
        )
    assert output.shape == (tokens, 1024)
    torch.testing.assert_close(output[:width], q[:width].reshape(width, -1) + 1)
    assert torch.count_nonzero(output[width:]) == 0
    args, kwargs = backend.flash_mla.call_args
    assert args[0].shape == (2 * width, 2, 576)
    assert kwargs["cache_seqlens"] is metadata.seq_lens
    assert kwargs["seqused_q"] is metadata.seqused_q
    assert kwargs["cu_seqlens_q"] is metadata.cu_seqlens_q
    assert kwargs["cu_seqlens_q"].tolist() == [0, width, 2 * width]
    assert kwargs["layout_kv"] == "PA_BBND"


def test_empty_batch_does_not_call_cann(backend):
    metadata = mod.ForwardMetadata(
        seq_lens=torch.empty(0, dtype=torch.int32),
        block_tables=torch.empty(0, 4, dtype=torch.int32),
    )
    batch = make_batch(ForwardMode.TARGET_VERIFY, [])
    backend._prepare_flash_mla_metadata(metadata, batch.seq_lens)
    backend.forward_metadata = metadata
    layer = SimpleNamespace(tp_q_head_num=2, v_head_dim=512)
    output = backend._forward_flash_mla(
        torch.empty(0, 2, 512), torch.empty(0, 2, 64), layer, batch
    )
    assert output.shape == (0, 1024)
    backend.flash_mla_metadata.assert_not_called()
    backend.flash_mla.assert_not_called()


@pytest.mark.parametrize("graph,use_fia", [(False, False), (False, True), (True, True)])
def test_non_speculative_decode_keeps_default_operators(backend, graph, use_fia):
    backend.use_flash_mla = False
    backend.needs_cpu_seq_lens = True
    backend.use_fia = use_fia
    backend.graph_mode = graph
    backend.tp_q_head_num = backend.q_head_num_padding = 8
    backend.token_to_kv_pool.get_key_buffer = Mock(
        return_value=torch.zeros(3, 128, 1, 512, dtype=torch.bfloat16)
    )
    backend.token_to_kv_pool.get_value_buffer = Mock(
        return_value=torch.zeros(3, 128, 1, 64, dtype=torch.bfloat16)
    )
    backend.token_to_kv_pool.get_kv_buffer = Mock(
        return_value=(
            backend.token_to_kv_pool.get_key_buffer.return_value,
            backend.token_to_kv_pool.get_value_buffer.return_value,
        )
    )
    mirror = torch.tensor([125, 0], dtype=torch.int32)
    backend.forward_metadata = mod.ForwardMetadata(
        seq_lens_cpu_int=None if graph else mirror,
        seq_lens_cpu_list=mirror.tolist() if graph else None,
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=8,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=576,
        v_head_dim=512,
        scaling=0.1,
    )
    q = torch.zeros(2, 8, 512, dtype=torch.bfloat16)
    rope = torch.zeros(2, 8, 64, dtype=torch.bfloat16)
    eager_fia = Mock(
        side_effect=lambda query, *args, **kwargs: (query.new_ones(query.shape), None)
    )
    graph_fia = Mock(side_effect=lambda *args, **kwargs: kwargs["out"][0].fill_(1))
    paged_mla = Mock(side_effect=lambda **kwargs: kwargs["out"].fill_(1))
    npu_ops = SimpleNamespace(
        _npu_fused_infer_attention_score_get_max_workspace=Mock(return_value=0),
        npu_fused_infer_attention_score=SimpleNamespace(out=graph_fia),
        _npu_paged_attention_mla=paged_mla,
    )
    with (
        patch.object(mod, "is_mla_preprocess_enabled", return_value=False),
        patch.object(mod, "is_fia_nz", return_value=False),
        patch.object(mod, "torch_npu", npu_ops),
        patch.object(
            torch.ops.npu, "npu_fused_infer_attention_score", eager_fia, create=True
        ),
    ):
        output = backend.forward_decode(
            q,
            None,
            None,
            layer,
            make_batch(ForwardMode.DECODE, [125, 0]),
            save_kv_cache=False,
            q_rope=rope,
        )
    torch.testing.assert_close(output, torch.ones(2, 4096, dtype=torch.bfloat16))
    backend.flash_mla.assert_not_called()
    if graph:
        graph_fia.assert_called_once()
        assert graph_fia.call_args.kwargs["actual_seq_lengths_kv"] == [125, 0]
        eager_fia.assert_not_called()
        paged_mla.assert_not_called()
    elif use_fia:
        eager_fia.assert_called_once()
        assert eager_fia.call_args.kwargs["actual_seq_lengths_kv"] is mirror
        paged_mla.assert_not_called()
    else:
        paged_mla.assert_called_once()
        assert paged_mla.call_args.kwargs["context_lens"] is mirror
        eager_fia.assert_not_called()


def test_non_speculative_graph_metadata_keeps_host_lengths(backend):
    backend.use_flash_mla = False
    backend.needs_cpu_seq_lens = True
    backend.speculative_num_draft_tokens = None
    batch = make_batch(ForwardMode.DECODE, [125, 0])
    batch.seq_lens_cpu = batch.seq_lens.clone()
    with patch.object(
        mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
    ):
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
    metadata = backend.forward_metadata
    assert metadata.seq_lens_cpu_list == [125, 0]
    assert metadata.seqused_q is None
    assert metadata.cu_seqlens_q is None
    assert metadata.flash_mla_metadata is None
    backend.flash_mla_metadata.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "dspark,is_draft", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("force_cpu", [False, True])
@pytest.mark.parametrize("paged_mtp", [False, True])
def test_cpu_length_gate_respects_backend_capability_and_explicit_override(
    dspark, is_draft, force_cpu, paged_mtp, enabled
):
    uses_flash_mla = enabled and dspark and not is_draft
    uses_paged_mtp = paged_mtp and is_draft and dspark
    needs_cpu = force_cpu or not (uses_flash_mla or uses_paged_mtp)
    spec = SimpleNamespace(
        is_dspark=lambda: dspark,
        is_none=lambda: not dspark,
        is_dflash_family=lambda: dspark,
        get_num_tokens_per_req_for_target_verify=lambda count, **kwargs: count,
    )
    model_config = SimpleNamespace(
        attention_arch=mod.AttentionArch.MHA if is_draft else mod.AttentionArch.MLA,
        dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        num_attention_heads=8,
        context_len=512,
        hf_config=SimpleNamespace(
            architectures=[
                "DSparkDraftModel" if is_draft else "KimiK3ForConditionalGeneration"
            ]
        ),
    )
    runner = SimpleNamespace(
        device="cpu",
        is_draft_worker=is_draft,
        page_size=128,
        model_config=model_config,
        kv_cache_dtype=torch.bfloat16,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(2, 512)),
        token_to_kv_pool=object(),
        use_mla_backend=not is_draft,
        spec_algorithm=spec,
        is_hybrid_swa=False,
        server_args=SimpleNamespace(enable_mixed_chunk=False),
    )
    masks = SimpleNamespace(
        mask=None, fia_mask=None, mtp_mask=torch.zeros(8, 8), mixed_chunk_attn_mask=None
    )
    original_tensor = torch.tensor

    def tensor_on_cpu(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    kernel_module = SimpleNamespace(flash_mla=Mock(), flash_mla_metadata=Mock())
    with (
        patch.object(mod, "flash_mla", kernel_module.flash_mla),
        patch.object(mod, "flash_mla_metadata", kernel_module.flash_mla_metadata),
        patch.object(mod.torch, "tensor", side_effect=tensor_on_cpu),
        patch.object(mod, "is_deepseek_dsa", return_value=False),
        patch.object(mod, "is_sparsity_driven_kv_offload_enabled", return_value=False),
        patch.object(mod, "AscendAttnMaskBuilder", return_value=masks),
        patch.object(mod.DllmConfig, "from_server_args", return_value=None),
        patch.object(
            mod,
            "get_bool_env_var",
            side_effect=lambda name, *args: (
                name == "ASCEND_USE_FIA"
                or (paged_mtp and name == "ASCEND_USE_PAGED_MTP")
                or (enabled and name == "SGLANG_NPU_USE_FLASH_MLA")
            ),
        ),
        patch.object(
            mod,
            "get_flags",
            return_value=SimpleNamespace(
                capture=SimpleNamespace(enable_torch_compile=False)
            ),
        ),
        patch.object(
            mod,
            "get_spec",
            return_value=SimpleNamespace(speculative_num_draft_tokens=8),
        ),
        patch.object(
            mod,
            "get_parallel",
            return_value=SimpleNamespace(
                attn_cp_size=1, attn_tp_size=4, dcp_enabled=False
            ),
        ),
        patch.object(mod.logger, "warning") as warning,
        patch.object(
            mod.envs.SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS,
            "get",
            return_value=force_cpu,
        ),
    ):
        obj = mod.AscendAttnBackend(runner)
    assert obj.use_flash_mla is uses_flash_mla
    assert obj.use_paged_mtp is uses_paged_mtp
    assert obj.needs_cpu_seq_lens is needs_cpu
    if not force_cpu and not (uses_flash_mla or uses_paged_mtp):
        warning.assert_called_once_with(
            "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0 is ignored: "
            "host-bound attention paths present; keeping the mirror."
        )
    else:
        warning.assert_not_called()
    if uses_flash_mla:
        assert obj.flash_mla is kernel_module.flash_mla
        assert obj.flash_mla_metadata is kernel_module.flash_mla_metadata


@pytest.mark.parametrize("draft_needs_cpu", [True, False])
def test_scheduler_keeps_cpu_mirror_until_draft_supports_device_lengths(
    draft_needs_cpu,
):
    from sglang.srt.managers import overlap_utils
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    target = SimpleNamespace(needs_cpu_seq_lens=False)
    draft = SimpleNamespace(needs_cpu_seq_lens=draft_needs_cpu)
    with (
        patch.object(
            overlap_utils,
            "get_exec",
            return_value=SimpleNamespace(
                overlap=SimpleNamespace(enable_two_batch_overlap=False)
            ),
        ),
        patch.object(
            overlap_utils,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm="DSPARK"),
        ),
        patch.object(
            SpeculativeAlgorithm,
            "from_string",
            return_value=SimpleNamespace(is_ngram=lambda: False),
        ),
    ):
        assert (
            overlap_utils.decide_needs_cpu_seq_lens((target, draft)) is draft_needs_cpu
        )


@pytest.fixture
def pool_factory():
    def init_cache(self, **kwargs):
        self.__dict__.update(kwargs)
        self.start_layer = kwargs["start_layer"] or 0
        self.store_dtype = self.dtype
        self.layer_transfer_counter = None
        self.memory_saver_adapter = SimpleNamespace(region=lambda _: nullcontext())

    def make_pool(fia_nz=False, dcp_size=1, **overrides):
        arguments = dict(
            size=256,
            page_size=128,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            use_combined_kv_buffer=True,
        )
        arguments.update(overrides)
        with (
            patch.object(KVCache, "__init__", init_cache),
            patch.object(pool_mod.NPUMLATokenToKVPool, "_finalize_allocation_log"),
            patch.object(
                pool_mod.envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD,
                "get",
                return_value=False,
            ),
            patch.object(pool_mod, "get_bool_env_var", return_value=fia_nz),
            patch.object(
                pool_mod,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=dcp_size, attn_dcp_rank=0),
            ),
        ):
            return pool_mod.NPUMLATokenToKVPool(**arguments)

    return make_pool


def test_merged_cache_preserves_split_views_and_page_boundary_writes(
    pool_factory,
):
    pool = pool_factory()
    locations = torch.tensor([127, 128, 255])
    key = torch.arange(3 * 512, dtype=torch.float32).view(3, 1, 512).to(torch.bfloat16)
    rope = torch.arange(3 * 64, dtype=torch.float32).view(3, 1, 64).to(torch.bfloat16)

    def scatter(destination, indices, updates):
        destination.index_copy_(0, indices.flatten().long(), updates)

    with patch.object(
        pool_mod,
        "torch_npu",
        SimpleNamespace(npu_scatter_nd_update_=scatter),
        create=True,
    ):
        pool.set_kv_buffer(SimpleNamespace(layer_id=0), locations, key, rope)
    combined = pool.get_combined_kv_buffer(0).view(-1, 1, 576)
    torch.testing.assert_close(combined[locations], torch.cat((key, rope), dim=-1))
    split_key, split_rope = pool.get_kv_buffer(0)
    assert (
        split_key.untyped_storage().data_ptr() == combined.untyped_storage().data_ptr()
    )
    assert (
        split_rope.untyped_storage().data_ptr() == combined.untyped_storage().data_ptr()
    )
    split_key[0, 0, 0, 0] = 7
    split_rope[0, 0, 0, 0] = 9
    assert combined[0, 0, 0] == 7
    assert combined[0, 0, 512] == 9
    assert pool.get_kv_size_bytes() == pool.kv_buffer.numel() * pool.dtype.itemsize
    pointers, lengths, items = pool.get_contiguous_buf_infos()
    assert len(pointers) == len(lengths) == len(items) == 2
    assert pool.get_kv_layer_ids() == [0, 1]
    info = pool._mla_buffer_info()
    info.validate(layer_ids=(0, 1))
    assert info.native_buffers is None
    assert all(tensor.is_contiguous() for tensor in info.buffers)
    pool._clear_buffers()
    assert not hasattr(pool, "kv_buffer")
    assert not hasattr(pool, "k_buffer")
    assert not hasattr(pool, "v_buffer")


@pytest.mark.parametrize("fia_nz", [False, True])
def test_disabled_combined_kv_buffer_keeps_separate_allocations(pool_factory, fia_nz):
    pool = pool_factory(use_combined_kv_buffer=False, fia_nz=fia_nz)
    assert pool.use_combined_kv_buffer is False
    key, rope = pool.get_kv_buffer(0)
    assert key.is_contiguous() and rope.is_contiguous()
    assert key.untyped_storage().data_ptr() != rope.untyped_storage().data_ptr()
    assert len(pool.get_contiguous_buf_infos()[0]) == 4
    assert pool.get_kv_layer_ids() == [0, 1, 0, 1]


@pytest.mark.parametrize(
    "dspark,is_draft", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("arch35", [False, True])
@pytest.mark.parametrize("flash_mla_env", [None, "0", "1"])
@pytest.mark.parametrize("hybrid", [False, True])
def test_pool_configurator_only_merges_dspark_target_cache(
    pool_factory, monkeypatch, dspark, is_draft, arch35, flash_mla_env, hybrid
):
    from sglang.srt.hardware_backend.npu import utils as npu_utils

    if flash_mla_env is None:
        monkeypatch.delenv("SGLANG_NPU_USE_FLASH_MLA", raising=False)
    else:
        monkeypatch.setenv("SGLANG_NPU_USE_FLASH_MLA", flash_mla_env)

    with patch.dict(
        sys.modules, {"sglang.srt.speculative.draft_parallel": MagicMock()}
    ):
        from sglang.srt.mem_cache import kv_cache_configurator as config_mod

    config = config_mod.KVCacheConfigurator.__new__(config_mod.KVCacheConfigurator)
    config.is_draft_worker = is_draft
    config.spec_algorithm = SimpleNamespace(is_dspark=lambda: dspark)
    config.model_config = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        head_dim=576,
        hf_config=SimpleNamespace(),
        get_num_kv_heads=lambda *_: 1,
    )
    config.kv_cache_dtype = torch.bfloat16
    config.kv_cache_dtype_str = "auto"
    config.use_mla_backend = True
    config.post_capture_kv_active = False
    config.mambaish_config = SimpleNamespace(full_attention_layer_ids=[0, 1])
    config.device = "cpu"
    config.layer_info = SimpleNamespace(
        start_layer=0, end_layer=2, num_effective_layers=2
    )
    constructor = Mock()
    constructor.return_value.get_kv_size_bytes.return_value = 0
    with (
        patch.object(pool_mod, "NPUMLATokenToKVPool", constructor),
        patch.object(config_mod, "_is_npu", True),
        patch.object(
            config_mod.KVCacheConfigurator, "_build_mha_quant_method", return_value=None
        ),
        patch.object(npu_utils, "is_npu_arch35", return_value=arch35),
        patch.object(
            config_mod,
            "get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1, attn_tp_size=1),
        ),
        patch.object(
            config_mod, "get_schedule", return_value=SimpleNamespace(page_size=128)
        ),
        patch.object(
            config_mod,
            "get_exec",
            return_value=SimpleNamespace(
                features=SimpleNamespace(enable_memory_saver=False)
            ),
        ),
        patch.object(
            config_mod,
            "get_model",
            return_value=SimpleNamespace(qsa_indexer_dtype="auto"),
        ),
        patch.object(
            config_mod,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm="DSPARK"),
        ),
    ):
        if hybrid:
            config._build_hybrid_linear_kv_pool(
                max_total_num_tokens=256,
                req_to_token_pool=SimpleNamespace(mamba_pool=Mock()),
                mha_pool_class=Mock(),
                mla_pool_class=Mock(),
                dsa_pool_class=Mock(),
            )
        else:
            config._build_ascend_mla_kv_pool(
                max_total_num_tokens=256, is_dsa_model=False
            )
    constructor.assert_called_once()
    requested = flash_mla_env == "1" and dspark and not is_draft
    assert (
        constructor.call_args.kwargs.get("use_combined_kv_buffer", False) is requested
    )
    pool = pool_factory(
        **{"use_combined_kv_buffer": False, **constructor.call_args.kwargs}
    )
    assert pool.use_combined_kv_buffer is requested
    key, rope = pool.get_kv_buffer(0)
    shares_storage = (
        key.untyped_storage().data_ptr() == rope.untyped_storage().data_ptr()
    )
    assert shares_storage is requested


def test_k3_hybrid_pool_maps_model_layer_to_combined_kv_cache(pool_factory):
    pool = pool_factory()
    hybrid = HybridLinearKVPool.__new__(HybridLinearKVPool)
    hybrid.full_kv_pool = pool
    hybrid.full_attention_layer_id_mapping = {3: 0, 7: 1}
    hybrid.start_layer = 0
    hybrid.layer_transfer_counter = Mock()
    assert hybrid.get_combined_kv_buffer(7).data_ptr() == pool.kv_buffer[1].data_ptr()
    hybrid.layer_transfer_counter.wait_until.assert_called_once_with(7)


def test_verify_mask_broadcasts_across_query_block(backend):
    metadata = mod.ForwardMetadata(
        seq_lens=torch.empty(3, dtype=torch.int32),
        block_tables=torch.zeros(3, 4, dtype=torch.int32),
    )
    mode = ForwardMode.TARGET_VERIFY
    batch = make_batch(mode, [125, 0, 40])
    backend._prepare_flash_mla_metadata(metadata, batch.seq_lens)
    backend.forward_metadata = metadata
    width = 8
    tokens = 3 * width
    q = torch.zeros(tokens, 2, 512, dtype=torch.bfloat16)
    rope = torch.zeros(tokens, 2, 64, dtype=torch.bfloat16)
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=2, v_head_dim=512, scaling=0.1)

    output = backend._forward_flash_mla(q, rope, layer, batch)

    assert output.shape == (tokens, 1024)
    torch.testing.assert_close(output[:width], torch.ones_like(output[:width]))
    assert torch.count_nonzero(output[width : 2 * width]) == 0
    torch.testing.assert_close(
        output[2 * width :], torch.ones_like(output[2 * width :])
    )
