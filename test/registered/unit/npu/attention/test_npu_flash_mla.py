"""CPU coverage for ordinary BF16 NPU FlashMLA integration."""

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
        # The CANN operator takes TND query (Q_T, Q_N, D) and returns NTD
        # attn_out (Q_N, Q_T, DV).
        assert kwargs["layout_q"] == "TND"
        assert kwargs["layout_out"] == "NTD"
        assert query.dim() == 3
        q_t, q_n, _ = query.shape
        output = torch.ones(q_n, q_t, 512, dtype=query.dtype)
        q_len = kwargs["cu_seqlens_q"].diff()
        inactive = (kwargs["seqused_q"] == 0).repeat_interleave(q_len)
        output[:, inactive] = float("nan")
        return output, torch.empty(0, dtype=torch.float32)

    obj.flash_mla = Mock(side_effect=attention)
    obj.token_to_kv_pool = SimpleNamespace(
        get_flash_mla_buffer=Mock(return_value=torch.zeros(3, 128, 1, 576)),
    )
    return obj


def make_batch(mode, lengths):
    return SimpleNamespace(
        batch_size=len(lengths),
        forward_mode=mode,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        seq_lens_cpu=None,
        req_pool_indices=torch.arange(len(lengths)),
        spec_info=SimpleNamespace(draft_token_num=8)
        if mode.is_target_verify()
        else None,
        extend_seq_lens=None,
        extend_seq_lens_cpu=[],
        out_cache_loc=None,
    )


@pytest.mark.parametrize(
    "mode,expected", [(ForwardMode.DECODE, 125), (ForwardMode.TARGET_VERIFY, 133)]
)
def test_eager_metadata_uses_device_lengths_without_cpu_mirror(backend, mode, expected):
    batch = make_batch(mode, [125, 0])
    with (
        patch.object(
            mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
    ):
        backend.init_forward_metadata(batch)
    metadata = backend.forward_metadata
    assert metadata.flash_mla_seq_lens.tolist() == [expected, 0]
    assert metadata.flash_mla_seqused_q.tolist() == [
        8 if mode.is_target_verify() else 1,
        0,
    ]
    assert batch.seq_lens.tolist() == [125, 0]
    assert batch.seq_lens_cpu is None


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
        names = ["flash_mla_seq_lens", "flash_mla_seqused_q", "flash_mla_metadata"]
        pointers = [getattr(metadata, name).data_ptr() for name in names]
        assert metadata.flash_mla_seq_lens.tolist() == [133, 0]
        batch.seq_lens[0] = 126
        backend.init_forward_metadata_out_graph(batch)
        assert metadata.flash_mla_seq_lens.tolist() == [134, 0]
        assert metadata.flash_mla_metadata.tolist() == [134, 0]
        assert metadata is backend.forward_metadata
        assert pointers == [getattr(metadata, name).data_ptr() for name in names]
    assert batch.seq_lens.tolist() == [126, 0]
    assert metadata.seq_lens.data_ptr() != batch.seq_lens.data_ptr()
    assert metadata.seq_lens_cpu_list is None


@pytest.mark.parametrize(
    "entry", ["forward_decode", "forward_decode_graph", "forward_extend"]
)
def test_decode_and_verify_dispatch_to_flash_mla_and_zero_padding(backend, entry):
    mode = (
        ForwardMode.TARGET_VERIFY if entry == "forward_extend" else ForwardMode.DECODE
    )
    batch = make_batch(mode, [125, 0])
    metadata = mod.ForwardMetadata(block_tables=torch.zeros(2, 4, dtype=torch.int32))
    backend._prepare_flash_mla_metadata(metadata, batch.seq_lens, mode, batch.spec_info)
    backend.forward_metadata = metadata
    width = 8 if mode.is_target_verify() else 1
    tokens = 2 * width + 2  # two request slots followed by DP token padding
    q = torch.zeros(tokens, 2, 512, dtype=torch.bfloat16)
    rope = torch.zeros(tokens, 2, 64, dtype=torch.bfloat16)
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=2, v_head_dim=512, scaling=0.1)
    with (
        patch.object(mod, "is_mla_preprocess_enabled", return_value=False),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
    ):
        output = getattr(backend, entry)(
            q, None, None, layer, batch, save_kv_cache=False, q_rope=rope
        )
    assert output.shape == (tokens, 1024)
    torch.testing.assert_close(output[:width], torch.ones_like(output[:width]))
    assert torch.count_nonzero(output[width:]) == 0
    args, kwargs = backend.flash_mla.call_args
    assert args[0].shape == (2 * width, 2, 576)
    assert kwargs["cache_seqlens"] is metadata.flash_mla_seq_lens
    assert kwargs["seqused_q"] is metadata.flash_mla_seqused_q
    assert kwargs["cu_seqlens_q"] is metadata.flash_mla_cu_seqlens_q
    assert kwargs["cu_seqlens_q"].tolist() == [0, width, 2 * width]
    assert kwargs["layout_kv"] == "PA_BBND"


def test_empty_batch_does_not_call_cann(backend):
    metadata = mod.ForwardMetadata(block_tables=torch.empty(0, 4, dtype=torch.int32))
    batch = make_batch(ForwardMode.DECODE, [])
    backend._prepare_flash_mla_metadata(
        metadata, batch.seq_lens, batch.forward_mode, None
    )
    backend.forward_metadata = metadata
    layer = SimpleNamespace(tp_q_head_num=2, v_head_dim=512)
    output = backend._forward_flash_mla(
        torch.empty(0, 2, 512), torch.empty(0, 2, 64), layer, batch
    )
    assert output.shape == (0, 1024)
    backend.flash_mla_metadata.assert_not_called()
    backend.flash_mla.assert_not_called()


def test_ragged_verify_is_rejected(backend):
    with pytest.raises(ValueError, match="static DSpark"):
        backend._prepare_flash_mla_metadata(
            mod.ForwardMetadata(),
            torch.tensor([125]),
            ForwardMode.TARGET_VERIFY,
            SimpleNamespace(ragged_verify_layout=object()),
        )


@pytest.mark.parametrize(
    "enabled,arch35,a3,dspark,is_draft,force_cpu,mojo,needs_cpu",
    [
        (False, False, False, False, False, False, False, True),
        (False, True, False, False, False, False, False, True),
        (True, False, False, False, False, False, False, True),
        (True, False, False, True, False, False, False, True),
        (True, True, False, False, False, False, False, False),
        (True, True, False, True, False, False, False, False),
        (True, True, False, True, True, False, False, True),
        (False, True, False, True, True, False, False, True),
        (True, True, False, False, False, True, False, True),
        (True, True, False, True, False, True, False, True),
        (True, True, False, True, True, False, True, False),
        (False, False, False, True, True, False, True, False),
        (True, True, False, True, True, True, True, True),
        (True, True, False, True, False, False, True, False),
        (False, False, True, False, False, False, False, True),
        (True, False, True, False, False, False, False, False),
        (True, False, True, True, False, False, False, False),
        (True, False, True, True, False, True, False, True),
        (True, False, True, True, True, False, False, True),
        (True, False, True, True, True, False, True, False),
        (True, False, True, True, True, True, True, True),
        (False, False, True, True, True, False, True, False),
        (True, True, True, False, False, False, False, False),
        (True, False, True, True, False, False, True, False),
    ],
)
def test_cpu_length_gate_respects_backend_capability_and_explicit_override(
    enabled, arch35, a3, dspark, is_draft, force_cpu, mojo, needs_cpu
):
    uses_flash_mla = enabled and (arch35 or a3) and not is_draft
    uses_mojo = mojo and is_draft and dspark
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

    cann_module = SimpleNamespace(flash_mla=Mock(), flash_mla_metadata=Mock())
    a3_module = SimpleNamespace(flash_mla=Mock(), flash_mla_metadata=Mock())
    with (
        patch.dict(
            sys.modules,
            {
                "cann_ops_transformer.ops.attention.flash_mla": (
                    cann_module if uses_flash_mla and arch35 else None
                ),
                "sgl_kernel_npu.attention.flash_mla": (
                    a3_module if uses_flash_mla and not arch35 else None
                ),
            },
        ),
        patch.object(mod, "is_npu_arch35", return_value=arch35),
        patch.object(mod, "is_npu_a3", return_value=a3),
        patch.object(mod.torch, "tensor", side_effect=tensor_on_cpu),
        patch.object(mod, "is_deepseek_dsa", return_value=False),
        patch.object(mod, "is_sparsity_driven_kv_offload_enabled", return_value=False),
        patch.object(mod, "AscendAttnMaskBuilder", return_value=masks),
        patch.object(mod.DllmConfig, "from_server_args", return_value=None),
        patch.object(
            mod,
            "get_bool_env_var",
            side_effect=lambda name, *args: mojo and name == "ASCEND_USE_MOJO_MTP",
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
        patch.object(mod.envs.SGLANG_NPU_USE_FLASH_MLA, "get", return_value=enabled),
        patch.object(
            mod.envs.SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS,
            "get",
            return_value=force_cpu,
        ),
    ):
        obj = mod.AscendAttnBackend(runner)
    assert obj.use_flash_mla is uses_flash_mla
    assert obj.use_mojo_mtp is uses_mojo
    assert obj.needs_cpu_seq_lens is needs_cpu
    assert obj.requires_host_seq_lens_update is not (uses_flash_mla or uses_mojo)
    if uses_flash_mla:
        provider = cann_module if arch35 else a3_module
        assert obj.flash_mla is provider.flash_mla
        assert obj.flash_mla_metadata is provider.flash_mla_metadata


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

    def make_pool(
        enabled=True, arch35=True, a3=False, fia_nz=False, dcp_size=1, **overrides
    ):
        arguments = dict(
            size=256,
            page_size=128,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        arguments.update(overrides)
        with (
            patch.object(KVCache, "__init__", init_cache),
            patch.object(pool_mod.NPUMLATokenToKVPool, "_finalize_allocation_log"),
            patch.object(pool_mod, "is_npu_arch35", return_value=arch35),
            patch.object(pool_mod, "is_npu_a3", return_value=a3),
            patch.object(
                pool_mod.envs.SGLANG_NPU_USE_FLASH_MLA, "get", return_value=enabled
            ),
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


@pytest.mark.parametrize("arch35,a3", [(True, False), (False, True)])
def test_merged_cache_preserves_split_views_and_page_boundary_writes(
    pool_factory, arch35, a3
):
    pool = pool_factory(arch35=arch35, a3=a3)
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
    combined = pool.get_flash_mla_buffer(0).view(-1, 1, 576)
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


@pytest.mark.parametrize(
    "enabled,arch35,a3",
    [
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, False),
    ],
)
@pytest.mark.parametrize("fia_nz", [False, True])
def test_fallback_keeps_separate_cache_allocations(
    pool_factory, enabled, arch35, a3, fia_nz
):
    pool = pool_factory(enabled=enabled, arch35=arch35, a3=a3, fia_nz=fia_nz)
    assert pool.use_flash_mla is False
    key, rope = pool.get_kv_buffer(0)
    assert key.is_contiguous() and rope.is_contiguous()
    assert key.untyped_storage().data_ptr() != rope.untyped_storage().data_ptr()
    assert len(pool.get_contiguous_buf_infos()[0]) == 4
    assert pool.get_kv_layer_ids() == [0, 1, 0, 1]


@pytest.mark.parametrize("arch35,a3", [(True, False), (False, True)])
def test_k3_hybrid_pool_maps_model_layer_to_flash_mla_cache(pool_factory, arch35, a3):
    pool = pool_factory(arch35=arch35, a3=a3)
    hybrid = HybridLinearKVPool.__new__(HybridLinearKVPool)
    hybrid.full_kv_pool = pool
    hybrid.full_attention_layer_id_mapping = {3: 0, 7: 1}
    hybrid.start_layer = 0
    hybrid.layer_transfer_counter = Mock()
    assert hybrid.get_flash_mla_buffer(7).data_ptr() == pool.kv_buffer[1].data_ptr()
    hybrid.layer_transfer_counter.wait_until.assert_called_once_with(7)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dtype": torch.float16},
        {"index_head_dim": 128},
        {"enable_memory_saver": True},
        {"fia_nz": True},
        {"dcp_size": 2},
    ],
)
@pytest.mark.parametrize("arch35,a3", [(True, False), (False, True)])
def test_unsupported_cache_configurations_fail_before_allocation(
    pool_factory, overrides, arch35, a3
):
    with pytest.raises(ValueError, match="ordinary BF16 MLA"):
        pool_factory(arch35=arch35, a3=a3, **overrides)


def test_verify_mask_broadcasts_across_query_block(backend):
    """seqused_q is per batch; the padding mask must cover the whole query block.

    DSpark verify runs query_len > 1, so a non-broadcast view over seqused_q
    would either raise or silently mask the wrong rows.
    """
    metadata = mod.ForwardMetadata(block_tables=torch.zeros(3, 4, dtype=torch.int32))
    mode = ForwardMode.TARGET_VERIFY
    batch = make_batch(mode, [125, 0, 40])
    backend._prepare_flash_mla_metadata(metadata, batch.seq_lens, mode, batch.spec_info)
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
