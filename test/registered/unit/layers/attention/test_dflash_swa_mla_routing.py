"""CPU coverage of the opt-in route; numerical kernels are tested separately.

Run the actual backend constructor and forward dispatcher with CPU tensors and
stub only device queries and leaf kernels. In particular, these tests must not
initialize CUDA just to decide whether a call belongs on the experimental path.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.kernels.ops.attention import decode_attention, extend_attention
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.layers.attention import triton_backend as tb
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture
def make_backend(monkeypatch):
    """Keep constructor eligibility real without asking CUDA about the host."""
    monkeypatch.setattr(tb, "_is_cuda", True)
    monkeypatch.setattr(tb, "_is_hip", False)
    monkeypatch.setattr(tb, "_is_gfx942", False)
    monkeypatch.setattr(tb, "_is_xpu", False)
    monkeypatch.setattr(tb, "is_gfx95_supported", lambda: False)
    monkeypatch.setattr(tb, "mambaish_config", lambda _: None)
    monkeypatch.setattr(tb, "get_device_core_count", lambda _: 1)
    monkeypatch.setattr(tb, "is_arch_support_pdl", lambda: False, raising=False)
    monkeypatch.setattr(tb, "cuda_graph_fully_disabled", lambda: True)
    monkeypatch.setattr(torch.compiler, "disable", lambda f: f)
    monkeypatch.setattr(
        decode_attention, "_lean_decode_launch_params", lambda *_: (1, False, 1)
    )
    monkeypatch.setattr(
        extend_attention,
        "_get_block_sizes_for_extend_attention",
        lambda **_: (64, 1, 1, 16, 4, 2),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.dllm.attention",
        SimpleNamespace(DllmFlashAttention=lambda _: object()),
    )
    # Fail immediately if a future eligibility check accidentally touches CUDA.
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", Mock(side_effect=AssertionError("CUDA initialized"))
    )

    def make(
        *,
        enabled=True,
        cuda=True,
        sm=(10, 3),
        draft=True,
        dflash=True,
        mla=True,
        dcp=1,
        deterministic=False,
        dllm_fa4=False,
        block=8,
        window=4095,
        heads=12,
        dtype=torch.bfloat16,
        max_bs=4,
    ):
        monkeypatch.setattr(tb, "_is_cuda", cuda)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: sm)
        monkeypatch.setattr(
            tb,
            "get_exec",
            lambda: SimpleNamespace(
                kernel=SimpleNamespace(
                    enable_lean_attention=False,
                    triton_attention_num_kv_splits=1,
                    triton_attention_split_tile_size=None,
                ),
                dllm=SimpleNamespace(dllm_algorithm=None),
                deterministic=SimpleNamespace(
                    enable_deterministic_inference=deterministic
                ),
            ),
        )
        monkeypatch.setattr(
            tb,
            "get_spec",
            lambda: SimpleNamespace(
                speculative_num_draft_tokens=block,
                speculative_num_steps=1,
                speculative_eagle_topk=1,
            ),
        )
        monkeypatch.setattr(
            tb,
            "get_parallel",
            lambda: SimpleNamespace(attn_dcp_size=dcp, attn_dcp_rank=0, attn_tp_size=1),
        )
        monkeypatch.setattr(
            tb, "get_schedule", lambda: SimpleNamespace(chunked_prefill_size=-1)
        )
        model_config = SimpleNamespace(
            attention_arch=AttentionArch.MLA if mla else AttentionArch.MHA,
            get_max_num_attention_heads=lambda: heads,
            get_num_kv_heads=lambda *_: 1,
            v_head_dim=512,
            swa_v_head_dim=512,
            context_len=4096,
            is_encoder_decoder=False,
            head_dim=576,
        )
        pool = SimpleNamespace(
            start_layer=0,
            get_key_buffer=lambda _: torch.empty((1, 1, 576), dtype=dtype),
            get_value_buffer=lambda _: torch.empty((1, 1, 512), dtype=dtype),
        )
        runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(size=max_bs, req_to_token=None),
            sliding_window_size=window,
            token_to_kv_pool=pool,
            token_to_kv_pool_allocator=None,
            kv_index_translator=None,
            page_size=64,
            decode_num_tokens_per_req=lambda: block,
            model_config=model_config,
            is_draft_worker=draft,
            spec_algorithm=SimpleNamespace(is_dflash=lambda: dflash),
            kv_cache_dtype=dtype,
            device="cpu",
            gpu_id=0,
        )
        with envs.SGLANG_OPT_DFLASH_SWA_MLA_VALUE_SPLIT.override(enabled):
            return tb.TritonAttnBackend(runner, dllm_fa4=dllm_fa4)

    return make


@pytest.mark.parametrize(
    "unsupported",
    [
        {"enabled": False},
        {"cuda": False},
        {"sm": (9, 0)},
        {"sm": (10, 0)},
        {"sm": (10, 2)},
        {"sm": (12, 0)},
        {"draft": False},
        {"dflash": False},
        {"mla": False},
        {"dcp": 2},
        {"deterministic": True},
        {"dllm_fa4": True},
        {"block": 7},
        {"window": 4096},
        {"heads": 16},
        {"heads": 96},
        {"dtype": torch.float16},
    ],
    ids=lambda x: next(iter(x)),
)
def test_ineligible_constructor_keeps_native_dispatch(make_backend, unsupported):
    backend = make_backend(**unsupported)
    assert not backend.use_dflash_swa_mla
    assert not hasattr(backend, "dflash_swa_mla_fwd")


def _forward_case(make_backend, *, batch=1, tokens_per_req=8, **constructor):
    backend = make_backend(**constructor)
    heads = constructor.get("heads", 12)
    backend.dflash_swa_mla_fwd = Mock()
    backend._forward_extend_kernel = Mock()
    backend._forward_extend_unified = Mock(return_value="deterministic")
    backend._forward_extend_dcp = Mock(return_value="dcp")
    tokens = batch * tokens_per_req
    backend.forward_metadata = tb.ForwardMetadata(
        attn_logits=None,
        attn_lse=None,
        max_extend_len=tokens_per_req,
        num_kv_splits=None,
        kv_indptr=torch.zeros(batch + 1, dtype=torch.int32),
        kv_indices=torch.empty(0, dtype=torch.int64),
        qo_indptr=torch.arange(batch + 1, dtype=torch.int64) * tokens_per_req,
        custom_mask=None,
        # A tree-mask pointer can remain allocated without a custom mask.
        mask_indptr=torch.zeros(batch + 1, dtype=torch.int64),
        window_kv_indptr=torch.zeros(batch + 1, dtype=torch.int32),
        window_kv_indices=torch.empty(0, dtype=torch.int64),
        window_num_kv_splits=None,
        window_kv_offsets=torch.zeros(batch, dtype=torch.int32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        qk_head_dim=576,
        v_head_dim=512,
        tp_q_head_num=heads,
        k_scale=None,
        v_scale=None,
        k_scale_float=1.0,
        v_scale_float=1.0,
        logit_capping_method="tanh",
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=4095,
        scaling=576**-0.5,
        xai_temperature_len=-1,
    )
    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        mha_one_shot=False,
        attn_attend_prefix_cache=None,
        extend_seq_lens_cpu=[tokens_per_req] * batch,
    )
    tensors = [
        torch.empty((tokens, heads * 576), dtype=torch.bfloat16),
        torch.empty((tokens, 1, 576), dtype=torch.bfloat16),
        torch.empty((tokens, 1, 512), dtype=torch.bfloat16),
    ]
    return backend, layer, fb, tensors


def _call(case, **kwargs):
    backend, layer, fb, tensors = case
    return backend.forward_extend(*tensors, layer, fb, save_kv_cache=False, **kwargs)


@pytest.mark.parametrize("heads,batch,splits", [(12, 1, 8), (24, 1, 4), (12, 4, 2)])
def test_qualified_shape_uses_window_rail_and_selected_partition(
    make_backend, heads, batch, splits
):
    case = _forward_case(make_backend, heads=heads, batch=batch)
    backend, layer, _, tensors = case
    out = _call(case)
    backend._forward_extend_kernel.assert_not_called()
    args, kwargs = backend.dflash_swa_mla_fwd.call_args
    assert args[0].shape == (batch * 8, heads, 576)
    assert args[0].data_ptr() == tensors[0].data_ptr()
    assert args[3].shape == (batch * 8, heads, 512)
    assert args[3].data_ptr() == out.data_ptr()
    assert args[6] is backend.forward_metadata.qo_indptr
    assert args[7] is backend.forward_metadata.window_kv_indptr
    assert args[8] is backend.forward_metadata.window_kv_indices
    assert kwargs["sm_scale"] == layer.scaling
    assert kwargs["page_size"] == 64
    assert kwargs["num_splits"] == splits


@pytest.mark.parametrize("heads,batch", [(12, 2), (12, 3), (24, 2), (24, 4), (96, 1)])
def test_unqualified_head_batch_pairs_keep_native(make_backend, heads, batch):
    case = _forward_case(make_backend, heads=heads, batch=batch)
    _call(case)
    case[0].dflash_swa_mla_fwd.assert_not_called()
    case[0]._forward_extend_kernel.assert_called_once()


def test_qualified_shape_beyond_backend_capacity_keeps_native(make_backend):
    case = _forward_case(make_backend, batch=4, max_bs=2)
    _call(case)
    case[0].dflash_swa_mla_fwd.assert_not_called()
    case[0]._forward_extend_kernel.assert_called_once()


@pytest.mark.parametrize(
    "mode",
    [ForwardMode.EXTEND, ForwardMode.DRAFT_EXTEND_V2, ForwardMode.DLLM_EXTEND],
)
def test_other_forward_modes_keep_native_extend(make_backend, mode):
    case = _forward_case(make_backend)
    case[2].forward_mode = mode
    _call(case)
    case[0].dflash_swa_mla_fwd.assert_not_called()
    case[0]._forward_extend_kernel.assert_called_once()


@pytest.mark.parametrize(
    "change",
    [
        "noncausal",
        "bidirectional",
        "window",
        "qk_dim",
        "v_dim",
        "heads",
        "custom_mask",
        "sinks",
        "score_mod",
        "aux_tensors",
        "soft_cap",
        "temperature",
        "k_scale",
        "v_scale",
        "k_heads",
        "v_heads",
        "q_dtype",
        "k_dtype",
        "v_dtype",
        "o_dtype",
    ],
)
def test_unsupported_attention_semantics_keep_native_extend(make_backend, change):
    case = _forward_case(make_backend)
    backend, layer, fb, tensors = case
    kwargs = {}
    if change == "noncausal":
        layer.is_cross_attention = True
    elif change == "bidirectional":
        layer.attn_type = AttentionType.DECODER_BIDIRECTIONAL
    elif change == "window":
        layer.sliding_window_size = 2047
    elif change == "qk_dim":
        layer.qk_head_dim = 512
        tensors[0] = torch.empty((8, 12 * 512), dtype=torch.bfloat16)
    elif change == "v_dim":
        layer.v_head_dim = 128
    elif change == "heads":
        layer.tp_q_head_num = 24
        tensors[0] = torch.empty((8, 24 * 576), dtype=torch.bfloat16)
    elif change == "custom_mask":
        backend.forward_metadata.custom_mask = torch.ones(64, dtype=torch.bool)
    elif change in ("sinks", "score_mod", "aux_tensors"):
        kwargs[change] = object()
    elif change == "soft_cap":
        layer.logit_cap = 1.0
    elif change == "temperature":
        layer.xai_temperature_len = 1
    elif change in ("k_scale", "v_scale"):
        setattr(layer, change, 1.0)
    elif change in ("k_heads", "v_heads"):
        i = 1 if change == "k_heads" else 2
        tensors[i] = tensors[i].expand(-1, 2, -1)
    elif change == "o_dtype":
        fb._attn_output = torch.empty((8, 12 * 512), dtype=torch.float16)
    else:
        i = {"q_dtype": 0, "k_dtype": 1, "v_dtype": 2}[change]
        tensors[i] = tensors[i].to(torch.float16)
    _call(case, **kwargs)
    backend.dflash_swa_mla_fwd.assert_not_called()
    backend._forward_extend_kernel.assert_called_once()


@pytest.mark.parametrize("batch,width", [(0, 8), (1, 7), (3, 8)])
def test_empty_ragged_or_oversize_batch_keeps_native(make_backend, batch, width):
    case = _forward_case(make_backend, batch=batch, tokens_per_req=width)
    _call(case)
    case[0].dflash_swa_mla_fwd.assert_not_called()
    case[0]._forward_extend_kernel.assert_called_once()


def test_metadata_width_alone_cannot_admit_short_query_tensor(make_backend):
    case = _forward_case(make_backend, tokens_per_req=7)
    case[0].forward_metadata.max_extend_len = 8
    _call(case)
    case[0].dflash_swa_mla_fwd.assert_not_called()
    case[0]._forward_extend_kernel.assert_called_once()


@pytest.mark.parametrize("branch", ["deterministic", "dcp"])
def test_existing_early_dispatch_takes_precedence(make_backend, branch):
    case = _forward_case(
        make_backend,
        **({"deterministic": True} if branch == "deterministic" else {"dcp": 2}),
    )
    # Even stale or externally modified eligibility must not bypass these paths.
    case[0].use_dflash_swa_mla = True
    assert _call(case) == branch
    case[0].dflash_swa_mla_fwd.assert_not_called()
    case[0]._forward_extend_kernel.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
