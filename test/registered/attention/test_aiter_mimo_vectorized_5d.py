import math
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import aiter_backend, aiter_utils
from sglang.srt.layers.attention.utils import (
    launch_gather_shuffle_5d_to_linear,
    launch_reshape_and_cache_shuffle_5d,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor import model_runner_kv_cache_mixin
from sglang.srt.models import dflash as dflash_model
from sglang.srt.models import mimo_v2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_simulated_acceptance,
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.triton_ops import fused_kv_materialize


def test_mimo_model_uses_native_v128_only_for_target_vectorized_5d(monkeypatch):
    server_args = SimpleNamespace(attention_backend="aiter")
    layout = SimpleNamespace(get=lambda: "vectorized_5d")
    monkeypatch.setattr(mimo_v2.envs, "SGLANG_AITER_KV_CACHE_LAYOUT", layout)

    kwargs = dict(
        head_dim=192,
        v_head_dim=128,
        num_kv_heads=1,
        server_args=server_args,
    )
    assert not mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=False)
    assert mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=True)

    kwargs["num_kv_heads"] = 2
    assert mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=False)
    kwargs["num_kv_heads"] = 1

    layout.get = lambda: "nhd"
    assert mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=False)

    server_args.attention_backend = "triton"
    assert not mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=False)


@pytest.mark.parametrize(
    "full_impl,swa_impl,native",
    [
        ("gluon", "gluon", False),
        ("flydsl", "gluon", False),
        ("gluon", "flydsl", False),
        ("flydsl", "flydsl", True),
    ],
)
def test_flash_tp4_native_v128_requires_both_flydsl_target_paths(
    monkeypatch, full_impl, swa_impl, native
):
    monkeypatch.setenv("SGLANG_AITER_PA_DECODE_IMPL", full_impl)
    monkeypatch.setenv("SGLANG_AITER_TARGET_VERIFY_SWA_IMPL", swa_impl)
    monkeypatch.setenv("SGLANG_AITER_KV_CACHE_LAYOUT", "vectorized_5d")
    monkeypatch.setattr(mimo_v2, "get_attention_tp_size", lambda: 4)
    server_args = SimpleNamespace(attention_backend="aiter")
    config = SimpleNamespace(
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
    )
    assert (
        mimo_v2._mimo_model_uses_native_v_cache(config=config, server_args=server_args)
        is native
    )
    assert (
        mimo_v2._mimo_needs_v_padding(
            head_dim=192,
            v_head_dim=128,
            num_kv_heads=2,
            server_args=server_args,
            force_v_pad=not native,
        )
        is not native
    )


@pytest.mark.parametrize(
    "full_kv_heads,swa_kv_heads,expected",
    [(1, 1, True), (2, 1, False), (1, 2, False)],
)
def test_mimo_pool_native_v128_requires_one_tp_local_kv_head(
    full_kv_heads, swa_kv_heads, expected
):
    text_config = SimpleNamespace(
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
    )
    model_config = SimpleNamespace(
        head_dim=192,
        get_num_kv_heads=lambda tp_size: full_kv_heads,
        get_swa_num_kv_heads=lambda tp_size: swa_kv_heads,
    )

    assert (
        model_runner_kv_cache_mixin._use_native_mimo_vectorized_v_cache(
            model_config=model_config,
            text_config=text_config,
            attention_backend="aiter",
            kv_cache_layout="vectorized_5d",
            tensor_parallel_size=8,
        )
        is expected
    )


def test_mimo_layers_pad_when_either_shared_pool_family_is_not_native(monkeypatch):
    server_args = SimpleNamespace(attention_backend="aiter")
    layout = SimpleNamespace(get=lambda: "vectorized_5d")
    monkeypatch.setattr(mimo_v2.envs, "SGLANG_AITER_KV_CACHE_LAYOUT", layout)
    monkeypatch.setattr(mimo_v2, "get_attention_tp_size", lambda: 8)

    # Full attention has two TP-local KV heads while SWA has one. The shared
    # pools therefore use padded V192, and the SWA layer must pad as well.
    config = SimpleNamespace(
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        num_key_value_heads=16,
        swa_num_key_value_heads=8,
    )
    native_v_cache = mimo_v2._mimo_model_uses_native_v_cache(
        config=config,
        server_args=server_args,
    )

    assert not native_v_cache
    assert mimo_v2._mimo_needs_v_padding(
        head_dim=192,
        v_head_dim=128,
        num_kv_heads=1,
        server_args=server_args,
        force_v_pad=not native_v_cache,
    )


def test_dflash_aiter_flypa_draft_keeps_vectorized_5d(monkeypatch):
    monkeypatch.setattr(model_runner_kv_cache_mixin, "_is_hip", True)
    monkeypatch.setenv("SGLANG_USE_AITER", "1")
    monkeypatch.setenv("SGLANG_AITER_KV_CACHE_LAYOUT", "vectorized_5d")
    monkeypatch.setenv("SGLANG_AITER_DFLASH_SWA_IMPL", "flydsl")
    runner = SimpleNamespace(
        is_draft_worker=True,
        spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
        server_args=SimpleNamespace(attention_backend="aiter"),
        page_size=64,
        kv_cache_dtype=torch.bfloat16,
    )

    assert (
        model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._get_mha_kv_cache_layout_override(
            runner
        )
        is None
    )

    runner.spec_algorithm = SimpleNamespace(is_dflash=lambda: False)
    assert (
        model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._get_mha_kv_cache_layout_override(
            runner
        )
        == "nhd"
    )


def test_mimo_dflash_capture_maps_post_layer_ids():
    model = object.__new__(mimo_v2.MiMoV2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.model = SimpleNamespace(layers_to_capture=[])
    model.capture_aux_hidden_states = False

    model.set_dflash_layers_to_capture([0, 15, 31, 47, 69])

    assert model.capture_aux_hidden_states
    assert model.model.layers_to_capture == [1, 16, 32, 48, 70]


def test_mimo_attention_sink_stays_fp32_under_bf16_default(monkeypatch):
    class FakeLinear(torch.nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    monkeypatch.setattr(mimo_v2, "get_attention_tp_rank", lambda: 0)
    monkeypatch.setattr(mimo_v2, "get_attention_tp_size", lambda: 1)
    monkeypatch.setattr(
        mimo_v2,
        "get_global_server_args",
        lambda: SimpleNamespace(attention_backend="aiter"),
    )
    monkeypatch.setattr(mimo_v2, "_mimo_needs_v_padding", lambda **_kwargs: False)
    monkeypatch.setattr(mimo_v2, "QKVParallelLinear", FakeLinear)
    monkeypatch.setattr(mimo_v2, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(mimo_v2, "get_rope", lambda *_args, **_kwargs: FakeLinear())
    monkeypatch.setattr(mimo_v2, "RadixAttention", FakeLinear)

    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        attention = mimo_v2.MiMoV2Attention(
            hidden_size=8,
            num_heads=1,
            num_kv_heads=1,
            head_dim=8,
            v_head_dim=8,
            attention_sink_bias=True,
        )
    finally:
        torch.set_default_dtype(original_dtype)

    assert attention.attention_sink_bias.dtype == torch.float32
    assert attention.attention_sink_bias.is_contiguous()


def test_mimo_model_returns_requested_intermediate_and_final_captures(monkeypatch):
    class FakeLayer:
        layer_scatter_modes = SimpleNamespace(layer_output_mode=None)

        def __call__(
            self,
            _positions,
            hidden_states,
            _forward_batch,
            residual,
            captured_last_layer_outputs=None,
        ):
            if captured_last_layer_outputs is not None:
                captured_last_layer_outputs.append(hidden_states.clone())
            return hidden_states + 1, residual

    class FakeNorm:
        def __call__(self, hidden_states, residual=None):
            if residual is None:
                return hidden_states
            return hidden_states + residual, None

    monkeypatch.setattr(
        mimo_v2,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_none=lambda: False),
    )
    model = object.__new__(mimo_v2.MiMoV2Model)
    torch.nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        world_size=1,
    )
    model.config = SimpleNamespace(num_hidden_layers=3)
    model.layers = [FakeLayer(), FakeLayer(), FakeLayer()]
    model.layers_to_capture = [1, 3]
    model.start_layer = 0
    model.end_layer = 3
    model.norm = FakeNorm()
    model._logged_no_ep_tbo = False
    model._logged_no_ep_tbo_fallback = False
    forward_batch = SimpleNamespace(
        can_run_tbo=False,
        return_hidden_states_before_norm=False,
    )
    input_embeds = torch.zeros((2, 4), dtype=torch.bfloat16)

    hidden_states, hidden_before_norm, captures = model.forward(
        input_ids=torch.zeros(2, dtype=torch.int64),
        positions=torch.arange(2),
        forward_batch=forward_batch,
        input_embeds=input_embeds,
    )

    assert hidden_before_norm is None
    torch.testing.assert_close(hidden_states, torch.full_like(input_embeds, 3))
    assert len(captures) == 2
    torch.testing.assert_close(captures[0], torch.full_like(input_embeds, 1))
    torch.testing.assert_close(captures[1], torch.full_like(input_embeds, 3))


def test_dflash_nested_swa_config_overrides_synthesized_full_layers():
    config = {
        "text_config": {
            "num_hidden_layers": 5,
            "layer_types": ["full_attention"] * 5,
        },
        "dflash_config": {
            "use_swa": True,
            "swa_window_size": 1024,
            "attention_sink_bias": True,
            "attention_value_scale": 0.612,
        },
    }

    assert get_dflash_layer_types(config) == ["sliding_attention"] * 5
    assert get_dflash_attention_sliding_window_size(config) == 1023
    parsed = parse_dflash_draft_config(draft_hf_config=config)
    assert parsed.attention_sink_bias
    assert parsed.attention_value_scale == pytest.approx(0.612)


def test_dflash_context_kv_materialization_applies_value_scale(monkeypatch):
    class FakeQKV(torch.nn.Module):
        def forward(self, hidden_states):
            rows = hidden_states.shape[0]
            q = torch.zeros((rows, 4), dtype=hidden_states.dtype)
            k = torch.ones((rows, 2), dtype=hidden_states.dtype)
            v = torch.full((rows, 2), 2, dtype=hidden_states.dtype)
            return torch.cat((q, k, v), dim=-1), None

    monkeypatch.setattr(
        dflash_model, "can_dflash_slice_qkv_weight", lambda _proj: (False, "test")
    )
    attention = object.__new__(dflash_model.DFlashAttention)
    torch.nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.v_scale = 0.5
    attention.qkv_proj = FakeQKV()

    key, value = attention.kv_proj_only(torch.zeros((3, 4), dtype=torch.bfloat16))

    torch.testing.assert_close(key, torch.ones_like(key))
    torch.testing.assert_close(value, torch.ones_like(value))


def test_dflash_attention_forwards_scaled_value_and_sink(monkeypatch):
    class FakeQKV(torch.nn.Module):
        def forward(self, hidden_states):
            rows = hidden_states.shape[0]
            q = torch.zeros((rows, 2), dtype=hidden_states.dtype)
            k = torch.ones((rows, 1), dtype=hidden_states.dtype)
            v = torch.full((rows, 1), 2, dtype=hidden_states.dtype)
            return torch.cat((q, k, v), dim=-1), None

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.captured = None

        def forward(self, q, k, v, forward_batch, sinks=None):
            self.captured = (q, k, v, forward_batch, sinks)
            return q

    class FakeOutput(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(dflash_model, "apply_qk_norm", lambda q, k, *_args: (q, k))
    attention = object.__new__(dflash_model.DFlashAttention)
    torch.nn.Module.__init__(attention)
    attention.q_size = 2
    attention.kv_size = 1
    attention.num_kv_heads = 1
    attention.head_dim = 1
    attention.v_scale = 0.5
    attention.attention_sink_bias = torch.nn.Parameter(
        torch.tensor([0.25], dtype=torch.float32), requires_grad=False
    )
    attention.qkv_proj = FakeQKV()
    attention.q_norm = FakeOutput()
    attention.k_norm = FakeOutput()
    attention.rotary_emb = lambda _positions, q, k: (q, k)
    attention.attn = FakeAttention()
    attention.o_proj = FakeOutput()
    hidden_states = torch.zeros((3, 4), dtype=torch.bfloat16)
    forward_batch = object()

    attention.forward(torch.arange(3), hidden_states, forward_batch)

    _, _, value, captured_batch, sinks = attention.attn.captured
    torch.testing.assert_close(value, torch.ones_like(value))
    assert captured_batch is forward_batch
    assert sinks is attention.attention_sink_bias
    assert sinks.dtype == torch.float32


def test_dflash_simulated_acceptance_uses_commit_length_semantics():
    candidates = torch.arange(16, dtype=torch.int64).view(2, 8)
    accept_len = torch.empty(2, dtype=torch.int32)
    commit_lens = torch.empty(2, dtype=torch.int32)
    bonus = torch.empty(2, dtype=torch.int64)
    out_tokens = torch.empty((2, 8), dtype=torch.int64)

    apply_dflash_simulated_acceptance(
        candidates=candidates,
        accept_len=accept_len,
        commit_lens=commit_lens,
        bonus=bonus,
        out_tokens=out_tokens,
        simulate_acc_len=4,
        simulate_acc_method="match-expected",
    )

    assert accept_len.tolist() == [3, 3]
    assert commit_lens.tolist() == [4, 4]
    assert bonus.tolist() == [100, 100]
    assert torch.all(out_tokens == 100)


def test_fused_dflash_kv_materialization_forwards_layer_value_scales(monkeypatch):
    rotary = SimpleNamespace(
        rotary_dim=2,
        is_neox_style=True,
        cos_sin_cache=torch.zeros((8, 4), dtype=torch.bfloat16),
    )

    def make_layer(value_scale):
        attention = SimpleNamespace(
            num_kv_heads=1,
            head_dim=4,
            rotary_emb=rotary,
            qkv_proj=SimpleNamespace(weight=torch.zeros((12, 4))),
            q_size=4,
            kv_size=4,
            k_norm=SimpleNamespace(
                weight=torch.ones(4),
                variance_epsilon=1e-5,
            ),
            v_scale=value_scale,
        )
        return SimpleNamespace(self_attn=attention)

    helper = fused_kv_materialize.FusedKVMaterializeHelper(
        layers=[make_layer(0.612), make_layer(None)],
        rotary_emb=rotary,
        num_kv_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
    )
    captured = {}

    def fake_norm_rope(
        kv,
        _k_norm_weight,
        _eps,
        value_scales,
        _cos_sin_cache,
        _positions,
        _num_kv_heads,
        _head_dim,
        _rotary_dim,
        *,
        k_out,
        v_out,
    ):
        captured["value_scales"] = value_scales.clone()
        return k_out.zero_(), v_out.zero_()

    monkeypatch.setattr(
        fused_kv_materialize, "_fused_norm_rope_stacked", fake_norm_rope
    )
    writes = []
    helper.materialize(
        ctx_hidden=torch.ones((2, 4)),
        positions=torch.tensor([0, 1], dtype=torch.int64),
        write_layer_kv=lambda layer, key, value: writes.append(
            (layer, key.shape, value.shape)
        ),
    )

    torch.testing.assert_close(captured["value_scales"], torch.tensor([0.612, 1.0]))
    assert writes == [
        (0, torch.Size([2, 1, 4]), torch.Size([2, 1, 4])),
        (1, torch.Size([2, 1, 4]), torch.Size([2, 1, 4])),
    ]


def test_dflash_trims_trailing_target_hidden_padding_before_kv_materialization(
    monkeypatch, caplog
):
    worker = object.__new__(DFlashWorkerV2)
    worker.model_runner = SimpleNamespace(device=torch.device("cpu"))
    worker._use_fused_kv_materialize = True
    worker._fused_kv_helper = object()
    worker._logged_padding_trim = False

    captured = {}

    def project_target_hidden(target_hidden):
        captured["project_input"] = target_hidden.clone()
        return target_hidden

    def append_target_hidden_fused(**kwargs):
        captured["fused_ctx_hidden"] = kwargs["ctx_hidden"].clone()
        captured["fused_positions"] = kwargs["ctx_positions"].clone()
        captured["fused_cache_loc"] = kwargs["ctx_cache_loc"].clone()

    worker.draft_model = SimpleNamespace(
        project_target_hidden=project_target_hidden,
    )
    monkeypatch.setattr(
        worker,
        "_append_target_hidden_fused",
        append_target_hidden_fused,
    )

    target_hidden = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    cache_loc = torch.arange(4, dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int64)

    worker._append_target_hidden_to_draft_kv_by_loc(
        target_hidden=target_hidden,
        cache_loc=cache_loc,
        positions=positions,
    )

    torch.testing.assert_close(captured["project_input"], target_hidden[:4])
    torch.testing.assert_close(captured["fused_ctx_hidden"], target_hidden[:4])
    torch.testing.assert_close(captured["fused_positions"], positions)
    torch.testing.assert_close(captured["fused_cache_loc"], cache_loc)
    assert worker._logged_padding_trim
    assert "trailing padding row(s)" in caplog.text


def test_dflash_merges_trained_mask_embedding_into_local_vocab_shard(tmp_path):
    mask_token_id = 7
    trained_embedding = torch.arange(4, dtype=torch.bfloat16)
    torch.save(
        {"mask_token_id": mask_token_id, "embedding": trained_embedding},
        tmp_path / "mask_embedding.pt",
    )
    embedding_module = torch.nn.Embedding(10, 4, dtype=torch.float32)
    embedding_module.weight.data.zero_()
    embedding_module.shard_indices = SimpleNamespace(
        org_vocab_start_index=5,
        org_vocab_end_index=15,
    )
    worker = object.__new__(DFlashWorkerV2)
    worker.server_args = SimpleNamespace(speculative_draft_model_path=str(tmp_path))
    worker.device = torch.device("cpu")
    worker._mask_token_id = mask_token_id
    worker.tp_rank = 0
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(
                get_input_embeddings=lambda: embedding_module,
            )
        )
    )

    worker._maybe_merge_trained_mask_embedding()

    torch.testing.assert_close(
        embedding_module.weight[mask_token_id - 5], trained_embedding.float()
    )


def _make_fresh_asm_case(sequence_length=256, batch_size=2):
    total_tokens = sequence_length * batch_size
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        logits_soft_cap=0.0,
        forward_metadata=SimpleNamespace(max_q_len=sequence_length),
        qo_indptr=torch.arange(
            0,
            total_tokens + 1,
            sequence_length,
            dtype=torch.int32,
        ),
    )
    layer = SimpleNamespace(
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0] * batch_size,
        extend_seq_lens_cpu=[sequence_length] * batch_size,
        seq_lens_cpu=torch.full(
            (batch_size,), sequence_length, dtype=torch.int32
        ),
    )
    q = torch.zeros((total_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((total_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((total_tokens, 1, 128), dtype=torch.bfloat16)
    return backend, layer, forward_batch, q, k, v


def _make_fresh_swa_case(lengths=(255, 257), sink_dtype=torch.bfloat16):
    total_tokens = sum(lengths)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        logits_soft_cap=0.0,
        forward_metadata=SimpleNamespace(max_q_len=max(lengths)),
        qo_indptr=torch.tensor(cumulative, dtype=torch.int32),
    )
    layer = SimpleNamespace(
        sliding_window_size=128,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
        mimo_original_v_head_dim=128,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0] * len(lengths),
        extend_seq_lens_cpu=list(lengths),
        seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
    )
    q = torch.zeros((total_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((total_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((total_tokens, 1, 128), dtype=torch.bfloat16)
    sinks = torch.linspace(-1.0, 1.0, 16, dtype=sink_dtype)
    return backend, layer, forward_batch, q, k, v, sinks


def _make_cached_bf16_chunk_case(prefix_len=256, extend_len=256):
    seq_len = prefix_len + extend_len
    num_blocks = (seq_len + 63) // 64
    k_buf = torch.zeros((num_blocks, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_buf = torch.zeros((num_blocks, 1, 8, 128, 8), dtype=torch.bfloat16)
    pool = SimpleNamespace(
        dtype=torch.bfloat16,
        store_dtype=torch.bfloat16,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    metadata = SimpleNamespace(
        swa_page_table=None,
        kv_indices=torch.arange(seq_len, dtype=torch.int32),
        kv_indptr=torch.tensor([0, seq_len], dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        max_q_len=extend_len,
        max_kv_len=seq_len,
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
        logits_soft_cap=0.0,
        token_to_kv_pool=pool,
        forward_metadata=metadata,
        qo_indptr=torch.tensor([0, extend_len], dtype=torch.int32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
        mimo_original_v_head_dim=128,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[prefix_len],
        extend_seq_lens_cpu=[extend_len],
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_sum=seq_len,
    )
    q = torch.zeros((extend_len, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((extend_len, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((extend_len, 1, 128), dtype=torch.bfloat16)
    return backend, layer, forward_batch, q, k, v


def _make_cached_bf16_ragged_case(
    prefix_lengths=(256, 0),
    extend_lengths=(257, 17),
):
    seq_lengths = [
        prefix_len + extend_len
        for prefix_len, extend_len in zip(prefix_lengths, extend_lengths)
    ]
    total_q = sum(extend_lengths)
    total_kv = sum(seq_lengths)
    num_blocks = (total_kv + 63) // 64
    k_buf = torch.zeros((num_blocks, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_buf = torch.zeros((num_blocks, 1, 8, 128, 8), dtype=torch.bfloat16)
    pool = SimpleNamespace(
        dtype=torch.bfloat16,
        store_dtype=torch.bfloat16,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )

    qo_indptr = [0]
    kv_indptr = [0]
    for extend_len, seq_len in zip(extend_lengths, seq_lengths):
        qo_indptr.append(qo_indptr[-1] + extend_len)
        kv_indptr.append(kv_indptr[-1] + seq_len)
    metadata = SimpleNamespace(
        swa_page_table=None,
        kv_indices=torch.arange(total_kv, dtype=torch.int32),
        kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        max_q_len=max(extend_lengths),
        max_kv_len=max(seq_lengths),
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
        logits_soft_cap=0.0,
        token_to_kv_pool=pool,
        forward_metadata=metadata,
        qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
        mimo_original_v_head_dim=128,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=list(prefix_lengths),
        extend_seq_lens_cpu=list(extend_lengths),
        seq_lens_cpu=torch.tensor(seq_lengths, dtype=torch.int32),
        seq_lens_sum=total_kv,
    )
    q = torch.zeros((total_q, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((total_q, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((total_q, 1, 128), dtype=torch.bfloat16)
    return backend, layer, forward_batch, q, k, v


@pytest.mark.parametrize("lengths", [(256, 256), (255, 257)])
def test_fresh_mimo_swa_uses_native_v128_ck_varlen(monkeypatch, lengths):
    backend, layer, forward_batch, q, k, v, sinks = _make_fresh_swa_case(lengths)
    captured = {}

    def fake_varlen(q_varlen, k_varlen, v_varlen, *args, **kwargs):
        captured.update(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_q=args[0],
            cu_k=args[1],
            max_q=args[2],
            max_k=args[3],
            kwargs=kwargs,
        )
        kwargs["out"].fill_(7.0)
        return kwargs["out"]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified fresh SWA input must not use CK prefill")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_SWA_VARLEN_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "flash_attn_varlen_func", fake_varlen)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=len(lengths) + 1,
        window_size=(128, -1),
        sinks=sinks,
    ).view(sum(lengths), 16, 128)

    assert captured["q"].shape == (sum(lengths), 16, 192)
    assert captured["k"].shape == (sum(lengths), 1, 192)
    assert captured["v"].shape == (sum(lengths), 1, 128)
    assert captured["v"].stride(-2) == 128
    assert captured["cu_q"].tolist() == [0, lengths[0], sum(lengths)]
    assert captured["cu_k"].tolist() == [0, lengths[0], sum(lengths)]
    assert captured["max_q"] == max(lengths)
    assert captured["max_k"] == max(lengths)
    assert captured["kwargs"]["min_seqlen_q"] == 0
    assert captured["kwargs"]["window_size"] == (128, 0, 0)
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["sink_ptr"].dtype == torch.float32
    assert captured["kwargs"]["out"].shape == (sum(lengths), 16, 128)
    assert captured["kwargs"]["out"].stride(-2) == 128
    assert torch.all(output == 7.0)


def test_fresh_mimo_swa_gfx942_fallback_uses_asymmetric_batch_prefill(
    monkeypatch,
):
    backend, layer, forward_batch, q, k, v, sinks = _make_fresh_swa_case()
    captured = {}

    def fake_batch_prefill(q_in, k_in, v_in, *args, **kwargs):
        captured.update(q=q_in, k=k_in, v=v_in, args=args, kwargs=kwargs)
        return torch.full(
            (q_in.shape[0], q_in.shape[1], 128),
            13.0,
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_SWA_VARLEN_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: False)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", fake_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=3,
        window_size=(128, -1),
        sinks=sinks,
    ).view(512, 16, 128)

    assert captured["q"].shape == (512, 16, 192)
    assert captured["k"].shape == (512, 1, 192)
    assert captured["v"].shape == (512, 1, 128)
    assert captured["args"][0].tolist() == [0, 255, 512]
    assert captured["args"][1].tolist() == [0, 255, 512]
    assert captured["args"][2].tolist() == list(range(512))
    assert captured["args"][3:5] == (257, 257)
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["window_size"] == (128, -1)
    assert captured["kwargs"]["sink_ptr"] is sinks
    assert output.shape == (512, 16, 128)
    assert torch.all(output == 13.0)


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(),
    reason="requires ROCm CK prefill",
)
@pytest.mark.parametrize("prefix_len", [0, 73])
@pytest.mark.parametrize("swa", [False, True])
@pytest.mark.parametrize("value_dim", [128, 192])
def test_gfx942_ck_linear_prefill_matches_torch(monkeypatch, prefix_len, swa, value_dim):
    if not aiter_utils.is_gfx942():
        pytest.skip("regression for the gfx942 CK LINEAR path")
    monkeypatch.setenv("SGLANG_FLYPA_MIMO_PREFILL", "0")
    monkeypatch.setenv("SGLANG_FLYDSL_MIMO_PREFILL", "0")
    torch.manual_seed(12345)
    device = torch.device("cuda")
    lengths = [255, 257]
    kv_lengths = [length + prefix_len for length in lengths]
    q_heads, kv_heads = 16, 2 if swa and value_dim == 192 else 1
    total_q, total_kv = sum(lengths), sum(kv_lengths)
    # Keep Q strided as it is after MiMo's fused QKV projection.
    q_storage = torch.randn(
        total_q, q_heads * 192 + 384, device=device, dtype=torch.bfloat16
    )
    q = q_storage[:, : q_heads * 192].view(total_q, q_heads, 192)
    k = torch.randn(total_kv, kv_heads, 192, device=device, dtype=torch.bfloat16)
    v = torch.randn(total_kv, kv_heads, value_dim, device=device, dtype=torch.bfloat16)
    sinks = torch.linspace(-1, 1, q_heads, device=device) if swa else None
    qo_indptr = torch.tensor([0, lengths[0], total_q], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor(
        [0, kv_lengths[0], total_kv], dtype=torch.int32, device=device
    )
    metadata = SimpleNamespace(
        max_q_len=max(lengths),
        max_kv_len=max(kv_lengths),
        kv_indptr=kv_indptr,
        kv_indices=torch.arange(total_kv, dtype=torch.int32, device=device),
        swa_page_table=None,
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
        logits_soft_cap=0.0,
        qo_indptr=qo_indptr,
        forward_metadata=metadata,
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=128 if swa else -1,
        tp_q_head_num=q_heads,
        tp_k_head_num=kv_heads,
        tp_v_head_num=kv_heads,
        qk_head_dim=192,
        head_dim=192,
        v_head_dim=value_dim,
        scaling=192**-0.5,
    )
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[prefix_len] * 2,
        extend_seq_lens_cpu=lengths,
        seq_lens_cpu=torch.tensor(kv_lengths),
        seq_lens_sum=total_kv,
    )
    if prefix_len:
        # Exercise the real SHUFFLE gather as well as the fresh-prompt path.
        pages = math.ceil(total_kv / 64)
        k_padded = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pages * 64 - total_kv))
        v_padded = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pages * 64 - total_kv))
        backend.token_to_kv_pool = SimpleNamespace(
            start_layer=0,
            dtype=torch.bfloat16,
            store_dtype=torch.bfloat16,
            k_buffer=[
                k_padded.view(pages, 64, kv_heads, 24, 8)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            ],
            v_buffer=[
                v_padded.view(pages, 8, 8, kv_heads, value_dim)
                .permute(0, 3, 1, 4, 2)
                .contiguous()
            ],
        )
        if swa:
            metadata.swa_page_table = metadata.kv_indices

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        batch,
        bs0=3,
        window_size=(128, -1) if swa else (-1, -1),
        sinks=sinks,
    ).view(total_q, q_heads, value_dim)
    torch.cuda.synchronize()

    q_start = kv_start = 0
    for q_len, kv_len in zip(lengths, kv_lengths):
        k_ref = (
            k[kv_start : kv_start + kv_len]
            .float()
            .repeat_interleave(q_heads // kv_heads, dim=1)
        )
        v_ref = (
            v[kv_start : kv_start + kv_len]
            .float()
            .repeat_interleave(q_heads // kv_heads, dim=1)
        )
        scores = torch.einsum(
            "qhd,khd->hqk", q[q_start : q_start + q_len].float(), k_ref
        ) * layer.scaling
        q_pos = torch.arange(q_len, device=device)[:, None] + prefix_len
        k_pos = torch.arange(kv_len, device=device)[None, :]
        mask = k_pos <= q_pos
        if swa:
            mask &= k_pos >= q_pos - 128
        scores.masked_fill_(~mask, float("-inf"))
        if swa:
            scores = torch.cat(
                [scores, sinks[:, None, None].expand(-1, q_len, 1)], dim=-1
            )
        probs = torch.softmax(scores, dim=-1)[..., :kv_len]
        expected = torch.einsum("hqk,khd->qhd", probs, v_ref)
        torch.testing.assert_close(
            output[q_start : q_start + q_len].float(), expected, atol=0.02, rtol=0.02
        )
        q_start += q_len
        kv_start += kv_len


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "wrong_arch",
        "short",
        "wrong_window",
        "no_sink",
        "wrong_sink_shape",
        "non_mimo",
        "wrong_v_shape",
        "logit_cap",
    ],
)
def test_fresh_mimo_swa_varlen_contract_guards_fall_back(monkeypatch, guard):
    backend, layer, forward_batch, q, k, v, sinks = _make_fresh_swa_case()
    enabled = guard != "disabled"
    is_gfx950 = guard != "wrong_arch"
    window_size = (128, -1)

    if guard == "short":
        forward_batch.extend_seq_lens_cpu = [128, 384]
    elif guard == "wrong_window":
        window_size = (127, -1)
    elif guard == "no_sink":
        sinks = None
    elif guard == "wrong_sink_shape":
        sinks = torch.zeros(8, dtype=torch.float32)
    elif guard == "non_mimo":
        layer.mimo_original_v_head_dim = None
    elif guard == "wrong_v_shape":
        v = torch.zeros((512, 1, 192), dtype=torch.bfloat16)
    elif guard == "logit_cap":
        backend.logits_soft_cap = 50.0

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_SWA_VARLEN_ENABLED", enabled)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: is_gfx950)
    monkeypatch.setattr(aiter_utils, "flash_attn_varlen_func", lambda: None)

    assert not aiter_utils.can_use_mimo_fresh_bf16_swa_varlen(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        window_size,
        sinks,
    )


def test_fresh_uniform_mimo_extend_uses_gfx950_bf16_asm(monkeypatch):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case()
    captured = {}

    def fake_asm(q_4d, k_4d, v_4d, *args):
        out = args[8]
        captured.update(q=q_4d, k=k_4d, v=v_4d, out=out)
        out.fill_(3.0)
        return [out]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified fresh uniform input must not use CK prefill")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_fwd", fake_asm)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=3,
        window_size=(-1, -1),
        sinks=None,
    ).view(2, 256, 16, 128)

    assert captured["q"].shape == (2, 256, 16, 192)
    assert captured["k"].shape == (2, 256, 1, 192)
    assert captured["v"].shape == (2, 256, 1, 128)
    assert captured["out"].shape == (2, 256, 16, 128)
    assert captured["out"].stride(-2) == 128
    assert torch.all(output == 3.0)
    assert torch.isfinite(output).all()


def test_short_fresh_mimo_extend_uses_gfx950_bf16_varlen(monkeypatch):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case(
        sequence_length=1, batch_size=1
    )
    captured = {}

    def fake_varlen(q_varlen, k_varlen, v_varlen, *args, **kwargs):
        captured.update(q=q_varlen, k=k_varlen, v=v_varlen, args=args, kwargs=kwargs)
        return torch.full((1, 16, 128), 17.0, dtype=torch.bfloat16)

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("gfx950 short BF16 extend must not use CK page-1 prefill")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "flash_attn_varlen_func", fake_varlen)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(-1, -1),
        sinks=None,
    ).view(1, 16, 128)

    assert captured["q"].shape == (1, 16, 192)
    assert captured["k"].shape == (1, 1, 192)
    assert captured["v"].shape == (1, 1, 128)
    assert captured["args"][0].tolist() == [0, 1]
    assert captured["args"][1].tolist() == [0, 1]
    assert captured["args"][2:4] == (1, 1)
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["window_size"] == (-1, -1, 0)
    assert torch.all(output == 17.0)


def test_tbo_padded_fresh_mimo_extend_uses_gfx950_bf16_asm(monkeypatch):
    logical_tokens = 257
    physical_tokens = 264
    backend, layer, forward_batch, _, _, _ = _make_fresh_asm_case(
        sequence_length=logical_tokens,
        batch_size=1,
    )
    q = torch.zeros((physical_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((physical_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((physical_tokens, 1, 128), dtype=torch.bfloat16)
    captured = {}

    def fake_asm(q_4d, k_4d, v_4d, *args):
        out = args[8]
        captured.update(q=q_4d, k=k_4d, v=v_4d, out=out)
        out.fill_(3.0)
        return [out]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("TBO padding must not force fresh attention to CK")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_fwd", fake_asm)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(-1, -1),
        sinks=None,
    ).view(physical_tokens, 16, 128)

    assert captured["q"].shape == (1, logical_tokens, 16, 192)
    assert captured["k"].shape == (1, logical_tokens, 1, 192)
    assert captured["v"].shape == (1, logical_tokens, 1, 128)
    assert captured["out"].shape == (1, logical_tokens, 16, 128)
    assert torch.all(output[:logical_tokens] == 3.0)
    assert torch.count_nonzero(output[logical_tokens:]) == 0


def test_fresh_ragged_mimo_extend_uses_gfx950_bf16_varlen_asm(monkeypatch):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case()
    forward_batch.extend_seq_lens_cpu = [255, 257]
    backend.qo_indptr = torch.tensor([0, 255, 512], dtype=torch.int32)
    captured = {}

    def fake_varlen_asm(q_varlen, k_varlen, v_varlen, *args):
        out = args[15]
        captured.update(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_q=args[0],
            cu_k=args[1],
            max_q=args[2],
            min_q=args[4],
            out=out,
        )
        out.fill_(5.0)
        return [out]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified fresh ragged input must not use CK prefill")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", fake_varlen_asm)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=3,
        window_size=(-1, -1),
        sinks=None,
    ).view(512, 16, 128)

    assert captured["q"].shape == (512, 16, 192)
    assert captured["k"].shape == (512, 1, 192)
    assert captured["v"].shape == (512, 1, 128)
    assert captured["cu_q"].tolist() == [0, 255, 512]
    assert captured["cu_k"].tolist() == [0, 255, 512]
    assert captured["max_q"] == 257
    assert captured["min_q"] == 255
    assert captured["out"].shape == (512, 16, 128)
    assert captured["out"].stride(-2) == 128
    assert torch.all(output == 5.0)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("reported_prefix_len", [256, 0])
def test_cached_bf16_chunk_prefill_uses_gfx950_opus_varlen_asm(
    monkeypatch, reported_prefix_len
):
    backend, layer, forward_batch, q, k, v = _make_cached_bf16_chunk_case()
    forward_batch.extend_prefix_lens_cpu = [reported_prefix_len]
    metadata = backend.forward_metadata
    captured = {}

    def fake_gather(k_buf, v_buf, slot_ids):
        captured["gather_slot_ids"] = slot_ids
        seq_len = slot_ids.numel()
        return (
            torch.zeros((seq_len, 1, 192), dtype=torch.bfloat16),
            torch.zeros((seq_len, 1, 128), dtype=torch.bfloat16),
        )

    def fake_opus_varlen_asm(q_varlen, k_varlen, v_varlen, **kwargs):
        out = kwargs["out"]
        captured.update(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_q=kwargs["seqstart_q"],
            cu_k=kwargs["seqstart_k"],
            max_q=kwargs["max_seqlen_q"],
            max_k=kwargs["max_seqlen_k"],
            causal=kwargs["causal"],
            out=out,
        )
        out.fill_(11.0)
        return out

    def reject_v3(*args, **kwargs):
        raise AssertionError("OPUS-qualified BF16 chunk must not use v3 ASM")

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified BF16 chunk must not use CK prefill")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.delenv("AITER_DISABLE_FMHA_OPUS", raising=False)
    monkeypatch.setattr(
        aiter_utils,
        "fmha_fwd_bf16_opus_varlen_fwd",
        fake_opus_varlen_asm,
    )
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", reject_v3)
    monkeypatch.setattr(aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(-1, -1),
        sinks=None,
    ).view(q.shape[0], 16, 128)

    assert torch.equal(captured["gather_slot_ids"], metadata.kv_indices)
    assert captured["q"].shape == (256, 16, 192)
    assert captured["k"].shape == (512, 1, 192)
    assert captured["v"].shape == (512, 1, 128)
    assert captured["v"].stride(-2) == 128
    assert captured["cu_q"].tolist() == [0, 256]
    assert captured["cu_k"].tolist() == [0, 512]
    assert captured["max_q"] == 256
    assert captured["max_k"] == 512
    assert captured["causal"] is True
    assert captured["out"].shape == (256, 16, 128)
    assert captured["out"].stride(-2) == 128
    assert torch.all(output == 11.0)


def test_ragged_cached_bf16_chunk_splits_dense_opus(monkeypatch):
    prefix_lengths = (256, 0)
    extend_lengths = (257, 17)
    logical_tokens = sum(extend_lengths)
    physical_tokens = 280
    backend, layer, forward_batch, _, _, _ = _make_cached_bf16_ragged_case(
        prefix_lengths=prefix_lengths,
        extend_lengths=extend_lengths,
    )
    q = torch.zeros((physical_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((physical_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((physical_tokens, 1, 128), dtype=torch.bfloat16)
    captured = []

    def fake_gather(k_buf, v_buf, slot_ids):
        total_kv = slot_ids.numel()
        return (
            torch.zeros((total_kv, 1, 192), dtype=torch.bfloat16),
            torch.zeros((total_kv, 1, 128), dtype=torch.bfloat16),
        )

    def fake_dense_opus(q_dense, k_dense, v_dense, **kwargs):
        captured.append((q_dense.shape, k_dense.shape, v_dense.shape))
        kwargs["out"].fill_(15.0)
        return kwargs["out"]

    def reject_group_opus(*args, **kwargs):
        raise AssertionError("qualified imbalanced batch must split dense OPUS")

    def reject_v3(*args, **kwargs):
        raise AssertionError("qualified OPUS batch must not use v3 ASM")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "MIMO_CHUNK_BF16_OPUS_SPLIT_MIN_Q", 0)
    monkeypatch.setattr(aiter_utils, "MIMO_CHUNK_BF16_OPUS_SPLIT_MIN_KV", 0)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.delenv("AITER_DISABLE_FMHA_OPUS", raising=False)
    monkeypatch.setattr(aiter_utils, "fmha_fwd_bf16_opus_fwd", fake_dense_opus)
    monkeypatch.setattr(
        aiter_utils,
        "fmha_fwd_bf16_opus_varlen_fwd",
        reject_group_opus,
    )
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", reject_v3)
    monkeypatch.setattr(aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=3,
        window_size=(-1, -1),
        sinks=None,
    ).view(physical_tokens, 16, 128)

    assert captured == [
        (
            torch.Size([1, 257, 16, 192]),
            torch.Size([1, 513, 1, 192]),
            torch.Size([1, 513, 1, 128]),
        ),
        (
            torch.Size([1, 17, 16, 192]),
            torch.Size([1, 17, 1, 192]),
            torch.Size([1, 17, 1, 128]),
        ),
    ]
    assert torch.all(output[:logical_tokens] == 15.0)
    assert torch.count_nonzero(output[logical_tokens:]) == 0


def test_tbo_padded_cached_bf16_chunk_uses_gfx950_varlen_asm(monkeypatch):
    logical_tokens = 257
    physical_tokens = 264
    backend, layer, forward_batch, _, _, _ = _make_cached_bf16_chunk_case(
        prefix_len=256,
        extend_len=logical_tokens,
    )
    q = torch.zeros((physical_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((physical_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((physical_tokens, 1, 128), dtype=torch.bfloat16)
    captured = {}

    def fake_gather(k_buf, v_buf, slot_ids):
        seq_len = slot_ids.numel()
        return (
            torch.zeros((seq_len, 1, 192), dtype=torch.bfloat16),
            torch.zeros((seq_len, 1, 128), dtype=torch.bfloat16),
        )

    def fake_varlen_asm(q_varlen, k_varlen, v_varlen, *args):
        out = args[15]
        captured.update(q=q_varlen, k=k_varlen, v=v_varlen, out=out)
        out.fill_(11.0)
        return [out]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("TBO padding must not force cached attention to CK")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setenv("AITER_DISABLE_FMHA_OPUS", "1")
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", fake_varlen_asm)
    monkeypatch.setattr(aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(-1, -1),
        sinks=None,
    ).view(physical_tokens, 16, 128)

    assert captured["q"].shape == (logical_tokens, 16, 192)
    assert captured["out"].shape == (logical_tokens, 16, 128)
    assert torch.all(output[:logical_tokens] == 11.0)
    assert torch.count_nonzero(output[logical_tokens:]) == 0


def test_tbo_ragged_cached_bf16_batch_uses_gfx950_varlen_asm(monkeypatch):
    prefix_lengths = (256, 0)
    extend_lengths = (257, 17)
    logical_tokens = sum(extend_lengths)
    physical_tokens = 280
    backend, layer, forward_batch, _, _, _ = _make_cached_bf16_ragged_case(
        prefix_lengths=prefix_lengths,
        extend_lengths=extend_lengths,
    )
    q = torch.zeros((physical_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((physical_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((physical_tokens, 1, 128), dtype=torch.bfloat16)
    captured = {}

    def fake_gather(k_buf, v_buf, slot_ids):
        captured["slot_ids"] = slot_ids
        total_kv = slot_ids.numel()
        return (
            torch.zeros((total_kv, 1, 192), dtype=torch.bfloat16),
            torch.zeros((total_kv, 1, 128), dtype=torch.bfloat16),
        )

    def fake_varlen_asm(q_varlen, k_varlen, v_varlen, *args):
        out = args[15]
        captured.update(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_q=args[0],
            cu_k=args[1],
            max_q=args[2],
            max_k=args[3],
            min_q=args[4],
            out=out,
        )
        out.fill_(13.0)
        return [out]

    def reject_flypa(*args, **kwargs):
        raise AssertionError("qualified ragged BF16 batch must not use FlyPA")

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified ragged BF16 batch must not use CK")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setenv("AITER_DISABLE_FMHA_OPUS", "1")
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", fake_varlen_asm)
    monkeypatch.setattr(aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather)
    monkeypatch.setattr(
        aiter_utils, "can_use_mimo_flypa_prefill", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(aiter_utils, "run_mimo_flypa_prefill", reject_flypa)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=3,
        window_size=(-1, -1),
        sinks=None,
    ).view(physical_tokens, 16, 128)

    assert captured["q"].shape == (logical_tokens, 16, 192)
    assert captured["k"].shape == (530, 1, 192)
    assert captured["v"].shape == (530, 1, 128)
    assert captured["cu_q"].tolist() == [0, 257, 274]
    assert captured["cu_k"].tolist() == [0, 513, 530]
    assert captured["max_q"] == 257
    assert captured["max_k"] == 513
    assert captured["min_q"] == 17
    assert torch.equal(captured["slot_ids"], backend.forward_metadata.kv_indices)
    assert torch.all(output[:logical_tokens] == 13.0)
    assert torch.count_nonzero(output[logical_tokens:]) == 0


def test_cached_bf16_swa_uses_flypa_with_translated_pages(monkeypatch):
    logical_tokens = 257
    physical_tokens = 264
    page_size = 64
    num_pages = 9
    k_buf = torch.zeros((num_pages + 1, 1, 24, page_size, 8), dtype=torch.bfloat16)
    v_buf = torch.zeros((num_pages + 1, 1, 8, 128, 8), dtype=torch.bfloat16)
    sub_pool = SimpleNamespace(
        dtype=torch.bfloat16,
        store_dtype=torch.bfloat16,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    parent_pool = SimpleNamespace(
        layers_mapping={0: (0, True)},
        swa_kv_pool=sub_pool,
        full_kv_pool=SimpleNamespace(),
        translate_loc_from_full_to_swa=lambda loc: loc + page_size,
    )
    metadata = SimpleNamespace(
        swa_page_table=torch.arange(513, dtype=torch.int32),
        kv_indices=torch.arange(513, dtype=torch.int32),
        kv_indptr=torch.tensor([0, 513], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, num_pages], dtype=torch.int32),
        paged_kv_indices=torch.arange(num_pages, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        max_q_len=logical_tokens,
        max_kv_len=513,
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=page_size,
        logits_soft_cap=0.0,
        token_to_kv_pool=parent_pool,
        forward_metadata=metadata,
        qo_indptr=torch.tensor([0, logical_tokens], dtype=torch.int32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=128,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
        mimo_original_v_head_dim=128,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[256],
        extend_seq_lens_cpu=[logical_tokens],
        seq_lens_cpu=torch.tensor([513], dtype=torch.int32),
        seq_lens_sum=513,
    )
    q = torch.zeros((physical_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((physical_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((physical_tokens, 1, 128), dtype=torch.bfloat16)
    sinks = torch.zeros(16, dtype=torch.float32)
    captured = {}

    def fake_flypa(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        kwargs["out"].fill_(17.0)
        return kwargs["out"]

    def reject_gather(*args, **kwargs):
        raise AssertionError("cached BF16 SWA FlyPA must consume the 5D cache")

    def reject_ck(*args, **kwargs):
        raise AssertionError("cached BF16 SWA FlyPA must not use CK prefill")

    import aiter.ops.flydsl as flydsl_ops

    monkeypatch.setenv("SGLANG_FLYPA_MIMO_PREFILL", "1")
    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(flydsl_ops, "flydsl_paged_attention_swa_bf16", fake_flypa)
    monkeypatch.setattr(
        aiter_utils, "launch_gather_shuffle_5d_to_linear", reject_gather
    )
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject_ck)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(128, -1),
        sinks=sinks,
    ).view(physical_tokens, 16, 128)

    args = captured["args"]
    assert args[0].shape == (logical_tokens, 16, 192)
    assert args[1].data_ptr() == k_buf.data_ptr()
    assert args[2].data_ptr() == v_buf.data_ptr()
    assert args[5].tolist() == list(range(1, num_pages + 1))
    assert captured["window_left"] == 128
    assert captured["sink_ptr"] is sinks
    assert captured["out"].shape == (logical_tokens, 16, 128)
    assert torch.all(output[:logical_tokens] == 17.0)
    assert torch.count_nonzero(output[logical_tokens:]) == 0


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "short",
        "swa",
        "sink",
        "window",
        "logit_cap",
        "wrong_shape",
        "wrong_total",
    ],
)
def test_fresh_bf16_varlen_asm_contract_guards_fall_back(monkeypatch, guard):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case()
    forward_batch.extend_seq_lens_cpu = [255, 257]
    backend.qo_indptr = torch.tensor([0, 255, 512], dtype=torch.int32)
    window_size = (-1, -1)
    sinks = None
    if guard == "short":
        forward_batch.extend_seq_lens_cpu = [128, 384]
    elif guard == "swa":
        layer.sliding_window_size = 4096
    elif guard == "sink":
        sinks = torch.zeros(16, dtype=torch.float32)
    elif guard == "window":
        window_size = (4096, -1)
    elif guard == "logit_cap":
        backend.logits_soft_cap = 50.0
    elif guard == "wrong_shape":
        layer.tp_q_head_num = 8
    elif guard == "wrong_total":
        forward_batch.extend_seq_lens_cpu = [255, 256]

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(
        aiter_utils,
        "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED",
        guard != "disabled",
    )
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", lambda *args: None)
    assert not aiter_utils.can_use_mimo_fresh_bf16_varlen_asm(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        window_size,
        sinks,
    )


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "nonuniform",
        "short",
        "swa",
        "sink",
        "window",
        "logit_cap",
        "wrong_shape",
    ],
)
def test_fresh_bf16_asm_contract_guards_fall_back(monkeypatch, guard):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case()
    enabled = guard != "disabled"
    window_size = (-1, -1)
    sinks = None
    if guard == "nonuniform":
        forward_batch.extend_seq_lens_cpu = [255, 257]
    elif guard == "short":
        backend, layer, forward_batch, q, k, v = _make_fresh_asm_case(
            sequence_length=128
        )
    elif guard == "swa":
        layer.sliding_window_size = 4096
    elif guard == "sink":
        sinks = torch.zeros(16, dtype=torch.float32)
    elif guard == "window":
        window_size = (4096, -1)
    elif guard == "logit_cap":
        backend.logits_soft_cap = 50.0
    elif guard == "wrong_shape":
        layer.tp_q_head_num = 8

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", enabled)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_fwd", lambda *args: [args[11]])
    assert not aiter_utils.can_use_mimo_fresh_bf16_asm(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        window_size,
        sinks,
    )


def _run_vectorized_prefill_scale_case(
    monkeypatch,
    *,
    direct_paged: bool,
    flydsl_prefill: bool = False,
    flydsl_model_marker: bool = True,
    flypa_env: bool = False,
    swa: bool = False,
):
    captured = {}
    q_descale = torch.tensor([0.125], dtype=torch.float32)
    k_descale = torch.tensor([0.25], dtype=torch.float32)
    v_descale = torch.tensor([0.5], dtype=torch.float32)

    def fake_quantize(q):
        return q.to(fp8_dtype), q_descale

    def fake_batch_prefill(q, k, v, *args, **kwargs):
        captured.update(q=q, k=k, v=v, kwargs=kwargs, selected="ck")
        return torch.zeros(
            (q.shape[0], q.shape[1], 128), dtype=torch.bfloat16, device=q.device
        )

    def fake_flydsl_prefill(q, k, v, *args, **kwargs):
        captured.update(q=q, k=k, v=v, args=args, kwargs=kwargs, selected="flydsl")
        return torch.zeros(
            (q.shape[0], q.shape[1], 128), dtype=torch.bfloat16, device=q.device
        )

    def fake_flypa(**compile_kwargs):
        def run(q, k, v, *args):
            captured.update(
                q=q,
                k=k,
                v=v,
                args=args,
                compile=compile_kwargs,
                selected="flypa",
            )
            return torch.zeros(
                (
                    q.shape[0],
                    compile_kwargs["num_qo_heads"],
                    compile_kwargs["head_dim_v"],
                ),
                dtype=torch.bfloat16,
            )

        return run

    def fake_gather(k_buf, v_buf, slot_ids, *, workspace=None):
        captured["gather_slot_ids"] = slot_ids
        return (
            torch.zeros((1, 1, 192), dtype=torch.uint8),
            torch.zeros((1, 1, 128), dtype=torch.uint8),
        )

    monkeypatch.setenv(aiter_utils.FLYPA_MIMO_PREFILL_ENV, "1" if flypa_env else "0")
    monkeypatch.setenv(
        aiter_utils.FLYDSL_MIMO_PREFILL_ENV, "1" if flydsl_prefill else "0"
    )
    monkeypatch.setattr(aiter_utils, "quantize_query_per_tensor_fp8", fake_quantize)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", fake_batch_prefill)
    monkeypatch.setattr(aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather)
    monkeypatch.setattr(aiter_utils, "flypa", fake_flypa)
    if flydsl_prefill or flypa_env:
        monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
        monkeypatch.setattr(aiter_utils, "is_gfx942", lambda: False)
    if flydsl_prefill:
        monkeypatch.setattr(
            aiter_utils,
            "load_flydsl_mimo_prefill_kernel",
            lambda: SimpleNamespace(run=fake_flydsl_prefill),
        )

    k_buf = torch.zeros((2, 1, 12, 64, 16), dtype=torch.uint8)
    v_buf = torch.zeros((2, 1, 4, 128, 16), dtype=torch.uint8)
    pool = SimpleNamespace(
        dtype=fp8_dtype,
        store_dtype=torch.uint8,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    metadata = SimpleNamespace(
        swa_page_table=(torch.tensor([1], dtype=torch.int32) if swa else None),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        max_q_len=4096 if flydsl_prefill else 1,
        max_kv_len=8192 if flydsl_prefill else 1,
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=fp8_dtype,
        page_size=64,
        logits_soft_cap=0.0,
        token_to_kv_pool=pool,
        forward_metadata=metadata,
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
        k_scale=k_descale,
        v_scale=v_descale,
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=128 if swa else -1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        k_scale=None,
        v_scale=None,
        mimo_original_v_head_dim=(
            128 if flydsl_prefill and flydsl_model_marker else None
        ),
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[1],
        seq_lens_sum=1,
    )
    q = torch.zeros((1, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((1, 192), dtype=torch.bfloat16)
    v = torch.zeros((1, 128), dtype=torch.bfloat16)
    sinks = None if direct_paged else torch.zeros(16, dtype=torch.float32)
    window_size = (128, -1) if swa else (-1, -1)

    out = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=window_size,
        sinks=sinks,
    )

    assert out.shape == (1, 16 * 128)
    assert captured["q"].dtype == fp8_dtype
    assert captured["kwargs"]["q_descale"] is q_descale
    assert captured["kwargs"]["k_descale"] is k_descale
    assert captured["kwargs"]["v_descale"] is v_descale
    if direct_paged:
        assert "gather_slot_ids" not in captured
        assert captured["k"].data_ptr() == k_buf.data_ptr()
        assert captured["v"].data_ptr() == v_buf.data_ptr()
    else:
        expected_slot_ids = metadata.swa_page_table if swa else metadata.kv_indices
        assert torch.equal(captured["gather_slot_ids"], expected_slot_ids)
    return captured


@pytest.mark.parametrize("direct_paged", [True, False])
def test_vectorized_prefill_forwards_independent_qkv_descales(
    monkeypatch, direct_paged
):
    _run_vectorized_prefill_scale_case(monkeypatch, direct_paged=direct_paged)


def test_cached_fp8_swa_gathers_v128_and_forwards_window_sink(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch,
        direct_paged=False,
        swa=True,
    )
    assert captured["gather_slot_ids"].tolist() == [1]
    assert captured["k"].shape == (1, 1, 192)
    assert captured["v"].shape == (1, 1, 128)
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["window_size"] == (128, -1)
    assert captured["kwargs"]["sink_ptr"].shape == (16,)


def test_qualified_long_mimo_prefill_selects_flydsl(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch, direct_paged=True, flydsl_prefill=True
    )
    assert captured["selected"] == "flydsl"
    assert captured["kwargs"]["max_seqlen_q"] == 4096
    assert captured["kwargs"]["max_seqlen_kv"] == 8192


def test_gfx950_fp8_flydsl_is_preferred_over_flypa(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch,
        direct_paged=True,
        flydsl_prefill=True,
        flypa_env=True,
    )
    assert captured["selected"] == "flydsl"
    assert captured["kwargs"]["max_seqlen_q"] == 4096
    assert captured["kwargs"]["max_seqlen_kv"] == 8192


def test_flydsl_env_falls_back_to_ck_without_v128_mimo_marker(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch,
        direct_paged=True,
        flydsl_prefill=True,
        flydsl_model_marker=False,
    )
    assert captured["selected"] == "ck"


def _run_flypa_prefill_case(
    monkeypatch,
    *,
    gfx942: bool,
    gfx950: bool,
    flypa_env: bool,
    kv_dtype,
    with_paged_metadata: bool = True,
    prefix_lens=None,
    flydsl_env: bool = False,
):
    captured = {}

    def fake_flypa(**compile_kwargs):
        def run(q, k, v, *args):
            captured.update(
                q=q,
                k=k,
                v=v,
                args=args,
                compile=compile_kwargs,
                selected="flypa",
            )
            return torch.zeros(
                (
                    q.shape[0],
                    compile_kwargs["num_qo_heads"],
                    compile_kwargs["head_dim_v"],
                ),
                dtype=torch.bfloat16,
            )

        return run

    def fake_batch_prefill(q, k, v, *args, **kwargs):
        captured.update(q=q, k=k, v=v, kwargs=kwargs, selected="ck")
        return torch.zeros(
            (q.shape[0], q.shape[1], 128), dtype=torch.bfloat16, device=q.device
        )

    def fake_gather(k_buf, v_buf, slot_ids, *, workspace=None):
        captured["gather_slot_ids"] = slot_ids
        return (
            torch.zeros((slot_ids.numel(), 1, 192), dtype=kv_dtype),
            torch.zeros((slot_ids.numel(), 1, 128), dtype=kv_dtype),
        )

    monkeypatch.setenv(aiter_utils.FLYPA_MIMO_PREFILL_ENV, "1" if flypa_env else "0")
    monkeypatch.setenv(aiter_utils.FLYDSL_MIMO_PREFILL_ENV, "1" if flydsl_env else "0")
    monkeypatch.setattr(aiter_utils, "is_gfx942", lambda: gfx942)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: gfx950)
    if gfx942 and not gfx950:

        def reject_logical_qkv_views(*args, **kwargs):
            raise AssertionError("gfx942 must not prepare gfx950 ASM Q/K/V views")

        monkeypatch.setattr(
            aiter_utils, "_mimo_logical_qkv_views", reject_logical_qkv_views
        )
    monkeypatch.setattr(aiter_utils, "flypa", fake_flypa)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", fake_batch_prefill)
    monkeypatch.setattr(aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather)
    if flydsl_env:

        def fake_flydsl_prefill(q, k, v, *args, **kwargs):
            captured.update(q=q, k=k, v=v, args=args, kwargs=kwargs, selected="flydsl")
            return torch.zeros(
                (q.shape[0], q.shape[1], 128),
                dtype=torch.bfloat16,
                device=q.device,
            )

        monkeypatch.setattr(
            aiter_utils,
            "load_flydsl_mimo_prefill_kernel",
            lambda: SimpleNamespace(run=fake_flydsl_prefill),
        )
    if kv_dtype == fp8_dtype:
        q_descale = torch.tensor([0.125], dtype=torch.float32)
        monkeypatch.setattr(
            aiter_utils,
            "quantize_query_per_tensor_fp8",
            lambda q: (q.to(fp8_dtype), q_descale),
        )
        pack = 16
        store_dtype = torch.uint8
        k_buf = torch.zeros((2, 1, 12, 64, 16), dtype=torch.uint8)
        v_buf = torch.zeros((2, 1, 4, 128, 16), dtype=torch.uint8)
    else:
        pack = 8
        store_dtype = torch.bfloat16
        k_buf = torch.zeros((2, 1, 24, 64, 8), dtype=torch.bfloat16)
        v_buf = torch.zeros((2, 1, 8, 128, 8), dtype=torch.bfloat16)

    pool = SimpleNamespace(
        dtype=kv_dtype,
        store_dtype=store_dtype,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    metadata = SimpleNamespace(
        swa_page_table=None,
        kv_indices=torch.arange(64, dtype=torch.int32),
        kv_indptr=torch.tensor([0, 64], dtype=torch.int32),
        paged_kv_indptr=(
            torch.tensor([0, 1], dtype=torch.int32) if with_paged_metadata else None
        ),
        paged_kv_indices=(
            torch.tensor([0], dtype=torch.int32) if with_paged_metadata else None
        ),
        paged_kv_last_page_len=(
            torch.tensor([64], dtype=torch.int32) if with_paged_metadata else None
        ),
        max_q_len=64,
        max_kv_len=64,
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        page_size=64,
        logits_soft_cap=0.0,
        token_to_kv_pool=pool,
        forward_metadata=metadata,
        qo_indptr=torch.tensor([0, 64], dtype=torch.int32),
        k_scale=torch.tensor([0.25], dtype=torch.float32),
        v_scale=torch.tensor([0.5], dtype=torch.float32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        k_scale=None,
        v_scale=None,
        mimo_original_v_head_dim=128,
    )
    prefix_lens = [64] if prefix_lens is None else list(prefix_lens)
    extend_len = 64
    seq_len = prefix_lens[0] + extend_len
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens_cpu=[extend_len],
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_sum=seq_len,
    )
    q = torch.zeros((64, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((64, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((64, 1, 128), dtype=torch.bfloat16)
    out = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(-1, -1),
        sinks=None,
    )
    assert out.shape == (64, 16 * 128)
    return captured, k_buf, v_buf, pack


def test_gfx942_bf16_flypa_prefill_skips_gather(monkeypatch):
    captured, k_buf, v_buf, pack = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
    )
    assert captured["selected"] == "flypa"
    assert "gather_slot_ids" not in captured
    assert captured["k"].data_ptr() == k_buf.data_ptr()
    assert captured["v"].data_ptr() == v_buf.data_ptr()
    assert captured["q"].shape == (64, 16, 192)
    assert captured["compile"]["head_dim_v"] == 128
    assert pack == 8


def test_gfx942_fp8_flypa_prefill_skips_ck(monkeypatch):
    captured, k_buf, v_buf, pack = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=True,
        kv_dtype=fp8_dtype,
    )
    assert captured["selected"] == "flypa"
    assert captured["k"].data_ptr() == k_buf.data_ptr()
    assert pack == 16
    assert captured["q"].dtype == fp8_dtype


def test_gfx950_fp8_uses_aiter_flydsl_at_small_sizes(monkeypatch):
    captured, k_buf, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=True,
        flypa_env=True,
        kv_dtype=fp8_dtype,
        flydsl_env=True,
    )
    assert captured["selected"] == "flydsl"
    assert captured["k"].data_ptr() == k_buf.data_ptr()
    assert "gather_slot_ids" not in captured


def test_gfx950_bf16_prefill_uses_direct_ck_page64(monkeypatch):
    captured, k_buf, v_buf, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=True,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
    )
    assert captured["selected"] == "ck"
    assert "gather_slot_ids" not in captured
    assert captured["k"].data_ptr() == k_buf.data_ptr()
    assert captured["v"].data_ptr() == v_buf.data_ptr()
    assert captured["q"].dtype == torch.bfloat16
    assert captured["kwargs"]["q_descale"] is None
    assert captured["kwargs"]["k_descale"] is None
    assert captured["kwargs"]["v_descale"] is None
    assert captured["kwargs"]["kv_last_page_lens"].tolist() == [64]


def test_gfx950_flypa_env_keeps_fresh_asm_shortcut(monkeypatch):
    selected = {}

    def fake_asm(q, k, v, layer, forward_batch):
        selected["path"] = "asm"
        return torch.zeros((q.shape[0], 16 * 128), dtype=torch.bfloat16)

    monkeypatch.setattr(
        aiter_utils, "can_use_mimo_fresh_bf16_asm", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(aiter_utils, "mimo_fresh_bf16_asm", fake_asm)
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=True,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
        prefix_lens=[0],
    )
    assert selected["path"] == "asm"
    assert captured.get("selected") != "flypa"


def test_gfx942_flypa_env_uses_paged_path_on_fresh_chunk(monkeypatch):
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
        prefix_lens=[0],
    )
    assert captured["selected"] == "flypa"


def test_gfx950_cached_asm_is_preferred_over_flypa(monkeypatch):
    selected = {}

    def fake_chunk_asm(*args, **kwargs):
        selected["path"] = "chunk_asm"
        q = args[1]
        return torch.zeros((q.shape[0], 16 * 128), dtype=torch.bfloat16)

    monkeypatch.setattr(
        aiter_utils, "can_use_mimo_chunk_bf16_varlen_asm", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(aiter_utils, "mimo_chunk_bf16_varlen_asm", fake_chunk_asm)
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=True,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
        prefix_lens=[256],
    )
    assert selected["path"] == "chunk_asm"
    assert captured.get("selected") != "flypa"


def test_flypa_env_off_gfx942_bf16_uses_direct_ck_page64(monkeypatch):
    captured, k_buf, v_buf, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=False,
        kv_dtype=torch.bfloat16,
    )
    assert captured["selected"] == "ck"
    assert "gather_slot_ids" not in captured
    assert captured["k"].data_ptr() == k_buf.data_ptr()
    assert captured["v"].data_ptr() == v_buf.data_ptr()


def test_flypa_ignored_without_gfx942_or_gfx950(monkeypatch):
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=False,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
    )
    assert captured["selected"] == "ck"
    assert "gather_slot_ids" in captured


def test_gfx942_bf16_flypa_requires_paged_kv_metadata(monkeypatch):
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
        with_paged_metadata=False,
    )
    assert captured["selected"] == "ck"
    assert "gather_slot_ids" in captured


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("q_row_padding", [0, 320])
@pytest.mark.parametrize(
    "page_size,query_length,kv_length",
    [
        (64, 1, 1),
        (64, 1, 257),
        (32, 16, 23),
        (32, 65, 141),
        (64, 16, 23),
        (64, 32, 70),
        (64, 129, 257),
        (128, 65, 141),
        (128, 257, 513),
    ],
)
def test_bf16_flypa_matches_torch_with_causal_partial_last_page(
    page_size, query_length, kv_length, q_row_padding
):
    arch = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.split(":", 1)[0]
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("BF16 FlyPA is supported on gfx942/gfx950")

    torch.manual_seed(20260819)
    device = torch.device("cuda")
    num_pages = math.ceil(kv_length / page_size)
    num_qo_heads = 16
    num_kv_heads = 1
    head_dim_qk = 192
    head_dim_v = 128
    pack = 8

    query = (
        torch.randn(
            query_length,
            num_qo_heads,
            head_dim_qk,
            dtype=torch.float32,
            device=device,
        )
        * 0.2
    ).to(torch.bfloat16)
    if q_row_padding:
        query_storage = torch.full(
            (query_length, num_qo_heads * head_dim_qk + q_row_padding),
            float("nan"), dtype=torch.bfloat16, device=device,
        )
        query_view = query_storage[:, :num_qo_heads * head_dim_qk].view_as(query)
        query_view.copy_(query)
        query = query_view
    key_valid = (
        torch.randn(
            kv_length,
            num_kv_heads,
            head_dim_qk,
            dtype=torch.float32,
            device=device,
        )
        * 0.2
    ).to(torch.bfloat16)
    value_valid = (
        torch.randn(
            kv_length,
            num_kv_heads,
            head_dim_v,
            dtype=torch.float32,
            device=device,
        )
        * 0.2
    ).to(torch.bfloat16)

    # Fill the invalid tail with a large sentinel. A missing last-page or causal
    # mask makes the comparison fail decisively instead of passing by chance.
    key_tokens = torch.full(
        (num_pages * page_size, num_kv_heads, head_dim_qk),
        64.0,
        dtype=torch.bfloat16,
        device=device,
    )
    value_tokens = torch.full(
        (num_pages * page_size, num_kv_heads, head_dim_v),
        64.0,
        dtype=torch.bfloat16,
        device=device,
    )
    key_tokens[:kv_length] = key_valid
    value_tokens[:kv_length] = value_valid
    key_cache = (
        key_tokens.view(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim_qk // pack,
            pack,
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    value_cache = (
        value_tokens.view(
            num_pages,
            page_size // pack,
            pack,
            num_kv_heads,
            head_dim_v,
        )
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )

    cu_seqlens_q = torch.tensor([0, query_length], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    # The prefetch pipeline looks ahead past the final logical page.
    kv_page_indices = torch.zeros(num_pages + 8, dtype=torch.int32, device=device)
    kv_page_indices[:num_pages] = torch.arange(
        num_pages, dtype=torch.int32, device=device
    )
    kv_last_page_lens = torch.tensor(
        [kv_length - (num_pages - 1) * page_size], dtype=torch.int32, device=device
    )
    unit_scale = torch.ones((), dtype=torch.float32, device=device)

    output = aiter_utils.flypa(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        page_size=page_size,
        is_causal=True,
        quant_query_mode="per-tensor",
    )(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        None,
        kv_indptr,
        kv_page_indices,
        query_length,
        kv_length,
        True,
        unit_scale,
        unit_scale,
        unit_scale,
        kv_last_page_lens,
    )

    key_ref = key_valid.float().expand(-1, num_qo_heads, -1)
    value_ref = value_valid.float().expand(-1, num_qo_heads, -1)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key_ref) / math.sqrt(
        head_dim_qk
    )
    rows = torch.arange(query_length, device=device)[:, None]
    columns = torch.arange(kv_length, device=device)[None, :]
    causal_mask = columns <= (kv_length - query_length + rows)
    scores.masked_fill_(~causal_mask[None, :, :], float("-inf"))
    reference = torch.einsum(
        "hqk,khd->qhd", torch.softmax(scores, dim=-1), value_ref
    ).to(torch.bfloat16)

    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    if q_row_padding:
        contiguous_output = aiter_utils.flypa(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            page_size=page_size,
            is_causal=True,
            quant_query_mode="per-tensor",
        )(
            query.contiguous(), key_cache, value_cache, cu_seqlens_q, None,
            kv_indptr, kv_page_indices, query_length, kv_length, True,
            unit_scale, unit_scale, unit_scale, kv_last_page_lens,
        )
        assert torch.equal(output, contiguous_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_tbo_ragged_cached_bf16_asm_matches_flypa(monkeypatch):
    if not aiter_utils.is_gfx950():
        pytest.skip("MiMo BF16 grouped ASM requires gfx950")

    torch.manual_seed(20260831)
    device = torch.device("cuda")
    query_lengths = [257, 17]
    prefix_lengths = [256, 0]
    kv_lengths = [513, 17]
    logical_tokens = sum(query_lengths)
    physical_tokens = 280
    num_pages = 10
    pack = 8

    query = torch.zeros(
        physical_tokens,
        16,
        192,
        dtype=torch.bfloat16,
        device=device,
    )
    query[:logical_tokens] = (
        torch.randn(logical_tokens, 16, 192, device=device) * 0.02
    ).to(torch.bfloat16)
    key_tokens = (torch.randn(num_pages * 64, 1, 192, device=device) * 0.02).to(
        torch.bfloat16
    )
    value_tokens = (torch.randn(num_pages * 64, 1, 128, device=device) * 0.02).to(
        torch.bfloat16
    )
    key_cache = (
        key_tokens.view(num_pages, 64, 1, 24, pack).permute(0, 2, 3, 1, 4).contiguous()
    )
    value_cache = (
        value_tokens.view(num_pages, 8, pack, 1, 128)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )

    slot_ids = torch.cat(
        (
            torch.arange(513, dtype=torch.int32, device=device),
            torch.arange(576, 593, dtype=torch.int32, device=device),
        )
    )
    qo_indptr = torch.tensor([0, 257, 274], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 513, 530], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, 9, 10], dtype=torch.int32, device=device)
    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page_lens = torch.tensor([1, 17], dtype=torch.int32, device=device)

    pool = SimpleNamespace(
        dtype=torch.bfloat16,
        store_dtype=torch.bfloat16,
        start_layer=0,
        k_buffer=[key_cache],
        v_buffer=[value_cache],
    )
    metadata = SimpleNamespace(
        swa_page_table=None,
        kv_indices=slot_ids,
        kv_indptr=kv_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=page_indices,
        paged_kv_last_page_len=last_page_lens,
        max_q_len=max(query_lengths),
        max_kv_len=max(kv_lengths),
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
        logits_soft_cap=0.0,
        token_to_kv_pool=pool,
        forward_metadata=metadata,
        qo_indptr=qo_indptr,
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
        mimo_original_v_head_dim=128,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=prefix_lengths,
        extend_seq_lens_cpu=query_lengths,
        seq_lens_cpu=torch.tensor(kv_lengths, dtype=torch.int32),
        seq_lens_sum=sum(kv_lengths),
    )
    current_key = torch.zeros(
        physical_tokens, 1, 192, dtype=torch.bfloat16, device=device
    )
    current_value = torch.zeros(
        physical_tokens, 1, 128, dtype=torch.bfloat16, device=device
    )

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED", True)

    actual = aiter_utils.forward_extend_vectorized_5d(
        backend,
        query.view(physical_tokens, -1),
        current_key,
        current_value,
        layer,
        forward_batch,
        bs0=3,
        window_size=(-1, -1),
        sinks=None,
    ).view(physical_tokens, 16, 128)

    unit_scale = torch.ones((), dtype=torch.float32, device=device)
    expected = aiter_utils.flypa(
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim_qk=192,
        head_dim_v=128,
        page_size=64,
        is_causal=True,
        quant_query_mode="per-tensor",
    )(
        query[:logical_tokens],
        key_cache,
        value_cache,
        qo_indptr,
        None,
        paged_kv_indptr,
        page_indices,
        max(query_lengths),
        max(kv_lengths),
        True,
        unit_scale,
        unit_scale,
        unit_scale,
        last_page_lens,
    )

    torch.testing.assert_close(actual[:logical_tokens], expected, rtol=2e-2, atol=2e-2)
    assert torch.count_nonzero(actual[logical_tokens:]) == 0


def _mimo_paged_metadata_kwargs(**overrides):
    kwargs = dict(
        kv_cache_is_vectorized_5d=True,
        page_size=64,
        kv_cache_dtype=torch.bfloat16,
        q_dtype=torch.bfloat16,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim=192,
    )
    kwargs.update(overrides)
    return kwargs


def test_paged_kv_metadata_enabled_for_bf16_and_fp8_shuffle_5d():
    assert aiter_utils.can_build_mimo_paged_kv_metadata(**_mimo_paged_metadata_kwargs())
    assert aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(kv_cache_dtype=fp8_dtype)
    )
    assert aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(num_kv_heads=2)
    )


def test_paged_kv_metadata_rejected_outside_mimo_contract():
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(kv_cache_is_vectorized_5d=False)
    )
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(page_size=16)
    )
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(kv_cache_dtype=torch.float16)
    )
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(num_kv_heads=3)
    )
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(num_qo_heads=8)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_query_fp8_quantization_returns_its_own_non_unit_descale():
    q = torch.linspace(
        -96.0,
        96.0,
        2 * 16 * 192,
        dtype=torch.bfloat16,
        device="cuda",
    ).view(2, 16 * 192)

    q_fp8, scale = aiter_utils.quantize_query_per_tensor_fp8(q)
    q_dequant = q_fp8.float() * scale

    assert q_fp8.dtype == fp8_dtype
    assert q_fp8.shape == q.shape
    assert scale.shape == (1,)
    assert scale.dtype == torch.float32
    assert scale.item() != pytest.approx(1.0)
    expected_scale = q.abs().max().float() / torch.finfo(fp8_dtype).max
    torch.testing.assert_close(scale, expected_scale.view(1), rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(q_dequant, q.float(), rtol=0.125, atol=0.5)


def test_query_fp8_quantization_rejects_prequantized_input():
    q = torch.zeros((1, 16 * 192), dtype=fp8_dtype)
    with pytest.raises(ValueError, match="non-FP8"):
        aiter_utils.quantize_query_per_tensor_fp8(q)


@pytest.mark.parametrize(
    "configured_query_length,decode",
    [(1, True), (4, True), (8, True), (4, False), (8, False)],
)
def test_flydsl_full_attention_uses_correct_context_lengths(
    monkeypatch, configured_query_length, decode
):
    query_length = 1 if decode else configured_query_length
    batch_size = 2
    partitions = 8
    equivalent_group = query_length * 16
    scalar_numel = batch_size * partitions * configured_query_length * 16
    captured = {}

    def fake_pa_decode_tile(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(3.0)

    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    if decode:
        block_tables = torch.cat((block_tables, torch.zeros_like(block_tables)))
    backend = SimpleNamespace(
        _use_flydsl_pa_decode=True,
        _flydsl_pa_decode_tile=fake_pa_decode_tile,
        _flydsl_pa_decode_num_partitions=partitions,
        _flydsl_pa_decode_pmax=torch.empty(scalar_numel, dtype=torch.float32),
        _flydsl_pa_decode_psum=torch.empty(scalar_numel, dtype=torch.float32),
        _flydsl_pa_decode_pout=torch.empty(scalar_numel * 128, dtype=torch.bfloat16),
        _flydsl_pa_decode_context_lengths=torch.empty(batch_size, dtype=torch.int32),
        _flydsl_pa_decode_workspace_max_bs=batch_size,
        _flydsl_pa_decode_query_length=configured_query_length,
        forward_metadata=SimpleNamespace(
            max_q_len=query_length,
            kv_indices=block_tables,
        ),
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
    )
    layer = SimpleNamespace(
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        scaling=192**-0.5,
        logit_cap=0.0,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor([100, 120], dtype=torch.int64),
    )
    q = torch.zeros((batch_size * query_length, 16 * 192), dtype=torch.bfloat16)
    output = torch.empty((batch_size * query_length, 16 * 128), dtype=torch.bfloat16)
    k_cache = torch.zeros((4, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_cache = torch.zeros((4, 1, 8, 128, 8), dtype=torch.bfloat16)

    def unexpected_gluon(**kwargs):
        pytest.fail("Explicit FlyDSL full decode must not enter Gluon")

    monkeypatch.setattr(aiter_utils, "pa_decode_gluon", unexpected_gluon)
    forward = (
        aiter_utils.forward_decode_vectorized_5d
        if decode
        else aiter_utils.forward_target_verify_flydsl_5d
    )
    forward(
        backend,
        q,
        layer,
        forward_batch,
        k_cache,
        v_cache,
        output,
        sinks=None,
    )

    assert captured["key_cache"] is k_cache
    assert captured["value_cache"] is v_cache
    assert captured["key_scale"] is None
    assert captured["value_scale"] is None
    assert captured["block_tables"].shape[0] == batch_size
    assert captured["context_lengths"].tolist() == [
        100 + (0 if decode else query_length),
        120 + (0 if decode else query_length),
    ]
    assert captured["query"].shape == (batch_size * query_length, 16, 192)
    assert captured["output"].shape == (batch_size * query_length, 16, 128)
    assert captured["pout"].shape == (
        batch_size,
        1,
        partitions,
        query_length * 16,
        128,
    )
    assert torch.all(output == 3.0)


@pytest.mark.parametrize(
    "is_draft,qk_dim,window_size,env_name",
    [
        (True, 128, 1024, "SGLANG_AITER_DFLASH_SWA_IMPL"),
        (False, 192, 128, "SGLANG_AITER_TARGET_VERIFY_SWA_IMPL"),
    ],
)
def test_flydsl_swa_prepares_graph_stable_metadata(
    monkeypatch, is_draft, qk_dim, window_size, env_name
):
    monkeypatch.setenv(env_name, "flydsl")
    backend = SimpleNamespace(
        is_draft_worker=is_draft,
        use_sliding_window_kv_pool=not is_draft,
        page_size=64,
        kv_cache_dtype=torch.bfloat16,
        num_head=16,
        num_kv_head=1,
        head_dim=qk_dim,
        v_head_dim=128,
        device=torch.device("cpu"),
    )

    if is_draft:
        aiter_utils.prepare_flydsl_dflash_swa(
            backend, max_batch_size=4, query_length=8, draft_window_size=window_size
        )
    else:
        aiter_utils.prepare_flydsl_target_verify_swa(
            backend, max_batch_size=4, query_length=8, window_size=window_size
        )

    expected_max_kv_len = window_size + 63 + 8
    assert backend._flydsl_dflash_swa_max_kv_len == expected_max_kv_len
    assert backend._flydsl_dflash_swa_max_pages == math.ceil(expected_max_kv_len / 64)
    assert backend._flydsl_dflash_swa_context_lens.shape == (4,)
    assert backend._flydsl_dflash_swa_kv_indptr.shape == (5,)


@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("query_length", [1, 8])
@pytest.mark.parametrize("context_length", [64, 65, 129, 65536])
def test_flydsl_target_swa_compacts_context_table(
    monkeypatch, num_kv_heads, query_length, context_length
):
    monkeypatch.delenv("SGLANG_AITER_VEC5D_SWA_VERIFY_QLEN1", raising=False)
    monkeypatch.setenv("SGLANG_AITER_TARGET_VERIFY_SWA_IMPL", "flydsl")
    batch_size = 1
    backend = SimpleNamespace(
        is_draft_worker=False,
        use_sliding_window_kv_pool=True,
        page_size=64,
        kv_cache_dtype=torch.bfloat16,
        num_head=16,
        num_kv_head=1,
        head_dim=192,
        v_head_dim=128,
        device=torch.device("cpu"),
        logits_soft_cap=0.0,
        _flydsl_target_verify_swa_enabled=True,
        _flydsl_dflash_swa_enabled=False,
    )
    num_pages = math.ceil(context_length / 64)
    page_table = torch.arange(num_pages, dtype=torch.int32).view(1, -1)
    backend.forward_metadata = SimpleNamespace(
        max_q_len=query_length,
        qo_indptr=torch.tensor([0, query_length], dtype=torch.int32),
        kv_indices=page_table,
        swa_page_table=page_table,
    )
    q = torch.zeros((query_length, 16 * 192), dtype=torch.bfloat16)
    output = torch.empty((query_length, 16 * 128), dtype=torch.bfloat16)
    k_cache = torch.empty((num_pages, num_kv_heads, 24, 64, 8), dtype=torch.bfloat16)
    v_cache = torch.empty((num_pages, num_kv_heads, 8, 128, 8), dtype=torch.bfloat16)
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor(
            [context_length if query_length == 1 else context_length - query_length],
            dtype=torch.int64,
        ),
    )
    layer = SimpleNamespace(
        sliding_window_size=128,
        tp_q_head_num=16,
        tp_k_head_num=num_kv_heads,
        qk_head_dim=192,
        v_head_dim=128,
        scaling=192**-0.5,
        logit_cap=0.0,
    )
    sinks = torch.zeros(16, dtype=torch.float32)
    captured = {"flatten_calls": 0, "flydsl_calls": 0}

    class FakeFlattenKernel:
        def __getitem__(self, _grid):
            def launch(table, indptr, indices, starts, _stride, **_kwargs):
                captured["flatten_calls"] += 1
                start = int(indptr[0])
                end = int(indptr[1])
                source_start = int(starts[0])
                indices[start:end].copy_(
                    table[0, source_start : source_start + end - start]
                )

            return launch

    def fake_flydsl(*args, **kwargs):
        captured["flydsl_calls"] += 1
        captured["args"] = args
        captured.update(kwargs)
        kwargs["out"].fill_(5)

    import aiter.ops.flydsl as flydsl_ops

    monkeypatch.setattr(
        aiter_utils, "_flatten_page_table_to_csr_kernel", FakeFlattenKernel()
    )
    monkeypatch.setattr(flydsl_ops, "flydsl_paged_attention_swa_bf16", fake_flydsl)
    monkeypatch.setattr(aiter_utils, "pa_decode_gluon", None)
    monkeypatch.setattr(aiter_utils, "get_recommended_splits", None)

    aiter_utils.prepare_flydsl_target_verify_swa(
        backend, batch_size, query_length, window_size=128
    )
    aiter_utils.prepare_flydsl_swa_forward_metadata(
        backend,
        batch_size,
        forward_batch.seq_lens,
        context_includes_query=query_length == 1,
    )

    forward = (
        aiter_utils.forward_decode_vectorized_5d
        if query_length == 1
        else aiter_utils.forward_target_verify_vectorized_5d
    )
    forward(backend, q, layer, forward_batch, k_cache, v_cache, output, sinks)
    forward(backend, q, layer, forward_batch, k_cache, v_cache, output, sinks)

    args = captured["args"]
    assert captured["flatten_calls"] == 1
    assert captured["flydsl_calls"] == 2
    expected_pages = min(num_pages, 3 if query_length == 1 else 4)
    assert args[4].tolist() == [0, expected_pages]
    assert args[5][:expected_pages].tolist() == list(
        range(num_pages - expected_pages, num_pages)
    )
    assert captured["kv_last_page_lens"].tolist() == [(context_length - 1) % 64 + 1]
    assert args[7] == 128 + 63 + query_length
    assert captured["window_left"] == (127 if num_kv_heads == 2 else 128)
    assert torch.all(output == 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "qk_dim,window_size,total_kv,num_kv_heads",
    [
        (128, 1024, 2048, 1),
        (192, 128, 4096, 1),
        (192, 128, 4096, 2),
        (128, 1024, 2048, 2),
    ],
    ids=["draft-d128", "target-d192", "flash-tp4-target", "flash-tp4-draft"],
)
def test_flydsl_swa_qlen8_matches_torch_and_replays_graph(
    monkeypatch, qk_dim, window_size, total_kv, num_kv_heads
):
    if not (aiter_utils.is_gfx950() or aiter_utils.is_gfx942()):
        pytest.skip("DFlash FlyPA SWA requires gfx942 or gfx950")

    torch.manual_seed(20260918 + qk_dim)
    device = torch.device("cuda")
    query_length, page_size = 8, 64
    num_q_heads, value_dim = 16, 128
    num_pages = total_kv // page_size
    prefix_length = total_kv - query_length

    q = (torch.randn(query_length, num_q_heads, qk_dim, device=device) * 0.2).to(
        torch.bfloat16
    )
    k_linear = (torch.randn(total_kv, num_kv_heads, qk_dim, device=device) * 0.2).to(
        torch.bfloat16
    )
    v_linear = (torch.randn(total_kv, num_kv_heads, value_dim, device=device) * 0.2).to(
        torch.bfloat16
    )
    k_cache = (
        k_linear.view(num_pages, page_size, num_kv_heads, qk_dim // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_linear.view(num_pages, page_size // 8, 8, num_kv_heads, value_dim)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, -1)
    backend = SimpleNamespace(
        is_draft_worker=qk_dim == 128,
        page_size=page_size,
        kv_cache_dtype=torch.bfloat16,
        num_head=num_q_heads,
        num_kv_head=num_kv_heads if qk_dim == 128 else 1,
        head_dim=qk_dim,
        v_head_dim=value_dim,
        device=device,
        logits_soft_cap=0.0,
        _flydsl_dflash_swa_enabled=qk_dim == 128,
        _flydsl_target_verify_swa_enabled=qk_dim == 192,
    )
    if qk_dim == 128 and num_kv_heads == 2:
        monkeypatch.setenv("SGLANG_AITER_DFLASH_SWA_IMPL", "flydsl")
        aiter_utils.prepare_flydsl_dflash_swa(backend, 1, query_length, window_size)
    else:
        aiter_utils._prepare_flydsl_swa(
            backend,
            max_batch_size=1,
            query_length=query_length,
            window_size=window_size,
            qk_head_dim=qk_dim,
            env_name="gpu-test",
        )
    backend.forward_metadata = SimpleNamespace(
        max_q_len=query_length,
        qo_indptr=torch.tensor([0, query_length], dtype=torch.int32, device=device),
        kv_indices=page_table,
        swa_page_table=page_table if qk_dim == 192 else None,
    )
    forward_batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([prefix_length], dtype=torch.int64, device=device),
    )
    aiter_utils.prepare_flydsl_swa_forward_metadata(
        backend, 1, forward_batch.seq_lens
    )
    layer = SimpleNamespace(
        sliding_window_size=window_size,
        tp_q_head_num=num_q_heads,
        tp_k_head_num=num_kv_heads,
        qk_head_dim=qk_dim,
        v_head_dim=value_dim,
        scaling=qk_dim**-0.5,
        logit_cap=0.0,
    )
    sinks = torch.linspace(-0.4, 0.3, num_q_heads, dtype=torch.float32, device=device)
    output = torch.empty(
        query_length,
        num_q_heads,
        value_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    def run():
        aiter_utils.forward_target_verify_vectorized_5d(
            backend,
            q.view(query_length, -1),
            layer,
            forward_batch,
            k_cache,
            v_cache,
            output.view(query_length, -1),
            sinks,
        )
        return output

    actual = run()
    torch.cuda.synchronize()

    def reference(prefix_len):
        context_len = prefix_len + query_length
        context_pages = math.ceil(context_len / page_size)
        first_page = max(
            0, context_pages - backend._flydsl_dflash_swa_max_pages
        )
        first_token = first_page * page_size
        k_tail = (
            k_linear[first_token:context_len]
            .float()
            .repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        )
        v_tail = (
            v_linear[first_token:context_len]
            .float()
            .repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        )
        scores = torch.einsum("qhd,khd->hqk", q.float(), k_tail) * layer.scaling
        q_positions = torch.arange(
            prefix_len, context_len, device=device
        )
        k_positions = torch.arange(first_token, context_len, device=device)
        visible = (k_positions[None, :] <= q_positions[:, None]) & (
            k_positions[None, :]
            >= (q_positions - window_size + int(num_kv_heads == 2))[:, None]
        )
        scores.masked_fill_(~visible.unsqueeze(0), float("-inf"))
        sink_scores = sinks[:, None, None].expand(num_q_heads, query_length, 1)
        probabilities = torch.softmax(
            torch.cat((scores, sink_scores), dim=-1), dim=-1
        )[..., :-1]
        return torch.einsum("hqk,khd->qhd", probabilities, v_tail)

    expected = reference(prefix_length)
    torch.testing.assert_close(actual.float(), expected, rtol=0.005, atol=0.005)
    eager = actual.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, eager, rtol=0, atol=0)

    replay_prefix = prefix_length - page_size
    forward_batch.seq_lens.fill_(replay_prefix)
    aiter_utils.prepare_flydsl_swa_forward_metadata(
        backend, 1, forward_batch.seq_lens
    )
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        graph_output.float(), reference(replay_prefix), rtol=0.005, atol=0.005
    )


@pytest.mark.parametrize("sliding_window", [-1, 128], ids=["full", "swa"])
def test_gluon_decode_uses_native_v128_workspace(monkeypatch, sliding_window):
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(2.0)

    monkeypatch.setattr(aiter_utils, "pa_decode_gluon", fake_decode)
    monkeypatch.setattr(aiter_utils, "get_recommended_splits", lambda *_: 2)

    batch_size = 2
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            kv_indices=torch.tensor([[0], [1]], dtype=torch.int32),
            swa_page_table=(
                torch.tensor([[1], [0]], dtype=torch.int32)
                if sliding_window > 0
                else None
            ),
        ),
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
    )
    layer = SimpleNamespace(
        sliding_window_size=sliding_window,
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        scaling=192**-0.5,
        k_scale=None,
        v_scale=None,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor([64, 96], dtype=torch.int32),
    )
    q = torch.zeros((batch_size, 16 * 192), dtype=torch.bfloat16)
    output = torch.empty((batch_size, 16 * 128), dtype=torch.bfloat16)
    k_cache = torch.zeros((2, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_cache = torch.zeros((2, 1, 8, 128, 8), dtype=torch.bfloat16)

    aiter_utils.forward_decode_vectorized_5d(
        backend,
        q,
        layer,
        forward_batch,
        k_cache,
        v_cache,
        output,
        sinks=None,
    )

    assert captured["output"].shape == (batch_size, 16, 128)
    assert captured["query"].shape == (batch_size, 16, 192)
    assert captured["temporary_output"].shape[-1] == 128
    assert captured["sliding_window"] == max(sliding_window, 0)
    assert torch.all(output == 2.0)


@pytest.mark.parametrize("sliding_window", [-1, 128], ids=["full", "swa"])
def test_gluon_target_verify_accepts_native_v128(monkeypatch, sliding_window):
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(4.0)

    monkeypatch.setattr(aiter_utils, "pa_decode_gluon", fake_decode)
    monkeypatch.setattr(aiter_utils, "get_recommended_splits", lambda *_: 2)

    batch_size = 2
    query_length = 4
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            max_q_len=query_length,
            kv_indices=torch.tensor([[0], [1]], dtype=torch.int32),
            swa_page_table=(
                torch.tensor([[1], [0]], dtype=torch.int32)
                if sliding_window > 0
                else None
            ),
        ),
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
    )
    layer = SimpleNamespace(
        sliding_window_size=sliding_window,
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        scaling=192**-0.5,
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor([64, 96], dtype=torch.int32),
    )
    q = torch.zeros((batch_size * query_length, 16 * 192), dtype=torch.bfloat16)
    output = torch.empty((batch_size * query_length, 16 * 128), dtype=torch.bfloat16)
    k_cache = torch.zeros((2, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_cache = torch.zeros((2, 1, 8, 128, 8), dtype=torch.bfloat16)

    aiter_utils.forward_target_verify_vectorized_5d(
        backend,
        q,
        layer,
        forward_batch,
        k_cache,
        v_cache,
        output,
        sinks=None,
    )

    assert captured["output"].shape == (
        batch_size * query_length,
        16,
        128,
    )
    assert captured["query"].shape == (
        batch_size * query_length,
        16,
        192,
    )
    assert captured["temporary_output"].shape[-1] == 128
    assert captured["context_lengths"].tolist() == [68, 100]
    assert captured["sliding_window"] == max(sliding_window, 0)
    assert torch.all(output == 4.0)


@pytest.mark.parametrize("query_length", [4, 8])
def test_gluon_swa_target_verify_splits_queries_with_stable_buffers(
    monkeypatch, query_length
):
    batch_size = 2
    q_heads, kv_heads, qk_dim, v_dim = 16, 1, 192, 128
    page_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(5.0)

    monkeypatch.setattr(aiter_utils, "pa_decode_gluon", fake_decode)
    monkeypatch.setenv("SGLANG_AITER_VEC5D_SWA_VERIFY_QLEN1", "1")

    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            max_q_len=query_length,
            kv_indices=torch.zeros_like(page_table),
            swa_page_table=page_table,
        ),
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
        k_scale=torch.ones(1, dtype=torch.float32),
        v_scale=torch.ones(1, dtype=torch.float32),
    )
    layer = SimpleNamespace(
        sliding_window_size=128,
        tp_q_head_num=q_heads,
        tp_k_head_num=kv_heads,
        qk_head_dim=qk_dim,
        v_head_dim=v_dim,
        scaling=qk_dim**-0.5,
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor([64, 96], dtype=torch.int32),
    )
    q = torch.zeros((batch_size * query_length, q_heads * qk_dim), dtype=torch.bfloat16)
    output = torch.empty(
        (batch_size * query_length, q_heads * v_dim), dtype=torch.bfloat16
    )
    k_cache = torch.zeros((4, kv_heads, 24, 64, 8), dtype=torch.bfloat16)
    v_cache = torch.zeros((4, kv_heads, 8, v_dim, 8), dtype=torch.bfloat16)

    def run():
        aiter_utils.forward_target_verify_vectorized_5d(
            backend,
            q,
            layer,
            forward_batch,
            k_cache,
            v_cache,
            output,
            sinks=torch.zeros(q_heads, dtype=torch.float32),
        )

    run()
    assert captured["query_length"] == 1
    assert captured["context_lengths"].tolist() == [
        *(64 + offset for offset in range(1, query_length + 1)),
        *(96 + offset for offset in range(1, query_length + 1)),
    ]
    assert captured["block_tables"].tolist() == (
        [[1, 2]] * query_length + [[3, 4]] * query_length
    )
    assert captured["temporary_output"].shape == (
        batch_size * query_length,
        kv_heads,
        1,
        q_heads // kv_heads,
        v_dim,
    )
    assert torch.all(output == 5.0)

    pointers = (
        captured["context_lengths"].data_ptr(),
        captured["block_tables"].data_ptr(),
        captured["temporary_output"].data_ptr(),
    )
    forward_batch.seq_lens.copy_(torch.tensor([32, 128], dtype=torch.int32))
    page_table.copy_(torch.tensor([[5, 6], [7, 8]], dtype=torch.int32))
    run()
    assert pointers == (
        captured["context_lengths"].data_ptr(),
        captured["block_tables"].data_ptr(),
        captured["temporary_output"].data_ptr(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "dtype,vector_width",
    [(torch.bfloat16, 8), (torch.uint8, 16)],
    ids=["bf16", "fp8-storage"],
)
def test_shuffle_5d_asymmetric_writer_gather_round_trip(dtype, vector_width):
    device = "cuda"
    num_tokens = 5
    num_heads = 1
    page_size = 64
    slots = torch.tensor([0, 65, 3, 127, 64], dtype=torch.int64, device=device)
    key = (
        torch.arange(
            num_tokens * num_heads * 192,
            dtype=torch.int32,
            device=device,
        )
        .remainder(251)
        .to(dtype)
        .view(num_tokens, num_heads, 192)
    )
    value = (
        torch.arange(
            num_tokens * num_heads * 128,
            dtype=torch.int32,
            device=device,
        )
        .add(17)
        .remainder(251)
        .to(dtype)
        .view(num_tokens, num_heads, 128)
    )
    key_cache = torch.zeros(
        (2, num_heads, 192 // vector_width, page_size, vector_width),
        dtype=dtype,
        device=device,
    )
    value_cache = torch.zeros(
        (2, num_heads, page_size // vector_width, 128, vector_width),
        dtype=dtype,
        device=device,
    )

    launch_reshape_and_cache_shuffle_5d(
        key,
        value,
        key_cache,
        value_cache,
        slots,
    )
    gathered_key, gathered_value = launch_gather_shuffle_5d_to_linear(
        key_cache,
        value_cache,
        slots,
    )

    assert gathered_key.shape == key.shape
    assert gathered_value.shape == value.shape
    torch.testing.assert_close(gathered_key, key, rtol=0, atol=0)
    torch.testing.assert_close(gathered_value, value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_vectorized_kv_pool_prefix_valid_commit_round_trip():
    batch_size, verify_block, num_heads = 3, 8, 1
    page_size, head_dim = 64, 128
    num_tokens = batch_size * verify_block
    commit_lens = torch.tensor([1, 4, 8], dtype=torch.int32, device="cuda")
    slots_2d = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [65, 66, 67, 68, 69, 70, 71, 72],
            [129, 130, 131, 132, 133, 134, 135, 136],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    key = (
        torch.arange(num_tokens * head_dim, dtype=torch.int32, device="cuda")
        .remainder(251)
        .to(torch.bfloat16)
        .view(num_tokens, num_heads, head_dim)
    )
    value = key.add(17)
    vector_width = 8
    key_cache = torch.zeros(
        (3, num_heads, head_dim // vector_width, page_size, vector_width),
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache = torch.zeros(
        (3, num_heads, page_size // vector_width, head_dim, vector_width),
        dtype=torch.bfloat16,
        device="cuda",
    )
    draft_key = key.add(1000)
    draft_value = value.add(1000)
    launch_reshape_and_cache_shuffle_5d(
        draft_key,
        draft_value,
        key_cache,
        value_cache,
        slots_2d.reshape(-1),
    )
    pool = object.__new__(MHATokenToKVPool)
    pool.kv_cache_layout = "vectorized_5d"
    pool.dtype = torch.bfloat16
    pool.store_dtype = torch.bfloat16
    pool.row_dim = num_heads * head_dim
    pool.v_row_dim = num_heads * head_dim
    pool.start_layer = 0
    pool.k_buffer = [key_cache]
    pool.v_buffer = [value_cache]

    pool.set_kv_buffer_prefix_valid(
        SimpleNamespace(layer_id=0), slots_2d, commit_lens, key, value
    )

    valid_mask = (
        torch.arange(verify_block, device="cuda")[None, :] < commit_lens[:, None]
    )
    valid_rows = valid_mask.reshape(-1)
    gathered_key, gathered_value = launch_gather_shuffle_5d_to_linear(
        key_cache, value_cache, slots_2d.reshape(-1)
    )
    expected_key = draft_key.clone()
    expected_value = draft_value.clone()
    expected_key[valid_rows] = key[valid_rows]
    expected_value[valid_rows] = value[valid_rows]
    torch.testing.assert_close(gathered_key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(gathered_value, expected_value, rtol=0, atol=0)


def test_vectorized_kv_pool_prefix_valid_routes_to_shuffle_writer(monkeypatch):
    captured = {}

    def fake_prefix_valid_writer(
        key, value, key_cache, value_cache, slot_mapping_2d, commit_lens
    ):
        captured.update(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping_2d=slot_mapping_2d,
            commit_lens=commit_lens,
        )

    monkeypatch.setattr(
        "sglang.srt.layers.attention.utils."
        "launch_reshape_and_cache_shuffle_5d_prefix_valid",
        fake_prefix_valid_writer,
    )
    pool = object.__new__(MHATokenToKVPool)
    pool.kv_cache_layout = "vectorized_5d"
    pool.dtype = torch.bfloat16
    pool.store_dtype = torch.bfloat16
    pool.row_dim = 128
    pool.v_row_dim = 128
    pool.start_layer = 0
    pool.k_buffer = [torch.empty((2, 1, 16, 64, 8), dtype=torch.bfloat16)]
    pool.v_buffer = [torch.empty((2, 1, 8, 128, 8), dtype=torch.bfloat16)]
    slots_2d = torch.tensor([[1, 2], [65, 66]], dtype=torch.int64)
    commit_lens = torch.tensor([1, 2], dtype=torch.int32)
    key = torch.randn((4, 1, 128), dtype=torch.bfloat16)
    value = torch.randn_like(key)

    pool.set_kv_buffer_prefix_valid(
        SimpleNamespace(layer_id=0), slots_2d, commit_lens, key, value
    )

    assert captured["key"] is key
    assert captured["value"] is value
    assert captured["key_cache"] is pool.k_buffer[0]
    assert captured["value_cache"] is pool.v_buffer[0]
    assert captured["slot_mapping_2d"] is slots_2d
    assert captured["commit_lens"] is commit_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_vectorized_kv_pool_move_round_trip_with_overlap():
    num_heads, page_size, head_dim = 1, 64, 128
    vector_width = 8
    key_cache = torch.zeros(
        (3, num_heads, head_dim // vector_width, page_size, vector_width),
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache = torch.zeros(
        (3, num_heads, page_size // vector_width, head_dim, vector_width),
        dtype=torch.bfloat16,
        device="cuda",
    )
    src_slots = torch.tensor([1, 65, 129], dtype=torch.int64, device="cuda")
    dst_slots = torch.tensor([65, 129, 2], dtype=torch.int64, device="cuda")
    key = (
        torch.arange(src_slots.numel() * head_dim, dtype=torch.int32, device="cuda")
        .remainder(251)
        .to(torch.bfloat16)
        .view(-1, num_heads, head_dim)
    )
    value = key.add(29)
    launch_reshape_and_cache_shuffle_5d(key, value, key_cache, value_cache, src_slots)

    pool = object.__new__(MHATokenToKVPool)
    pool.kv_cache_layout = "vectorized_5d"
    pool.layer_num = 1
    pool.page_size = page_size
    pool.size = 2 * page_size
    pool.k_buffer = [key_cache]
    pool.v_buffer = [value_cache]
    pool.move_kv_cache(dst_slots, src_slots)

    moved_key, moved_value = launch_gather_shuffle_5d_to_linear(
        key_cache, value_cache, dst_slots
    )
    torch.testing.assert_close(moved_key, key, rtol=0, atol=0)
    torch.testing.assert_close(moved_value, value, rtol=0, atol=0)


def test_vectorized_kv_pool_move_routes_through_layout_helpers(monkeypatch):
    calls = []
    gathered_key = torch.ones((2, 1, 128), dtype=torch.bfloat16)
    gathered_value = torch.full_like(gathered_key, 2)

    def fake_gather(key_cache, value_cache, slots):
        calls.append(("gather", key_cache, value_cache, slots))
        return gathered_key, gathered_value

    def fake_scatter(key, value, key_cache, value_cache, slots):
        calls.append(("scatter", key, value, key_cache, value_cache, slots))

    monkeypatch.setattr(
        "sglang.srt.layers.attention.utils.launch_gather_shuffle_5d_to_linear",
        fake_gather,
    )
    monkeypatch.setattr(
        "sglang.srt.layers.attention.utils.launch_reshape_and_cache_shuffle_5d",
        fake_scatter,
    )
    pool = object.__new__(MHATokenToKVPool)
    pool.kv_cache_layout = "vectorized_5d"
    pool.layer_num = 1
    pool.page_size = 64
    pool.size = 128
    pool.k_buffer = [torch.empty((3, 1, 16, 64, 8), dtype=torch.bfloat16)]
    pool.v_buffer = [torch.empty((3, 1, 8, 128, 8), dtype=torch.bfloat16)]
    src_slots = torch.tensor([1, 65], dtype=torch.int64)
    dst_slots = torch.tensor([65, 2], dtype=torch.int64)

    pool.move_kv_cache(dst_slots, src_slots)

    assert calls[0] == ("gather", pool.k_buffer[0], pool.v_buffer[0], src_slots)
    assert calls[1] == (
        "scatter",
        gathered_key,
        gathered_value,
        pool.k_buffer[0],
        pool.v_buffer[0],
        dst_slots,
    )


def _make_flydsl_bf16_backend(query_length=4):
    backend = object.__new__(aiter_backend.AiterAttnBackend)
    backend.kv_cache_is_vectorized_5d = True
    backend.use_mla = False
    backend.topk = 1
    backend.page_size = 64
    backend.max_context_len = 1_048_576
    backend.num_head = 16
    backend.num_kv_head = 1
    backend.head_dim = 192
    backend.v_head_dim = 128
    backend.input_dtype = torch.bfloat16
    backend.kv_cache_dtype = torch.bfloat16
    backend.num_draft_tokens = query_length
    backend._flydsl_pa_decode_workspace_max_bs = 0
    backend._flydsl_pa_decode_query_length = query_length
    backend._flydsl_pa_decode_compiled = False
    return backend


def _patch_flydsl_backend_dependencies(monkeypatch, *, gfx942, gfx950):
    captured = {}

    def fake_compile_pa_decode_tile(**kwargs):
        captured["tile"] = kwargs
        captured.setdefault("tiles", []).append(kwargs)

    def fake_compile_pa_decode_reduce(**kwargs):
        captured["reduce"] = kwargs
        captured.setdefault("reduces", []).append(kwargs)

    kernels = SimpleNamespace(
        pa_decode_tile=lambda **kwargs: None,
        compile_pa_decode_tile=fake_compile_pa_decode_tile,
        compile_pa_decode_reduce=fake_compile_pa_decode_reduce,
        kv_dtype_parameter="kv_dtype",
        reduce_uses_runtime_query_length=True,
        version="test",
        runtime_path="test-runtime",
        kernel_path="test-kernel",
    )
    monkeypatch.setattr(
        aiter_backend.envs,
        "SGLANG_AITER_PA_DECODE_IMPL",
        SimpleNamespace(get=lambda: "flydsl"),
    )
    monkeypatch.setattr(aiter_backend, "is_gfx942_supported", lambda: gfx942)
    monkeypatch.setattr(aiter_backend, "is_gfx95_supported", lambda: gfx950)
    monkeypatch.setattr(aiter_backend, "get_flydsl_mimo_num_partitions", lambda: 8)
    monkeypatch.setattr(aiter_backend, "load_flydsl_pa_decode_kernels", lambda: kernels)
    monkeypatch.setattr(
        aiter_backend,
        "warmup_flydsl_full_decode",
        lambda backend: captured.setdefault("warmups", []).append(backend),
    )
    monkeypatch.setattr(
        aiter_backend.AiterAttnBackend,
        "_ensure_flydsl_pa_decode_workspace",
        lambda self, max_bs: None,
    )
    return captured


@pytest.mark.parametrize("query_length", [1, 4, 8])
@pytest.mark.parametrize(
    "gfx942,gfx950",
    [(True, False), (False, True)],
    ids=["gfx942", "gfx950"],
)
def test_flydsl_bf16_decode_configuration_accepts_supported_arches(
    monkeypatch, gfx942, gfx950, query_length
):
    captured = _patch_flydsl_backend_dependencies(
        monkeypatch, gfx942=gfx942, gfx950=gfx950
    )
    backend = _make_flydsl_bf16_backend(query_length)

    backend._configure_flydsl_pa_decode(max_bs=1)
    backend._compile_flydsl_pa_decode()

    assert backend._use_flydsl_pa_decode
    assert backend._flydsl_pa_decode_num_partitions == 8
    assert backend._flydsl_pa_decode_tile is not None
    assert captured["tile"]["kv_dtype"] == "bf16"
    assert captured["tile"]["head_dim"] == 192
    assert captured["tile"]["v_head_dim"] == 128
    assert captured["tile"]["query_length"] == query_length
    assert [item["query_length"] for item in captured["tiles"]] == sorted(
        {1, query_length}
    )
    assert captured["reduce"]["head_size"] == 128
    assert captured["reduce"]["query_group_size"] == 16
    assert "query_seq_len" not in captured["reduce"]
    # The prepared flag prevents duplicate work on subsequent graph setup calls.
    count = len(captured["tiles"])
    backend._compile_flydsl_pa_decode()
    assert len(captured["tiles"]) == count
    assert captured["warmups"] == [backend]


def test_flydsl_fp8_decode_compile_disables_bf16_kv(monkeypatch):
    captured = _patch_flydsl_backend_dependencies(
        monkeypatch, gfx942=False, gfx950=True
    )
    backend = _make_flydsl_bf16_backend()
    backend.kv_cache_dtype = fp8_dtype

    backend._configure_flydsl_pa_decode(max_bs=1)
    backend._compile_flydsl_pa_decode()

    assert captured["tile"]["kv_dtype"] == "fp8"


def test_flydsl_pa_decode_loader_rejects_legacy_compile_api(monkeypatch):
    def legacy_compile_pa_decode_tile(*, head_dim):
        pass

    modules = {
        "flydsl": SimpleNamespace(__version__="0.3.2", __file__="test-runtime"),
        "aiter.ops.flydsl.pa_decode": SimpleNamespace(
            __file__="legacy-pa-decode-tile",
            pa_decode_tile=lambda **kwargs: None,
            compile_pa_decode_tile=legacy_compile_pa_decode_tile,
        ),
        "aiter.ops.flydsl.kernels.pa_decode_reduce": SimpleNamespace(
            compile_pa_decode_ps_reduce=lambda **kwargs: None
        ),
    }
    monkeypatch.setattr(
        aiter_utils.importlib, "import_module", lambda name: modules[name]
    )
    aiter_utils.load_flydsl_pa_decode_kernels.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="v_head_dim"):
            aiter_utils.load_flydsl_pa_decode_kernels()
    finally:
        aiter_utils.load_flydsl_pa_decode_kernels.cache_clear()


def test_flydsl_pa_decode_loader_accepts_optimized_tile_api(monkeypatch):
    def compile_pa_decode_tile(*, v_head_dim, kv_dtype):
        pass

    modules = {
        "flydsl": SimpleNamespace(__version__="0.3.2", __file__="test-runtime"),
        "aiter.ops.flydsl.pa_decode": SimpleNamespace(
            __file__="pa-decode-tile-032",
            pa_decode_tile=lambda **kwargs: None,
            compile_pa_decode_tile=compile_pa_decode_tile,
        ),
        "aiter.ops.flydsl.kernels.pa_decode_reduce": SimpleNamespace(
            compile_pa_decode_ps_reduce=lambda *, query_seq_len, **kwargs: None
        ),
    }
    monkeypatch.setattr(
        aiter_utils.importlib, "import_module", lambda name: modules[name]
    )
    aiter_utils.load_flydsl_pa_decode_kernels.cache_clear()

    try:
        kernels = aiter_utils.load_flydsl_pa_decode_kernels()
        assert kernels.kv_dtype_parameter == "kv_dtype"
        assert not kernels.reduce_uses_runtime_query_length
    finally:
        aiter_utils.load_flydsl_pa_decode_kernels.cache_clear()


def test_flydsl_bf16_decode_configuration_rejects_unsupported_arch(monkeypatch):
    _patch_flydsl_backend_dependencies(monkeypatch, gfx942=False, gfx950=False)
    backend = _make_flydsl_bf16_backend()

    with pytest.raises(RuntimeError, match="validated only on gfx942 or gfx950"):
        backend._configure_flydsl_pa_decode(max_bs=1)


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm"
)
def test_split_verify_mixed_kv_heads_matches_torch_and_graph(monkeypatch):
    import weakref

    monkeypatch.setenv("SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1", "1")
    torch.manual_seed(721)
    device = torch.device("cuda")
    qlen, q_heads, dim, total_kv, page = 8, 16, 192, 384, 64
    table = torch.arange(total_kv // page, dtype=torch.int32, device=device)[None]
    backend = SimpleNamespace(
        page_size=page,
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        forward_metadata=SimpleNamespace(
            max_q_len=qlen, kv_indices=table, swa_page_table=table
        ),
    )
    batch = SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([300], dtype=torch.int64, device=device),
    )
    cases = []
    for kv_heads, window in [(1, -1), (2, 128)]:
        q = torch.randn(qlen, q_heads, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(total_kv, kv_heads, dim, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        k_cache = (
            k.view(-1, page, kv_heads, dim // 8, 8)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        v_cache = (
            v.view(-1, page // 8, 8, kv_heads, dim)
            .permute(0, 3, 1, 4, 2)
            .contiguous()
        )
        layer = SimpleNamespace(
            sliding_window_size=window,
            tp_q_head_num=q_heads,
            tp_k_head_num=kv_heads,
            qk_head_dim=dim,
            v_head_dim=dim,
            scaling=dim**-0.5,
            logit_cap=0.0,
        )
        sinks = (
            torch.linspace(-0.4, 0.3, q_heads, device=device) if window > 0 else None
        )
        cases.append((q, k, v, k_cache, v_cache, layer, sinks, torch.empty_like(q)))

    def run():
        # MiMo alternates full attention (one KV head) and SWA (two KV heads)
        # on the same backend, including within one captured graph.
        for i in (0, 1, 0):
            q, _, _, k_cache, v_cache, layer, sinks, output = cases[i]
            aiter_utils.forward_target_verify_vectorized_5d(
                backend, q, layer, batch, k_cache, v_cache, output, sinks
            )

    def check(prefix):
        for q, k, v, _, _, layer, sinks, output in cases:
            repeats = q_heads // layer.tp_k_head_num
            k_ref = k.float().repeat_interleave(repeats, dim=1)
            v_ref = v.float().repeat_interleave(repeats, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q.float(), k_ref) * layer.scaling
            q_pos = prefix + torch.arange(qlen, device=device)[:, None]
            k_pos = torch.arange(total_kv, device=device)[None, :]
            visible = k_pos <= q_pos
            if layer.sliding_window_size > 0:
                visible &= k_pos > q_pos - layer.sliding_window_size
            scores.masked_fill_(~visible, float("-inf"))
            if sinks is not None:
                scores = torch.cat(
                    [scores, sinks[:, None, None].expand(-1, qlen, 1)], -1
                )
            probs = torch.softmax(scores, -1)[..., :total_kv]
            expected = torch.einsum("hqk,khd->qhd", probs, v_ref)
            torch.testing.assert_close(
                output.float(),
                expected,
                atol=0.003,
                rtol=0.02,
                msg=lambda msg: (
                    f"KV heads={layer.tp_k_head_num}, "
                    f"window={layer.sliding_window_size}, prefix={prefix}: {msg}"
                ),
            )

    run()
    torch.cuda.synchronize()
    check(300)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    old_context = weakref.ref(backend._swa_target_verify_context_lengths)
    aiter_utils._ensure_swa_target_verify_workspace(
        backend,
        max_batch_size=2,
        query_length=qlen,
        num_q_heads=q_heads,
        num_kv_heads=1,
        value_head_dim=dim,
        max_blocks_per_sequence=12,
        max_partitions=8,
        device=device,
        output_dtype=torch.bfloat16,
        context_dtype=batch.seq_lens.dtype,
    )
    assert old_context() is not None, "captured graph still needs the old allocation"
    batch.seq_lens.fill_(180)
    for case in cases:
        case[-1].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(180)


@pytest.mark.parametrize("capture_aux", [False, True])
def test_mimo_multimodal_aux_capture_embeds_placeholders_first(monkeypatch, capture_aux):
    from sglang.srt.managers import mm_utils
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 4)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(
            self, input_ids=None, positions=None, forward_batch=None,
            input_embeds=None, **kwargs
        ):
            # A raw multimodal placeholder would raise an embedding OOB here.
            hidden = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            if capture_aux:
                return hidden + 1, None, [hidden + 2]
            return hidden + 1, None

    embedded = torch.full((2, 4), 7.0)
    monkeypatch.setattr(mm_utils, "embed_mm_inputs", lambda **kwargs: (embedded, {}))
    monkeypatch.setattr(mm_utils, "get_global_server_args", lambda: None)
    model = object.__new__(mimo_v2.MiMoV2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(encoder_only=False)
    model.capture_aux_hidden_states = capture_aux
    model._is_multimodal = True
    model.model = LanguageModel()
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.lm_head = None
    model.logits_processor = lambda _ids, hidden, _head, _batch, **kwargs: (
        hidden,
        kwargs["aux_hidden_states"],
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        contains_mm_inputs=lambda: True,
        mm_inputs=[SimpleNamespace(mm_items=[])],
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[2],
        input_embeds=None,
    )
    hidden, auxiliary = model.forward(
        torch.tensor([7, 123456789]), torch.arange(2), batch
    )
    torch.testing.assert_close(hidden, embedded + 1)
    assert batch.mm_inputs is None
    if capture_aux:
        assert len(auxiliary) == 1
        torch.testing.assert_close(auxiliary[0], embedded + 2)
    else:
        assert auxiliary is None
