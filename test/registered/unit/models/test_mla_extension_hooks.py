"""CPU contracts for MLA arithmetic overrides within the SRT forward lifecycle."""

import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MLA = forward_mla.DeepseekMLAForwardMixin


@pytest.mark.parametrize(
    "mode", [ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY]
)
@pytest.mark.parametrize("reuse_indices", [False, True])
@pytest.mark.parametrize("born_fp8", [False, True])
def test_prepare_dispatches_hooks_and_preserves_index_reuse(
    monkeypatch, mode, reuse_indices, born_fp8
):
    torch.manual_seed(0)
    rows, heads, rank, rope = 3, 2, 128 if born_fp8 else 4, 2
    latent = torch.randn(rows, rank + rank + rope)
    query = torch.randn(rows, heads, rank + rope).to(torch.bfloat16)
    positions = torch.arange(rows)
    hidden = torch.randn(rows, 8)
    batch = SimpleNamespace(forward_mode=mode, attn_cp_metadata=None)
    previous = torch.tensor([[0, 1], [0, 2], [1, 2]])
    indices = torch.tensor([[1, 0], [2, 0], [2, 1]])
    rotary = Mock(side_effect=lambda pos, q, k: (q + 10, k + 20))
    attn = SimpleNamespace(
        q_lora_rank=rank,
        kv_lora_rank=rank,
        qk_nope_head_dim=rank,
        qk_rope_head_dim=rope,
        num_local_heads=heads,
        layer_id=7,
        dcp_size=1,
        alt_stream=None,
        use_dsa=True,
        use_deep_gemm_bmm=False,
        skip_topk=reuse_indices,
        is_nextn=False,
        q_a_layernorm=lambda x: x,
        kv_a_layernorm=lambda x: x,
        q_b_proj_forward=lambda x: query,
        w_kc=torch.randn(heads, rank, rank).to(torch.bfloat16),
        rotary_emb=rotary,
        indexer=Mock(return_value=indices),
        current_attention_backend="dsa",
        is_dcp_q_replicate_phase=lambda batch: False,
        _fuse_rope_for_trtllm_mla=lambda batch: False,
    )
    for name in (
        "_split_q_nope_pe",
        "should_run_indexer",
        "_can_fuse_bmm_into_attention",
        "_make_mla_bmm_fusion_plan",
        "_q8kv8_born_fp8_q_backend",
    ):
        setattr(attn, name, MethodType(getattr(MLA, name), attn))
    attn.run_mla_indexer = Mock(wraps=MethodType(MLA.run_mla_indexer, attn))
    prepared = []

    def prepare(pos, q, q_pe, k, k_pe, forward_batch):
        q, q_pe, k, k_pe = MLA.prepare_mla_qkv(
            attn, pos, q, q_pe, k, k_pe, forward_batch
        )
        result = (q + 1, q_pe, k + 2, k_pe)
        prepared.append(result)
        return result

    attn.prepare_mla_qkv = Mock(side_effect=prepare)
    attn.needs_unfused_mla_qkv = True
    monkeypatch.setattr(
        forward_mla,
        "get_attn_tp_context",
        lambda: SimpleNamespace(fetch_qkv_latent=lambda: latent),
    )
    monkeypatch.setattr(forward_mla, "is_kv_b_lora_active", lambda _: False)
    monkeypatch.setattr(
        forward_mla, "is_graph_dsa_split_op_surface", lambda _: not born_fp8
    )
    monkeypatch.setattr(forward_mla, "_SGLANG_EXPERIMENTAL_LORA_OPTI", False)
    monkeypatch.setattr(forward_mla, "_ENABLE_DSA_Q8KV8_BORN_FP8_Q", born_fp8)
    monkeypatch.setattr(forward_mla, "_is_cpu", False)
    monkeypatch.setattr(forward_mla, "_is_hip", False)
    monkeypatch.setattr(forward_mla, "is_in_full_prefill_graph", lambda: False)
    monkeypatch.setattr(forward_mla, "is_in_breakable_cuda_graph", lambda: False)
    backend = SimpleNamespace(q8kv8_born_fp8_q_eligible=lambda batch, heads: True)
    monkeypatch.setattr(forward_mla, "get_attn_backend", lambda: backend)
    monkeypatch.setattr(
        forward_mla, "maybe_capture_indexer_topk", lambda layer, value: value
    )
    kv_prepare = Mock(side_effect=lambda module, batch, k, k_pe, **kwargs: (k, k_pe))
    monkeypatch.setattr(forward_mla, "prepare_kv_for_attention", kv_prepare)

    result = MLA.forward_absorb_prepare(
        attn, positions, hidden, batch, None, prev_topk_indices=previous
    )

    assert result[7] is (previous if reuse_indices else indices)
    assert attn.run_mla_indexer.call_count == (0 if reuse_indices else 1)
    if not reuse_indices:
        kwargs = attn.indexer.call_args.kwargs
        assert kwargs["x"] is hidden and kwargs["forward_batch"] is batch
        assert kwargs["positions"] is positions and kwargs["layer_id"] == 7
        torch.testing.assert_close(kwargs["q_lora"], latent[:, :rank])
    attn.prepare_mla_qkv.assert_called_once()
    rotary.assert_called_once()
    q, q_pe, k, k_pe = prepared[0]
    assert result[0] is q_pe and result[1] is k_pe
    assert result[2] is q and result[3] is k
    assert kv_prepare.call_args.args[2] is k
    assert kv_prepare.call_args.args[3] is k_pe
    torch.testing.assert_close(
        q, torch.bmm(query[..., :rank].transpose(0, 1), attn.w_kc).transpose(0, 1) + 1
    )
    torch.testing.assert_close(q_pe, query[..., rank:] + 10)
    torch.testing.assert_close(k, latent[:, rank : 2 * rank].unsqueeze(1) + 2)
    torch.testing.assert_close(k_pe, latent[:, 2 * rank :].unsqueeze(1) + 20)


def test_default_prepare_keeps_bmm_attention_fusion(monkeypatch):
    attn = SimpleNamespace(
        dcp_size=1,
        use_dsa=True,
        use_deep_gemm_bmm=False,
        w_kc=torch.empty(2, 4, 4),
        current_attention_backend="dsa",
        needs_unfused_mla_qkv=False,
    )
    monkeypatch.setattr(forward_mla, "is_kv_b_lora_active", lambda _: False)
    monkeypatch.setattr(forward_mla, "is_graph_dsa_split_op_surface", lambda _: True)
    monkeypatch.setattr(forward_mla, "is_in_prefill_graph_scope", lambda: True)
    assert MLA._can_fuse_bmm_into_attention(attn, SimpleNamespace())


@pytest.mark.parametrize("custom_prepare", [False, True])
def test_born_fp8_requires_default_query_preparation(monkeypatch, custom_prepare):
    attn = SimpleNamespace(
        current_attention_backend="dsa",
        use_deep_gemm_bmm=False,
        w_kc=torch.empty(2, 128, 128, dtype=torch.bfloat16),
        rotary_emb=Mock(),
        _fuse_rope_for_trtllm_mla=lambda batch: False,
        dcp_size=1,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        kv_lora_rank=128,
        num_local_heads=2,
    )
    attn.needs_unfused_mla_qkv = custom_prepare
    monkeypatch.setattr(forward_mla, "_is_cpu", False)
    monkeypatch.setattr(forward_mla, "_is_hip", False)
    monkeypatch.setattr(forward_mla, "is_kv_b_lora_active", lambda _: False)
    monkeypatch.setattr(forward_mla, "_SGLANG_EXPERIMENTAL_LORA_OPTI", False)
    monkeypatch.setattr(forward_mla, "is_graph_dsa_split_op_surface", lambda _: False)
    monkeypatch.setattr(forward_mla, "is_in_full_prefill_graph", lambda: False)
    monkeypatch.setattr(forward_mla, "is_in_breakable_cuda_graph", lambda: False)
    backend = SimpleNamespace(q8kv8_born_fp8_q_eligible=lambda batch, heads: True)
    monkeypatch.setattr(forward_mla, "get_attn_backend", lambda: backend)

    result = MLA._q8kv8_born_fp8_q_backend(attn, SimpleNamespace(), None)

    assert result is (None if custom_prepare else backend)


@pytest.mark.parametrize("fp8", [False, True])
def test_default_value_projection_matches_bmm(monkeypatch, fp8):
    torch.manual_seed(1)
    weight = torch.randn(2, 4, 3)
    attn = SimpleNamespace(
        use_deep_gemm_bmm=False,
        num_local_heads=2,
        v_head_dim=3,
        w_vc=weight.to(torch.float8_e4m3fn) if fp8 else weight,
        w_scale=torch.tensor(0.5),
    )
    values = torch.randn(5, 2, 4)
    monkeypatch.setattr(forward_mla, "_is_cpu", True)
    monkeypatch.setattr(forward_mla, "_is_musa", False)
    expected = (
        torch.bmm(
            values.to(torch.bfloat16).transpose(0, 1),
            attn.w_vc.to(torch.bfloat16) * attn.w_scale,
        )
        if fp8
        else torch.bmm(values.transpose(0, 1), weight)
    )
    output = MLA.project_mla_values(attn, values, None)
    torch.testing.assert_close(
        output, expected.transpose(0, 1).flatten(1, 2), rtol=0, atol=0
    )


@pytest.mark.parametrize("carry_indices", [None, False, True])
@pytest.mark.parametrize("lora_mode", ["legacy", "v2", "experimental"])
def test_projection_override_keeps_lora_gate_and_output_order(
    monkeypatch, carry_indices, lora_mode
):
    values = torch.ones(3, 2, 4)
    indices = torch.zeros(3, 2, dtype=torch.int32)
    events = []

    def projection(value, allocator):
        events.append("projection")
        torch.testing.assert_close(value, values)
        return torch.full((3, 6), 2.0)

    prepared = object()

    def prepare_lora(module, value):
        events.append("prepare_lora")
        assert module is attn
        torch.testing.assert_close(value, values)
        return prepared

    def apply_lora(projected):
        events.append("lora")
        return projected + 3

    def apply_v2(state, projected):
        assert state is prepared
        return apply_lora(projected)

    def apply_experimental(module, value, projected, state):
        assert state is prepared
        return apply_lora(projected)

    def gate(projected, gate):
        events.append("gate")
        return projected * gate

    def output(projected):
        events.append("output")
        return projected + 7, None

    attn = SimpleNamespace(
        current_attention_backend="dsa",
        num_local_heads=2,
        kv_lora_rank=4,
        next_skip_topk=carry_indices,
        attn_mqa=Mock(return_value=values),
        _fuse_rope_for_trtllm_mla=lambda batch: False,
        is_dcp_mla_decode_phase=lambda batch: False,
        project_mla_values=projection,
        apply_attention_output_gate=gate,
        o_proj=output,
    )
    monkeypatch.setattr(
        forward_mla, "_SGLANG_EXPERIMENTAL_LORA_OPTI", lora_mode == "experimental"
    )
    monkeypatch.setattr(forward_mla, "kv_b_lora_correction", lambda _: lora_mode)
    if lora_mode == "legacy":
        from sglang.srt.lora import deepseek_mla_correction

        monkeypatch.setattr(
            deepseek_mla_correction,
            "apply_v_correction",
            lambda module, value, projected: apply_lora(projected),
        )
    elif lora_mode == "v2":
        from sglang.srt.lora.dense import mla_correction

        monkeypatch.setattr(mla_correction, "prepare_v_correction", prepare_lora)
        monkeypatch.setattr(mla_correction, "apply_correction", apply_v2)
    else:
        from sglang.srt.lora.trtllm_lora_temp import deepseek_mla_correction

        monkeypatch.setattr(
            deepseek_mla_correction, "kv_b_lora_v_prepare", prepare_lora
        )
        monkeypatch.setattr(
            deepseek_mla_correction, "kv_b_lora_v_apply", apply_experimental
        )
    result = MLA.forward_absorb_core(
        attn,
        None,
        None,
        None,
        None,
        SimpleNamespace(),
        None,
        None,
        indices,
        None,
        attention_output_gate=torch.tensor(5.0),
    )
    expected = [] if lora_mode == "legacy" else ["prepare_lora"]
    assert events == expected + ["projection", "lora", "gate", "output"]
    if carry_indices is not None:
        result, returned_indices = result
        assert returned_indices is (indices if carry_indices else None)
    torch.testing.assert_close(result, torch.full((3, 6), 32.0))
    assert attn.attn_mqa.call_args.kwargs["topk_indices"] is indices


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
