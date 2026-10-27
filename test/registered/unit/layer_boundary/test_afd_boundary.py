"""AFD contracts over native stage boundaries and invocation-owned residuals."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang.srt.afd.contracts import AFDRole
from sglang.srt.afd.model_adapters.base import AFDDecoderAdapter
from sglang.srt.afd.model_adapters.glm5 import Glm5AFDAdapter
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.boundary_fixtures import build_stages
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("family", ["qwen", "glm"])
@pytest.mark.parametrize("rows", [1, 5])
def test_two_microbatches_use_native_residuals_once_and_reset_each_invocation(
    monkeypatch, family, rows
):
    from sglang.srt.models import glm4_moe, qwen3_moe
    from sglang.test.boundary_fixtures import Norm, parallel_of, planning

    module = qwen3_moe if family == "qwen" else glm4_moe
    model_type = (
        module.Qwen3MoeDecoderLayer
        if family == "qwen"
        else module.GlmMoeDsaAFDDecoderLayer
    )
    monkeypatch.setattr(module, "afd_execution_mode", lambda: "attention")
    adapter = object.__new__(AFDDecoderAdapter if family == "qwen" else Glm5AFDAdapter)
    adapter.role = AFDRole.ATTENTION
    adapter._step_id = 0
    adapter._stage_states = {}
    adapter.num_layers = 2
    layers = []
    with planning(parallel_of(attn_dp=8, attn_tp=1)):
        for index in range(2):
            from sglang.srt.layers.layer_boundary.factories import (
                declare_attn,
                declare_ffn,
            )

            ffn = declare_ffn(dense_tp_size=1)
            attn_stage, ffn_stage = build_stages(
                (declare_attn(), Norm()),
                (ffn, Norm()),
                previous=None if index == 0 else ffn,
                terminal=index == 1,
            )

            class Attention:
                skip_topk = next_skip_topk = False

                def __call__(self, *, hidden_states, **kw):
                    return hidden_states + self.offset

                def maybe_use_decode_attn_tp(self, batch):
                    return nullcontext()

            attention = Attention()
            attention.offset = index + 1
            layer = SimpleNamespace(
                attn_boundary=attn_stage,
                ffn_boundary=ffn_stage,
                self_attn=attention,
                _resolve_gfx95_quant_format=lambda: "",
            )
            layer.forward_attention_for_afd = MethodType(
                model_type.forward_attention_for_afd, layer
            )
            layers.append(layer)
        adapter.inner = SimpleNamespace(layers=layers)
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=rows,
            input_ids=torch.arange(rows),
            positions=torch.arange(rows),
            req_pool_indices=torch.arange(rows),
            seq_lens=torch.ones(rows, dtype=torch.int32),
            seq_lens_sum=rows,
            out_cache_loc=torch.arange(rows),
        )
        initial = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
        stages = adapter.split_step(
            hidden_states=initial,
            residual=None,
            positions=batch.positions,
            forward_batch=batch,
            stages=2,
        )
        adapter._step_id = 1
        adapter._stage_states = {
            i: SimpleNamespace(zero_allocator=None) for i in range(2)
        }
        # Reuse the same stage batches as capture warmup does, with new inputs.
        for shift in (0, 17):
            for stage in stages:
                stage.hidden_states = (
                    initial[stage.token_start : stage.token_stop] + shift
                )
                stage.residual = None
            for index in range(2):
                for stage in stages:
                    h, r = adapter.local_compute(
                        layer=index,
                        stage=stage,
                        hidden_states=stage.hidden_states,
                        residual=stage.residual,
                    )
                    stage.hidden_states, stage.residual = adapter.finish_layer(
                        layer=index, stage=stage, ffn_output=h * 0.75, residual=r
                    )
                if index == 0:
                    assert (
                        stages[0].forward_batch.residual_stream
                        is not stages[1].forward_batch.residual_stream
                    )
                else:
                    assert all(s.forward_batch.residual_stream is None for s in stages)
            h, r = adapter.join_step(stages=stages)
            actual = Norm()(h, r)[0]
            expected = initial + shift
            for offset in (1, 2):
                expected = 2.5 * (3 * expected + offset)
            torch.testing.assert_close(actual, expected * 2)
        assert batch.residual_stream is None


@pytest.mark.parametrize("mode", ["auto", "direct", "fixed", "legacy_false"])
def test_qwen_constructor_retains_router_without_native_expert_collectives(
    monkeypatch, mode
):
    from sglang.srt.afd import model_hooks
    from sglang.srt.afd.config import AFDConfig
    from sglang.srt.layers.layer_boundary.factories import layer_stack
    from sglang.srt.models import qwen3_moe
    from sglang.test.boundary_fixtures import Norm, parallel_of, planning

    class Router(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def route(self, hidden_states):
            return hidden_states

    afd_config = (
        AFDConfig(experimental_direct=False)
        if mode == "legacy_false"
        else AFDConfig(ffn_backend=mode)
    )
    monkeypatch.setattr(
        model_hooks,
        "get_server_args",
        lambda: SimpleNamespace(afd_config=afd_config, moe_a2a_backend="none"),
    )

    monkeypatch.setattr(qwen3_moe, "afd_execution_mode", lambda: "attention")
    monkeypatch.setattr(qwen3_moe, "Qwen3AFDRouter", Router)
    monkeypatch.setattr(qwen3_moe, "get_rope_config", lambda cfg: (10000, None))
    monkeypatch.setattr(qwen3_moe, "Qwen3MoeAttention", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(qwen3_moe, "RMSNorm", lambda *a, **kw: Norm())
    config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    with planning(parallel_of(attn_dp=8, attn_tp=1)), layer_stack():
        layer = qwen3_moe.Qwen3MoeDecoderLayer(config, 0)
    assert layer.ffn_boundary.declaration.dense_tp_size == 1
    assert not layer.ffn_boundary.declaration.sparse
    assert layer.is_layer_sparse
    assert isinstance(layer.mlp, Router) == (mode in ("auto", "direct"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
