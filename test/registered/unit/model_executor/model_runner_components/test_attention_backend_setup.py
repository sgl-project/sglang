import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.model_runner_components import (
    attention_backend_setup,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    ResolvedAttentionBackendStr,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True


def test_split_full_attention_applies_model_wrapper_once():
    # The hybrid backend takes the speculative attention mode from the
    # published configuration.
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(speculative_attention_mode="prefill")
    override.install()
    try:
        runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_attention_mode="prefill"),
            model_config=SimpleNamespace(context_len=2048),
            kv_cache_dtype=None,
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            kv_index_translator=None,
            init_new_workspace=None,
        )
        wrapper_inputs = []
        wrapped_backend = object()

        def wrap_once(model_runner, backend):
            assert model_runner is runner
            wrapper_inputs.append(backend)
            return wrapped_backend

        constructors = {
            "decode-test": lambda model_runner: _FakeBackend("decode"),
            "prefill-test": lambda model_runner: _FakeBackend("prefill"),
        }
        resolved = ResolvedAttentionBackendStr(
            decode="decode-test", prefill="prefill-test"
        )

        with (
            patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
            patch.object(
                attention_backend_setup,
                "attn_backend_wrapper",
                side_effect=wrap_once,
            ),
        ):
            result = attention_backend_setup._build_resolved_backend(
                model_runner=runner,
                resolved=resolved,
                init_new_workspace=True,
            )

        assert result is wrapped_backend
        assert len(wrapper_inputs) == 1
        split_backend = wrapper_inputs[0]
        assert isinstance(split_backend, HybridAttnBackend)
        assert split_backend.decode_backend.name == "decode"
        assert split_backend.prefill_backend.name == "prefill"
        assert runner.init_new_workspace is True
    finally:
        override.restore()


def test_equal_resolved_backends_ignore_stale_global_backend():
    runner = SimpleNamespace(
        server_args=SimpleNamespace(
            attention_backend="global-test",
            speculative_attention_mode="prefill",
        ),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )
    constructors = {
        "global-test": lambda _runner: _FakeBackend("global"),
        "resolved-test": lambda _runner: _FakeBackend("resolved"),
    }
    resolved = ResolvedAttentionBackendStr(
        decode="resolved-test",
        prefill="resolved-test",
    )

    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=lambda _runner, backend: backend,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=False,
        )

    assert result.name == "resolved"


def test_kimi_k3_dflash_captures_the_dspark_post_layer_taps():
    """DFLASH target_layer_ids name layer outputs. K3 taps already capture layer
    outputs, so DFLASH must reach them unshifted (targets that tap layer inputs
    add 1) and land on the same states as DSPARK. K3 used to expose only the
    DSPARK hook, so DFLASH was rejected here at startup."""
    import torch

    from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
    from sglang.srt.models import kimi_k3
    from sglang.srt.models.kimi_k3 import (
        KimiK3ForConditionalGeneration,
        KimiK3LinearForCausalLM,
        KimiK3LinearModel,
    )

    num_tokens, hidden_size, layer_ids = 3, 4, [1, 3]
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1)

    def make_layer(idx):
        def layer(*, hidden_states, residual, **_):
            return hidden_states + (idx + 1), residual, False

        return layer

    captured = {}
    for is_dspark in (False, True):
        model = SimpleNamespace(
            config=SimpleNamespace(attn_res_block_size=None),
            pp_group=pp_group,
            start_layer=0,
            end_layer=4,
            layers=[make_layer(i) for i in range(4)],
            norm=lambda hidden_states: hidden_states,
            dspark_layers_to_capture=None,
            _trim_padded_attn=False,
        )
        model._dspark_capture_stream = MethodType(
            KimiK3LinearModel._dspark_capture_stream, model
        )
        lm = KimiK3LinearForCausalLM.__new__(KimiK3LinearForCausalLM)
        torch.nn.Module.__init__(lm)
        lm.model = model
        lm.capture_aux_hidden_states = False
        # The checkpoint architecture is the multimodal wrapper around the LM.
        target = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
        torch.nn.Module.__init__(target)
        target.language_model = lm

        attention_backend_setup.configure_aux_hidden_state_capture(
            model=target,
            eagle_use_aux_hidden_state=False,
            eagle_aux_hidden_state_layer_ids=None,
            dflash_use_aux_hidden_state=True,
            dflash_target_layer_ids=list(layer_ids),
            is_dspark=is_dspark,
        )
        assert lm.capture_aux_hidden_states

        with patch.object(
            kimi_k3, "get_parallel", return_value=SimpleNamespace(pp_group=pp_group)
        ):
            _, aux_hidden_states = KimiK3LinearModel.forward(
                model,
                None,
                torch.arange(num_tokens),
                SimpleNamespace(),
                inputs_embeds=torch.zeros(num_tokens, hidden_size),
            )
        captured[is_dspark] = pack_aux_hidden_states(aux_hidden_states)

    # Layer i adds i + 1, so the outputs of layers 1 and 3 hold 3 and 10;
    # the draft reads them packed in target_layer_ids order.
    expected = torch.cat(
        [torch.full((num_tokens, hidden_size), v) for v in (3.0, 10.0)], dim=-1
    )
    for packed in captured.values():
        torch.testing.assert_close(packed, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
