"""Unit tests for --enable-return-aux-hidden-states (aux capture without a draft)."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    resolve_spec_aux_hidden_state_config,
)
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

VOCAB, HIDDEN, NUM_AUX = 11, 4, 3


def _resolve(*, is_draft_worker=False, num_layers=32, num_loops=1):
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_hidden_layers=num_layers, num_loops=num_loops
        )
    )
    return resolve_spec_aux_hidden_state_config(
        server_args=None,
        model_config=model_config,
        spec_algorithm=SpeculativeAlgorithm.from_string(None),
        is_draft_worker=is_draft_worker,
    )


# --- server arguments --------------------------------------------------------


def _resolved(**fields):
    args = ServerArgs(model_path="dummy", **fields)
    args.resolve_once()
    return resolving_view(args)


def test_cli_maps_the_flag_and_algorithm_to_the_capture_method():
    args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--speculative-algorithm",
            "dflash",
            "--enable-return-aux-hidden-states",
            "--return-aux-hidden-state-layer-ids",
            "1",
            "16",
            "31",
        ]
    )
    args.resolve_once()
    resolved = resolving_view(args)
    assert resolved.aux_hidden_state_capture == "dflash"
    # The algorithm only named the convention; the speculative stack stays off.
    assert resolved.speculative_algorithm is None
    assert args.return_aux_hidden_state_layer_ids == [1, 16, 31]


@pytest.mark.parametrize(
    "fields,match",
    [
        (
            dict(return_aux_hidden_state_layer_ids=[1, 2]),
            "requires --enable-return-aux-hidden-states",
        ),
        (
            dict(enable_return_aux_hidden_states=True),
            "needs --speculative-algorithm",
        ),
        (
            dict(enable_return_aux_hidden_states=True, speculative_algorithm="EAGLE"),
            "needs --speculative-algorithm",
        ),
        (
            dict(
                enable_return_aux_hidden_states=True,
                speculative_algorithm="EAGLE3",
                speculative_draft_model_path="draft",
            ),
            "drop --speculative-draft-model-path",
        ),
        (
            dict(enable_return_aux_hidden_states=True, speculative_algorithm="DFLASH"),
            "requires --return-aux-hidden-state-layer-ids",
        ),
        (
            dict(enable_return_aux_hidden_states=True, speculative_algorithm="DSPARK"),
            "requires --return-aux-hidden-state-layer-ids",
        ),
        (
            dict(
                enable_return_aux_hidden_states=True,
                speculative_algorithm="EAGLE3",
                pp_size=2,
            ),
            "only supports pp_size == 1",
        ),
        (
            dict(
                enable_return_aux_hidden_states=True,
                speculative_algorithm="EAGLE3",
                return_aux_hidden_state_layer_ids=[],
            ),
            "must not be empty",
        ),
        (
            dict(
                enable_return_aux_hidden_states=True,
                speculative_algorithm="DFLASH",
                return_aux_hidden_state_layer_ids=[1, 1],
            ),
            "must not repeat",
        ),
        (
            dict(
                enable_return_aux_hidden_states=True,
                speculative_algorithm="DFLASH",
                return_aux_hidden_state_layer_ids=[-1],
            ),
            "must be non-negative",
        ),
    ],
)
def test_invalid_combinations_are_rejected(fields, match):
    with pytest.raises(ValueError, match=match):
        ServerArgs(model_path="dummy", **fields).resolve_once()


def test_eagle3_defaults_to_model_layers():
    resolved = _resolved(
        enable_return_aux_hidden_states=True, speculative_algorithm="EAGLE3"
    )
    assert resolved.aux_hidden_state_capture == "eagle3"
    assert resolved.speculative_algorithm is None


def test_capture_defaults_return_hidden_states_mode_to_full():
    resolved = _resolved(
        enable_return_aux_hidden_states=True, speculative_algorithm="EAGLE3"
    )
    assert resolved.return_hidden_states_mode == "full"


def test_capture_keeps_an_explicit_return_hidden_states_mode():
    resolved = _resolved(
        enable_return_aux_hidden_states=True,
        speculative_algorithm="EAGLE3",
        return_hidden_states_mode="last",
    )
    assert resolved.return_hidden_states_mode == "last"


# --- aux capture config ------------------------------------------------------


def test_disabled_by_default():
    with published_topology():
        config = _resolve()
    assert not config.eagle_use_aux_hidden_state
    assert not config.dflash_use_aux_hidden_state
    assert not config.is_dspark


def test_eagle3_capture_without_draft():
    with published_topology(
        enable_return_aux_hidden_states=True,
        speculative_algorithm="EAGLE3",
        return_aux_hidden_state_layer_ids=[1, 5],
    ):
        config = _resolve()
    assert config.eagle_use_aux_hidden_state
    assert config.eagle_aux_hidden_state_layer_ids == [1, 5]
    assert not config.dflash_use_aux_hidden_state
    # No draft runs, so the KV pool must not reserve draft layers.
    assert config.eagle_draft_num_layers is None
    assert config.dflash_draft_num_layers is None


@pytest.mark.parametrize("method,is_dspark", [("dflash", False), ("dspark", True)])
def test_dflash_family_capture_without_draft(method, is_dspark):
    with published_topology(
        enable_return_aux_hidden_states=True,
        speculative_algorithm=method.upper(),
        return_aux_hidden_state_layer_ids=[1, 16, 31],
    ):
        config = _resolve()
    assert config.dflash_use_aux_hidden_state
    assert config.dflash_target_layer_ids == [1, 16, 31]
    assert config.is_dspark is is_dspark
    assert not config.eagle_use_aux_hidden_state
    assert config.dflash_draft_num_layers is None


def test_draft_runner_never_captures():
    with published_topology(
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=[1],
    ):
        config = _resolve(is_draft_worker=True)
    assert not config.dflash_use_aux_hidden_state


def test_layer_id_beyond_target_depth_is_rejected():
    with published_topology(
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=[1, 8],
    ):
        with pytest.raises(ValueError, match="out of range"):
            _resolve(num_layers=8)


def test_loop_models_span_num_hidden_layers_times_num_loops():
    with published_topology(
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=[1, 12],
    ):
        config = _resolve(num_layers=8, num_loops=2)
    assert config.dflash_target_layer_ids == [1, 12]


class _Model:
    def __init__(self, *hooks):
        self.calls = []
        for hook in hooks:
            setattr(self, hook, lambda ids, _hook=hook: self.calls.append((_hook, ids)))


@pytest.mark.parametrize(
    "is_dspark,hooks,expected",
    [
        (
            True,
            ("set_dspark_layers_to_capture", "set_dflash_layers_to_capture"),
            "set_dspark_layers_to_capture",
        ),
        (
            False,
            ("set_dspark_layers_to_capture", "set_dflash_layers_to_capture"),
            "set_dflash_layers_to_capture",
        ),
        # Without a native DSPARK hook, DSPARK rides the DFLASH capture.
        (True, ("set_dflash_layers_to_capture",), "set_dflash_layers_to_capture"),
    ],
)
def test_dflash_family_hook_routing(is_dspark, hooks, expected):
    model = _Model(*hooks)
    configure_aux_hidden_state_capture(
        model=model,
        eagle_use_aux_hidden_state=False,
        eagle_aux_hidden_state_layer_ids=None,
        dflash_use_aux_hidden_state=True,
        dflash_target_layer_ids=[2, 3],
        is_dspark=is_dspark,
    )
    assert model.calls == [(expected, [2, 3])]


def test_eagle3_hook_routing():
    model = _Model("set_eagle3_layers_to_capture")
    configure_aux_hidden_state_capture(
        model=model,
        eagle_use_aux_hidden_state=True,
        eagle_aux_hidden_state_layer_ids=[1, 5],
        dflash_use_aux_hidden_state=False,
        dflash_target_layer_ids=None,
        is_dspark=False,
    )
    assert model.calls == [("set_eagle3_layers_to_capture", [1, 5])]


def test_model_without_eagle3_hook_is_rejected():
    with pytest.raises(ValueError, match="set_eagle3_layers_to_capture"):
        configure_aux_hidden_state_capture(
            model=_Model("set_dflash_layers_to_capture"),
            eagle_use_aux_hidden_state=True,
            eagle_aux_hidden_state_layer_ids=None,
            dflash_use_aux_hidden_state=False,
            dflash_target_layer_ids=None,
            is_dspark=False,
        )


# --- logits processor --------------------------------------------------------


def _logits_metadata(capture_hidden_mode, extend_seq_lens):
    return LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        capture_hidden_mode=capture_hidden_mode,
        extend_seq_lens=torch.tensor(extend_seq_lens),
        extend_seq_lens_cpu=list(extend_seq_lens),
    )


def _run_logits_processor(capture_hidden_mode, *, with_aux=True, **server_fields):
    extend_seq_lens = [3, 2]
    num_tokens = sum(extend_seq_lens)
    hidden_states = torch.randn(num_tokens, HIDDEN)
    aux = [torch.randn(num_tokens, HIDDEN) for _ in range(NUM_AUX)]
    lm_head_weight = torch.randn(VOCAB, HIDDEN)
    with published_topology(**server_fields):
        processor = LogitsProcessor(SimpleNamespace(vocab_size=VOCAB))
        with patch.object(
            processor,
            "_get_logits",
            side_effect=lambda states, *args, **kwargs: states @ lm_head_weight.T,
        ):
            output = processor(
                torch.zeros(num_tokens, dtype=torch.long),
                hidden_states,
                None,
                _logits_metadata(capture_hidden_mode, extend_seq_lens),
                aux_hidden_states=aux if with_aux else None,
            )
    return output, hidden_states, aux


def test_full_capture_keeps_last_hidden_next_to_aux():
    output, hidden_states, aux = _run_logits_processor(
        CaptureHiddenMode.FULL,
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=[0, 1, 2],
    )
    torch.testing.assert_close(output.hidden_states, torch.cat(aux, dim=-1))
    torch.testing.assert_close(output.last_hidden_states, hidden_states)


def test_fewer_captured_layers_than_requested_fails_fast():
    with pytest.raises(ValueError, match="captured 3 aux hidden states"):
        _run_logits_processor(
            CaptureHiddenMode.FULL,
            enable_return_aux_hidden_states=True,
            speculative_algorithm="DFLASH",
            return_aux_hidden_state_layer_ids=[0, 1, 2, 3],
        )


def test_missing_aux_in_capture_only_mode_fails_fast():
    with pytest.raises(ValueError, match="captured 0 aux hidden states"):
        _run_logits_processor(
            CaptureHiddenMode.NULL,
            with_aux=False,
            enable_return_aux_hidden_states=True,
            speculative_algorithm="DFLASH",
            return_aux_hidden_state_layer_ids=[0, 1, 2],
        )


def test_last_hidden_is_not_kept_without_the_flag():
    # Stock EAGLE3/DFLASH serving also captures FULL + aux; it must not change.
    output, _, aux = _run_logits_processor(CaptureHiddenMode.FULL)
    torch.testing.assert_close(output.hidden_states, torch.cat(aux, dim=-1))
    assert output.last_hidden_states is None


@pytest.mark.parametrize(
    "capture_hidden_mode,with_aux",
    [
        (CaptureHiddenMode.LAST, True),
        (CaptureHiddenMode.NULL, True),
        (CaptureHiddenMode.FULL, False),
    ],
)
def test_last_hidden_only_under_full_aux_capture(capture_hidden_mode, with_aux):
    output, _, _ = _run_logits_processor(
        capture_hidden_mode,
        with_aux=with_aux,
        enable_return_aux_hidden_states=True,
        speculative_algorithm="EAGLE3",
    )
    assert output.last_hidden_states is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
