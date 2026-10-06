"""Unit tests for --aux-hidden-state-capture (aux capture without a draft)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

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


def _resolve(*, is_draft_worker=False):
    return resolve_spec_aux_hidden_state_config(
        server_args=None,
        model_config=None,
        spec_algorithm=SpeculativeAlgorithm.from_string(None),
        is_draft_worker=is_draft_worker,
    )


# --- server arguments --------------------------------------------------------


def test_cli_parses_method_and_layer_ids():
    args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--aux-hidden-state-capture",
            "dflash",
            "--aux-hidden-state-layer-ids",
            "1",
            "16",
            "31",
        ]
    )
    args.resolve_once()
    assert args.aux_hidden_state_capture == "dflash"
    assert args.aux_hidden_state_layer_ids == [1, 16, 31]


@pytest.mark.parametrize(
    "fields,match",
    [
        (
            dict(aux_hidden_state_layer_ids=[1, 2]),
            "requires --aux-hidden-state-capture",
        ),
        (
            dict(aux_hidden_state_capture="dflash"),
            "requires --aux-hidden-state-layer-ids",
        ),
        (
            dict(aux_hidden_state_capture="dspark"),
            "requires --aux-hidden-state-layer-ids",
        ),
        (
            dict(aux_hidden_state_capture="eagle3", speculative_algorithm="EAGLE3"),
            "cannot be combined with --speculative-algorithm",
        ),
    ],
)
def test_invalid_combinations_are_rejected(fields, match):
    with pytest.raises(ValueError, match=match):
        ServerArgs(model_path="dummy", **fields).resolve_once()


def test_eagle3_defaults_to_model_layers():
    ServerArgs(model_path="dummy", aux_hidden_state_capture="eagle3").resolve_once()


# --- aux capture config ------------------------------------------------------


def test_disabled_by_default():
    with published_topology():
        config = _resolve()
    assert not config.eagle_use_aux_hidden_state
    assert not config.dflash_use_aux_hidden_state
    assert not config.is_dspark


def test_eagle3_capture_without_draft():
    with published_topology(
        aux_hidden_state_capture="eagle3", aux_hidden_state_layer_ids=[1, 5]
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
        aux_hidden_state_capture=method, aux_hidden_state_layer_ids=[1, 16, 31]
    ):
        config = _resolve()
    assert config.dflash_use_aux_hidden_state
    assert config.dflash_target_layer_ids == [1, 16, 31]
    assert config.is_dspark is is_dspark
    assert not config.eagle_use_aux_hidden_state
    assert config.dflash_draft_num_layers is None


def test_draft_runner_never_captures():
    with published_topology(
        aux_hidden_state_capture="dflash", aux_hidden_state_layer_ids=[1]
    ):
        config = _resolve(is_draft_worker=True)
    assert not config.dflash_use_aux_hidden_state


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
        aux_hidden_state_capture="dflash",
        aux_hidden_state_layer_ids=[0, 1, 2],
    )
    torch.testing.assert_close(output.hidden_states, torch.cat(aux, dim=-1))
    torch.testing.assert_close(output.last_hidden_states, hidden_states)


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
        aux_hidden_state_capture="eagle3",
    )
    assert output.last_hidden_states is None
