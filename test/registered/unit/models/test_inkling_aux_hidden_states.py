"""Inkling auxiliary capture keeps final states before LM-head-only scaling."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.arg_groups.layernorm_sp_hook import validate_layernorm_sp
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _inkling_forward(embedding_output):
    # Execute the production host method without importing its CUDA kernels.
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/models/inkling.py"
    model = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "InklingForConditionalGeneration"
    )
    method = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    scope = {
        "torch": torch,
        "general_mm_embed_routine": lambda **kwargs: embedding_output,
    }
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, method],
        type_ignores=[],
    )
    exec(compile(module, str(path), "exec"), scope)
    return scope["forward"]


@pytest.mark.parametrize("multiplier", [None, 1.0, 2.0])
@pytest.mark.parametrize(
    "capture_mode,capture_only,with_aux",
    [
        (CaptureHiddenMode.FULL, True, True),
        (CaptureHiddenMode.LAST, True, True),
        (CaptureHiddenMode.NULL, True, True),
        (CaptureHiddenMode.FULL, False, True),
        (CaptureHiddenMode.FULL, False, False),
    ],
)
def test_capture_preserves_unscaled_final_states(
    multiplier, capture_mode, capture_only, with_aux
):
    pre_norm = torch.arange(1, 21, dtype=torch.float32).reshape(5, 4)
    hidden_states = torch.nn.functional.rms_norm(pre_norm, (4,))
    aux = [torch.arange(20, dtype=torch.float32).reshape(5, 4)]
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    embedding_output = (hidden_states, aux) if with_aux else hidden_states
    forward = _inkling_forward(embedding_output)
    server_fields = (
        dict(
            enable_return_aux_hidden_states=True,
            speculative_algorithm="DFLASH",
            return_aux_hidden_state_layer_ids=[0],
        )
        if capture_only
        else {}
    )
    with published_topology(**server_fields):
        processor = LogitsProcessor(SimpleNamespace(vocab_size=3))
        model = SimpleNamespace(
            audio=None,
            visual=None,
            llm=SimpleNamespace(
                _dflash_layers_to_capture={0} if with_aux else set(),
                lm_head=None,
                logits_processor=processor,
            ),
            config=SimpleNamespace(
                text_config=SimpleNamespace(logits_mup_width_multiplier=multiplier)
            ),
        )
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=capture_mode,
            extend_seq_lens=torch.tensor([3, 2]),
            extend_seq_lens_cpu=[3, 2],
        )
        with patch.object(
            processor,
            "_get_logits",
            side_effect=lambda states, *args, **kwargs: states @ weight.T,
        ):
            output = forward(
                model, torch.zeros(5, dtype=torch.long), torch.arange(5), metadata
            )

    assert isinstance(output, LogitsProcessorOutput)
    logits_input = hidden_states if not multiplier else hidden_states / multiplier
    torch.testing.assert_close(
        output.next_token_logits, logits_input[[2, 4]] @ weight.T
    )
    if capture_only and capture_mode.is_full():
        torch.testing.assert_close(output.last_hidden_states, hidden_states)
    else:
        assert output.last_hidden_states is None
    if capture_mode == CaptureHiddenMode.NULL:
        assert output.hidden_states is None
    else:
        expected = torch.cat(aux, dim=-1) if with_aux else hidden_states
        if capture_mode.is_last():
            expected = expected[[2, 4]]
        torch.testing.assert_close(output.hidden_states, expected)


def test_inkling_rejects_sequence_sharded_layernorm():
    with pytest.raises(ValueError, match="only supported"):
        validate_layernorm_sp(
            architecture="InklingForConditionalGeneration",
            tp_size=2,
            attn_dp_enabled=False,
            speculative_algorithm=None,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
