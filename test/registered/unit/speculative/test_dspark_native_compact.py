"""Native compact must keep the strided fallback and defer KV-pool lookup."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_constructor_does_not_resolve_uninitialized_pool(monkeypatch):
    monkeypatch.setenv("SGLANG_DSPARK_COMPACT_NATIVE_ACCEPT", "1")
    monkeypatch.setenv("SGLANG_DSPARK_COMPACT_POST_FOLD", "1")
    monkeypatch.setattr(torch, "device", lambda _: SimpleNamespace(type="cuda"))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    resolve_pool = Mock(side_effect=AssertionError("draft KV pool not initialized"))
    epilogue = dspark_verify.DsparkVerifyEpilogue(
        max_bs=2,
        verify_num_draft_tokens=3,
        device="cpu",
        tp_sync=SimpleNamespace(enabled=lambda _: False),
        commit_ctx=SimpleNamespace(resolve_pool=resolve_pool),
    )
    assert epilogue.native_compact
    assert epilogue.native_post_fold
    resolve_pool.assert_not_called()


def test_capture_falls_back_for_unsupported_pool():
    calls = []

    class Epilogue:
        native_compact = True
        native_post_fold = True
        folds_commit = False

        def __call__(self, **kwargs):
            calls.append(kwargs)

    epilogue = Epilogue()
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(is_draft_worker=False), ragged_verify_mode=True
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        input_ids=torch.tensor([1, 2]),
        seq_lens=torch.tensor([10]),
        req_pool_indices=torch.tensor([1]),
        batch_size=1,
    )
    output = LogitsProcessorOutput(
        next_token_logits=torch.zeros(2, 8), hidden_states=torch.zeros(2, 4)
    )
    dspark_verify.DsparkVerifyEpilogue.capture_hook(
        epilogue, runner, output, forward_batch, 2
    )
    assert not epilogue.native_compact
    assert not epilogue.native_post_fold
    assert len(calls) == 1
    assert calls[0]["compact_logits"] is output.next_token_logits


@pytest.mark.parametrize("armed,graph", [(True, True), (False, True), (True, False)])
def test_packed_result_only_when_native_graph_accept_is_armed(
    monkeypatch, armed, graph
):
    monkeypatch.setenv("SGLANG_DSPARK_COMPACT_DIRECT_INPUTS", "0")
    logits = torch.arange(40, dtype=torch.float32).view(5, 8)
    hidden = torch.arange(20, dtype=torch.float32).view(5, 4)
    output = LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)
    layout = SimpleNamespace(verify_lens=torch.tensor([2, 3]), graph_num_tokens=5)
    window = object()
    monkeypatch.setattr(
        dspark_verify.BuildRaggedVerifyWindow, "execute", Mock(return_value=window)
    )
    adjustments = Mock()
    monkeypatch.setattr(dspark_verify, "apply_logits_adjustments_strided", adjustments)
    scatter_calls = []

    def scatter(*, compact, layout, fill_value, verify_num_draft_tokens):
        scatter_calls.append(compact)
        assert layout.verify_lens.tolist() == [2, 3]
        assert verify_num_draft_tokens == 3
        dense = compact.new_full((6, compact.shape[1]), fill_value)
        dense[:2] = compact[:2]
        dense[3:] = compact[2:]
        return dense

    monkeypatch.setattr(dspark_verify.ScatterCompactToStrided, "execute", scatter)
    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.model_runner = SimpleNamespace(decode_cuda_graph_runner=None)
    executor.verify_num_draft_tokens = 3
    executor.verify_epilogue = SimpleNamespace(native_compact=True, begin_step=Mock())
    executor._run_ragged = Mock(
        return_value=dspark_verify.TargetVerifyResult(
            logits_output=output, can_run_cuda_graph=graph
        )
    )
    result, strided_hidden = executor.run_compact(
        batch=object(),
        layout=layout,
        draft_block_ids=torch.zeros(2, 1, dtype=torch.int64),
        draft_tokens=torch.zeros(2, 2, dtype=torch.int64),
        bs=2,
        device="cpu",
        sampling_info=None,
        verify_window=SimpleNamespace(iteration_ids=None),
        inject_gate=armed,
    )
    executor.verify_epilogue.begin_step.assert_called_once_with(
        layout.verify_lens, armed=armed
    )
    assert executor._run_ragged.call_args.kwargs["ragged_window"] is window
    assert result.native_compact == (armed and graph)
    if armed and graph:
        assert not scatter_calls
        adjustments.assert_not_called()
        assert result.logits_output.next_token_logits is logits
        assert strided_hidden is None
    else:
        assert len(scatter_calls) == 2
        adjustments.assert_called_once()
        assert result.logits_output.next_token_logits.shape == (6, 8)
        torch.testing.assert_close(strided_hidden[:2], hidden[:2])
        torch.testing.assert_close(strided_hidden[3:], hidden[2:])
        assert not strided_hidden[2].any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
