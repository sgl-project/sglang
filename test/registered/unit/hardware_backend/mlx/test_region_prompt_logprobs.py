"""One fixed graph must follow live score starts, lengths and target IDs."""

import dataclasses
from unittest import mock

import pytest
import torch
from test_region_shared_logits import Body, _graph, _runner

from sglang.srt.hardware_backend.mlx.export_validation import (
    ServingForwardArg,
    ServingForwardExportWrapper,
    serving_forward_args,
)
from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxLoweringRegistry,
    build_mlx_fx_plan,
    make_mlx_fx_executor,
)
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_mps_ci(est_time=2, suite="stage-a-unit-test-mps")


def batch(runner, length, start, prefix=0):
    value = runner._synthetic_batch(("extend", length))
    return dataclasses.replace(
        value,
        return_logprob=True,
        extend_seq_lens_cpu=[length],
        extend_prefix_lens_cpu=[prefix],
        extend_prefix_lens=torch.tensor(
            [prefix], dtype=torch.int32, device=value.input_ids.device
        ),
        extend_logprob_start_lens_cpu=[start],
        extend_input_logprob_token_ids_gpu=(
            torch.arange(length - start, device=value.input_ids.device) + 11
        )
        % 61,
        top_logprobs_nums=[0],
        token_ids_logprobs=[None],
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("scale,fp32", [(None, False), (0.25, False), (None, True)])
def test_one_graph_follows_live_prompt_score_layout(device, fast, scale, fp32):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    with get_context().override_server_args():
        torch.manual_seed(73)
        runner = _runner(device, scale, fp32)
        model = runner.model_runner.model
        model.logits_processor.input_logprob_processor.enable_fast_input_logprobs = fast
        hidden = torch.randn(8, 8).to(device, torch.bfloat16)
        model.body = Body(hidden)
        captured = runner._synthetic_batch(("extend_logprob", 8))
        wrapper = ServingForwardExportWrapper(model, runner.model_runner, captured)
        inputs = serving_forward_args(captured)
        graph = _graph(wrapper, inputs)
        plan = build_mlx_fx_plan(graph, MlxFxLoweringRegistry.standard_export_decoder())
        plan.require_fully_supported()
        execute = make_mlx_fx_executor(plan, list(inputs)) if device == "mps" else graph
        runner._executors[("extend_logprob", 8)] = type(
            "Executor", (), {"execute": staticmethod(execute)}
        )()
        for length, start, prefix in [(5, 0, 0), (5, 2, 0), (3, 1, 17), (7, 6, 0)]:
            live = batch(runner, length, start, prefix)
            assert runner._executor_key(live) == ("extend_logprob", 8)
            metadata = LogitsMetadata(
                forward_mode=live.forward_mode,
                extend_return_logprob=True,
                extend_seq_lens_cpu=[length],
                extend_logprob_start_lens_cpu=[start],
                extend_logprob_pruned_lens_cpu=[length - start],
                extend_input_logprob_token_ids_gpu=live.extend_input_logprob_token_ids_gpu,
            )
            expected = model.logits_processor(
                live.input_ids, hidden[:length], model.lm_head, metadata
            )
            with mock.patch.object(
                model.logits_processor,
                "forward",
                side_effect=AssertionError("no eager processor after launch"),
            ):
                actual = runner.execute(live)
            assert actual.next_token_logits.shape == (1, 61)
            assert actual.input_token_logprobs.shape == (length - start,)
            assert actual.input_token_logprobs.dtype == torch.float32
            torch.testing.assert_close(
                actual.next_token_logits.cpu(),
                expected.next_token_logits.cpu(),
                atol=0.008,
                rtol=0.03,
            )
            torch.testing.assert_close(
                actual.input_token_logprobs.cpu(),
                expected.input_token_logprobs.cpu(),
                atol=0.01,
                rtol=0,
            )


def test_scoring_signature_contains_only_live_tensor_layout():
    with get_context().override_server_args():
        runner = _runner("cpu", None, False)
        first = serving_forward_args(runner._pad_extend_batch(batch(runner, 5, 0), 8))
        changed = serving_forward_args(runner._pad_extend_batch(batch(runner, 5, 2), 8))
        assert first[ServingForwardArg.LOGPROB_PRUNED_INDICES].tolist() == [
            0,
            1,
            2,
            3,
            4,
            4,
            4,
            4,
        ]
        assert changed[ServingForwardArg.LOGPROB_PRUNED_INDICES].tolist() == [
            2,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
        ]
        assert changed[ServingForwardArg.LOGPROB_TOKEN_IDS].tolist() == [
            11,
            12,
            13,
            0,
            0,
            0,
            0,
            0,
        ]
        assert (
            first[ServingForwardArg.LOGPROB_TOKEN_IDS].shape
            == changed[ServingForwardArg.LOGPROB_TOKEN_IDS].shape
        )


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": 2},
        {"top_logprobs_nums": [3]},
        {"token_ids_logprobs": [[2]]},
        {"extend_logprob_start_lens_cpu": [-1]},
        {"extend_logprob_start_lens_cpu": [6]},
        {"extend_input_logprob_token_ids_gpu": None},
        {"extend_input_logprob_token_ids_gpu": torch.zeros(5, dtype=torch.float32)},
        {"extend_input_logprob_token_ids_gpu": torch.zeros(3, dtype=torch.int64)},
        {"token_indices_to_pool": torch.tensor([1])},
        {"multi_item_delimiter_indices": torch.tensor([1])},
    ],
)
def test_unsupported_scoring_metadata_keeps_eager_fallback(change):
    with get_context().override_server_args():
        runner = _runner("cpu", None, False)
        assert (
            runner._executor_key(dataclasses.replace(batch(runner, 5, 0), **change))
            is None
        )


def test_score_capture_is_bounded_by_shared_chunk_size():
    with get_context().override_server_args():
        runner = _runner("cpu", None, False)
        processor = runner.model_runner.model.logits_processor.input_logprob_processor
        processor.logprobs_chunk_size = 16
        assert runner._logprob_prefill_keys() == [("extend_logprob", 8)]
        assert runner._executor_key(batch(runner, 17, 0)) is None
        assert runner._executor_key(batch(runner, 5, 5)) == ("extend", 8)
        assert runner._executor_key(batch(runner, 0, 0)) is None
