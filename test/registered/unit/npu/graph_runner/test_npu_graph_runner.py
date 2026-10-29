"""CPU coverage for NPU decode-graph replay orchestration."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner as mod
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def test_replay_updates_sequence_lengths_and_slices_logits_to_raw_batch():
    runner = mod.NPUGraphRunner.__new__(mod.NPUGraphRunner)
    runner.require_mlp_tp_gather = False
    runner.capture_bs = [2]
    runner.captured_req_width = 1
    runner.deepep_adapter = SimpleNamespace(replay=mock.Mock())
    runner.buffers = SimpleNamespace(
        input_ids=torch.empty(2, dtype=torch.int64),
        positions=torch.empty(2, dtype=torch.int64),
        seq_lens=torch.empty(2, dtype=torch.int32),
        seq_lens_cpu=torch.empty(2, dtype=torch.int32),
        req_pool_indices=torch.empty(2, dtype=torch.int32),
    )
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
        is_draft_worker=False,
        model_config=SimpleNamespace(hf_config=object()),
    )
    runner.seq_len_fill_value = 0
    runner.capture_forward_mode = object()
    runner.is_encoder_decoder = False
    runner._pad_to_bucket = lambda raw_bs, buckets: raw_bs
    runner._make_graph_key = lambda bs: ("graph", bs)
    attention = SimpleNamespace(init_forward_metadata_out_graph=mock.Mock())
    runner._replay_attn_backend = lambda: attention
    runner.backend = SimpleNamespace(
        replay_with_input_update=mock.Mock(
            return_value=LogitsProcessorOutput(
                next_token_logits=torch.arange(3, dtype=torch.float32).view(3, 1),
                hidden_states=torch.arange(3, dtype=torch.float32).view(3, 1),
            )
        )
    )
    runner.is_dllm = False
    runner.if_use_v2 = False
    runner.use_fias_v2_bsnd = False
    runner.capture_forward_mode = SimpleNamespace(is_target_verify=lambda: False)
    runner._init_arch_map()

    forward_batch = SimpleNamespace(
        needs_forward_metadata_init=lambda: False,
        batch_size=2,
        input_ids=torch.tensor([4, 5]),
        positions=torch.tensor([8, 9]),
        seq_lens=torch.tensor([10, 11]),
        seq_lens_cpu=torch.tensor([10, 11], dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        out_cache_loc=None,
        input_embeds=None,
        mrope_positions=None,
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
    )

    with (
        mock.patch.object(mod, "enable_num_token_non_padded", return_value=False),
        mock.patch.object(
            mod.envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM, "get", return_value=False
        ),
        mock.patch.object(mod, "build_replay_fb_view", return_value="replay-view"),
        mock.patch.object(mod, "is_deepseek_dsa", return_value=False),
        mock.patch.object(mod, "is_deepseek_v4", return_value=False),
    ):
        output = runner.execute(forward_batch)

    runner.deepep_adapter.replay.assert_called_once_with()
    attention.init_forward_metadata_out_graph.assert_called_once_with("replay-view")
    runner.backend.replay_with_input_update.assert_called_once_with(
        ("graph", 2),
        seq_lens=[10, 11],
        attr_name=runner._get_update_attr_name(),
        attr_type=runner._get_update_attr_type(),
    )
    torch.testing.assert_close(output.next_token_logits, torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(output.hidden_states, torch.tensor([[0.0], [1.0]]))


def _make_runner(raw_bs, graph_bs, width):
    runner = mod.NPUGraphRunner.__new__(mod.NPUGraphRunner)
    runner.raw_bs = raw_bs
    runner.bs = graph_bs
    runner.raw_num_token = raw_bs * width
    runner.captured_req_width = width
    # The shared storage can be larger than the graph selected for this batch.
    capacity = (graph_bs + 1) * width
    runner.buffers = SimpleNamespace(
        input_ids=torch.full((capacity,), -1, dtype=torch.int64),
        positions=torch.full((capacity,), -1, dtype=torch.int64),
        pp_proxy_tensors={
            "hidden_states": torch.full((capacity, 8), -1.0),
            "residual": torch.full((capacity, 8), -2.0),
            "topk_indices": torch.full((capacity, 4), -3, dtype=torch.int32),
        },
    )
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_dflash=lambda: False),
        model_config=SimpleNamespace(hf_config=object()),
    )
    runner._make_graph_key = lambda bs: ("graph", bs)
    runner.load_batch = mock.Mock()
    runner.backend = SimpleNamespace(replay=mock.Mock())
    batch = SimpleNamespace(
        needs_forward_metadata_init=lambda: False,
        input_ids=torch.arange(runner.raw_num_token),
        positions=torch.arange(runner.raw_num_token) + 10,
        mrope_positions=None,
    )
    return runner, batch


@pytest.fixture
def _dsa_replay_path():
    # GLM-5.2 uses DSA, whose lengths are device tensors; no NPU graph.update
    # or accelerator kernel is needed to exercise the replay orchestration.
    with (
        mock.patch.object(mod, "is_deepseek_dsa", return_value=True),
        mock.patch.object(
            mod.envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM, "get", return_value=False
        ),
    ):
        yield


@pytest.mark.parametrize("raw_bs,graph_bs", [(1, 1), (2, 2), (3, 4)])
def test_preplanned_replay_refreshes_pp_inputs_in_place_before_each_replay(
    raw_bs, graph_bs, _dsa_replay_path
):
    runner, batch = _make_runner(raw_bs, graph_bs, width=4)
    buffers = runner.buffers.pp_proxy_tensors
    pointers = {name: buf.data_ptr() for name, buf in buffers.items()}
    previous = {name: buf.clone() for name, buf in buffers.items()}
    # Snapshot what the captured graph actually sees at replay time.
    runner.backend.replay.side_effect = lambda *args: PPProxyTensors(
        {name: buf.clone() for name, buf in buffers.items()}
    )
    rows = graph_bs * runner.captured_req_width

    for iteration in (1, 2):
        received = {
            name: torch.arange(rows * buf.shape[1], dtype=buf.dtype).reshape(
                rows, buf.shape[1]
            )
            + iteration * 100
            for name, buf in buffers.items()
        }
        # PP transport markers are not graph tensor inputs.
        proxy = PPProxyTensors({**received, "__msg_type__": "proxy"})
        output = runner.execute(batch, pp_proxy_tensors=proxy)

        for name, expected in received.items():
            assert buffers[name].data_ptr() == pointers[name]
            torch.testing.assert_close(buffers[name][:rows], expected)
            torch.testing.assert_close(output.tensors[name], expected)
            torch.testing.assert_close(buffers[name][rows:], previous[name][rows:])
        torch.testing.assert_close(
            runner.buffers.input_ids[: runner.raw_num_token], batch.input_ids
        )
        torch.testing.assert_close(
            runner.buffers.positions[: runner.raw_num_token], batch.positions
        )

    # Refreshing PP inputs must not re-plan attention or add a second load_batch.
    runner.load_batch.assert_not_called()
    assert runner.backend.replay.call_count == 2


@pytest.mark.parametrize("raw_bs,graph_bs", [(1, 1), (2, 2), (3, 4)])
@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("preplanned", [False, True])
def test_pp_output_preserves_all_token_rows(
    raw_bs, graph_bs, width, preplanned, _dsa_replay_path
):
    # width=1: ordinary decode; width=2/4: one/three-step MTP.
    runner, batch = _make_runner(raw_bs, graph_bs, width)
    batch.needs_forward_metadata_init = lambda: not preplanned
    graph_output = PPProxyTensors(
        {
            name: torch.arange(buf.numel(), dtype=buf.dtype).reshape(buf.shape)
            for name, buf in runner.buffers.pp_proxy_tensors.items()
        }
    )
    runner.backend.replay.return_value = graph_output

    # The first PP stage has no incoming proxy, including during pre-planned
    # verify; its output still carries every token in the padded graph batch.
    output = runner.execute(batch)

    rows = graph_bs * width
    for name, value in output.tensors.items():
        assert value.shape[0] == rows
        torch.testing.assert_close(value, graph_output.tensors[name][:rows])
    if preplanned:
        runner.load_batch.assert_not_called()
    else:
        runner.load_batch.assert_called_once_with(batch, None)
    runner.backend.replay.assert_called_once_with(("graph", graph_bs), batch)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
