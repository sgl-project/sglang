"""CPU regression coverage for NPU backend policies used by the common runner."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as mod
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as runner_mod
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=12, suite="base-a-test-npu")


def make_backend(*, dflash=False, dspark=False, width=1, bs=4, raw_bs=2):
    backend = mod.NPUCudaGraphBackend.__new__(mod.NPUCudaGraphBackend)
    attention = SimpleNamespace(forward_metadata=None)
    runner = runner_mod.DecodeCudaGraphRunner.__new__(runner_mod.DecodeCudaGraphRunner)
    runner.__dict__.update(
        raw_bs=raw_bs,
        bs=bs,
        captured_req_width=width,
        capture_forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
            ),
            spec_algorithm=SimpleNamespace(
                is_dflash=lambda: dflash,
                is_dspark=lambda: dspark,
                is_dflash_family=lambda: dflash or dspark,
            ),
        ),
        attn_backend=attention,
        _make_graph_key=lambda bs: runner_mod.ShapeKey(size=bs),
    )
    backend._decode_runner = runner
    return backend, runner, attention


def test_cpu_lengths_are_padded_without_reading_device_lengths():
    backend, _, _ = make_backend()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        seq_lens_cpu=torch.tensor([10, 11, 999]),
        seq_lens=mock.Mock(),
    )
    assert backend._decode_update_lengths(batch) == [10, 11, 0, 0]
    batch.seq_lens.cpu.assert_not_called()
    assert backend.cache_loc_dtype == torch.int32


@pytest.mark.parametrize(
    "dflash,dspark", [(False, False), (True, False), (False, True)]
)
def test_verify_cpu_lengths_include_block_only_once(dflash, dspark):
    backend, _, _ = make_backend(dflash=dflash, dspark=dspark, width=3)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        seq_lens_cpu=torch.tensor([13, 14] if dflash or dspark else [10, 11]),
    )
    assert backend._decode_update_lengths(batch) == [13, 14, 0, 0]


def test_dflash_replay_metadata_lengths_are_not_double_added():
    backend, _, attention = make_backend(dflash=True, width=3)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        seq_lens_cpu=torch.tensor([13, 14]),
    )
    attention.forward_metadata = SimpleNamespace(seq_lens_cpu_list=[13, 14, 0, 0])
    assert backend._decode_update_lengths(batch) == [13, 14, 0, 0]


def test_v2_update_key_comes_from_capture_protocol():
    backend, runner, _ = make_backend()
    runner.model_runner.model_config.hf_config.architectures = ["MiMoV2ForCausalLM"]
    assert backend._decode_update_attr_name() == "actual_seq_kvlen"


def test_dsa_replay_does_not_request_host_lengths():
    backend, _, _ = make_backend()
    graph = SimpleNamespace(replay=mock.Mock())
    backend._graphs = {"key": graph}
    backend._outputs = {"key": object()}
    with mock.patch.object(mod, "is_deepseek_dsa", return_value=True):
        result = backend.replay("key", object())
    assert result is backend._outputs["key"]
    graph.replay.assert_called_once_with()


def test_common_replay_builds_host_payload_but_dsa_skips_host_update():
    backend, _, _ = make_backend()
    received = []

    def updated_output(key, **kwargs):
        received.append((key, kwargs))
        return kwargs["seq_lens"]

    backend.replay_with_input_update = updated_output
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        seq_lens_cpu=torch.tensor([10, 11]),
    )
    with (
        mock.patch.object(mod, "is_deepseek_dsa", return_value=False),
        mock.patch.object(mod, "is_deepseek_v4", return_value=False),
        mock.patch.object(
            mod.envs.SGLANG_NPU_USE_FIAS_V2_BSND, "get", return_value=False
        ),
    ):
        assert backend.replay("variant-key", batch) == [10, 11, 0, 0]
    assert received[0] == (
        "variant-key",
        {
            "seq_lens": [10, 11, 0, 0],
            "attr_name": "actual_seq_lengths_kv",
            "attr_type": [],
        },
    )
    graph = SimpleNamespace(replay=mock.Mock())
    backend._graphs = {"draft-key": graph}
    backend._outputs = {"draft-key": "legacy-output"}
    with mock.patch.object(mod, "is_deepseek_dsa", return_value=True):
        assert backend.replay("draft-key", object()) == "legacy-output"
    assert len(received) == 1


def test_dflash_preparation_refreshes_lengths_and_cache_slots():
    backend, runner, attention = make_backend(dflash=True)
    runner.require_mlp_tp_gather = False
    runner.require_gathered_buffer = False
    runner.capture_bs = [4]
    runner._pad_to_bucket = lambda size, buckets: 4
    runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
    runner.model_runner.is_draft_worker = False
    runner.seq_len_fill_value = 0
    runner.is_encoder_decoder = False
    runner.buffers = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.int64),
        positions=torch.zeros(4, dtype=torch.int64),
        seq_lens=torch.full((4,), 99, dtype=torch.int32),
        seq_lens_cpu=torch.full((4,), 99, dtype=torch.int32),
        req_pool_indices=torch.full((4,), 99, dtype=torch.int32),
        out_cache_loc=torch.full((4,), 99, dtype=torch.int32),
    )
    seen_lengths = []
    attention.init_forward_metadata_out_graph = lambda view: seen_lengths.append(
        runner.buffers.seq_lens.clone()
    )
    batch = SimpleNamespace(
        needs_forward_metadata_init=lambda: False,
        batch_size=2,
        input_ids=torch.tensor([4, 5]),
        positions=torch.tensor([8, 9]),
        seq_lens_cpu=torch.tensor([10, 11], dtype=torch.int32),
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
        out_cache_loc=torch.tensor([20, 21], dtype=torch.int32),
        input_embeds=None,
        mrope_positions=None,
    )
    with (
        mock.patch(
            "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
            return_value=False,
        ),
        mock.patch.object(runner_mod, "build_replay_fb_view", return_value=object()),
        mock.patch.object(
            mod.envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM, "get", return_value=False
        ),
    ):
        assert backend.prepare_replay(batch)
    torch.testing.assert_close(
        seen_lengths[0], torch.tensor([10, 11, 0, 0], dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.buffers.out_cache_loc,
        torch.tensor([20, 21, 0, 0], dtype=torch.int32),
    )
    assert (runner.raw_bs, runner.bs, runner.raw_num_token) == (2, 4, 2)
    assert runner._replay_graph_key == runner_mod.ShapeKey(size=4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
