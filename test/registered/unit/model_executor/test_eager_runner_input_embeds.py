from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _runner(*, capture_input_embeds: bool) -> tuple[EagerRunner, Mock]:
    runner = object.__new__(EagerRunner)
    model = Mock()
    runner.model_runner = SimpleNamespace(
        forward_input_embeds_to_decode=capture_input_embeds,
        model=model,
        attn_backend=Mock(),
        device_timer=None,
        _pp_kwargs=Mock(return_value={"pp_proxy_tensors": "proxy"}),
    )
    runner.enable_pdmux = False
    runner.load_batch = lambda forward_batch, _proxy: forward_batch
    return runner, model


def _forward_batch() -> SimpleNamespace:
    return SimpleNamespace(
        input_ids="ids",
        input_embeds="embeds",
        positions="positions",
        needs_forward_metadata_init=lambda: False,
    )


def test_eager_decode_passes_captured_input_embeddings() -> None:
    runner, model = _runner(capture_input_embeds=True)
    forward_batch = _forward_batch()

    runner._execute_decode(forward_batch, pp_proxy_tensors="proxy")

    model.forward.assert_called_once_with(
        "ids",
        "positions",
        forward_batch,
        pp_proxy_tensors="proxy",
        input_embeds="embeds",
    )


def test_eager_decode_keeps_default_input_id_path() -> None:
    runner, model = _runner(capture_input_embeds=False)
    forward_batch = _forward_batch()

    runner._execute_decode(forward_batch, pp_proxy_tensors="proxy")

    model.forward.assert_called_once_with(
        "ids",
        "positions",
        forward_batch,
        pp_proxy_tensors="proxy",
    )


@pytest.mark.parametrize("forward_input_embeds_to_decode", [False, True])
def test_dp_padding_gates_composed_input_embeddings(
    forward_input_embeds_to_decode: bool,
) -> None:
    forward_batch = object.__new__(ForwardBatch)
    forward_batch.input_ids = torch.tensor([10, 11])
    forward_batch.input_embeds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    forward_batch.req_pool_indices = torch.tensor([0])
    forward_batch.lora_ids = None
    forward_batch.seq_lens_sum = 2
    forward_batch.seq_lens = torch.tensor([2])
    forward_batch.seq_lens_cpu = None
    forward_batch.out_cache_loc = torch.tensor([20, 21])
    forward_batch.encoder_lens = None
    forward_batch.positions = torch.tensor([0, 1])
    forward_batch.mamba_track_indices = None
    forward_batch.mamba_track_mask = None
    forward_batch.mamba_track_seqlens = None
    forward_batch.mrope_positions = None
    forward_batch.extend_seq_lens = None
    forward_batch.rids_int = None
    forward_batch.sampling_info = None
    forward_batch.bootstrap_room_ids_int = None
    forward_batch.spec_info = None
    original_input_embeds = forward_batch.input_embeds
    model_runner = SimpleNamespace(
        forward_input_embeds_to_decode=forward_input_embeds_to_decode,
        attn_backend=SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: 0),
    )

    forward_batch._pad_inputs_to_size(model_runner, num_tokens=4, bs=2)

    if forward_input_embeds_to_decode:
        assert torch.equal(
            forward_batch.input_embeds,
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]),
        )
    else:
        assert forward_batch.input_embeds is original_input_embeds


def test_tp_worker_invokes_forward_batch_customization_hook() -> None:
    worker = object.__new__(TpModelWorker)
    worker.set_hicache_consumer = Mock()
    worker.customize_forward_batch = Mock()
    worker.is_dllm = Mock(return_value=False)
    worker.pp_group = SimpleNamespace(is_last_rank=False)
    worker._model_runner = SimpleNamespace(
        forward=Mock(
            return_value=SimpleNamespace(
                logits_output="proxy",
                can_run_graph=False,
                expert_distribution_metrics=None,
            )
        )
    )
    schedule_batch = SimpleNamespace(hicache_consumer_index=0)
    forward_batch = SimpleNamespace(apply_deprecated_skip_attn_backend_init=Mock())

    with patch.object(ForwardBatch, "init_new", return_value=forward_batch):
        worker.forward_batch_generation(schedule_batch)

    worker.customize_forward_batch.assert_called_once_with(
        schedule_batch, forward_batch
    )
    worker.model_runner.forward.assert_called_once_with(
        forward_batch, pp_proxy_tensors=None
    )
