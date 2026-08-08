from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.speculative.dvr.worker as dvr_worker_module
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.dvr.worker import (
    DecodeVerifyRollbackWorker,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize(
    "capture_mode",
    [
        CaptureHiddenMode.NULL,
        CaptureHiddenMode.FULL,
    ],
)
def test_short_prefix_uses_one_root_verify_sentinel(capture_mode):
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.draft_backend = SimpleNamespace(
        target_capture_hidden_mode=capture_mode,
    )
    worker.device = "cpu"
    worker.num_draft_tokens = 4
    worker.chain_retrieve_index = torch.arange(8).view(2, 4)
    worker.chain_retrieve_sibling = torch.full((2, 4), -1)
    worker.chain_position_offsets = torch.arange(4)
    batch = SimpleNamespace(
        spec_info=dvr_worker_module.EagleDraftInput(bonus_tokens=torch.tensor([6, 7])),
        seq_lens=torch.tensor([1, 65]),
        seq_lens_cpu=torch.tensor([1, 65]),
        seq_lens_sum=66,
    )

    verify_input = worker.build_root_only_verify_input(batch)

    assert verify_input.draft_token.tolist() == [6] * 4 + [7] * 4
    assert verify_input.spec_steps == 0
    assert verify_input.retrieve_next_token.eq(-1).all()
    assert verify_input.positions.tolist() == [1, 2, 3, 4, 65, 66, 67, 68]


def test_short_prefix_tracks_only_new_prefill_requests():
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.seed_verify_slots = set()
    worker.draft_backend = SimpleNamespace(
        requires_short_prompt_verify=True,
        target_capture_hidden_mode=CaptureHiddenMode.NULL,
        finish_prefill=lambda _batch, _result: "next-draft",
    )
    worker.state_lifecycle = SimpleNamespace(
        prepare_target_extend=lambda _batch: None,
        finish_target_extend=lambda _batch: None,
        prepare_for_cache_release=lambda _req: None,
    )
    worker.target_model_worker = SimpleNamespace(
        forward_batch_generation=lambda _batch, **_kwargs: SimpleNamespace()
    )
    new_req = SimpleNamespace(rid="new", req_pool_idx=1, origin_input_ids=[7])
    running_req = SimpleNamespace(rid="running", req_pool_idx=2, origin_input_ids=[8])
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=True,
        reqs=[new_req, running_req],
        decoding_reqs=[running_req],
        seq_lens=torch.tensor([1, 1]),
    )

    result = worker.forward_batch_generation(batch)

    assert worker.seed_verify_slots == {1}
    assert result.next_draft_input == "next-draft"
    worker.prepare_for_kv_cache_release(new_req)
    assert not worker.seed_verify_slots


def test_short_prefix_mixed_batch_masks_only_seed_row_during_sampling():
    calls = []
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.device = "cpu"
    worker.num_draft_tokens = 4
    worker.seed_verify_slots = {1}
    worker.chain_retrieve_index = torch.arange(8).view(2, 4)
    worker.chain_retrieve_sibling = torch.full((2, 4), -1)
    worker.chain_position_offsets = torch.arange(4)
    verify_input = dvr_worker_module.EagleVerifyInput.create_idle_input(1, 3, 4, "cpu")
    verify_input.draft_token = torch.arange(8)
    verify_input.positions = torch.arange(8)
    verify_input.retrieve_index = worker.chain_retrieve_index
    verify_input.retrieve_next_token = torch.tensor([[1, 2, 3, -1], [5, 6, 7, -1]])
    verify_input.retrieve_next_sibling = worker.chain_retrieve_sibling
    verify_input.seq_lens_cpu = torch.tensor([1, 65])
    verify_input.seq_lens_sum = 66
    model_runner = SimpleNamespace(war_fastpath_read_done_event=object())
    worker.draft_backend = SimpleNamespace(
        context=nullcontext,
        idle_input=lambda: None,
        propose=lambda _batch: calls.append("draft") or verify_input,
        commit_draft_state=lambda _batch, _result: calls.append("finish"),
        war_fastpath_runner=model_runner,
    )
    worker.state_lifecycle = SimpleNamespace(
        prepare_rollback=lambda _batch: None,
    )
    worker.target_model_worker = SimpleNamespace(
        model_runner=model_runner,
    )
    worker.rollback_done_event = None

    def verify(_batch, _input, **kwargs):
        calls.append(("mask", kwargs["root_only_mask"].tolist()))
        model_runner.war_fastpath_read_done_event = object()
        return SimpleNamespace()

    worker.verify = verify
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            penalizer_orchestrator=SimpleNamespace(is_required=False),
        ),
        spec_info=SimpleNamespace(),
        reqs=[
            SimpleNamespace(req_pool_idx=1),
            SimpleNamespace(req_pool_idx=2),
        ],
        device="cpu",
        batch_size=lambda: 2,
    )

    worker.forward_batch_generation(batch)

    assert calls == ["draft", ("mask", [True, False]), "finish"]
    assert not worker.seed_verify_slots


@pytest.mark.parametrize(
    ("additive_penalties", "scaling_penalties", "orchestrator_required"),
    [
        (torch.zeros(1, 2), None, False),
        (None, torch.ones(1, 2), False),
        (None, None, True),
    ],
)
def test_dvr_rejects_dynamic_token_penalties(
    additive_penalties, scaling_penalties, orchestrator_required
):
    worker = object.__new__(DecodeVerifyRollbackWorker)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(
            acc_additive_penalties=additive_penalties,
            acc_scaling_penalties=scaling_penalties,
            penalizer_orchestrator=SimpleNamespace(is_required=orchestrator_required),
        ),
    )

    with pytest.raises(ValueError, match="dynamic token penalties"):
        worker.forward_batch_generation(batch)


@pytest.mark.parametrize("updates_draft", [False, True])
def test_weight_update_recaptures_the_complete_dvr_draft_graph_set(
    updates_draft,
):
    calls = []
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.draft_graph_buffers = {"stale": object()}
    worker.init_cuda_graphs = lambda: calls.append(("recapture",))

    def update_draft_weights(model_path, load_format, **kwargs):
        calls.append(("load", model_path, load_format, kwargs))
        return True, "loaded"

    backend = SimpleNamespace(
        graph_runner=object(),
        update_weights_from_disk=(
            lambda req: (
                update_draft_weights(
                    req.model_path,
                    req.load_format,
                    recapture_cuda_graph=False,
                )
                if updates_draft
                else (True, "loaded")
            )
        ),
        reset_cuda_graphs=lambda: (
            calls.append(("reset",)),
            setattr(backend, "graph_runner", None),
        ),
    )
    worker.draft_backend = backend
    request = SimpleNamespace(
        model_path="updated-model",
        load_format="auto",
        recapture_cuda_graph=True,
    )

    success, _ = worker.update_weights_from_disk(request)

    assert success
    if updates_draft:
        assert calls == [
            (
                "load",
                "updated-model",
                "auto",
                {"recapture_cuda_graph": False},
            ),
            ("reset",),
            ("recapture",),
        ]
    else:
        assert calls == [("reset",), ("recapture",)]
    assert worker.draft_backend.graph_runner is None
    assert not worker.draft_graph_buffers


def test_cache_release_waits_for_pending_dvr_rollback(monkeypatch):
    calls = []
    read_done = object()
    state_done = object()
    stream = SimpleNamespace(wait_event=lambda value: calls.append(("wait", value)))
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: SimpleNamespace(current_stream=lambda: stream),
    )
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.device = "cuda"
    worker.seed_verify_slots = {3}
    runner = SimpleNamespace(war_fastpath_read_done_event=read_done)
    worker.target_model_worker = SimpleNamespace(model_runner=runner)
    worker.draft_backend = SimpleNamespace(war_fastpath_runner=runner)
    worker.rollback_done_event = state_done
    worker.state_lifecycle = SimpleNamespace(
        prepare_for_cache_release=lambda req: calls.append(("release", req.rid))
    )
    req = SimpleNamespace(rid="done", req_pool_idx=3)

    worker.prepare_for_kv_cache_release(req)

    assert calls == [
        ("wait", read_done),
        ("wait", state_done),
        ("release", "done"),
    ]
    assert not worker.seed_verify_slots


def test_verify_publishes_lengths_before_commit_and_preserves_war_snapshot(
    monkeypatch,
):
    calls = []

    class Event:
        def record(self):
            calls.append("record_commit")

    class FillBonusTokens:
        def __getitem__(self, _grid):
            def run(tokens, _accept_lens, output, _width):
                calls.append("fill_bonus")
                output.copy_(tokens.reshape(-1)[: output.shape[0]])

            return run

    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda _device: SimpleNamespace(
            Event=Event, current_stream=lambda: SimpleNamespace()
        ),
    )
    monkeypatch.setattr(
        dvr_worker_module,
        "eagle_prepare_for_verify",
        lambda *_args, **_kwargs: (SimpleNamespace(), False),
    )
    monkeypatch.setattr(dvr_worker_module, "fill_bonus_tokens", FillBonusTokens())

    snapshot_done = object()
    runner = SimpleNamespace(war_fastpath_read_done_event=None)
    logits_output = SimpleNamespace(
        next_token_logits=torch.tensor([[1.0, 0.0]]),
        hidden_states=None,
    )

    def target_forward(**_kwargs):
        runner.war_fastpath_read_done_event = snapshot_done
        return SimpleNamespace(
            logits_output=logits_output,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.device = "cpu"
    worker.num_draft_tokens = 2
    worker.verify_plan_stream = None
    worker.req_to_token_pool = object()
    worker.target_model_worker = SimpleNamespace(
        model_runner=runner,
        forward_batch_generation=target_forward,
    )
    worker.draft_backend = SimpleNamespace(
        prepare_target_verify=lambda batch, spec_info: setattr(
            batch, "seq_lens_cpu_cache", spec_info.seq_lens_cpu
        ),
        finish_target_verify_prepare=lambda _batch, _stream: None,
        validate_target_output=lambda _output: None,
    )
    worker.state_lifecycle = SimpleNamespace(
        rollback=lambda **_kwargs: calls.append("commit")
    )
    worker.sample_verified_tokens = lambda *_args: (
        torch.tensor([7, 8], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([64], dtype=torch.int32),
        seq_lens_cpu_cache=None,
        forward_mode=ForwardMode.DECODE,
        return_logprob=False,
    )
    spec_info = SimpleNamespace(
        is_verify_input=lambda: True,
        draft_token_num=2,
        num_tokens_per_req=None,
        custom_mask=object(),
        seq_lens_cpu=torch.tensor([64], dtype=torch.int32),
        spec_steps=1,
    )

    worker.verify(
        batch,
        spec_info,
        rollback_plan=object(),
        on_publish=lambda _seq_lens: calls.append("publish"),
    )

    assert calls == ["fill_bonus", "publish", "commit", "record_commit"]
    assert runner.war_fastpath_read_done_event is snapshot_done
    assert worker.rollback_done_event is not snapshot_done
