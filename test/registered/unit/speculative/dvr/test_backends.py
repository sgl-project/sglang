from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.speculative.dvr.dflash as dflash_module
import sglang.srt.speculative.dvr.draft as self_draft_module
import sglang.srt.speculative.dvr.eagle as eagle_module
import sglang.srt.speculative.dvr.sampling as sampling_module
import sglang.srt.speculative.dvr.worker as worker_module
import sglang.srt.speculative.eagle_worker_v2 as eagle_worker_module
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dvr.worker import DecodeVerifyRollbackWorker
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@pytest.fixture(autouse=True)
def use_pytorch_sampling_backend(monkeypatch):
    monkeypatch.setattr(
        sampling_module,
        "get_server_args",
        lambda: SimpleNamespace(sampling_backend="pytorch"),
    )


def sampling_info(vocab_size):
    return SimpleNamespace(
        top_ks=torch.tensor([vocab_size]),
        top_ps=torch.tensor([1.0]),
        min_ps=torch.tensor([0.0]),
        temperatures=torch.ones((1, 1)),
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        is_all_greedy=False,
        apply_logits_bias=lambda _logits: None,
        sampling_seed=torch.tensor([2026]),
    )


def self_draft_owner(num_draft_tokens=2):
    worker = object.__new__(DecodeVerifyRollbackWorker)
    worker.num_draft_tokens = num_draft_tokens
    worker.num_draft_steps = num_draft_tokens - 1
    worker.topk = 1
    worker.model_runner = object()
    worker.chain_position_offsets = torch.arange(num_draft_tokens)
    worker.chain_retrieve_index = torch.arange(num_draft_tokens).view(1, -1)
    worker.chain_retrieve_next = torch.cat(
        (torch.arange(1, num_draft_tokens), torch.tensor([-1]))
    ).view(1, -1)
    worker.chain_retrieve_sibling = torch.full((1, num_draft_tokens), -1)
    return worker


def test_eagle_checks_loaded_draft_layers_not_target_config(monkeypatch):
    class LinearStateLayer:
        pass

    draft_modules = []

    class DraftWorker:
        def __init__(self, *_args, **_kwargs):
            self.draft_runner = SimpleNamespace(
                mambaish_config=object(),
                model=SimpleNamespace(modules=lambda: draft_modules),
            )

    monkeypatch.setattr(eagle_module, "DVREagleDraftWorker", DraftWorker)
    monkeypatch.setattr(eagle_module, "RadixLinearAttention", LinearStateLayer)
    monkeypatch.setattr(
        self_draft_module.envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM,
        "get",
        lambda: False,
    )
    server_args = SimpleNamespace(
        device="cpu",
        speculative_num_steps=1,
        speculative_num_draft_tokens=2,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=2)),
        max_running_requests=2,
        context_length=128,
    )
    server_args.override = lambda *_args, **values: vars(server_args).update(values)
    target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE,
            model_config=SimpleNamespace(context_len=128),
        ),
        get_memory_pool=lambda: (None, None),
    )
    worker_args = (server_args, 0, None, 29500, target_worker)

    assert DecodeVerifyRollbackWorker(*worker_args).draft_worker is not None
    draft_modules.append(LinearStateLayer())
    with pytest.raises(NotImplementedError, match="linear-attention draft-model state"):
        DecodeVerifyRollbackWorker(*worker_args)


def test_eagle_rejection_hook_delegates_to_upstream_sampling(monkeypatch):
    worker = object.__new__(eagle_worker_module.EagleDraftWorker)
    logits = torch.tensor([[1.0, 2.0]])
    info = sampling_info(2)
    proposal = (
        torch.tensor([[0.25, 0.75]]),
        torch.tensor([[0.75]]),
        torch.tensor([[1]]),
    )
    captured = {}

    def sample(next_token_logits, temperatures):
        captured["logits"] = next_token_logits
        captured["temperatures"] = temperatures
        return proposal

    monkeypatch.setattr(
        eagle_worker_module,
        "sample_draft_proposal",
        sample,
    )
    probs, topk_p, topk_index = worker._sample_rejection_proposal(
        logits,
        info,
        torch.tensor([64]),
        position_offset=1,
    )

    assert captured["logits"] is logits
    assert captured["temperatures"] is info.temperatures
    assert probs is proposal[0]
    assert topk_p is proposal[1]
    assert topk_index is proposal[2]


def test_eagle_proposal_uses_request_seed_and_position(monkeypatch):
    worker = object.__new__(eagle_module.DVREagleDraftWorker)
    worker.proposal_sampling_seeds = torch.tensor([2026, 2030])
    captured = {}

    def sample(probs, seeds, positions, *, position_offset):
        captured.update(
            probs=probs,
            seeds=seeds.clone(),
            positions=positions.clone(),
            position_offset=position_offset,
        )
        return torch.tensor([1, 0])

    monkeypatch.setattr(eagle_module, "dvr_sample_from_probs", sample)
    logits = torch.tensor([[0.25, 0.75], [0.60, 0.40]])
    probs, topk_p, topk_index = worker._sample_rejection_proposal(
        logits,
        SimpleNamespace(
            sampling_seed=None,
            temperatures=torch.ones((2, 1)),
            top_ks=torch.tensor([1, 1]),
            top_ps=torch.tensor([1.0, 1.0]),
            min_ps=torch.tensor([0.0, 0.0]),
            need_top_k_sampling=True,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
        ),
        torch.tensor([64, 127]),
        position_offset=1,
    )

    assert captured["probs"] is probs
    torch.testing.assert_close(probs, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert captured["seeds"].tolist() == [2026, 2030]
    assert captured["positions"].tolist() == [64, 127]
    assert captured["position_offset"] == 1
    assert topk_index.tolist() == [[1], [0]]
    torch.testing.assert_close(topk_p, probs.gather(1, topk_index))


def test_eagle_refreshes_graph_seeds_before_draft():
    calls = []
    seeds = torch.tensor([2026, 2030])
    worker = SimpleNamespace(
        set_sampling_seeds=lambda value: calls.append(("seed", value)),
        draft=lambda batch: calls.append(("draft", batch)) or "verify-input",
    )
    backend = eagle_module.EagleDraftBackend(SimpleNamespace(), worker)
    batch = SimpleNamespace(sampling_info=SimpleNamespace(sampling_seed=seeds))

    assert backend.propose(batch) == "verify-input"
    assert calls == [("seed", seeds), ("draft", batch)]


def test_dflash_draft_input_uses_request_allocation_as_host_bound():
    worker = object.__new__(dflash_module.DVRDFlashDraftWorker)
    worker.reserved_seq_lens_cpu = torch.empty(4, dtype=torch.int32)
    batch = SimpleNamespace(
        batch_size=lambda: 2,
        reqs=[
            SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=96)),
            SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=160)),
        ],
        seq_lens=torch.tensor([65, 129], dtype=torch.int32),
    )

    draft_input = worker.make_draft_input(
        batch, torch.tensor([7, 11], dtype=torch.int64)
    )

    assert draft_input.reserved_seq_lens_cpu.tolist() == [96, 160]
    assert draft_input.reserved_seq_lens_sum == 256
    assert draft_input.new_seq_lens.tolist() == [65, 129]


def test_dflash_cuda_graph_recapture_replaces_only_its_hook(monkeypatch):
    worker = object.__new__(dflash_module.DVRDFlashDraftWorker)
    unrelated_hook, old_hook, first_hook, second_hook = (
        object(),
        object(),
        object(),
        object(),
    )
    worker.capture_hooks = [old_hook]
    worker._draft_sampler = object()
    worker.draft_model_runner = SimpleNamespace(
        capture_tail_hooks=[unrelated_hook, old_hook]
    )
    new_hooks = iter((first_hook, second_hook))

    def capture_graph(draft_worker):
        draft_worker.draft_model_runner.capture_tail_hooks.append(next(new_hooks))

    monkeypatch.setattr(
        dflash_module.DFlashWorkerV2,
        "init_cuda_graphs",
        capture_graph,
    )

    worker.init_cuda_graphs()
    worker.init_cuda_graphs()

    assert worker.draft_model_runner.capture_tail_hooks == [
        unrelated_hook,
        second_hook,
    ]
    assert worker.capture_hooks == [second_hook]


def make_weight_backend(backend_type, *, target_path, draft_path):
    calls = []
    updater = SimpleNamespace(
        update_weights_from_disk=lambda *args, **kwargs: (
            calls.append(("disk", args, kwargs)) or (True, "ok")
        ),
        update_weights_from_ipc=lambda request: (
            calls.append(("ipc", request)) or (True, "ok")
        ),
    )
    runner = SimpleNamespace(weight_updater=updater)
    owner = SimpleNamespace(
        server_args=SimpleNamespace(
            model_path=target_path,
            load_format="target-format",
            speculative_draft_model_path=draft_path,
            speculative_draft_load_format="draft-format",
        )
    )
    if backend_type == "eagle":
        backend = eagle_module.EagleDraftBackend(
            owner, SimpleNamespace(draft_runner=runner)
        )
    else:
        backend = dflash_module.DFlashDraftBackend(
            owner, SimpleNamespace(draft_model_runner=runner)
        )
    return backend, calls


@pytest.mark.parametrize("backend_type", ["eagle", "dflash"])
@pytest.mark.parametrize("shared_checkpoint", [False, True])
def test_draft_weight_reload_uses_runner_weight_updater(
    backend_type, shared_checkpoint
):
    target_path = "target"
    draft_path = target_path if shared_checkpoint else "draft"
    backend, calls = make_weight_backend(
        backend_type, target_path=target_path, draft_path=draft_path
    )
    request = SimpleNamespace(
        model_path="updated-target",
        load_format="updated-format",
    )

    assert backend.update_weights_from_disk(request) == (True, "ok")

    expected = (
        ("updated-target", "updated-format")
        if shared_checkpoint
        else ("draft", "draft-format")
    )
    assert calls == [("disk", expected, {"recapture_cuda_graph": False})]


@pytest.mark.parametrize("backend_type", ["eagle", "dflash"])
@pytest.mark.parametrize("shared_checkpoint", [False, True])
def test_draft_ipc_reload_preserves_checkpoint_ownership(
    backend_type, shared_checkpoint
):
    target_path = "target"
    draft_path = target_path if shared_checkpoint else "draft"
    backend, calls = make_weight_backend(
        backend_type, target_path=target_path, draft_path=draft_path
    )
    request = SimpleNamespace(zmq_handles={"device": "handle"})

    assert backend.update_weights_from_ipc(request) == (True, "ok")

    if shared_checkpoint:
        assert calls == [("ipc", request)]
    else:
        assert calls == [
            (
                "disk",
                ("draft", "draft-format"),
                {"recapture_cuda_graph": False},
            )
        ]


def test_dflash_private_commit_keeps_target_verify_war_snapshot(monkeypatch):
    snapshot_done = object()
    runner = SimpleNamespace(war_fastpath_read_done_event=snapshot_done)
    commit_calls = []
    worker = SimpleNamespace(
        commit_accepted=lambda **kwargs: commit_calls.append(kwargs),
    )
    owner = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=runner),
    )
    backend = dflash_module.DFlashDraftBackend(owner, worker)
    backend.pending_draft_block = object()
    backend.context = nullcontext
    monkeypatch.setattr(dflash_module, "spec_stage_span", lambda _name: nullcontext())
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )
    batch_result = SimpleNamespace(
        logits_output=object(),
        accept_lens=torch.tensor([4], dtype=torch.int32),
        next_draft_input=SimpleNamespace(
            bonus_tokens=torch.tensor([17], dtype=torch.int64)
        ),
    )

    backend.commit_draft_state(batch, batch_result)

    assert len(commit_calls) == 1
    assert runner.war_fastpath_read_done_event is snapshot_done
    assert backend.pending_draft_block is None
    assert batch_result.next_draft_input.bonus_tokens.tolist() == [17]


def test_self_draft_copies_each_graph_proposal_before_next_replay(monkeypatch):
    worker = self_draft_owner(3)
    backend = worker_module.SelfDraftBackend(worker)
    backend.proposal_prob_buffer = torch.empty((1, 2, 3))
    worker.model_runner = SimpleNamespace(
        ngram_embedding_manager=SimpleNamespace(
            update_after_decode=lambda **_kwargs: None
        )
    )
    sampled_tokens = iter((torch.tensor([0]), torch.tensor([2])))
    monkeypatch.setattr(
        sampling_module,
        "dvr_sample_from_probs",
        lambda *_args: next(sampled_tokens),
    )
    static_logits = torch.empty((1, 3))
    per_step = iter(
        (
            torch.tensor([[0.6, 0.3, 0.1]]).log(),
            torch.tensor([[0.1, 0.2, 0.7]]).log(),
        )
    )

    def draft_forward(_batch):
        static_logits.copy_(next(per_step))
        return LogitsProcessorOutput(next_token_logits=static_logits)

    backend.decode_forward = draft_forward
    forward_batch = SimpleNamespace(
        spec_info=worker_module.EagleDraftInput(bonus_tokens=torch.tensor([1])),
        out_cache_loc=torch.arange(3),
        batch_size=1,
        seq_lens=torch.tensor([10]),
        seq_lens_cpu=torch.tensor([10]),
        seq_lens_sum=10,
        positions=torch.tensor([10]),
        sampling_info=sampling_info(3),
    )

    _, proposals = backend.draft_tokens(forward_batch)

    torch.testing.assert_close(proposals[0, 0], torch.tensor([0.6, 0.3, 0.1]))
    torch.testing.assert_close(proposals[0, 1], torch.tensor([0.1, 0.2, 0.7]))


def test_self_draft_restores_batch_output_flags(monkeypatch):
    worker = self_draft_owner()
    backend = worker_module.SelfDraftBackend(worker)
    backend.draft_tokens = lambda _batch: (_ for _ in ()).throw(
        RuntimeError("draft failed")
    )
    monkeypatch.setattr(
        self_draft_module.ForwardBatch,
        "init_new",
        lambda _batch, _runner, *, return_hidden_states_before_norm: object(),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=worker_module.EagleDraftInput(bonus_tokens=torch.tensor([1])),
        seq_lens=torch.tensor([4]),
        req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(8).view(1, 8)),
        return_logprob=True,
        return_hidden_states=True,
    )

    with pytest.raises(RuntimeError, match="draft failed"):
        backend.propose(batch)
    assert batch.return_logprob is True
    assert batch.return_hidden_states is True


def test_self_draft_positions_do_not_alias_sequence_lengths(monkeypatch):
    worker = self_draft_owner()

    def init_new(batch, _runner, *, return_hidden_states_before_norm):
        assert not return_hidden_states_before_norm
        assert batch.spec_info.positions.data_ptr() != batch.seq_lens.data_ptr()
        batch.spec_info.positions.add_(1)
        torch.testing.assert_close(batch.seq_lens, torch.tensor([4]))
        raise RuntimeError("stop after initialization")

    monkeypatch.setattr(self_draft_module.ForwardBatch, "init_new", init_new)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=worker_module.EagleDraftInput(bonus_tokens=torch.tensor([1])),
        seq_lens=torch.tensor([4]),
        req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(8).view(1, 8)),
        return_logprob=False,
        return_hidden_states=False,
    )

    with pytest.raises(RuntimeError, match="stop after initialization"):
        worker_module.SelfDraftBackend(worker).propose(batch)


@pytest.mark.parametrize(
    ("algorithm", "expected_hidden"),
    [
        (SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK, "NULL"),
        (SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_EAGLE, "FULL"),
        (SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK_DFLASH, "FULL"),
    ],
)
def test_dummy_verify_input_matches_draft_backend(algorithm, expected_hidden):
    spec_info = create_dummy_verify_input(
        spec_algorithm=algorithm,
        server_args=SimpleNamespace(
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        ),
        custom_mask=torch.ones(1, dtype=torch.bool),
        num_tokens_per_req=4,
        is_draft_worker=False,
    )

    assert spec_info is not None
    assert spec_info.capture_hidden_mode.name == expected_hidden
