"""Real PP transport with production worker phases and deterministic model edges."""

import multiprocessing as mp
import tempfile
import time
from array import array
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    DraftProposal,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_pp_coordinator import (
    DSparkPPCoordinator,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_inject import (
    ProjectedContextState,
    TargetKVInjector,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkDecodeStep,
    DSparkWorkerV2,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.dspark_shared_modules_utils import make_groups

WORKER = "sglang.srt.speculative.dspark_components.dspark_worker_v2"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"


class StageGroup:
    all_gather_object = GroupCoordinator.all_gather_object
    broadcast = GroupCoordinator.broadcast
    send_object = GroupCoordinator.send_object
    recv_object = GroupCoordinator.recv_object
    recv_tensor_dict = GroupCoordinator.recv_tensor_dict

    def __init__(self, group):
        self.__dict__.update(vars(group))
        self.bad_frame = False

    def send_tensor_dict(self, tensor_dict, **kwargs):
        if self.bad_frame:
            tensor_dict = dict(tensor_dict)
            signature, stage = tensor_dict["frame"]
            signature = (signature[0], signature[1] + 1, *signature[2:])
            tensor_dict["frame"] = (signature, stage)
        return GroupCoordinator.send_tensor_dict(self, tensor_dict, **kwargs)


def fixture(world, tp, pp, device, *, failure=None, sampling=False):
    rank, stage = world.rank_in_group, pp.rank_in_group
    owner = (pp.world_size - 1) * tp.world_size
    state = SimpleNamespace(events=[], committed={}, forward_done=False, inputs=None)
    lengths = torch.tensor([5, 9], device=device)
    slots = torch.tensor([[7, 3, 1, 9], [8, 2, 6, 4]], device=device) + rank * 100
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                rid=f"request-{i}",
                kv_committed_len=n,
                origin_input_ids=array("I", range(n)),
                output_ids=[10 * (i + 1)],
                dspark_projected_context=ProjectedContextState("fixture:0", n),
            )
            for i, n in enumerate((5, 9))
        ],
        spec_algorithm=SpeculativeAlgorithm.DSPARK,
        forward_mode=ForwardMode.DECODE,
        return_logprob=False,
        return_hidden_states=False,
        seq_lens=lengths,
        seq_lens_cpu=lengths.cpu().clone(),
        seq_lens_sum=14,
        input_ids=torch.arange(14, device=device),
        prefix_lens=[0, 0],
        extend_lens=[5, 9],
        spec_info=make_next_draft_input(
            bonus_tokens=torch.tensor([10, 20], device=device), new_seq_lens=lengths
        ),
        req_pool_indices=torch.tensor([3, 1], device=device) + rank * 10,
        has_grammar=True,
        forward_iter=0,
        spec_verify_tier_num_tokens=8,
        global_num_tokens=None,
    )

    def forward(batch=None, *, forward_batch=None, pp_proxy_tensors=None, **kwargs):
        if failure == "forward" and stage == 1 and tp.rank_in_group == 0:
            raise RuntimeError("injected target forward failure")
        inputs = batch.input_ids if forward_batch is None else forward_batch.input_ids
        if stage == 0:
            assert pp_proxy_tensors is None
            hidden = inputs.float()[:, None].repeat(1, 2)
        else:
            hidden = pp_proxy_tensors["hidden_states"]
            expected = inputs.float()[:, None].repeat(1, 2) + stage * (stage + 1) / 2
            torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
        hidden = hidden + stage + 1
        if failure == "activation_rows" and stage == 0:
            hidden = hidden[:1]
        state.inputs = inputs.clone()
        state.forward_done = True
        state.events.append("forward")
        final = stage == pp.world_size - 1
        return GenerationBatchResult(
            logits_output=(
                LogitsProcessorOutput(next_token_logits=hidden.repeat(1, 32))
                if final
                else None
            ),
            next_token_ids=(
                torch.tensor(
                    [64, 20] if failure == "prefill_sample" else [10, 20], device=device
                )
                + tp.rank_in_group
                if final and forward_batch is None
                else None
            ),
            pp_hidden_states_proxy_tensors=(
                None if final else PPProxyTensors({"hidden_states": hidden})
            ),
            can_run_cuda_graph=False,
        )

    def commit(*, verify_window=None, commit_lens=None, **kwargs):
        assert all(world.all_gather_object(state.forward_done))
        if commit_lens is not None:
            values = state.inputs.view(2, 4)
            for row, count in enumerate(commit_lens.tolist()):
                for col in range(count):
                    slot = int(verify_window.verify_cache_loc_2d[row, col])
                    state.committed[slot] = int(values[row, col])
                req = kwargs["batch"].reqs[row]
                req.dspark_projected_context = ProjectedContextState(
                    injector.weight_version,
                    int(kwargs["batch"].seq_lens_cpu[row]) + count,
                )
        state.events.append("commit")

    def capture_forward(*args, **kwargs):
        state.events.append("capture-forward")
        return "raw-ticket"

    def capture_accept(*args, **kwargs):
        state.events.append("capture-accept")
        return "accepted-ticket"

    target = SimpleNamespace(
        forward_batch_generation=forward,
        training_capture=SimpleNamespace(
            after_verify_forward=capture_forward, after_verify_accept=capture_accept
        ),
    )
    injector = TargetKVInjector.__new__(TargetKVInjector)
    injector.weights_digest = (
        "fixture-other" if failure == "weights" and rank == 0 else "fixture"
    )
    injector.epoch = 0
    injector.invalidation_ct = 0

    def ensure_context(batch):
        starts = [
            req.dspark_projected_context.end if req.dspark_projected_context else 0
            for req in batch.reqs
        ]
        ends = batch.seq_lens_cpu.tolist()
        if starts != ends:
            state.projection_starts = starts
        for req, start, end in zip(batch.reqs, starts, ends, strict=True):
            if start != end:
                copies = world.all_gather_object((req.rid, start, end))
                assert all(item == (req.rid, start, end) for item in copies)
                req.dspark_projected_context = ProjectedContextState(
                    injector.weight_version, end
                )
        if batch.forward_mode.is_extend():
            commit()

    injector.ensure_context = ensure_context
    injector.inject_verify = commit
    executor = TargetVerifyExecutor(
        target_worker=target,
        gamma=3,
        verify_num_draft_tokens=4,
        model_runner=None,
        kv_injector=injector,
    )
    executor._verify_backend_self_adds_seq_lens_cache = False

    def accept(**kwargs):
        assert rank == owner
        block = kwargs["draft_block"]
        if sampling:
            assert bool((block.corrected_logits == owner + 0.5).all())
        out = torch.full((2, 4), -999, dtype=torch.int64, device=device)
        out[0, 0] = 30
        out[1, :2] = block.draft_tokens[1, :2]
        out[1, 2] = 31
        if failure == "accepted_token":
            out[1, 0] += 1
        commits = torch.tensor([1, 3], device=device, dtype=torch.int32)
        if failure == "accepted_length":
            commits[0] = 5
        state.events.append("accept")
        return AcceptOuts(
            correct_len=commits - 1,
            bonus=torch.tensor([30, 31], device=device),
            cap_trim_lens=torch.zeros(2, dtype=torch.int32, device=device),
            commit_lens=commits,
            new_seq_lens=kwargs["prefix_lens"] + commits,
            out_tokens=out,
        )

    executor.accept_and_finalize = accept
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.ps = ParallelState.trivial(
        pp_size=pp.world_size,
        pp_rank=stage,
        tp_size=tp.world_size,
        tp_rank=tp.rank_in_group,
    )
    worker.device = str(device)
    worker.model_runner = SimpleNamespace(model_config=SimpleNamespace(vocab_size=64))
    worker.verify_num_draft_tokens = 4
    worker._pending_prefill_step = worker._pending_decode_step = None
    worker._target_kv_contract = object()
    worker._is_pd_prefill = False
    worker._capture_hidden_mode = CaptureHiddenMode.NULL
    worker._need_mamba_verify_commit = False
    worker._draft_is_moe = False
    worker._target_worker = target
    worker._kv_injector = injector
    worker._verify_executor = executor
    worker._verify_planner = SimpleNamespace(
        mode_value="static", carries_confidence=False
    )
    worker._observers = SimpleNamespace(
        segment=lambda *args: nullcontext(), observe_verify_step=lambda **kwargs: None
    )

    def prepare(batch):
        worker._check_no_pending_step()
        if failure == "prepare" and rank == 0:
            raise RuntimeError("injected proposal preparation failure")
        injector.ensure_context(batch)
        ids = torch.full((2, 3), 63, dtype=torch.int64, device=device)
        ids[:, 0] = batch.spec_info.bonus_tokens
        if failure == "anchor" and rank == 0:
            ids[0, 0] += 1
        tokens = torch.tensor([[11, 12, 13], [21, 22, 23]], device=device) + rank
        if failure == "proposal_token" and rank == owner:
            tokens[0, 0] = 64
        proposal = DraftProposal(
            draft_block_ids=ids,
            draft_block=DraftBlockResult(
                draft_tokens=tokens,
                corrected_logits=(
                    torch.full((2, 3, 64), rank + 0.5, device=device)
                    if sampling
                    else None
                ),
                greedy_mask=torch.tensor([True, not sampling], device=device),
                temperatures=torch.ones(2, device=device),
            ),
            draft_hidden=None,
        )
        step = DSparkDecodeStep(
            batch=batch,
            draft_input=batch.spec_info,
            prefix_lens=batch.seq_lens,
            verify_window=VerifyWindow(
                positions_2d=batch.seq_lens[:, None] + torch.arange(4, device=device),
                verify_cache_loc=slots.flatten(),
                verify_cache_loc_2d=slots,
            ),
            sampling_info=None,
            proposal=proposal,
            confidence=None,
            verify_token_budget=None,
            layout=None,
            run_compact=False,
            verify_ids_2d=torch.cat([ids[:, :1], tokens], dim=1),
            grammar_tree=None,
            fold_eligible=False,
        )
        worker._pending_decode_step = step
        state.step = step
        return step

    worker.prepare_decode_step = prepare
    coordinator = DSparkPPCoordinator(
        worker=worker, world_group=world, tp_group=tp, pp_group=pp
    )
    return coordinator, batch, state, slots


def run_isolated(coordinator, batch, state):
    state.forward_done = False
    snapshot = vars(batch).copy()
    try:
        return coordinator.run_batch(batch)
    finally:
        vars(batch).clear()
        vars(batch).update(snapshot)


def run_cases(world, tp, pp, device):
    def prepare_verify(verify_input, batch, target):
        return SimpleNamespace(input_ids=verify_input.draft_token), None

    def grammar(*, tree, **kwargs):
        assert tree is not None
        tree.resolve()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(DFlashVerifyInput, "prepare_for_verify", prepare_verify)
        )
        stack.enter_context(patch(f"{WORKER}.prepare_mamba_track_for_verify"))
        stack.enter_context(patch(f"{WORKER}.build_grammar_vocab_mask", grammar))
        stack.enter_context(patch(f"{VERIFY}.apply_dflash_verify_logits_adjustments"))
        for sampling in (False, True):
            coordinator, batch, state, slots = fixture(
                world, tp, pp, device, sampling=sampling
            )
            assert coordinator.run_batch(None) is None
            batch.forward_mode = ForwardMode.EXTEND
            for req in batch.reqs:
                req.dspark_projected_context = None
            result = run_isolated(coordinator, batch, state)
            assert result.next_token_ids.tolist() == [10, 20]
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_info = result.next_draft_input
            for _ in range(2):
                state.events.clear()
                state.committed.clear()
                result = run_isolated(coordinator, batch, state)
                assert result.accept_lens.tolist() == [1, 3]
                assert result.training_capture == "accepted-ticket"
                step = state.step
                expected = [10, 20] if coordinator.sequence == 3 else [30, 31]
                assert state.inputs.view(2, 4)[:, 0].tolist() == expected
                torch.testing.assert_close(
                    step.grammar_tree.resolve()[2], step.verify_ids_2d.cpu()
                )
                assert len(state.committed) == 4
                for row, count in enumerate((1, 3)):
                    for col in range(4):
                        slot = int(slots[row, col])
                        if col < count:
                            assert state.committed[slot] == int(
                                state.inputs.view(2, 4)[row, col]
                            )
                        else:
                            assert slot not in state.committed
                assert state.events.index("capture-forward") < state.events.index(
                    "capture-accept"
                )
                assert state.events.index("capture-accept") < state.events.index(
                    "commit"
                )
                assert state.events.count("accept") == int(coordinator._is_owner)
                assert coordinator.worker._pending_decode_step is None
                batch.spec_info = result.next_draft_input
                batch.seq_lens = result.new_seq_lens
                batch.seq_lens_cpu = result.new_seq_lens.cpu()
                for i, req in enumerate(batch.reqs):
                    req.kv_committed_len = int(batch.seq_lens[i])
                    req.output_ids.extend(
                        result.next_token_ids.view(2, 4)[i, : (1, 3)[i]].tolist()
                    )
            assert coordinator.sequence == 4
        for changed in (
            None,
            ProjectedContextState("fixture:0", 2),
            ProjectedContextState("stale:0", 5),
            ProjectedContextState("fixture:0", 6),
        ):
            coordinator, batch, state, _ = fixture(world, tp, pp, device)
            if world.rank_in_group == 0:
                batch.reqs[0].dspark_projected_context = changed
            run_isolated(coordinator, batch, state)
            assert state.projection_starts == [0, 9]
            assert coordinator.worker._kv_injector.invalidation_ct == 1
        failures = (
            "order",
            "history",
            "idle",
            "prefix",
            "prepare",
            "anchor",
            "proposal_token",
            "forward",
            "activation_rows",
            "frame",
            "accepted_token",
            "accepted_length",
            "prefill_sample",
            "allocate",
            "weights",
        )
        for failure in failures:
            coordinator, batch, state, _ = fixture(
                world, tp, pp, device, failure=failure
            )
            if failure == "prefill_sample":
                batch.forward_mode = ForwardMode.EXTEND
            if world.rank_in_group == 0:
                if failure == "order":
                    batch.reqs.reverse()
                elif failure == "history":
                    batch.reqs[0].origin_input_ids[0] += 1
                elif failure == "idle":
                    batch = None
                elif failure == "prefix":
                    batch.seq_lens_cpu[0] += 1
                elif failure == "frame":
                    pp.bad_frame = True
            original_empty = torch.empty

            def allocate(*args, failure=failure, empty=original_empty, **kwargs):
                if (
                    failure == "allocate"
                    and world.rank_in_group == 0
                    and args == ((2, 3),)
                ):
                    raise MemoryError("injected receiver allocation failure")
                return empty(*args, **kwargs)

            try:
                try:
                    with patch.object(torch, "empty", allocate):
                        coordinator.run_batch(batch)
                except RuntimeError as error:
                    assert "DSpark PP" in str(error), error
                else:
                    raise AssertionError(f"accepted failure: {failure}")
                assert coordinator.failed
                assert "commit" not in state.events
                try:
                    coordinator.run_batch(None)
                except RuntimeError as error:
                    assert "restart" in str(error), error
                else:
                    raise AssertionError("reused a failed coordinator")
            finally:
                pp.bad_frame = False
        coordinator, batch, state, _ = fixture(world, tp, pp, device)
        assert run_isolated(coordinator, batch, state).accept_lens.tolist() == [1, 3]


def exercise(rank, root, *, tp_size, pp_size, cuda):
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/init",
        rank=rank,
        world_size=tp_size * pp_size,
        timeout=timedelta(seconds=60),
    )
    try:
        groups = make_groups(rank, tp_size=tp_size, pp_size=pp_size, cuda=cuda)
        world, tp, pp = (StageGroup(g) for g in groups)
        if cuda:
            world.device_group = dist.new_group(
                backend="nccl", timeout=timedelta(seconds=45)
            )
        run_cases(world, tp, pp, device)
    finally:
        dist.destroy_process_group()


def launch_coordinator_test(test, *, tp_size, pp_size, cuda=False):
    with tempfile.TemporaryDirectory() as root:
        context = mp.get_context("spawn")
        workers = [
            context.Process(
                target=exercise,
                args=(rank, root),
                kwargs={"tp_size": tp_size, "pp_size": pp_size, "cuda": cuda},
            )
            for rank in range(tp_size * pp_size)
        ]
        try:
            for worker in workers:
                worker.start()
            deadline = time.monotonic() + 240
            for worker in workers:
                worker.join(timeout=max(0, deadline - time.monotonic()))
            test.assertEqual(
                [worker.exitcode for worker in workers], [0] * len(workers)
            )
        finally:
            for worker in workers:
                if worker.pid is not None and worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=5)
