"""Result relay fixtures shared by CPU transport and CUDA D2H tests."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class PPResultTestGroup:
    send_object = GroupCoordinator.send_object
    recv_object = GroupCoordinator.recv_object
    send_tensor_dict = GroupCoordinator.send_tensor_dict
    recv_tensor_dict = GroupCoordinator.recv_tensor_dict

    def __init__(self, rank, world_size, device_group):
        self.world_size = world_size
        self.rank_in_group = rank
        self.ranks = list(range(world_size))
        self.cpu_group = dist.group.WORLD
        self.device_group = device_group


def relay_pp_result(rank, root, world_size=4, use_cuda=False):
    torch.set_num_threads(1)
    if use_cuda:
        torch.cuda.set_device(rank)
    device = "cuda" if use_cuda else "cpu"
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    device_group = (
        dist.new_group(backend="nccl", timeout=timedelta(seconds=30))
        if use_cuda
        else dist.group.WORLD
    )
    group = PPResultTestGroup(rank, world_size, device_group)
    try:
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="null"),
        ):
            for case in ("prefill", "decode", "uncapped", "copied"):
                prefill = case == "prefill"
                scheduler, batch, source, observed = make_pp_result_fixture(
                    prefill=prefill, rank=rank, device=device
                )
                if case == "uncapped":
                    source.cap_lens = None
                    source.next_token_ids = source.next_token_ids.int()
                if case == "copied":
                    source.copy_done = scheduler.device_module.Event()
                    source.copy_to_cpu(return_logprob=False, return_hidden_states=False)
                if rank == world_size - 1:
                    packet = SchedulerPPMixin._pp_prepare_tensor_dict(
                        scheduler, source, batch
                    )
                    group.send_tensor_dict(packet)
                packet = group.recv_tensor_dict()
                result = receive_pp_result(scheduler, batch, packet)
                check_pp_result(scheduler, batch, result, observed, prefill=prefill)
                if rank != world_size - 1:
                    # Forward the original frame; local request state is advanced.
                    group.send_tensor_dict(packet)
                dist.barrier()
                if rank == 0:
                    print(
                        f"PP{world_size} {device} result relay passed: {case}",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


class CPUEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def make_pp_result_fixture(*, prefill=False, device="cpu", rank=0):
    prefixes = [5, 9, 13]
    reqs = [
        SimpleNamespace(
            rid=f"request-{i}",
            kv_committed_len=length,
            output_ids=[] if prefill else [i + 1],
            is_retracted=False,
            finished=lambda: False,
            grammar=None,
            spec_verify_ct=0,
            spec_num_correct_drafts=0,
            spec_num_block_accept_tokens=0,
            spec_num_cap_tokens=0,
            update_spec_correct_drafts_histogram=Mock(),
            update_spec_cap_lens_histogram=Mock(),
        )
        for i, length in enumerate(prefixes)
    ]
    batch = SimpleNamespace(
        reqs=reqs,
        spec_algorithm=SpeculativeAlgorithm.DSPARK,
        forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
        return_logprob=False,
        req_pool_indices=torch.tensor([7, 2, 11], device=device) + rank,
        seq_lens=torch.tensor(prefixes, device=device),
        seq_lens_cpu=torch.tensor(prefixes),
        seq_lens_sum=sum(prefixes),
        spec_info=None,
        input_ids=torch.tensor([1, 2, 3], device=device),
    )
    if prefill:
        tokens = torch.tensor([31, 41, 51], device=device)
        bonus = tokens.clone()
        lengths = batch.seq_lens.clone()
        accepts = blocks = caps = None
    else:
        tokens = torch.tensor(
            [31, -999, -999, -999, 41, 42, 43, -999, 51, 52, 53, 54], device=device
        )
        bonus = torch.tensor([31, 43, 54], device=device)
        accepts = torch.tensor([1, 3, 4], dtype=torch.int32, device=device)
        blocks = torch.tensor([2, 3, 4], dtype=torch.int32, device=device)
        caps = torch.tensor([2, 3, 4], dtype=torch.int32, device=device)
        lengths = batch.seq_lens + accepts
    result = GenerationBatchResult(
        next_token_ids=tokens,
        accept_lens=accepts,
        block_accept_lens=blocks,
        cap_lens=caps,
        new_seq_lens=lengths,
        next_draft_input=make_draft_input_v2(bonus_tokens=bonus, new_seq_lens=lengths),
        speculative_num_draft_tokens=None if prefill else 4,
        can_run_cuda_graph=True,
    )
    scheduler = SimpleNamespace(
        future_map=SimpleNamespace(stash=Mock()),
        device_module=(
            torch.cuda
            if torch.device(device).type == "cuda"
            else SimpleNamespace(Event=CPUEvent)
        ),
        tp_worker=SimpleNamespace(training_capture=None),
    )
    processor = SimpleNamespace(
        model_worker=SimpleNamespace(on_verify_complete_cpu=Mock()),
        advance_grammar_fsm=Mock(),
    )
    observed = []

    def process(received_batch, received):
        if prefill:
            observed.append(received.next_token_ids.tolist())
        else:
            observed.append(
                SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
                    processor, received, received_batch
                )
            )

    scheduler.process_batch_result = process
    return scheduler, batch, result, observed


def receive_pp_result(scheduler, batch, packet):
    result = SchedulerPPMixin._pp_prep_batch_result(
        scheduler, batch, PPBatchMetadata(True), PPProxyTensors(packet)
    )
    result.copy_done.synchronize()
    return result


def check_pp_result(scheduler, batch, result, observed, *, prefill):
    slots, payload = scheduler.future_map.stash.call_args.args
    torch.testing.assert_close(slots, batch.req_pool_indices)
    expected_bonus = [31, 41, 51] if prefill else [31, 43, 54]
    assert payload.bonus_tokens.tolist() == expected_bonus
    assert payload.bonus_tokens.device == batch.seq_lens.device
    assert result.next_token_ids.is_cpu
    assert batch.input_ids is None
    assert result.can_run_cuda_graph
    if not prefill:
        assert result.accept_lens.is_cpu
        assert result.block_accept_lens.is_cpu
        assert result.cap_lens is None or result.cap_lens.is_cpu
    SchedulerPPMixin._pp_process_batch_result(scheduler, batch, result)
    assert batch.spec_info is result.next_draft_input
    expected_lengths = [5, 9, 13] if prefill else [6, 12, 17]
    assert batch.seq_lens.tolist() == expected_lengths
    assert batch.seq_lens_cpu.tolist() == expected_lengths
    assert batch.seq_lens_sum == sum(expected_lengths)
    if prefill:
        assert observed == [[31, 41, 51]]
    else:
        assert observed == [[[31], [41, 42, 43], [51, 52, 53, 54]]]
        assert [req.kv_committed_len for req in batch.reqs] == expected_lengths
        assert result.num_correct_drafts_per_req_cpu == [0, 2, 3]
        assert result.num_correct_drafts == 5
        assert result.num_block_accept_tokens == 9
        assert result.num_cap_tokens == (0 if result.cap_lens is None else 9)
