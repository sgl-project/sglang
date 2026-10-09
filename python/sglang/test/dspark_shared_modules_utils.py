"""Native vocabulary modules and real collectives for PP draft binding tests."""

import multiprocessing as mp
import tempfile
import time
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers import vocab_parallel_embedding as vocab
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.speculative.dspark_components import dspark_shared_modules as shared


class TestGroup:
    all_gather_object = GroupCoordinator.all_gather_object
    broadcast = GroupCoordinator.broadcast

    def __init__(self, ranks, rank, cpu_group, device_group=None):
        self.ranks = ranks
        self.rank_in_group = ranks.index(rank)
        self.world_size = len(ranks)
        self.cpu_group = cpu_group
        self.device_group = device_group if device_group is not None else cpu_group

    def all_reduce(self, value):
        dist.all_reduce(value, group=self.device_group)
        return value

    def all_gather(self, value, dim):
        values = [torch.empty_like(value) for _ in self.ranks]
        dist.all_gather(values, value, group=self.device_group)
        return torch.cat(values, dim=dim)


def make_groups(rank, *, tp_size, pp_size, cuda):
    world_size = tp_size * pp_size
    world = TestGroup(list(range(world_size)), rank, dist.group.WORLD)
    groups = []
    for axis in ("tp", "pp"):
        selected = None
        for index in range(pp_size if axis == "tp" else tp_size):
            ranks = (
                list(range(index * tp_size, (index + 1) * tp_size))
                if axis == "tp"
                else list(range(index, world_size, tp_size))
            )
            cpu = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=45))
            device = (
                dist.new_group(ranks, backend="nccl", timeout=timedelta(seconds=45))
                if cuda
                else cpu
            )
            if rank in ranks:
                selected = TestGroup(ranks, rank, cpu, device)
        groups.append(selected)
    return world, *groups


def exercise(rank, root, *, tp_size, pp_size, cuda=False):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/init",
        rank=rank,
        world_size=tp_size * pp_size,
        timeout=timedelta(seconds=60),
    )
    try:
        world, tp, pp = make_groups(rank, tp_size=tp_size, pp_size=pp_size, cuda=cuda)
        parallel = SimpleNamespace(tp_rank=tp.rank_in_group, tp_size=tp.world_size)
        with ExitStack() as stack:
            for name, value in (
                ("get_parallel", lambda: parallel),
                ("get_tp_group", lambda: tp),
                ("is_allocation_symmetric", lambda: False),
                ("tensor_model_parallel_all_reduce", tp.all_reduce),
            ):
                stack.enter_context(patch.object(vocab, name, value))
            stack.enter_context(patch.object(shared, "get_world_group", lambda: world))
            stack.enter_context(
                patch(
                    "sglang.srt.models.dspark.tensor_model_parallel_all_gather",
                    tp.all_gather,
                )
            )
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                for tied in (False, True):
                    run_case(tp, pp, device, dtype, tied=tied)
            for failure in (
                "missing_embedding",
                "head_dtype",
                "allocation",
                "shard",
                "tp_metadata",
            ):
                if failure in ("shard", "tp_metadata") and tp_size == 1:
                    continue
                try:
                    run_case(tp, pp, device, torch.float32, failure=failure)
                except RuntimeError as error:
                    assert "PP DSpark" in str(error), error
                else:
                    raise AssertionError(f"rank {rank} accepted {failure}")
                # A coordinated pre-collective failure must leave groups reusable.
                run_case(tp, pp, device, torch.float32)
    finally:
        dist.destroy_process_group()


def run_case(tp, pp, device, dtype, *, tied=False, failure=None):
    size, hidden = 67, 8
    full_embedding = (torch.arange(size * hidden).reshape(size, hidden) % 19 - 9).to(
        device=device, dtype=dtype
    )
    full_head = full_embedding if tied else full_embedding.flip(0).contiguous()

    def make(cls, weights):
        with torch.device(device):
            module = cls(size, hidden, params_dtype=dtype, padding_size=64)
        module.weight_loader(module.weight, weights)
        return module

    embedding = (
        make(vocab.VocabParallelEmbedding, full_embedding)
        if pp.rank_in_group == 0
        else None
    )
    head = (
        make(vocab.ParallelLMHead, full_head)
        if pp.rank_in_group == pp.world_size - 1
        else None
    )
    original_embedding, original_head = embedding, head
    if (
        failure == "missing_embedding"
        and pp.rank_in_group == 0
        and tp.rank_in_group == 0
    ):
        embedding = None
    if (
        failure == "head_dtype"
        and pp.rank_in_group == pp.world_size - 1
        and tp.rank_in_group == 0
    ):
        head.weight = torch.nn.Parameter(head.weight.double(), requires_grad=False)
    if failure == "shard" and pp.rank_in_group == 0 and tp.rank_in_group == 1:
        embedding.shard_indices = embedding._get_indices(
            128, 128, size, size, 0, tp.world_size
        )
    if failure == "tp_metadata" and pp.rank_in_group == 0 and tp.rank_in_group == 1:
        embedding.padding_size = 128
    target = SimpleNamespace(get_input_embeddings=lambda: embedding, lm_head=head)
    make_replica = shared._make_replica

    def allocate(*args, **kwargs):
        if failure == "allocation" and pp.rank_in_group == 1 and tp.rank_in_group == 0:
            raise MemoryError("injected replica allocation failure")
        return make_replica(*args, **kwargs)

    with patch.object(shared, "_make_replica", allocate):
        bound_embedding, bound_head = shared.resolve_dspark_shared_modules(
            target_model=target, pp_group=pp, tp_group=tp, device=device
        )
    if original_embedding is not None:
        assert bound_embedding is original_embedding
    else:
        assert target.get_input_embeddings() is None
    if original_head is not None:
        assert bound_head is original_head
    else:
        assert target.lm_head is None
    assert bound_embedding.weight.device == device
    assert bound_head.weight.device == device
    assert bound_embedding.weight.dtype == dtype
    assert not bound_embedding.weight.requires_grad
    input_ids = torch.tensor([0, 33, 63, 64, 66], device=device)
    torch.testing.assert_close(
        bound_embedding(input_ids), full_embedding[input_ids], rtol=0, atol=0
    )
    draft = SimpleNamespace(
        embed_tokens=bound_embedding,
        lm_head=bound_head,
        logits_mup_width_multiplier=None,
    )
    hidden_states = torch.arange(24, device=device, dtype=dtype).reshape(3, hidden) % 7
    logits, _ = DSparkDraftMixin.compute_base_logits(draft, hidden_states)
    torch.testing.assert_close(logits, hidden_states @ full_head.T, rtol=0, atol=0)
    assert logits.shape == (3, size)


def launch_shared_module_test(test, *, tp_size, pp_size, cuda=False):
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
            deadline = time.monotonic() + 180
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
