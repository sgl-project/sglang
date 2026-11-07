"""The sdma exchange of the HIP prefill indexer's row-sharded selections across attention TP.

``SGLANG_OPT_DSV4_INDEXER_TP_SHARD_GATHER=sdma`` moves ``low_ratio_backend_hip._ShardExchange``'s
rows with mori's ``SdmaAllGatherOp``: ``start`` posts one copy-engine put per peer and returns,
so the layer's q_b GEMM and KV store keep every CU; ``wait`` runs before attention. Single node.
Every rank falls back to the RCCL all-gather when any rank cannot set it up, and per exchange
when it is larger than the window (``SGLANG_OPT_DSV4_INDEXER_TP_SHARD_SDMA_BYTES``).

Contract: every rank calls ``prepare`` / ``start`` / ``wait`` in the same order with the same
``rows_per_rank`` / ``cols``, and reads the region ``prepare`` returned before the next one.
mori's ``faulted`` syncs, so it is checked at setup (the self-test), not per exchange.
"""

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

MODES = ("rccl", "sdma")


def gather_mode() -> str:
    mode = envs.SGLANG_OPT_DSV4_INDEXER_TP_SHARD_GATHER.get()
    if mode not in MODES:
        raise ValueError(
            f"SGLANG_OPT_DSV4_INDEXER_TP_SHARD_GATHER={mode!r} not in {MODES}"
        )
    return mode


class SdmaGather:
    """``[world * rows_per_rank, cols]`` int32 regions of one ``SdmaAllGatherOp`` window,
    gathered in place: rank r's block is rows ``[r * rows_per_rank, (r + 1) * rows_per_rank)``."""

    def __init__(self, op):
        self.op = op
        self._block_bytes = 0

    def prepare(self, rows_per_rank: int, cols: int) -> torch.Tensor:
        # an exchange started and never waited for (an exception in between)
        self.op.drain()
        self._block_bytes = rows_per_rank * cols * 4
        return self.op.region(self._block_bytes).view(torch.int32).view(-1, cols)

    def start(self) -> None:
        self.op.start(self._block_bytes)

    def wait(self) -> None:
        self.op.wait()


_sdma: Dict[str, Optional[SdmaGather]] = {}


def _all_ranks(group: "GroupCoordinator", ok: bool) -> bool:
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return bool(flag.item())


def _setup_steps(group: "GroupCoordinator", built: dict) -> List[Tuple[str, Callable]]:
    """Run in order on every rank, which agree on each before the next: a local failure
    makes every rank fall back instead of leaving its peers in a collective."""
    capacity = envs.SGLANG_OPT_DSV4_INDEXER_TP_SHARD_SDMA_BYTES.get()

    def same_node():
        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(group.cpu_group, source_rank=0)):
            raise RuntimeError("the group spans nodes")

    def uid():
        from mori.cco import Communicator

        if group.rank_in_group == 0:
            built["uid"] = bytes(Communicator.get_unique_id())

    def communicator():
        from mori.cco import Communicator, UniqueId
        from mori.ops.sdma_allgather import SdmaAllGatherOp

        uid = [built.get("uid")]
        torch.distributed.broadcast_object_list(
            uid, src=group.ranks[0], group=group.cpu_group
        )
        built["comm"] = Communicator.init(
            group.world_size,
            group.rank_in_group,
            UniqueId.from_bytes(uid[0]),
            per_rank_vmm=SdmaAllGatherOp.per_rank_vmm(capacity),
        )

    def op():
        from mori.ops.sdma_allgather import SdmaAllGatherOp

        def agree(ok):
            # the op votes on each of its own steps; a failed vote already told every rank
            built["voted_out"] = not _all_ranks(group, ok)
            return not built["voted_out"]

        built["op"] = SdmaAllGatherOp(built["comm"], capacity, agree=agree)
        built["op"].self_test()

    return [
        ("same_node", same_node),
        ("uid", uid),
        ("communicator", communicator),
        ("op", op),
    ]


def _teardown(built: dict) -> None:
    try:
        if "op" in built:
            built["op"].close()
        if "comm" in built:
            built["comm"].destroy()
    except Exception:  # noqa: BLE001
        logger.warning("indexer shard gather: sdma teardown failed", exc_info=True)


def init_shard_gather(group: "GroupCoordinator") -> None:
    """Collective over the group's CPU group: builds its sdma exchange when every rank can,
    else every rank frees what it built and falls back to RCCL."""
    key = group.unique_name
    if key not in _sdma:
        built, why = {}, None
        for step, run in _setup_steps(group, built):
            try:
                run()
            except Exception as e:  # noqa: BLE001
                why = f"{step}: {e!r}"
            if built.pop("voted_out", False) or not _all_ranks(group, why is None):
                why = why or f"{step} failed on another rank"
                break
        if why is None:
            _sdma[key] = SdmaGather(built["op"])
            logger.info("indexer shard gather: sdma ready, world %d", group.world_size)
        else:
            _teardown(built)
            _sdma[key] = None
            logger.warning(
                "indexer shard gather: sdma unavailable (%s); using rccl", why
            )


def sdma_ready(group: "GroupCoordinator") -> bool:
    return _sdma.get(group.unique_name) is not None


def sdma_for(group: "GroupCoordinator", nbytes: int) -> Optional[SdmaGather]:
    """The group's sdma exchange when it is ready and its window holds nbytes; every input is
    rank-uniform, so every rank decides the same."""
    gather = _sdma.get(group.unique_name)
    return gather if gather is not None and nbytes <= gather.op.capacity else None
