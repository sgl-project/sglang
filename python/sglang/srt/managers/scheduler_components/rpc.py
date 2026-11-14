"""Execute a scheduler RPC and agree on its result before replying."""

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput

if TYPE_CHECKING:
    from sglang.srt.distributed import GroupCoordinator

logger = logging.getLogger(__name__)


def execute_rpc_request(
    target: Any, request: RpcReqInput, group: "GroupCoordinator"
) -> RpcReqOutput:
    logger.info("handle_rpc_request: %s, param: %s", request.method, request.parameters)
    error_message = ""
    failed = False
    try:
        method = getattr(target, request.method)
        method(**(request.parameters or {}))
    except Exception as error:
        failed = True
        error_message = str(error)
        logger.error("Failed to call rpc %s: %s", request.method, error_message)

    # Replaces the completion barrier and includes failures on non-IO ranks.
    # Only RPC calls pay this cost; the scheduler's normal receive loop does not.
    no_failure = torch.iinfo(torch.int64).max
    failed_rank = torch.tensor(
        [group.rank if failed else no_failure], dtype=torch.int64
    )
    dist.all_reduce(failed_rank, op=dist.ReduceOp.MIN, group=group.cpu_group)
    source = int(failed_rank[0])
    if source == no_failure:
        return RpcReqOutput(success=True, message="")

    # Broadcast UTF-8 bytes rather than pickling an exception or Python object.
    encoded = error_message.encode("utf-8") if group.rank == source else b""
    length = torch.tensor([len(encoded)], dtype=torch.int64)
    dist.broadcast(length, src=source, group=group.cpu_group)
    size = int(length[0])
    payload = torch.empty(size, dtype=torch.uint8)
    if group.rank == source:
        payload[:] = torch.tensor(list(encoded), dtype=torch.uint8)
    if size:
        dist.broadcast(payload, src=source, group=group.cpu_group)
    return RpcReqOutput(success=False, message=bytes(payload.tolist()).decode("utf-8"))
