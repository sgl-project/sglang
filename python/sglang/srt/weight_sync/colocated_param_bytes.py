"""Wire protocol and engine-side copy loop of `update_weights_from_colocated_param_bytes`.

A trainer process on the same GPU holds parameter bytes already in this rank's own layout (its replica of this
engine ran this engine's loader) in CUDA-IPC transfer buffers. Each TP rank pulls them over a node-local ZMQ PAIR
socket and copies every byte range into the weight location it published: the same bytes a P2P transfer-engine
write covers, so the session's post-load pass finalizes them the same way.

One sync on the socket of a (trainer rank, engine rank) pair:
    engine  -> Hello                      trainer checks it serves this GPU
    trainer -> Shared                     transfer buffers and the params the loader writes
    trainer -> Write, engine -> Ack       once per filled buffer; the trainer reuses a buffer after its Ack
    trainer -> End, engine -> Closed      the engine closed the IPC handles; the trainer may free the buffers
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import msgspec
import torch
import zmq

from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions


class Hello(msgspec.Struct, frozen=True):
    device_uuid: str


class Shared(msgspec.Struct, frozen=True):
    # MultiprocessingSerializer bytes of the transfer buffers, opened once per sync
    serialized_buffers: bytes
    # by runner role, every param the engine's loader writes; a sync that leaves one unwritten fails
    expected_names_by_role: Dict[str, FrozenSet[str]]


class Write(msgspec.Struct, frozen=True):
    buffer_index: int
    role: str
    # (param name, byte offset in the buffer, nbytes)
    params: List[Tuple[str, int, int]]


class Ack(msgspec.Struct, frozen=True):
    error: Optional[str]


class End(msgspec.Struct, frozen=True):
    pass


class Closed(msgspec.Struct, frozen=True):
    pass


def pull_param_bytes(
    *,
    sock: zmq.Socket,
    device_uuid: str,
    params_by_role: Dict[str, Dict[str, torch.nn.Parameter]],
    published_nbytes_by_role: Dict[str, Dict[str, int]],
) -> Optional[str]:
    """Runs one sync's pull on `sock` and returns the first error, or None.

    An error is acknowledged to the trainer and returned, never raised, so the trainer stops sending and every TP
    rank still reaches the scheduler's barrier. A silent trainer surfaces as `zmq.Again` from the socket's timeout.
    """
    sock.send_pyobj(Hello(device_uuid))
    shared: Shared = sock.recv_pyobj()
    # the trainer and this rank may index the same GPU differently; the patch resolves it by UUID
    monkey_patch_torch_reductions()
    buffers = MultiprocessingSerializer.deserialize(shared.serialized_buffers)
    written_names_by_role: Dict[str, Set[str]] = {
        role: set() for role in params_by_role
    }
    error = None
    while not isinstance(write := sock.recv_pyobj(), End):
        if error is None:
            error = _copy_write(
                buffer=buffers[write.buffer_index],
                write=write,
                params_by_role=params_by_role,
                published_nbytes_by_role=published_nbytes_by_role,
                written_names_by_role=written_names_by_role,
            )
        sock.send_pyobj(Ack(error))
    del buffers
    torch.cuda.ipc_collect()
    if error is None:
        error = _unwritten_params_error(
            expected_names_by_role=shared.expected_names_by_role,
            written_names_by_role=written_names_by_role,
        )
    sock.send_pyobj(Closed())
    return error


def _copy_write(
    *,
    buffer: torch.Tensor,
    write: Write,
    params_by_role: Dict[str, Dict[str, torch.nn.Parameter]],
    published_nbytes_by_role: Dict[str, Dict[str, int]],
    written_names_by_role: Dict[str, Set[str]],
) -> Optional[str]:
    if write.role not in params_by_role:
        return f"bytes for runner {write.role!r}, which this session does not update ({sorted(params_by_role)})"
    params = params_by_role[write.role]
    published_nbytes = published_nbytes_by_role[write.role]
    dsts, srcs = [], []
    for name, offset, nbytes in write.params:
        if name not in published_nbytes:
            return f"{write.role}.{name} is not a published weight of this rank"
        if nbytes != published_nbytes[name]:
            return f"{write.role}.{name}: {nbytes} bytes sent, {published_nbytes[name]} published"
        dsts.append(_published_byte_range(param=params[name], nbytes=nbytes))
        srcs.append(buffer[offset : offset + nbytes])
        written_names_by_role[write.role].add(name)
    torch._foreach_copy_(dsts, srcs)
    # the trainer refills the buffer once it has the Ack
    torch.cuda.current_stream().synchronize()
    return None


def _published_byte_range(*, param: torch.Tensor, nbytes: int) -> torch.Tensor:
    # the raw bytes at data_ptr() that the rank published, not a view of the param's elements
    storage = param.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=param.device).set_(
        storage, param.data_ptr() - storage.data_ptr(), (nbytes,)
    )


def _unwritten_params_error(
    *,
    expected_names_by_role: Dict[str, FrozenSet[str]],
    written_names_by_role: Dict[str, Set[str]],
) -> Optional[str]:
    for role, expected_names in expected_names_by_role.items():
        unwritten = sorted(expected_names - written_names_by_role.get(role, set()))
        if unwritten:
            return f"{role}: {len(unwritten)} params the loader writes got no bytes this sync, e.g. {unwritten[:5]}"
    return None
