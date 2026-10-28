from sglang.srt.disaggregation.nixl.buffers import BufferKeys, BufferKind, NixlBufferKey
from sglang.srt.disaggregation.nixl.conn import PeerBuffers
from sglang.srt.disaggregation.nixl.transport import NixlTransport, Registration


def _device(key: NixlBufferKey, gpu_id: int) -> int:
    return -1 if key.kind == BufferKind.AUX else gpu_id


def attach_nixl_transport(
    manager,
    agent,
    *,
    gpu_id=0,
    kv=(),
    aux=(),
    state=(),
    peers=(),
    peer_kv=(),
    peer_aux=(),
    peer_state=(),
    peer_gpu_id=0,
):
    """Give `manager` a real NixlTransport over `agent`: this side's buffers
    registered, and every agent name in `peers` importing the peer's buffers.
    Aux buffers are host memory; the others sit on the given GPU."""
    transport = NixlTransport(agent=agent)
    local = (list(kv), list(aux), [list(s) for s in state])
    local_keys = BufferKeys.for_buffers(*local)
    transport.register_all(
        [
            Registration(key=key, ptr=ptr, length=1, device=_device(key, gpu_id))
            for key, ptr in local_keys.pair(*local)
        ]
    )
    peer = (list(peer_kv), list(peer_aux), [list(s) for s in peer_state])
    peer_keys = BufferKeys.for_buffers(*peer)
    peer_bases = dict(peer_keys.pair(*peer))
    manager.transport = transport
    manager._staging_keys = {}
    manager._local_keys = local_keys
    manager._peers = {
        name: PeerBuffers(
            remote=transport.import_registration(
                agent_name=name,
                metadata=name.encode("ascii"),
                bases=peer_bases,
                devices={key: _device(key, peer_gpu_id) for key in peer_bases},
            ),
            keys=peer_keys,
        )
        for name in peers
    }
    return transport
