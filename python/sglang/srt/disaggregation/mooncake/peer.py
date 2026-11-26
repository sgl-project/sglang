"""What a Mooncake prefill holds about each decode registration."""

from __future__ import annotations

import struct
from typing import List, Tuple

import msgspec

from sglang.srt.disaggregation.common.peer import Peer, PeerLayout
from sglang.srt.disaggregation.common.utils import pack_int_lists, unpack_int_lists
from sglang.srt.disaggregation.mooncake.buffers import (
    PEER_STAGING_KEY,
    BufferKeys,
    MooncakeBufferKey,
)
from sglang.srt.disaggregation.mooncake.transport import (
    MooncakeBufferExport,
    MooncakeRemoteBuffers,
)


def export_frames(
    export: MooncakeBufferExport, keys: BufferKeys
) -> Tuple[bytes, bytes, bytes, bytes, bytes]:
    """The registration frames that carry `export`, which covers `keys` in
    flat() order and then the staging ring if any: the session id, the KV, aux
    and state base addresses, and the staging ring's (empty without one)."""
    num_keys = len(keys.flat())
    kv, aux, state = keys.unflatten(export.bases[:num_keys])
    staging = export.bases[num_keys:]
    return (
        export.session_id.encode("ascii"),
        b"".join(struct.pack("Q", ptr) for ptr in kv),
        b"".join(struct.pack("Q", ptr) for ptr in aux),
        pack_int_lists(state, "Q"),
        struct.pack("Q", staging[0]) if staging else b"",
    )


class KVArgsRegisterInfo(msgspec.Struct, frozen=True):
    """A decode's registration message. Its frames interleave the transport's
    export with the PeerLayout; only this codec and export_frames split them."""

    # The name the decode's room messages carry: its mooncake session id.
    name: str
    # The protocol's names for the decode's KV, aux and state buffers.
    keys: BufferKeys
    # Covers `keys` in flat() order, then the staging ring when has_staging.
    export: MooncakeBufferExport
    has_staging: bool
    layout: PeerLayout

    @property
    def import_keys(self) -> List[MooncakeBufferKey]:
        """The names to import `export` under."""
        return self.keys.flat() + ([PEER_STAGING_KEY] if self.has_staging else [])

    @classmethod
    def from_zmq(cls, msg: List[bytes]) -> KVArgsRegisterInfo:
        layout = PeerLayout(
            dst_aux_item_lens=(
                list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])) if msg[6] else None
            ),
            dst_tp_rank=int(msg[8].decode("ascii")),
            dst_attn_tp_size=int(msg[9].decode("ascii")),
            dst_kv_item_len=int(msg[10].decode("ascii")),
            dst_state_item_lens=(
                unpack_int_lists(msg[11], "I") if len(msg) > 11 else []
            ),
            dst_state_dim_per_tensor=(
                unpack_int_lists(msg[12], "I") if len(msg) > 12 else []
            ),
            dst_kv_layer_ids=(
                list(struct.unpack(f"{len(msg[13]) // 4}I", msg[13]))
                if len(msg) > 13 and msg[13] != b""
                else []
            ),
            dst_state_layer_ids=(
                unpack_int_lists(msg[14], "I")
                if len(msg) > 14 and msg[14] != b""
                else []
            ),
            staging_total_size=(
                int(msg[16].decode("ascii")) if len(msg) > 16 and msg[16] != b"" else 0
            ),
            dst_dcp_size=(
                int(msg[17].decode("ascii")) if len(msg) > 17 and msg[17] != b"" else 1
            ),
            dst_dcp_rank=(
                int(msg[18].decode("ascii")) if len(msg) > 18 and msg[18] != b"" else 0
            ),
            staging_slot_layer_ids=(
                list(struct.unpack(f"{len(msg[19]) // 8}Q", msg[19]))
                if len(msg) > 19 and msg[19]
                else []
            ),
            dst_kv_item_lens=(
                list(struct.unpack(f"{len(msg[20]) // 8}Q", msg[20]))
                if len(msg) > 20 and msg[20]
                else []
            ),
            dst_draft_rows_dcp_collapsed=(len(msg) > 21 and msg[21] == b"1"),
            num_draft_entries=(
                int(msg[22].decode("ascii"))
                if len(msg) > 22 and msg[22] != b""
                else None
            ),
            dst_needs_full_draft_kv=len(msg) <= 23 or msg[23] != b"0",
        )
        ptrs = (
            list(struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])),
            list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            unpack_int_lists(msg[7], "Q"),
        )
        staging_base = (
            struct.unpack("Q", msg[15])[0] if len(msg) > 15 and len(msg[15]) == 8 else 0
        )
        keys = BufferKeys.for_buffers(*ptrs)
        bases = [ptr for _, ptr in keys.pair(*ptrs)]
        if staging_base:
            bases.append(staging_base)
        session_id = msg[3].decode("ascii")
        return cls(
            name=session_id,
            keys=keys,
            export=MooncakeBufferExport(session_id=session_id, bases=bases),
            has_staging=bool(staging_base),
            layout=layout,
        )


class MooncakePeer(Peer, frozen=True, eq=False):
    """A Mooncake decode registration; `name` is its mooncake session id."""

    remote: MooncakeRemoteBuffers
    keys: BufferKeys
    # The registration exported a staging ring under PEER_STAGING_KEY.
    has_staging: bool
    # Registration cleared the state strides; skip the per-transfer check.
    state_strides_validated: bool
