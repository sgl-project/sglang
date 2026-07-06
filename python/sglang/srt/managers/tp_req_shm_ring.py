"""Shared-memory ring buffer for broadcasting scheduler work-requests to peer
TP ranks on a single node, avoiding a per-iteration gloo broadcast.

Motivation
----------
With ``tp_size > 1`` the scheduler runs a blocking gloo ``broadcast_pyobj`` on
*every* iteration to ship freshly-received requests from rank 0 to peer ranks --
even when zero requests arrived (an empty broadcast still costs one collective,
~50us on this box).  This module lets rank 0 write length-prefixed pickle frames
into a ``/dev/shm`` region and publish a monotonic sequence counter; peer ranks
poll the counter (a single 8-byte load when empty) and only read + deserialize
when it advances.  gloo is untouched for the tensor path and cross-node.

Correctness contract
---------------------
* Single writer (rank 0), N-1 readers (peers).  SPMC.
* ``seq`` is a monotonically increasing count of published *batches*.  Each peer
  tracks its own ``last_seq`` and reads exactly the batches in
  ``(last_seq, seq]`` -- no lost / duplicated / reordered batches.
* The writer fully writes a slot (payload + per-slot ready-seq marker) *before*
  publishing the global ``seq``.  Readers load the global ``seq`` first, then
  read the slot and verify the slot's marker matches; a mismatch means the
  writer lapped the reader (ring overrun) and the caller must fall back.
* Ring capacity is fixed (``num_slots``); if a reader falls behind by more than
  ``num_slots`` batches the overrun is detected and reported so the caller can
  fall back to gloo rather than silently corrupt.
* ``close()`` unmaps; rank 0 additionally ``unlink``s the file for clean
  shutdown.

This is intra-node only (shared memory does not cross nodes).

.. warning::
   **Not safe as a drop-in replacement for the gloo request broadcast in live
   serving.**  The gloo broadcast keeps request admission *lock-step*: every TP
   rank obtains the same set of new requests in the same scheduler iteration,
   which the batch builder + NCCL forward collectives rely on.  This ring
   delivers requests asynchronously (rank 0 writes and continues; peers poll
   independently), so a peer can see a request one iteration late and build a
   different batch than rank 0 -- desyncing the forward and hanging.  Making it
   safe requires a per-iteration sync, which reintroduces the collective the
   ring was meant to remove.  The module is a correctness-validated *prototype*
   (see test_shm_ring) kept for offline micro-timing only; profiling (Experiment
   A) showed the empty broadcast is fully overlapped behind CUDA sync anyway, so
   removing it yields no ITL/throughput gain for this workload.
"""

from __future__ import annotations

import hashlib
import mmap
import os
import struct
from typing import List, Optional

MAGIC = 0x54505247  # 'TPRG'
VERSION = 1

# Header: magic(u32) version(u32) num_slots(u32) slot_size(u32) seq(u64)
_HEADER = struct.Struct("<IIIIQ")
_SEQ_OFFSET = 16  # byte offset of the u64 seq inside the header
_SEQ_STRUCT = struct.Struct("<Q")

# Per-slot prefix: ready_seq(u64) payload_len(u32)
_SLOT_PREFIX = struct.Struct("<QI")


def shm_path_for(ipc_name: str, tag: str = "tpreq") -> str:
    base = os.path.basename((ipc_name or "default").rstrip("/")) or "default"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)
    digest = hashlib.blake2s((ipc_name or "").encode(), digest_size=4).hexdigest()
    return f"/dev/shm/sglang_{tag}_{safe}_{digest}.shm"


def _file_size(num_slots: int, slot_size: int) -> int:
    return _HEADER.size + num_slots * slot_size


class TpReqShmRingWriter:
    """Rank-0 writer."""

    def __init__(self, path: str, num_slots: int, slot_size: int):
        self.path = path
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.seq = 0
        size = _file_size(num_slots, slot_size)
        # O_TRUNC so a stale file from a crashed prior run is reset.
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.ftruncate(self.fd, size)
        self.mmap = mmap.mmap(self.fd, size, access=mmap.ACCESS_WRITE)
        _HEADER.pack_into(self.mmap, 0, MAGIC, VERSION, num_slots, slot_size, 0)

    def max_payload(self) -> int:
        return self.slot_size - _SLOT_PREFIX.size

    def write_batch(self, frames: List[bytes]) -> bool:
        """Publish a batch of raw pickle frames.  Returns False if the encoded
        batch does not fit a slot (caller should fall back to gloo)."""
        # Encode: num_frames(u32) then (len(u32), bytes) per frame.
        parts = [struct.pack("<I", len(frames))]
        for f in frames:
            parts.append(struct.pack("<I", len(f)))
            parts.append(f)
        payload = b"".join(parts)
        if len(payload) > self.max_payload():
            return False

        next_seq = self.seq + 1
        slot = next_seq % self.num_slots
        off = _HEADER.size + slot * self.slot_size
        # Write payload + slot marker BEFORE publishing global seq.
        _SLOT_PREFIX.pack_into(self.mmap, off, next_seq, len(payload))
        pstart = off + _SLOT_PREFIX.size
        self.mmap[pstart : pstart + len(payload)] = payload
        # Publish: bump the global seq last so a reader that sees the new seq is
        # guaranteed to see a fully written slot.
        _SEQ_STRUCT.pack_into(self.mmap, _SEQ_OFFSET, next_seq)
        self.seq = next_seq
        return True

    def close(self, unlink: bool = True) -> None:
        try:
            self.mmap.close()
        finally:
            os.close(self.fd)
            if unlink:
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass


class TpReqShmRingReader:
    """Peer-rank reader."""

    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        self.mmap = mmap.mmap(self.fd, size, access=mmap.ACCESS_READ)
        magic, version, self.num_slots, self.slot_size, seq = _HEADER.unpack_from(
            self.mmap, 0
        )
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"tp req ring header mismatch at {path}")
        # Start from the current published seq: only NEW batches are of interest.
        self.last_seq = seq

    def current_seq(self) -> int:
        return _SEQ_STRUCT.unpack_from(self.mmap, _SEQ_OFFSET)[0]

    def poll(self) -> Optional[List[bytes]]:
        """Return the concatenated frames of all batches published since the last
        poll (in order), or [] if none.  Returns None on ring overrun (caller
        must fall back to gloo)."""
        seq = self.current_seq()
        if seq == self.last_seq:
            return []
        # Overrun: writer advanced by more than the ring can hold since we last
        # read, so some slots we needed were overwritten.
        if seq - self.last_seq > self.num_slots:
            return None

        frames: List[bytes] = []
        for s in range(self.last_seq + 1, seq + 1):
            slot = s % self.num_slots
            off = _HEADER.size + slot * self.slot_size
            ready_seq, plen = _SLOT_PREFIX.unpack_from(self.mmap, off)
            # If the slot marker does not match the batch we expect, the writer
            # lapped us mid-read -> overrun.
            if ready_seq != s:
                return None
            pstart = off + _SLOT_PREFIX.size
            payload = bytes(self.mmap[pstart : pstart + plen])
            (nf,) = struct.unpack_from("<I", payload, 0)
            o = 4
            for _ in range(nf):
                (fl,) = struct.unpack_from("<I", payload, o)
                o += 4
                frames.append(payload[o : o + fl])
                o += fl
        self.last_seq = seq
        return frames

    def close(self) -> None:
        try:
            self.mmap.close()
        finally:
            os.close(self.fd)
