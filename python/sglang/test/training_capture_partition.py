"""Logical-head partitions and an independent registered-buffer Store writer."""

import argparse
import json
import socket
from pathlib import Path

import msgspec
from safetensors.torch import load_file

from sglang.srt.training_capture.catalog import CaptureLease, HTTPCaptureCatalog
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import (
    canonical_bytes,
    decode_manifest,
    validate_tensors,
)
from sglang.srt.training_capture.snapshot import (
    SnapshotMetadata,
    assemble_snapshot,
    prepare_snapshot_partition,
)
from sglang.srt.training_capture.snapshot_writer import (
    PublicationJournal,
    SnapshotWriter,
)
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.training_capture_utils import Registrar, make_snapshot


def make_partitioned_snapshot():
    original_manifest, original = make_snapshot(response_length=4)
    layout = plan_capture_layout(
        original_manifest.kv, tp_size=2, pp_layer_ranges=[(0, 4)], aux_tp_rank=1
    )
    metadata = SnapshotMetadata(
        **{
            name: getattr(original_manifest, name)
            for name in SnapshotMetadata.__struct_fields__
        }
    )
    metadata = msgspec.structs.replace(
        metadata,
        generation_id="partitioned",
        topology=msgspec.structs.replace(
            layout.topology, owners=list(reversed(layout.topology.owners))
        ),
    )
    prepared, tensors = [], {}
    for partition in reversed(layout.partitions):
        pool = HostBufferPool(
            kv=metadata.kv,
            max_tokens=metadata.sequence.total_length,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            pin_memory=False,
            partition=partition,
        )
        slot = pool.acquire()
        try:
            ranges = {part.layer_id: part for part in partition.heads}
            for obj in original_manifest.objects:
                if obj.kind == "aux" and partition.include_aux:
                    slot.tensors[obj.name][: obj.shape[0]].copy_(original[obj.key])
                elif obj.kind == "kv" and obj.layer_id in ranges:
                    heads = ranges[obj.layer_id]
                    start, end = obj.token_range
                    slot.tensors[obj.name][start:end].copy_(
                        original[obj.key][:, heads.start : heads.end]
                    )
            part, payloads = prepare_snapshot_partition(
                metadata,
                slot.tensors,
                valid_kv_tokens=metadata.sequence.total_length - 1,
                partition=partition,
            )
            prepared.append(
                msgspec.structs.replace(part, objects=tuple(reversed(part.objects)))
            )
            tensors.update(payloads)
        finally:
            pool.release(slot, transfer_complete=True)
            pool.close()
    manifest = assemble_snapshot(metadata, prepared, layout=layout)
    validate_tensors(manifest, tensors)
    return manifest, tensors


def write_owner(args):
    manifest = decode_manifest(Path(args.manifest).read_bytes())
    lease = msgspec.json.decode(Path(args.lease).read_bytes(), type=CaptureLease)
    tensors = load_file(args.tensors)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    store = MooncakeSnapshotStore.connect(
        {
            "local_hostname": address,
            "master_server_addr": args.master,
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
            "rdma_devices": "",
            "global_segment_size": 0,
            "local_buffer_size": 16 << 20,
        }
    )
    journal = None
    try:
        for tensor in tensors.values():
            store.register(tensor)
        journal = PublicationJournal(args.journal)
        writer = SnapshotWriter(store, HTTPCaptureCatalog(args.catalog), journal)
        receipt = writer.write_partition(manifest, tensors, lease, owner_id=args.owner)
        Path(args.receipt).write_bytes(canonical_bytes(receipt))
        print(json.dumps({"owner": args.owner, "objects": len(tensors)}), flush=True)
    finally:
        store.close()
        if journal is not None:
            journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "master",
        "catalog",
        "manifest",
        "lease",
        "tensors",
        "owner",
        "receipt",
        "journal",
    ):
        parser.add_argument("--" + name, required=True)
    write_owner(parser.parse_args())
