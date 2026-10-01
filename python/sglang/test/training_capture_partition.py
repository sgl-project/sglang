"""Logical-head partitions and an independent registered-buffer Store writer."""

import argparse
import json
import socket
from pathlib import Path

import msgspec
from safetensors.torch import load_file

from sglang.srt.training_capture.catalog import CaptureLease, HTTPCaptureCatalog
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import (
    Topology,
    canonical_bytes,
    decode_manifest,
    digest_bytes,
    tensor_bytes,
    validate_tensors,
)
from sglang.srt.training_capture.snapshot_writer import (
    PublicationJournal,
    SnapshotWriter,
)
from sglang.test.training_capture_utils import make_snapshot


def make_partitioned_snapshot():
    manifest, original = make_snapshot(response_length=4)
    manifest = msgspec.structs.replace(manifest, generation_id="partitioned")
    owners = ["dp0-pp0-tp0", "dp0-pp0-tp1"]
    objects, tensors = [], {}
    for obj in manifest.objects:
        if obj.kind == "aux":
            key = manifest.key_prefix + "aux/" + obj.name
            objects.append(msgspec.structs.replace(obj, key=key, owner_id=owners[1]))
            tensors[key] = original[obj.key]
            continue
        for head, owner in enumerate(owners):
            tensor = original[obj.key][:, head : head + 1].contiguous()
            chunk = obj.token_range[0] // manifest.kv.storage_chunk_tokens
            suffix = f"kv/{obj.layer_id}/{owner}/{chunk}/{obj.component}"
            key = manifest.key_prefix + suffix
            objects.append(
                msgspec.structs.replace(
                    obj,
                    object_id=suffix.replace("/", "-"),
                    key=key,
                    shape=list(tensor.shape),
                    nbytes=tensor.numel() * tensor.element_size(),
                    sha256=digest_bytes(tensor_bytes(tensor)),
                    owner_id=owner,
                    head_range=(head, head + 1),
                )
            )
            tensors[key] = tensor
    manifest = msgspec.structs.replace(
        manifest,
        objects=list(reversed(objects)),
        topology=Topology(
            tp_size=2, owners=list(reversed(owners)), aux_owner=owners[1]
        ),
    )
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
