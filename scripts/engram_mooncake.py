#!/usr/bin/env python3
"""Upload V4.1 FP8 Engram tables and keep a local Mooncake memory segment alive."""

import argparse
import gc
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mooncake.store import (
    EngramStore,
    EngramStoreConfig,
    MooncakeDistributedStore,
    ReplicateConfig,
)
from safetensors import safe_open

from sglang.srt.layers.engram import build_engram_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--master", default="127.0.0.1:50051")
    parser.add_argument("--metadata", default="http://127.0.0.1:50052/metadata")
    parser.add_argument("--protocol", choices=("tcp", "rdma"), default="tcp")
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--local-hostname", default="127.0.0.1")
    parser.add_argument(
        "--output", required=True, help="Connection/layout manifest for SGLang"
    )
    parser.add_argument("--pool-gib", type=int, default=256)
    args = parser.parse_args()
    model = Path(args.model)
    checkpoint_config = json.loads((model / "config.json").read_text())
    config = SimpleNamespace(**checkpoint_config["text_config"])
    layout = build_engram_layout(config)
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    connection = dict(
        local_hostname=args.local_hostname,
        metadata_server=args.metadata,
        global_segment_size=0,
        local_buffer_size=64 * 1024**2,
        protocol=args.protocol,
        rdma_devices=args.rdma_devices,
        master_server_addr=args.master,
    )
    store = MooncakeDistributedStore()
    server_connection = dict(connection, global_segment_size=args.pool_gib * 1024**3)
    rc = store.setup(**server_connection)
    if rc != 0:
        raise RuntimeError(f"Mooncake setup failed: {rc}")
    replicate = ReplicateConfig()
    replicate.with_hard_pin = True
    manifest = dict(connection=connection, layers={})
    ready_key = "engram:ready"
    if store.is_exist(ready_key):
        raise RuntimeError("Engram tables already published in this Store")
    layers = {}
    for layer_index, layer in enumerate(layout.layer_ids):
        cfg = EngramStoreConfig()
        cfg.table_vocab_sizes = [
            p for group in layout.primes[layer_index] for p in group
        ]
        cfg.row_bytes = layout.head_dim + layout.head_dim // 32
        layers[layer] = cfg
    table = EngramStore(layers, store)
    try:
        for layer_index, layer in enumerate(layout.layer_ids):
            cfg = layers[layer]
            weight_name = f"layers.{layer}.engram.embed.weight"
            scale_name = f"layers.{layer}.engram.embed.scale"
            with (
                safe_open(
                    model / index[weight_name], framework="pt", device="cpu"
                ) as wf,
                safe_open(
                    model / index[scale_name], framework="pt", device="cpu"
                ) as sf,
            ):
                weight = wf.get_slice(weight_name)
                scale = sf.get_slice(scale_name)
                buffers, offset = [], 0
                for head, rows in enumerate(cfg.table_vocab_sizes):
                    packed = np.empty((rows, cfg.row_bytes), dtype=np.uint8)
                    # Bound temporary copies while retaining one layer's upload buffers.
                    for start in range(0, rows, 65536):
                        end = min(start + 65536, rows)
                        packed[start:end, : layout.head_dim] = (
                            weight[offset + start : offset + end]
                            .view(torch.uint8)
                            .numpy()
                        )
                        packed[start:end, layout.head_dim :] = (
                            scale[offset + start : offset + end]
                            .view(torch.uint8)
                            .numpy()
                        )
                    buffers.append(packed)
                    offset += rows
                    print(
                        f"Packed layer={layer} head={head} bytes={packed.nbytes}",
                        flush=True,
                    )
                if offset != layout.num_embeddings[layer_index]:
                    raise ValueError("Head sizes do not match checkpoint table")
                table.populate(layer, buffers, replicate)
                # Validate rows at both ends, including byte offsets above 2 GiB.
                ids = np.array(
                    [
                        [
                            np.zeros(len(buffers), dtype=np.int64),
                            np.array(cfg.table_vocab_sizes, dtype=np.int64) - 1,
                        ]
                    ]
                )
                actual = np.empty((*ids.shape, cfg.row_bytes), dtype=np.uint8)
                if store.register_buffer(actual.ctypes.data, actual.nbytes) != 0:
                    raise RuntimeError("Could not register verification buffer")
                try:
                    table.lookup_into(layer, ids, actual)
                finally:
                    if store.unregister_buffer(actual.ctypes.data) != 0:
                        raise RuntimeError("Could not unregister verification buffer")
                for head, packed in enumerate(buffers):
                    np.testing.assert_array_equal(actual[0, 0, head], packed[0])
                    np.testing.assert_array_equal(actual[0, 1, head], packed[-1])
                del packed, buffers, weight, scale
                gc.collect()
            manifest["layers"][str(layer)] = dict(
                table_vocab_sizes=cfg.table_vocab_sizes,
                head_dim=layout.head_dim,
                row_bytes=cfg.row_bytes,
            )
            print(f"Uploaded and verified layer={layer}", flush=True)
        rc = store.put(
            ready_key,
            json.dumps(manifest["layers"], sort_keys=True).encode(),
            replicate,
        )
        if rc != 0:
            raise RuntimeError(f"Publishing ready marker failed: {rc}")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output)
        print(f"READY: {output}; keep this process alive while serving", flush=True)
        signal.signal(
            signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        )
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__ == "__main__":
    main()
