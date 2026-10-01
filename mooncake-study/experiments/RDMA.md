# Remote RDMA Store Validation

This lane exercises the existing registered-buffer snapshot adapter against a
Store segment on another physical node. It uses two nodes:

- A: one GPU for real Qwen3 P/D serving, the test driver and independent readers.
- B: Mooncake Master and a 256 MiB CPU Store segment. The segment needs no GPU.

P/D share A's GPU and use TCP for their own KV handoff. The snapshot write/read
path between A and B uses RDMA. This lane does not establish cross-node PD
transport, GPUDirect, multi-replica recovery, production Catalog retention,
training quality or performance SLOs. Its Catalog is a test double and its
target-KV draft is synthetic and untrained.

## Environment

Use the runtime in `h100-runtime-lock.json`, the implementation checkout on A
and B, and `mooncake_master` on B's `PATH`. Both nodes need accessible HCAs and
reachable TransferEngine control endpoints. On the tested RoCE cluster, the
pods need an RDMA resource allocation and host networking: device files alone
inside an isolated network namespace failed QP RTR with `No such device`.
Verify the physical node names differ. A pod IP difference alone is insufficient.

Choose the HCA and GID index for the actual deployment. The commands below use
the tested `mlx5_00` and index 3; they are not portable defaults. All clients use
`protocol=rdma`, an explicit device and `global_segment_size=0`. Only B mounts a
storage segment. `MC_STORE_MEMCPY=0` disables the same-host memory-copy shortcut.
Master/TransferEngine control RPCs still use TCP.

Run from the checkout with its `python` directory on `PYTHONPATH`. Set `PATH`
to include the experiment environment. Keep the Store in a supervised foreground
process and retain its log and terminal exit status:

```bash
# On B. STORE_IP must be reachable from A.
env MC_STORE_MEMCPY=0 MC_GID_INDEX=3 MC_TE_METRIC=1 \
  MC_TE_METRIC_INTERVAL_SECONDS=1 \
  python -m sglang.test.training_capture_rdma serve \
  --host "$STORE_IP" --devices mlx5_00 \
  --ready /tmp/rdma-store-ready.json --lifetime 1800
```

The ready JSON records the fixture PID, Master PID and client setup. Create a
JSON setup file on A using that `setup` object, replacing only `local_hostname`
with A's reachable address. Preserve the generated Master endpoint and the zero
global segment size. Example structure:

```json
{
  "local_hostname": "PRODUCER_IP",
  "master_server_addr": "STORE_IP:MASTER_PORT",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "rdma",
  "rdma_devices": "mlx5_00",
  "global_segment_size": 0,
  "local_buffer_size": 16777216
}
```

## Probe And Real Capture

Run the probe first on A. It writes a registered 1 MiB buffer, closes the writer,
then checks the digest using a new client. Cleanup waits for the normal Store
read lease to expire and uses `remove(force=False)` on its own unique key.

```bash
env MC_STORE_MEMCPY=0 MC_GID_INDEX=3 MC_TE_METRIC=1 \
  MC_TE_METRIC_INTERVAL_SECONDS=1 \
  python -m sglang.test.training_capture_rdma probe \
  --setup /tmp/rdma-client.json

env MC_STORE_MEMCPY=0 MC_GID_INDEX=3 MC_TE_METRIC=1 \
  MC_TE_METRIC_INTERVAL_SECONDS=1 \
  TRAINING_CAPTURE_RDMA_SETUP=/tmp/rdma-client.json \
  TRAINING_CAPTURE_TEST_MODEL=/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  python test/registered/storage/test_training_capture_rdma.py -v -f
```

Run the whole file. Four cases cover AR and target-KV DSpark with eager execution
and decode/verify CUDA graphs. Each checks the actual online source KV and raw
teacher rows, positions, response masks and KV validity, including chunked
prefill, prefix reuse, one-token replies, real decode batches and cancellation.
Missing/stale PD teacher handoffs must fail capture without failing generation.
After each case, serving processes exit and a separate interpreter reads every
published manifest and tensor through RDMA. It validates object hashes and
coverage, so producer memory cannot provide the retrieved values.

The CI registration is explicitly disabled in ordinary single-node runners;
the environment variable enables this dedicated manual lane. Retain the test
exit code, complete logs, source hashes, HCA/GID selection and physical node
identities with results. A registered test or successful SDK setup alone is not
evidence of working RDMA transfers.

Stop the Store only after clients have exited. The fixture reinstalls its Python
signal handlers after native SDK initialization so SIGTERM can close its segment
and reap its own Master. Verify both recorded PIDs are gone, the foreground
command has ended and serving GPU allocations are released before deleting the
temporary jobs. Use a dedicated ephemeral Master/segment for this lane; its
hard-pinned test samples are discarded with that segment.
