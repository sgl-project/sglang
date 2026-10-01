# Cross-Node P/D RDMA Capture

This lane places P and the CPU Store segment on node B, with D, the Catalog test
double and readers on node A. Each node needs one GPU and a working RDMA setup.
P sends serving KV to D through Mooncake RDMA. D exports selected KV and raw
teacher summaries to B's Store using the registered Host-buffer RDMA adapter.
The first teacher row travels in the existing bounded PD control message;
control RPC/ZMQ traffic still uses TCP.

Use the dependency lock and environment preparation in [RDMA.md](RDMA.md).
Verify distinct physical node assignments, RDMA resource allocation, HCA/GID
selection and reachable control addresses. The tested RoCE setup uses host
networking. Start the separate Store fixture on B and create the zero-segment
client setup for A as described there. Run its 1 MiB probe before serving.

Both nodes must see the same model path and shared test-observation directory.
Shared files contain test configuration, source observations and manifest
references; serving KV and Store payloads do not use the filesystem. The
source observer saves the actual online values without another target forward.

## Start The External P

Run this foreground command on B. Use a new root directory for each fixture.
The setup file is A's client setup; P does not connect to the Store or Catalog.
Its intentionally unreachable Catalog endpoint detects accidental reservation
ownership on P. The root, ready file and client setup must be visible on A.

```bash
env MC_STORE_MEMCPY=0 MC_GID_INDEX=3 MC_TE_METRIC=1 \
  MC_TE_METRIC_INTERVAL_SECONDS=1 \
  TRAINING_CAPTURE_TEST_MODEL=/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  python -m sglang.test.pd_capture_remote \
  --host "$PREFILL_IP" --devices mlx5_00 \
  --setup /shared/run/rdma-client.json \
  --root /shared/run/prefill \
  --ready /shared/run/prefill-ready.json --lifetime 1800
```

The ready file is written after HTTP health succeeds. It records the owning
supervisor and server PIDs, model, hostname, URL, bootstrap port and observation
path. Check the foreground process is still alive; a ready file alone is not
liveness evidence. P runs the target with overlap and retains its prefix cache.
Target-KV DSpark requires no draft on P.

## Run D And Verify

Run the complete registered test file on A:

```bash
env MC_STORE_MEMCPY=0 MC_GID_INDEX=3 MC_TE_METRIC=1 \
  MC_TE_METRIC_INTERVAL_SECONDS=1 \
  TRAINING_CAPTURE_TEST_MODEL=/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  TRAINING_CAPTURE_RDMA_SETUP=/shared/run/rdma-client.json \
  TRAINING_CAPTURE_PD_PREFILL=/shared/run/prefill-ready.json \
  TRAINING_CAPTURE_RDMA_PUBLICATIONS=/shared/run/publications.json \
  python test/registered/storage/test_training_capture_pd_rdma.py -v -f
```

The four cases run AR and target-KV DSpark with eager D and decode/verify CUDA
graphs. D binds its reachable address and explicitly selects RDMA and the
configured HCA. The external P remains alive across cases; each D exits before
a fresh interpreter reads every published snapshot. The assertions cover raw
top-128 IDs/logits, LSE, selected-layer KV, masks, positions, validity, chunked
prompts, prefix reuse, actual batches, accepted/rejected verify paths and three
fault/cancellation requests per case. P's HTTP liveness check raises on a
timeout instead of treating an observation failure as process exit.

After the test driver exits, send SIGTERM to the recorded P supervisor PID and
wait for its foreground command to exit. Verify all P/D serving processes and
GPU allocations have been released. Keep Store alive and run this on A:

```bash
env MC_STORE_MEMCPY=0 MC_GID_INDEX=3 \
  python -m sglang.test.training_capture_rdma read \
  --setup /shared/run/rdma-client.json \
  --publications /shared/run/publications.json \
  --output /shared/run/after-pd-exit.json
```

This final read must validate every retained publication after both serving
roles have exited. Then stop the Store fixture, verify its process tree has
exited and release temporary jobs. Retain commands, process observations, source
and log hashes, and all test exit codes. This is a dedicated manual RDMA lane;
ordinary single-node CI registration is explicitly disabled.

The synthetic draft is untrained, the Catalog is a test double and this fixture
uses TP=PP=DP=1. It does not establish wider distributed topology correctness,
production retention/consumer replay, useful draft acceptance, throughput SLOs,
or direct trainer GPU reads from the Store.
