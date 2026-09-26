# Native NIXL peer recovery fault test

This two-host test uses real GPU buffers and the SGLang recovery helper. It
breaks RC queue pairs while four 2 GiB writes are outstanding, requires the
batch to fail, then verifies fresh writes to the same live decode agent and an
independent healthy agent. It checks the destination after handle retirement
and uses a different payload for the new request to detect delayed old writes.
A run where the fault misses the transfer fails the test.

Requirements: two isolated CUDA/RDMA hosts, NIXL 1.4.1, PyTorch, requests,
SGLang from this checkout, a C compiler, and libibverbs development headers.
Run only on disposable test workers. No engine restart or metadata deletion is
used as the injected fault.

On both hosts, from the repository root:

```bash
gcc -shared -fPIC -O2 -o /tmp/nixl-fault.so \
  test/manual/disaggregation/nixl_recovery/fault_qp.c -ldl -lpthread
export PYTHONPATH="$PWD/python"
export UCX_TLS=rc_verbs,ud_verbs,cuda_copy,self
export UCX_NET_DEVICES=mlx5_0:1  # choose an RDMA device on each host
```

Start decode, then prefill (replace `DECODE_IP` with its reachable address):

```bash
python test/manual/disaggregation/nixl_recovery/probe.py \
  --role decode --decode-host DECODE_IP --fault-library /tmp/nixl-fault.so
```

```bash
LD_PRELOAD=/tmp/nixl-fault.so \
python test/manual/disaggregation/nixl_recovery/probe.py \
  --role prefill --decode-host DECODE_IP --fault-library /tmp/nixl-fault.so
```

Success requires `(settled, failed) == (True, True)`, exact byte comparisons for
both peers, unchanged process IDs, and both processes exiting with status zero.
The decode process times out if the prefill exits early. This is a native
transport regression test; serving output checks and throughput benchmarks
should additionally exercise a router with actual prefill/decode engines.
