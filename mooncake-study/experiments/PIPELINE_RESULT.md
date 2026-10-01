# Pipeline DSpark Result Channel

This is a prerequisite for pipeline speculative execution, whose serving gates
remain closed. It extends the scheduler's existing PP output channel without
changing Mooncake object formats or adding a separate transport service.

Run the complete CPU and CUDA files:

```bash
PYTHONPATH=python python test/registered/unit/spec/test_dspark_pp_result.py -v -f
PYTHONPATH=python python test/registered/unit/training_capture/test_pd_capture.py -v -f
PYTHONPATH=python python test/registered/storage/test_dspark_pp_result_cuda.py -v -f
PYTHONPATH=python python test/registered/storage/test_dspark_pp_result_nccl.py -v -f
```

The CUDA copy-stream test requires one GPU; the NCCL test requires two. The CPU
test uses four processes. Transport fixtures call production
`GroupCoordinator.send_tensor_dict` / `recv_tensor_dict` in a ring from the last
stage to the first and back. Metadata uses Gloo; tensor payloads use Gloo for CPU
or NCCL for CUDA. Every stage has different physical request-pool indices.

Each ring tests prefill, accepted decode blocks, optional cap omission with int32
token IDs, and sender results already copied to CPU. The last case mixes CPU
accepted tokens/lengths with device bonus/sequence tensors. Each receiver calls
the scheduler's prepare/process methods and the actual spec-v2 token resolver.

Decode fixtures contain three requests with accepted lengths `[1, 3, 4]` and
poisoned padding. Assertions require:

- FutureMap receives exactly one bonus per request in local request slots.
- Only accepted prefixes reach token resolution; poisoned padding is excluded.
- Sequence lengths advance from `[5, 9, 13]` to `[6, 12, 17]`; metrics retain
  per-request correct-draft, block-accept and cap counts.
- A retracted request does not receive a second request-level KV commit.
- Bad versions, stale/reordered requests, malformed payloads, invalid acceptance
  lengths, wrong bonuses and inconsistent sequence lengths are rejected.
- A sender's pending D2H completion is observed before CPU payload consumption.
- CUDA receive copies use pinned CPU destinations on the copy stream, while
  next-draft tensors remain on the GPU.
- DSpark chunked prefill retains output communication and P/D teacher handoffs.

The shared PP scheduler path also has a real-model regression on two GPUs:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_pp.py -v -f
```

It runs ordinary AR with PP2 on both P and D, in eager and graph modes, using
actual Mooncake TCP P/D and Store plus an HTTP Catalog test double. This guards
existing PP serving after the result-channel edit. Deterministic DSpark result
fixtures do not prove PP speculative model execution, source-KV NCCL assembly,
trained-draft quality, full mixed-microbatch scheduling or performance SLOs.
Retained results and source hashes are in `pipeline-dspark-result.json`.
