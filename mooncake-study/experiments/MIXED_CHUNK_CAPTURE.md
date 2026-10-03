# Mixed Prefill And Decode Capture

Training capture now accepts ordinary autoregressive mixed-chunk scheduling.
The native scheduler appends ongoing decode requests to a chunked prefill
batch. Capture uses each request's actual extend length to locate fresh
positions and its own sequence length to decide whether a teacher row predicts
a response token. Incomplete prompt chunks contribute KV without teacher rows;
decode requests in the same batch still contribute their next-token teacher.

This reuses the existing owned KV/teacher staging, overlap commit ledger and
Mooncake publication protocol. It does not change the Store schema or require
tensor-specific fields in Mooncake Master. Mixed-chunk speculative execution
remains rejected, consistent with the serving scheduler's capability gate.

## Runtime Matrix

`test_training_capture_mixed.py` uses the real Qwen3-0.6B model, FlashInfer,
a real TCP Mooncake Store and the HTTP test Catalog on one H100. It runs:

- Synchronous eager and overlap eager.
- Synchronous and overlap with decode CUDA graphs.
- Breakable, Full and torch.compile piecewise prefill with overlap/decode graphs.

Each configuration starts a capture-off process, runs the complete workload,
then starts the capture-on process with the same configuration. It requires
exact output token equality, including unbiased requests. Both processes must
actually enable mixed-chunk scheduling. Piecewise uses eager compile debug
mode; Full prefill remains experimental.

| Request | Prompt Tokens | Response Tokens | Capture |
| --- | ---: | ---: | --- |
| Ongoing decode | 17 | 24 | Yes |
| Chunked prefill | 271 | 6 | Yes |
| Short prefill | 33 | 1 | Yes |
| Over-limit prefill | 529 | 1 | Excluded by the 512-token sample limit |
| Cached prompt | 271 | 4 | Yes |
| Fresh request | 40 | 4 | Yes |

A test-only receiver waits at an ongoing decode boundary until the three new
requests arrive, then passes them through the unchanged native scheduler. It
does not manufacture capture tickets, change the chosen batches, force model
outputs or replace the model/Store. Per-request logit bias exercises separation
of raw teacher values from sampling transforms. The short and fresh requests
are unbiased.

Assertions require real `MIXED` forward observations for the ongoing decode,
both incomplete and final chunks of the long prompt, and native mixed batches
containing the excluded request. The cached request must hit at least 270
prefix tokens. Graph configurations must observe mixed prefill replay and
ordinary decode replay. All five captured requests must publish, the sixth
must be explicitly excluded for length, no capture failure is allowed, and
all four reservations must become available without quarantined buffers.

After the serving process exits, a new Store client reads all objects and
checks selected KV, raw top-128 values/IDs, full-vocabulary LSE, token IDs,
positions, masks and final-token validity against online observations. The
observer reads source pools and raw model logits before reuse; it is not an
independent implementation of the attention backend's pool writes.

## Reproduction

Use the runtime environment recorded in `h100-runtime-lock.json`, a local
immutable Qwen3-0.6B model and `mooncake_master` on `PATH`:

```bash
export TRAINING_CAPTURE_TEST_MODEL=/models/Qwen3-0.6B
export PYTHONPATH=python
export OMP_NUM_THREADS=1
python -m pytest -q test/registered/unit/training_capture/test_config.py test/registered/unit/training_capture/test_coordinator.py
python test/registered/storage/test_training_capture_mixed.py -v -f
```

The unit test places unselected prefill rows before a selected decode row,
uses distinct logits for each request, and overwrites source KV/logits before
publication. It checks that row selection, position offsets and owned data
remain correct. Existing capability exclusions and coordinator lifecycle tests
run alongside it.

This lane does not certify mixed scheduling under TP/PP, P/D or RDMA, trained
draft quality, saturated load or serving SLOs. The receiver barrier and source
observations make it a correctness experiment, not a latency benchmark.

## Verified Results

On the resident H100, configuration/coordinator tests pass 69 tests and 70
subtests in 53.82 seconds. The initial synchronous smoke passes in 91.233
seconds. The final runtime suite adds resource-return assertions and passes
all seven methods in 599.592 seconds:

- 35 complete snapshots after producer exit, 826 tensor objects and
  57,941,350 tensor bytes.
- 42 capture-off requests with exactly matching capture-on output tokens.
- Seven explicitly excluded over-limit requests and zero capture failures.
- 123 observed mixed request-frames, including 54 prefill graph replay frames.
- All four reservations available and zero quarantined buffers in every cell.

The [machine-readable report](mixed-chunk-capture.json) retains job commands,
results, per-configuration observations and source/log hashes. Its archive has
25 artifacts under
`/gpfs/user/fuxuanwei/mooncake-lab-archive/mixed-chunk-20261003`. Source audit
checks 3,363 Python/source test files. The final runtime's only difference
from the committed code is its runtime-no-op CI declaration, corrected from
an unconfigured suite to `extra-a-test-1-gpu-small`; the exact single-line
replacement and actual CI parser result are checked separately. The unit
tests and production code are identical to their executed sources.

All three jobs finish successfully, the experiment queue is empty, and the
resident worker and resumed idle load are verified live. No new GPU allocation
was required, and the existing resident allocation remains available.
