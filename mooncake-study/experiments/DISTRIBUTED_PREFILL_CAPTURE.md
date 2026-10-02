# Distributed Prefill Graph Capture

Pipeline prefill now owns the activations received from the preceding stage.
The prefill input registry stages hidden states, residuals and optional routing
indices on the token axis, clears padded rows, and rejects missing or malformed
non-first-stage inputs before replay. Decode and prefill share the activation
allocation geometry, including blocked residuals and mHC hidden widths.

Full and Breakable graphs clone staged activations inside the captured body.
Torch.compile piecewise uses stable views of the staging buffers, refreshed
before each replay. Its CUDA subgraphs bind input addresses, so making a new
clone outside the compiled model would silently replay with stale pointers.
Fused norms may mutate staged residuals; the next load restores current inputs.
Every intermediate `PPProxyTensors` output is trimmed to the live token count
before the scheduler transfers it to the next stage. Piecewise runtime tests
enable the compiler's captured-input-address assertion.

Layer discovery retains `PPMissingLayer` entries as holes in the attention
table. Split attention operators index that table with global layer IDs. The
previous compacted table both disabled prefill graphs and would misaddress
later-stage layers if only the graph eligibility check were relaxed. Unknown
local layers still fail the existing complete-discovery check.

## Reproduction

Use two CUDA devices, matching SGLang dependencies, the pinned Mooncake SDK,
and `mooncake_master` on `PATH`. See `h100-runtime-lock.json`. On the recorded
base image, remove its optional old FlashInfer cubin/JIT-cache wheels from the
container environment before using the newer dependency overlay.

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pipeline_prefill_graph.py -v -f

PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_tensor_prefill_graph.py -v -f
```

Each file runs AR and static KV-input DSpark against Full, Breakable and
torch.compile piecewise prefill graphs. PP2 uses synchronous scheduling; TP2
uses overlap. Piecewise retains the default `tc_compiler=eager`. Decode or
target verify uses Full graphs. A separate eager run with both graph modes
disabled establishes greedy output references and the target contract for
the synthetic two-layer draft.

Each graph case publishes ten samples: a chunked 271-token prompt, a cached
one-token response, a cached extension, two three-request batches, and an
unbiased six-token response. The test compares generation with the eager
reference, requires actual prefill replay on every rank, checks PP output row
counts, token/request padding, prefix lengths 128/256/271 and buffer reuse.
DSpark additionally checks acceptance, rejection, verify replay, encoded KV
and writes into the draft's KV pool on every rank.

The fixture uses BF16 Qwen3-0.6B, target FlashInfer, draft Triton, selected
layers 0/14/27, a 4096-token KV pool, 128-token prefill chunks and graph buckets
16/32/64/128. KV and compact teacher D2H batches contain 16 rows and share an
8 MiB device budget. Raw teacher values are observed before serving bias.

After the producer exits, a new Mooncake client validates manifests, object
digests, topology and completeness. Selected KV and raw top-128 values must
equal the independent online observations exactly; IDs, full-vocabulary LSE,
accepted tokens, loss masks, positions and terminal KV validity are checked
as well. No target forward reconstructs missing data. Catalog is an HTTP
test double, and Store transport is TCP.

The single-GPU AR and DSpark prefill files remain regression gates. CPU tests
cover global layer indexing, owned activation buffers, padding reuse, input
validation, output trimming, decode geometry and existing graph helpers.

## Results

The primary PP2 matrix passes six tests in 284.237s; TP2 passes six in
330.710s. Each publishes 60 graph samples and ten eager baseline samples,
with 22 observed prefill replay frames per graph case across both ranks.
Every sample passes exact post-exit Store comparison. The ordinary single-GPU
four-case regression passes in 148.103s, and DSpark passes in 190.909s.

The initial PP matrix's unbiased request exposed stale piecewise input
addresses: generated IDs started with 0 instead of the eager reference's 8.
The corrected matrix passes without relaxing the output or tensor checks.
Earlier import/setup and test-invocation failures are retained separately.

CPU verification passes 83 distinct methods: 37 registry, 21 other runner,
six final PP runner, 11 prefill helper, three layer-discovery and five prefix
gate methods. Source hashes, command lines, test summaries, supplemental
address-check runs, failed attempts and allocation cleanup are retained in
`distributed-prefill-capture.json`. The TP2 and single-GPU full matrices ran
before the PP-only pointer fix; those PP1 paths are unchanged. Their piecewise
cases additionally run with the final source and input-address assertions.

The final supplemental address checks pass two TP2 tests in 138.814s, one
single-GPU AR test in 58.250s and one single-GPU DSpark test in 90.940s.
Across primary and supplemental acceptance runs, 283 complete snapshots pass
post-exit readback, including independently generated eager/contract seeds.
All 22 submitted jobs are terminal; expected baseline failures and preliminary
passes are excluded from acceptance totals. The temporary two-H100 allocation
was deleted and its pod is absent. The resident H100 resumed idle load with
no queued or active experiment.

## Limits

These instrumented tests copy reference tensors and synchronize the device;
their duration is not serving-performance evidence. The draft is synthetic
and untrained. Mixed TP/PP prefill graphs, P/D prefill graphs, wider model
families, MLA prefix graphs, asynchronous PP, Inductor, this combination over
RDMA, production Catalog retention, trained quality and workload SLOs need
their own validation. Existing decode/P/D/RDMA evidence remains separate.
