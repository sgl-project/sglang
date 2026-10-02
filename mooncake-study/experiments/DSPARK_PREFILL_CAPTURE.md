# Target-KV DSpark Prefill Graph Verification

The prefill graph runner now respects the resolved speculative hidden-state
requirement. A target-KV DSpark checkpoint leaves auxiliary hidden capture off;
the graph uses the configured server return mode (normally `NULL`). Hidden-input
DFLASH/DSpark targets still require `FULL`. Breakable EAGLE target/draft behavior
and explicit server return modes remain covered by constructor regression tests.

Run with one CUDA GPU, the matching SGLang dependencies, Mooncake SDK and
`mooncake_master` on `PATH`:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_dspark_prefill_graph.py -v -f
```

The fixture first verifies an ordinary AR Full-graph run through the real TCP
Store, then uses one saved sample to bind a synthetic KV-input draft to that
target's teacher/KV identity. Each case exports a separate two-layer draft with
a fixed Markov proposal. This checkpoint is a correctness fixture, not a trained
model or evidence of useful acceptance rates.

| Prefill Backend | Overlap | Target Verify Backend |
| --- | --- | --- |
| `breakable` | On | `full` |
| `full` | On | `full` |
| `full` | Off | `full` |
| `tc_piecewise` | On | `full` |

Target attention is FlashInfer, draft attention is Triton, and verify uses the
static four-token window. The target runs BF16 with selected KV layers 0/14/27,
a 4096-token pool, 128-token prefill chunks and graph buckets 16/32/64/128.
Full prefill reserves four request slots; target verify uses request buckets
1/2/4. Piecewise uses the default `tc_compiler=eager`, not Inductor. Both KV
and compact teacher D2H batching use 16 rows under an 8 MiB device budget.

Each speculative case publishes ten samples: a 271-token chunked prompt, its
cached one-token response, a 19-token cached extension, two three-request
batches with prompt lengths 33/37/43 and responses 8/7/1, and an unbiased
six-token response. The first and extended prompts have eight-token responses.
Biased requests exercise full draft acceptance; the unbiased request exercises
rejection. Serving bias is applied after the raw teacher observation/capture.

The test requires actual prefill and verify graph replays, token/request
padding, prefix lengths 128/256/271, and reuse of the same graph input buffer
across distinct calls. It checks both the captured and runtime hidden modes
and the actual replay output: all must omit hidden states for this KV-input
configuration. The independent projection observer checks encoded target KV
and per-layer draft KV writes, as well as verify outputs without hidden states.

Overlap is checked through the runtime capture counter and actual forwarded
tokens ahead of the scheduler's delivered result. AR advances one token;
DSpark can advance several accepted tokens. An initial test incorrectly
required exactly one for both algorithms; retained forward records showed
DSpark's three-token lead, and the corrected assertion requires a positive
speculative lead rather than using the AR-specific value.

After each producer exits, a new Store client validates every manifest and
object digest, compares selected KV and raw top-128 logits exactly to online
source observations, checks top-k IDs, full-vocabulary LSE within 1e-6, tokens,
loss masks, positions, terminal KV validity and sample completeness. No target
model is rerun to reconstruct data. Rejected and terminally trimmed verify
rows must not appear in the snapshot.

The ordinary four-case AR regression is
`test/registered/storage/test_training_capture_prefill_graph.py`; constructor,
padding, chunked-prefix and wrapper regression tests live under
`test/registered/unit/model_executor/`. Retained job results, commands and
source/log hashes are in `dspark-prefill-capture.json`.

The final speculative matrix passes four tests in 222.429s (40 speculative
samples plus nine AR seed samples). The ordinary matrix passes four tests in
148.444s (36 samples). Final CPU checks pass 7 hidden-mode tests, including 27
constructor subcases, 5 chunked-prefix tests, 6 wrapper tests and 2 padding
tests. Earlier failures and preliminary passes are retained separately in the
evidence. AR ran before the speculative-lag assertion correction and extra
summary field, with identical production code; its recorded counters also
satisfy the added overlap-counter assertion. The last change to the hidden-mode
unit test explicitly binds a lambda loop variable and has its own passing run.

These instrumented tests synchronize and copy full vocabulary logits for
reference. Their timing is not serving-performance evidence. Coverage is
single-GPU Qwen3 MHA, static KV-input DSpark, TCP Store and an HTTP Catalog
test double. Distributed prefill graphs, MLA-specific prefix graphs, mixed
prefill/decode batches, Inductor, production Catalog retention, trained draft
quality and workload SLOs require separate validation. Existing cross-node
RDMA evidence does not establish this new combination's RDMA behavior.
