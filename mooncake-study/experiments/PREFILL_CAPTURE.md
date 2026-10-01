# Prefill Graph Capture Verification

Run the complete registered test file with one CUDA GPU, the matching SGLang
dependencies, the Mooncake SDK and `mooncake_master` on `PATH`:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_prefill_graph.py -v -f
```

The fixture starts a real TCP Store with a separate 256 MiB segment, an HTTP
Catalog test double and one ordinary serving process at a time. The model runs
BF16 with FlashInfer attention, selected KV layers 0/14/27, a 4096-token pool,
128-token prefill chunks and 64-token storage chunks. It exercises these modes:

| Prefill Backend | Overlap | Decode Backend |
| --- | --- | --- |
| `breakable` | On | `full` |
| `full` | On | `full` |
| `full` | Off | `full` |
| `tc_piecewise` | On | `full` |

Prefill buckets are 16/32/64/128 tokens. Full prefill captures four request
slots; decode uses request buckets 1/2/4. The piecewise case uses the default
`tc_compiler=eager`, which still captures and replays CUDA graph pieces; it
does not certify Inductor compilation.

Each case publishes nine samples: a 271-token chunked prompt, its cached
single-token reply, a 19-token cached-prefix extension, and two distinct
three-request batches with prompt lengths 33/37/43 and reply lengths 4/3/1.
Serving logit bias runs after the raw teacher observation and capture.

The test-only server records the actual static input buffer address, replay
sequence, padded token count and Full graph request capacity. The source
observer records the live forward mode and per-request prefix length. Assertions
require real prefill replay, token padding, a three-request prefill batch, Full
request padding, chunk boundaries, cached-prefix replay, decode replay and
reuse of one input buffer by distinct replay calls. The observed token shapes
include 15-to-16 and 19-to-32 padding and a 113-to-128 three-request batch.
The cached one-token extend may fall back to eager because the smallest bucket
would exceed the runner's padding limit; its sample is checked as well.

After the serving process exits, a newly connected Store reader validates every
manifest and tensor digest. It compares all selected KV and raw top-128 values
exactly against observations from the online target; checks top-k membership,
full-vocabulary LSE (`rtol=atol=1e-6`), tokens, masks, teacher positions and KV
validity; and rejects missing, duplicate or extra samples. No target model is
run for reconstruction. Synchronous and overlap cases can have different final
KV validity because overlap may already have forwarded the final output token.

The observer copies full source logits to CPU only in the instrumented test server. It is
not enabled in normal serving and its timing is not performance evidence.
This lane covers Qwen3 MHA on one GPU, ordinary AR, FlashInfer and TCP Store.
It does not certify MLA's distinct chunked-prefix graph topology, distributed
prefill graphs, speculative prefill graphs, mixed prefill/decode batches,
production Catalog retention or serving SLOs. Cross-node RDMA evidence is
recorded separately in `RDMA.md` and `PD_RDMA.md`.
