# Target-only prefill with speculative decode (experimental)

GEN can initialize EAGLE draft state after receiving target-only CTX KV and its already-sampled handoff token. It replays the committed prefix through the target model without resampling the handoff, then runs draft prefill to build real draft KV, hidden states and proposal probabilities. This adds full-prefix target and draft compute; it is a correctness fallback, **not a throughput optimization**.

## Configuration and contract

Leave speculative options absent on CTX. On GEN, set `SGLANG_RUST_SERVER=0` and use the usual single-layer EAGLE options plus:

```text
--speculative-use-rejection-sampling
--disaggregation-decode-draft-bootstrap
--disaggregation-decode-draft-bootstrap-max-tokens 65536
```

Supported configurations are TP-only DeepSeek V3 / GLM MoE text models with Mooncake and no decode radix cache, HiCache, KV offload or HiSparse. Constrained decoding, input embeddings and multimodal requests are rejected. The token limit rejects oversized prefixes; it never truncates them. Python egress is required because native egress does not carry replay accounting.

| CTX | GEN | Draft-state source |
| --- | --- | --- |
| No draft | No draft | None; unused rejection-sampling flags do not matter |
| EAGLE draft | EAGLE draft | Existing CTX-seeded path; rejection-sampling settings must agree |
| No draft | EAGLE + bootstrap | GEN full-prefix replay |

Both peers must advertise their draft capability. The receiver allocates its wire layout before registration, so one listener cannot mix seeded and target-only peers. Missing, invalid or inconsistent capabilities fail closed.

Unsupported requests are rejected before bootstrap model execution using existing prebuilt-request cleanup. Overlapping work is drained before runner buffers are reused. Incomplete KV mappings and runtime CUDA/NCCL/model failures remain worker-fatal; draft state is published only after initialization succeeds.

Response metadata reports additional target replay tokens as `pd_draft_bootstrap_tokens`, accumulated across rebootstrap. CTX `cached_tokens` is unchanged: computed-throughput accounting must include this extra work separately. The counter does not measure draft FLOPs or energy.

## CPU checks

Run `python test/manual/disaggregation/pd_draft_bootstrap/run_cpu_tests.py` with PyTorch installed. This source-isolated runner loads the checked-in handoff tests while bypassing the SGLang package's CUDA import chain. It does not exercise real model runners or transport.

In a full SGLang development environment, also run:

```bash
python -m pytest test/registered/unit/disaggregation/test_draft_bootstrap.py test/registered/unit/disaggregation/test_draft_bootstrap_args.py test/registered/unit/disaggregation/test_disaggregation_wire.py test/registered/unit/disaggregation/test_register_to_bootstrap.py
```

## Validation scope

Unit tests exercise capability negotiation, replay boundaries, handoff preservation, failed initialization and rejected-request cleanup. Source-isolated tests use CPU tensors and mocked model runners; they do not establish GPU, transport or model-distribution correctness.

Before enabling this experimental path in production, run seeded and target-only-prefill configurations with the same model and sampling settings, both with and without overlap. Include concurrent requests, rebootstrap, oversized prefixes, and unsupported requests. Verify terminal usage, unchanged handoff tokens/logprobs, replay counters, and sampling quality. No upstream GPU results are claimed by this port.
