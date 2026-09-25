# Rawsystemone implementation report

## Baseline and environment

- Specification: `sglang_rawsystemone_spec_v1_2.md`, revision 1.2.
- Local baseline commit: `182f62d2deb2fa2ff335f1ea74cf5020c0d65f11`.
- Existing untracked `kotlin/` and the specification were left intact.
- Local Python: 3.12.9; Pydantic: 2.11.4.
- No PyTorch or Transformers installation, no cached checkpoint, and no working
  NVIDIA driver (`nvidia-smi` cannot communicate with the driver).
- No model checkpoint, precision, tokenizer, or GPU hardware was used for the
  local mocked tests. They must not be represented as inference benchmarks.

## Implemented

- Native `/v1/rawsystemone` in the existing authenticated HTTP server/lifespan;
  no Engine creation, loopback inference HTTP, extra model, or compiler call.
- Strict schemas; native prefix tokenization once and separately encoded options
  without added special tokens; conditional suffix means followed by stable
  softmax over original options, stable host summation, complete
  token-position auditing, ordered duplicates and first-index ties.
- Entire-request validation against option, cumulative token, native context,
  and scheduler KV-pool limits before GPU work; no automatic truncation.
- Deterministic deduplication/token-prefix planning, one prefix-only native
  operation, request-local shared likelihood records, and independent native
  branch batches. Dispatch follows `TokenizerManager.score_request`'s native
  `GenerateReqInput` batch precedent without its label-token/MIS semantics.
- Count/token partitioning and work-conserving bounded workers; FIFO weighted
  global admission, bounded parent queue, deadlines, fail-fast cancellation,
  unique child IDs, generator closure, and admission release.
- Full-sequence batched reference/fallback for disabled caching, short common
  spans, and one unique candidate. The private reference switch is not an HTTP
  request field and never uses shared-prefix merging.
- A monotonic tokenizer-manager epoch counts weight-control dispatches and
  replies, including same-path updates, failed/partial updates, and session
  begin/end. The service checks the epoch between/after phases without nested
  reader locks; existing per-child weight locks remain unchanged.
- Default per-parent cache isolation, inherited middleware salts, no-content
  logging, native trace propagation, and count/timing/cache diagnostics.
- Two narrow supporting fixes: honor the existing `no_logs` field in request
  logging, and preserve the last valid vocabulary ID in native input-logprob
  metadata (the previous upper-bound check replaced it with zero).
- API documentation, runnable client, live reference/teacher-forcing suite,
  and reproducible HTTP benchmark harness.

## Validation status

Local results:

- **37 CPU tests passed**, no skips, with
  `PYTHONPATH=/tmp/rawsystemone-tools python test/registered/unit/entrypoints/test_rawsystemone.py`.
  The temporary path contains msgspec and formatting tools; normal server
  environments already provide msgspec. Tests cover native sampling defaults,
  CLI declarations/runtime initialization, isolated HTTP routing, actual native
  manager batch dispatch and logprob assembly, and mocked composite lifecycle.
  The September 25 amendment also covers fixed prefix boundaries, conditional
  suffix sums, unequal option lengths, stable softmax, duplicate weights, and
  zero-token prefix/option errors.
- `python test/manual/test_rawsystemone.py`: **4 tests skipped** because no
  `RAWSYSTEMONE_URL` was supplied. This is discovery validation, not GPU parity.
- Ruff 0.15.1 (`F401,F821,UP037` and format), isort 7.0.0, and Python 3.10 syntax
  parsing passed for the changed Python files. `git diff --check` passed.
- Client and benchmark `--help` commands passed. A mocked HTTP smoke run exercised
  six benchmark modes and score comparisons; its synthetic timings are not
  published as performance measurements.
- Mintlify `validate` and `broken-links` passed after installing the CLI on
  September 25. This supersedes the earlier dependency-installation failure.

**Not run:** real server startup, GPU/native-model parity, independent
teacher-forced model inference, authenticated live endpoint regressions,
chunked-prefill/tensor-parallel device tests, real GPU disconnect/update cleanup,
or latency/throughput/prefill benchmarks. No measured speedup or physical prefix
reuse result is claimed. The specification's hardware acceptance/definition of
done remains pending these runs; runnable commands are in [README.md](README.md).

## Explicit limits

Initial admitted architectures: text-only Llama, Qwen2, Qwen3, and Mistral causal
decoders. Other architectures are rejected instead of inventing likelihoods.
Data/pipeline parallelism, multiple tokenizer workers, disaggregation,
speculative/diffusion/MIS, multimodal/encoder-decoder, and non-generation modes
are rejected. One tokenizer worker is necessary for the process-wide admission
and weight epoch to cover the entire endpoint. Tensor parallel and chunked
prefill use the existing native path but are not yet empirically validated.

Cache eviction/page alignment can reduce reuse without altering score coverage.
Native cached/uncached-input counters are diagnostics, not FLOP or GPU-overlap
measurements. The default salt prevents reuse across parents. Global persistent
prefix-score caching, single-token gathering, and the optional LLM compiler are
not implemented.
