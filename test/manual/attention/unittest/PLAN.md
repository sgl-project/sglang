# Attention Backend Unit Test Plan

## Current progress

Last updated: 2026-05-26

Implemented:
- Shared dense MHA/GQA correctness helpers exist in
  `test/manual/attention/unittest/common/attention_methods/dense_attention.py`.
- Dense attention backend correctness files exist under
  `test/manual/attention/unittest/dense/` for `torch_native`, `triton`,
  `flashinfer`, `fa3`, `fa4`, `flex_attention`, and decode-only `trtllm_mha`.
  Dense/SWA expected paths now use separate HF-style reference modules with
  copied random projection weights instead of calling projection helpers on the
  SGLang actual module.
- SWA attention backend correctness files exist under
  `test/manual/attention/unittest/swa/` for `triton` and `flashinfer`.
- MLA attention backend correctness exists under
  `test/manual/attention/unittest/mla/` for `triton`, `flashinfer`, and
  `flashmla`; its actual path now uses a small DeepSeek-shaped absorb-MLA module
  that explicitly writes latent KV through `get_token_to_kv_pool()` before calling
  `attn_mqa`, while its expected path is a separate HF-style PyTorch reference
  module with copied random weights and no `RadixAttention` or backend calls. The
  MLA fixture now supports nonzero `qk_rope_head_dim` and passes a full
  `[latent, rope]` query tensor so shape-constrained MLA kernels can be tested
  with realistic DeepSeek-like `kv_lora_rank=512`, `qk_rope_head_dim=64`
  metadata.
- GDN hybrid-linear attention backend correctness exists under
  `test/manual/attention/unittest/gdn/` for full-attention backends `torch_native`,
  `triton`, and `flashinfer` with linear-attention kernel backend `triton`; its
  expected path is now a separate reference module plus pure PyTorch gated-delta
  recurrence, not the Triton/FLA GDN kernels. The FlashInfer GDN file uses 64-dim
  heads to satisfy FlashInfer SM90 prefill kernel constraints. GDN tree-verify
  tests use a scoped `5e-2` absolute tolerance for bf16 recurrent-kernel
  accumulation differences; chain verify and non-spec GDN paths keep `3e-2`.
- Each attention-method folder now has a local `README.md` capability matrix and
  progress summary. Implemented folders are `dense/`, `swa/`, `mla/`, and `gdn/`.
  Placeholder folders now exist for not-yet-implemented method fixtures:
  `dual_chunk/`, `dsa/`, and `dsv4/`.
- Phase 3 dense runner integration is implemented for representative attention
  backends: eager mode for `torch_native`, and CUDA-graph metadata capture/replay
  decode mode for `triton` and `flashinfer`. Runner coverage now includes MHA,
  GQA, and MQA decode variants; the CUDA-graph tests capture against a fixed padded
  decode batch and replay distinct request metadata/input tensors, so capture data
  is not identical to forward data.
- Phase 3 CUDA graph decode replay now also covers SWA for `triton` and
  `flashinfer`, MLA for `triton`, `flashinfer`, and `flashmla`, plus GDN for
  `triton` and `flashinfer`. The SWA graph cases currently use decode lengths within the
  configured window; an above-window Triton SWA decode case exposes a
  backend/reference semantic mismatch before graph replay and should be
  investigated separately from runner coverage.
- Phase 3 PCG/BCG split-op replay now covers representative dense MHA/GQA
  extend for `triton`, `flashinfer`, `fa3`, `fa4`, and `flex_attention`, plus SWA
  extend, GDN extend paths for both `triton` and `flashinfer`, and MLA extend
  paths for `triton`, `flashinfer`, and `flashmla`. These tests use live backend
  metadata with a larger static token buffer to verify `num_token_non_padded_cpu`
  slicing, not just exact-shape eager behavior.
- MLA split-op coverage exercises the absorb-MLA cached-KV path where
  `RadixAttention` receives `k/v=None`; the split-op wrapper now preserves that
  contract instead of assuming materialized K/V tensors.
- Phase 4 synthetic speculative metadata coverage has started for representative
  backends. Dense `triton` and `flashinfer` now cover `TARGET_VERIFY` chain
  (`topk=1`) and tree (`topk=2`) masks for `EagleVerifyInput`, plus chain verify
  metadata for `FrozenKVMTPVerifyInput`, `DFlashVerifyInput`, and
  `NgramVerifyInput`. Dense `flashinfer` also covers `DRAFT_EXTEND` with ragged
  accepted-token counts for both EAGLE and Frozen-KV MTP draft-extend input tags.
  SWA `triton` covers EAGLE `TARGET_VERIFY` chain and tree masks combined with a
  finite sliding window. MLA `triton` covers EAGLE `TARGET_VERIFY` chain masks;
  MLA `flashinfer` and `flashmla` cover EAGLE chain verify and EAGLE draft-extend
  on supported DeepSeek-like shapes. GDN `triton` and `flashinfer` cover EAGLE
  chain and tree verify against the pure PyTorch gated-delta recurrence
  reference.
- Phase 4 target-verify CUDA-graph-style replay now covers representative valid
  backends with fixed capture batches and distinct replay metadata/input tensors:
  dense `triton` covers EAGLE tree, DFlash chain, and NGRAM chain; dense
  `flashinfer` covers EAGLE tree, Frozen-KV MTP chain, and DFlash chain; SWA
  `triton` covers EAGLE tree with sliding-window metadata; MLA `triton` covers
  EAGLE tree with the absorb-MLA cached-KV path; MLA `flashinfer` and `flashmla`
  cover EAGLE chain with the absorb-MLA cached-KV path; GDN `triton` and
  `flashinfer` cover EAGLE chain and tree with speculative Mamba state buffers.
- Phase 4 draft-extend CUDA-graph-style replay now covers dense `flashinfer` for
  EAGLE and Frozen-KV MTP `DRAFT_EXTEND`, plus MLA `flashinfer` for EAGLE
  `DRAFT_EXTEND`, with ragged accepted-token counts. The capture batch uses a
  fixed max accepted-token count per request, while replay uses distinct ragged
  request metadata/input tensors.
- Phase 4 `DRAFT_EXTEND_V2` CUDA-graph-style replay now covers dense and MLA
  `triton` with a fixed token count per request, matching the multi-layer EAGLE
  v2 graph contract where attention metadata receives prefix lengths and the
  draft token count is carried separately by the graph/spec buffers. The MLA case
  exercises the absorb-MLA cached-KV path with distinct capture/replay metadata
  and input tensors.
- Phase 4 production draft-runner integration now uses the same adapter-based
  runner lifecycle as draft-extend and target-verify graph replay. The shared
  `common/runner_modes/eagle_draft_runner.py` helper owns production
  `EAGLEDraftCudaGraphRunner`, `EAGLEDraftExtendCudaGraphRunner`, and
  `FrozenKVMTPCudaGraphRunner` capture/replay, while attention-method wrappers
  provide fixture/input/state callbacks. Dense `triton` and `flashinfer` cover
  EAGLE draft decode for chain (`topk=1`) and tree (`topk=2`) layouts. MLA
  `triton`, `flashinfer`, and `flashmla` cover EAGLE draft decode on absorb-MLA
  fixtures. Dense `flashinfer` covers production EAGLE `DRAFT_EXTEND` and
  Frozen-KV MTP draft decode; dense `triton` covers production EAGLE
  `DRAFT_EXTEND_V2`; MLA `flashinfer` covers production EAGLE `DRAFT_EXTEND`;
  and MLA `triton` covers production EAGLE `DRAFT_EXTEND_V2`. These tests capture
  fixed padded batches, replay distinct metadata/input tensors, reset shared graph
  input buffers between independent runner instances, and compare graph replay
  against the eager worker path. The comparison intentionally targets runner
  buffer/metadata compatibility; attention math remains covered by independent
  HF-style Phase 2/3 references.
- The synthetic EAGLE verify helper uses realistic target-verify semantics:
  `ForwardBatch.seq_lens` represents prefix KV lengths, while `spec_info`
  supplies the draft tokens, positions, retrieve indices, and custom tree mask.
  The expected path is an HF-style PyTorch custom-mask reference, not a second
  backend call.
- Triton `DRAFT_EXTEND` is intentionally not enabled yet. The current fixture
  exposes a mismatch against the HF-style reference even for narrow accepted-token
  layouts, while FlashInfer `DRAFT_EXTEND` passes. Keep this as a focused follow-up
  on Triton draft-extend metadata/reference semantics rather than weakening Phase 4
  checks.
- FlashInfer SWA `TARGET_VERIFY` is intentionally not enabled yet. The current
  FlashInfer sliding-window metadata updater expects prefix lengths that are not
  supplied by the target-verify path (`prefix_lens=None`), so the Triton SWA spec
  tests cover the interaction for now.
- FlashInfer MLA EAGLE tree verify (`topk=2`) is intentionally not enabled yet.
  Chain verify (`topk=1`) passes, but the tree custom-mask path currently
  mismatches the HF-style PyTorch reference on realistic MLA shapes.
- FlashMLA MLA `DRAFT_EXTEND` CUDA-graph replay is intentionally not enabled yet.
  The eager path passes, but capture currently raises
  `AttributeError: 'FlashMLABackend' object has no attribute 'cuda_graph_qo_indptr'`
  from the inherited FlashInfer MLA capture metadata path.
- Cutlass MLA and TRT-LLM MLA are hardware-gated in this environment. Cutlass MLA
  decode reports support only for compute capability 10.0, while the TRT-LLM MLA
  XQA path reports an SM120a/SM121a requirement. Keep their tests hardware-gated
  rather than enabling them in the default SM90 sweep.
- Tokenspeed MLA is not enabled yet because it requires an FP8 KV-cache fixture
  (`kv_cache_dtype=fp8_e4m3`); the current MLA unit fixture intentionally uses
  fp16 KV cache for reference parity.
- `dual_chunk_flash_attn` should be modeled as its own attention method fixture,
  not as a dense backend swap. The backend expects a packed five-way query
  projection (`query`, `succ`, `inter`, and critical variants), so the dense
  Q/K/V module is structurally wrong for it even when short sequences would make
  the mask mathematically close to ordinary causal attention.
- FA3/FA4 CUDA-graph replay is intentionally not enabled yet. Dense eager and
  PCG/BCG split-op paths match the HF-style reference, but the shared decode
  CUDA-graph helper currently mismatches on replay for both FA backends. Local
  probes also show larger-than-tolerance mismatches for FA3/FA4 EAGLE
  `TARGET_VERIFY` graph replay and `DRAFT_EXTEND_V2` graph replay, so keep them
  as a focused FlashAttention graph-metadata follow-up rather than enabling
  partial speculative graph coverage.
- `trtllm_mha` dense coverage is decode-only for now. Local SM90 probes show MHA,
  GQA, MQA, and page-size-32 decode match the HF-style reference, while prefill
  goes through FlashInfer TRT-LLM Gen FMHA and reports `Unsupported architecture`.
  TRT-LLM MHA also rejects page size 1 (`16/32/64/128` only), and the shared
  CUDA-graph decode helper currently mismatches on replay. Treat those as
  backend-specific Phase 3 follow-ups.
- Flex attention dense coverage required a small backend compatibility fix:
  PyTorch `create_block_mask` expects a plain mask function, so the Flex backend
  now exposes its causal/decode mask callbacks as static functions instead of
  bound methods.
- Dense input-config cases now cover page size 1, zero-prefix exact page,
  zero-prefix input lengths below/equal/above a page, prefix-length exact page,
  total-length exact page, total-length crossing a page boundary, ragged
  below/equal/above page-boundary batches, representative page-size-32 crossing,
  decode page-boundary batches, and bsz=1 decode with nonzero prefix.
- Dense attention-config cases cover GQA and MQA separately from input-layout
  coverage.
- The `flashinfer` file uses `head_dim=64` because FlashInfer SM90 prefill kernels
  require value head dim in `{64, 128, 256}`.
- SWA input-config cases cover no-prefix lengths below/equal/above the window for
  `triton` and `flashinfer`; `triton` also covers prefix lengths below/equal/above
  the window. Prefix+SWA for FlashInfer needs a more faithful metadata fixture
  before enabling.
- MLA and GDN representative Triton cases now mirror the dense input-shape edge
  coverage more closely: page size 1, zero-prefix exact page, input lengths
  below/equal/above a page, prefix exact page, total exact page, page-boundary
  crossing, ragged page-boundary batches, page-size-32 crossing, decode
  page-boundary batches, and bsz=1 decode.
- The harness uses random shared weights, real `ForwardBatch` metadata, and real
  KV/request pools. Reference implementations must be independent HF-style
  PyTorch modules or functions; they may receive copied weights from the actual
  module, but must not call SGLang attention modules, SGLang backend wrappers, or
  SGLang kernel helpers.
- Shared helpers are split by responsibility. Attention-method fixtures live
  under `common/attention_methods/` and build modules, inputs, references, and
  metadata. Runner orchestration lives under `common/runner_modes/` and owns
  CUDA graph/PCG/BCG/speculative capture and replay flows.
  The folder is intentionally named `runner_modes`, not `graph_runners`, because
  it also hosts eager-style and split-op runner coverage rather than only CUDA
  graph execution.
- Runner-mode helpers must use a shared adapter/lifecycle framework whenever the
  same runner contract can apply to more than one attention method or backend.
  Do not add one-off `run_*_case` implementations inside attention-method files or
  backend-specific wrappers when a generic runner-mode helper can own capture,
  replay, eager comparison, and lifecycle setup. Attention-method files should
  provide only method-specific callbacks such as fixture construction, input
  creation, state preparation, forward-batch construction, and output comparison.
  `common/runner_modes/cuda_graph_decode_runner.py` now uses one CUDA graph
  decode lifecycle for dense/SWA, MLA, and GDN through attention-family adapters.
  Reusable family-specific callbacks such as case cloning, random capture inputs,
  padded replay inputs, forward calls, and reference-output adapters live in the
  attention-method helper files rather than in the CUDA graph runner. GDN only
  supplies the recurrent-cache snapshot and restore hooks from the runner side.
- Speculative runner helpers are split by speculative forward mode:
  `common/runner_modes/speculative_target_verify_runner.py` owns
  `TARGET_VERIFY` custom-mask/retrieve-index metadata and verify graph replay,
  while `common/runner_modes/speculative_draft_extend_runner.py` owns
  `DRAFT_EXTEND`/`DRAFT_EXTEND_V2` accepted-token metadata and draft graph
  replay. Their CUDA graph capture/replay lifecycle is de-duplicated in
  `common/runner_modes/speculative_cuda_graph_runner.py`, following the
  adapter-based shape of `cuda_graph_decode_runner.py`. There is no root-level
  speculative runner shim; in-tree tests import the runner-mode modules directly.
- Production EAGLE draft graph-runner coverage follows the same adapter pattern:
  `common/runner_modes/eagle_draft_runner.py` owns the fixed-capture-batch
  `EAGLEDraftCudaGraphRunner` lifecycle, and attention-method callbacks provide
  module construction, draft inputs, replay-state setup, forward-batch
  construction, and output comparison.
- RoPE is intentionally omitted from the current unit-level runner x attention
  tests. These tests feed post-RoPE-equivalent Q/K tensors because rotary math is
  orthogonal to runner/backend metadata compatibility.

In progress:
- Locally runnable Phase 4 production draft-runner coverage is complete for the
  representative valid backends listed above. Remaining Phase 4 work is limited
  to backend-specific blockers and hardware-gated paths documented in the
  implemented/deferred bullets.

Next implementation steps:
- Expand Phase 2 to additional attention methods/backends with method-specific
  fixtures rather than forcing them through the dense harness. Priority candidates:
  hardware-gated MLA kernels (`cutlass_mla`, `trtllm_mla`/`tokenspeed_mla` where
  hardware and KV dtype support them), dual-chunk attention with a packed-query
  fixture, and DSA/DSV4-style methods.
- Keep `torch_native` SWA out of the matrix until the backend honors
  `RadixAttention.sliding_window_size`.
- Defer Phase 2 expansion for additional backends until representative Phase 3 and
  Phase 4 tests are passing.

Latest verification:
- Added production draft-runner coverage for MLA EAGLE draft decode, EAGLE
  draft-extend graph runners, and dense Frozen-KV MTP draft decode. The graph
  runner helper now resets shared CUDA graph input buffers between independent
  unit-runner captures so stale larger buffers from one runner contract cannot
  leak into the next.
- `python -m py_compile test/manual/attention/unittest/common/runner_modes/eagle_draft_runner.py test/manual/attention/unittest/dense/test_flashinfer.py test/manual/attention/unittest/dense/test_triton.py test/manual/attention/unittest/mla/test_flashinfer.py test/manual/attention/unittest/mla/test_flashmla.py test/manual/attention/unittest/mla/test_triton.py`
- `python test/manual/attention/unittest/dense/test_flashinfer.py -v`
  - Ran 10 tests in 4.469s after adding dense FlashInfer production
    draft-extend and Frozen-KV MTP graph-runner coverage.
- `python test/manual/attention/unittest/dense/test_triton.py -v`
  - Ran 8 tests in 3.727s after adding dense Triton production
    `DRAFT_EXTEND_V2` graph-runner coverage.
- `python test/manual/attention/unittest/mla/test_flashinfer.py -v`
  - Ran 9 tests in 2.761s after adding MLA FlashInfer production EAGLE draft
    decode and draft-extend graph-runner coverage.
- `python test/manual/attention/unittest/mla/test_flashmla.py -v`
  - Ran 7 tests in 2.491s after adding MLA FlashMLA production EAGLE draft
    decode graph-runner coverage.
- `python test/manual/attention/unittest/mla/test_triton.py -v`
  - Ran 8 tests in 3.705s after adding MLA Triton production EAGLE draft decode
    and `DRAFT_EXTEND_V2` graph-runner coverage.
- `python -m unittest discover -s test/manual/attention/unittest/dense -p 'test_*.py' -v`
  - Ran 27 tests in 24.701s after isolating graph-runner shared buffer state.
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 71 tests in 27.382s after the production draft-runner expansion.
- Documented the runner-mode requirement that reusable runner contracts must be
  implemented through shared adapter/lifecycle helpers under `common/runner_modes/`
  instead of local one-off capture/replay wrappers.
- `git diff --check`
- Refactored production `EAGLEDraftCudaGraphRunner` coverage into the
  adapter-based lifecycle in `common/runner_modes/eagle_draft_runner.py`. Dense
  `triton` and `flashinfer` wrappers now only provide method-specific callbacks
  for fixture creation, draft-input construction, replay-state setup,
  forward-batch creation, and layout validation.
- `python -m py_compile test/manual/attention/unittest/common/runner_modes/eagle_draft_runner.py test/manual/attention/unittest/dense/test_triton.py test/manual/attention/unittest/dense/test_flashinfer.py`
- `python test/manual/attention/unittest/dense/test_triton.py -v`
  - Ran 7 tests in 3.558s after refactoring production EAGLE draft-runner
    coverage.
- `python test/manual/attention/unittest/dense/test_flashinfer.py -v`
  - Ran 8 tests in 3.881s after refactoring production EAGLE draft-runner
    coverage.
- `python -m py_compile $(find test/manual/attention/unittest/common test/manual/attention/unittest/dense test/manual/attention/unittest/swa test/manual/attention/unittest/mla test/manual/attention/unittest/gdn -name '*.py' -not -path '*/__pycache__/*')`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 63 tests in 25.042s after refactoring production EAGLE draft-runner
    coverage.
- Renamed runner orchestration helpers from `common/graph_runners/` to
  `common/runner_modes/` because the folder includes eager-style, PCG/BCG
  split-op, speculative, and CUDA graph runner modes.
- Renamed speculative files to remove misleading graph-only names:
  - `common/runner_modes/speculative_target_verify_runner.py`
  - `common/runner_modes/speculative_draft_extend_runner.py`
- Added `common/runner_modes/speculative_cuda_graph_runner.py` so
  `speculative_target_verify_runner.py` and `speculative_draft_extend_runner.py`
  share one fixed-capture-batch CUDA graph capture/replay lifecycle instead of
  duplicating `run_*_cuda_graph_case` logic.
- `python -m py_compile $(find test/manual/attention/unittest/common test/manual/attention/unittest/dense test/manual/attention/unittest/swa test/manual/attention/unittest/mla test/manual/attention/unittest/gdn -name '*.py' -not -path '*/__pycache__/*')`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 61 tests in 22.570s after renaming runner helpers and de-duplicating the
    speculative CUDA graph lifecycle.
- Previous broad verification after moving shared helpers into
  `common/attention_methods/` and `common/runner_modes/`:
  `python -m py_compile $(find test/manual/attention/unittest/common test/manual/attention/unittest/dense test/manual/attention/unittest/swa test/manual/attention/unittest/mla test/manual/attention/unittest/gdn -name '*.py' -not -path '*/__pycache__/*')`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 61 tests in 22.504s after moving attention methods and graph runners into
    separate folders.
- Added per-attention-method capability summaries:
  - `dense/README.md`, `swa/README.md`, `mla/README.md`, `gdn/README.md`
  - Placeholder method packages: `dual_chunk/`, `dsa/`, `dsv4/`
- `python -m py_compile test/manual/attention/unittest/dual_chunk/__init__.py test/manual/attention/unittest/dsa/__init__.py test/manual/attention/unittest/dsv4/__init__.py`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 61 tests in 22.442s after adding per-method summary files and placeholder
    method folders.
- `python -m py_compile test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/mla/test_flashinfer.py test/manual/attention/unittest/mla/test_flashmla.py`
- `python test/manual/attention/unittest/mla/test_flashinfer.py -v`
  - Ran 7 tests in 1.304s after adding FlashInfer MLA `DRAFT_EXTEND`
    CUDA-graph replay coverage.
- `python test/manual/attention/unittest/mla/test_flashmla.py -v`
  - Ran 6 tests in 1.354s after confirming FlashMLA remains on supported eager
    draft-extend and verify/decode graph paths.
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 61 tests in 22.582s after adding FlashInfer MLA `DRAFT_EXTEND`
    CUDA-graph replay coverage.
- Probed dense FA3/FA4 speculative CUDA-graph candidates:
  - `run_dense_draft_extend_v2_cuda_graph_case` mismatched the HF-style reference
    for both FA3 and FA4 (`max abs diff ~= 0.618`).
  - `run_dense_spec_verify_cuda_graph_case(..., topk=2, spec_kind="eagle")`
    mismatched the HF-style reference for both FA3 and FA4 (`max abs diff ~= 0.115`).
- `python -m py_compile test/manual/attention/unittest/common/attention_methods/gdn_attention.py test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/gdn/test_triton.py test/manual/attention/unittest/gdn/test_flashinfer.py`
- `python test/manual/attention/unittest/gdn/test_triton.py -v`
  - Ran 5 tests in 1.228s after adding GDN EAGLE tree verify/replay coverage.
- `python test/manual/attention/unittest/gdn/test_flashinfer.py -v`
  - Ran 5 tests in 1.329s after adding GDN EAGLE tree verify/replay coverage.
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 60 tests in 22.611s after adding GDN EAGLE tree verify/replay coverage.
- `python -m py_compile test/manual/attention/unittest/common/attention_methods/gdn_attention.py test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/gdn/test_triton.py test/manual/attention/unittest/gdn/test_flashinfer.py`
- `python test/manual/attention/unittest/gdn/test_triton.py -v`
  - Ran 5 tests in 1.179s after adding GDN EAGLE chain verify and CUDA-graph
    replay coverage.
- `python test/manual/attention/unittest/gdn/test_flashinfer.py -v`
  - Ran 5 tests in 1.276s after adding GDN EAGLE chain verify and CUDA-graph
    replay coverage.
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 60 tests in 22.530s after adding GDN EAGLE chain verify and CUDA-graph
    replay coverage.
- `python -m py_compile test/manual/attention/unittest/dense/test_trtllm_mha.py`
- `python test/manual/attention/unittest/dense/test_trtllm_mha.py -v`
  - Ran 1 test in 0.534s after adding decode-only dense `trtllm_mha` coverage.
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 56 tests in 22.370s after adding decode-only dense `trtllm_mha`
    coverage.
- `python -m py_compile test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/mla/test_triton.py`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
  - Ran 6 tests in 1.012s after adding MLA Triton `DRAFT_EXTEND_V2`
    CUDA-graph replay coverage.
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 55 tests in 22.229s after adding MLA Triton `DRAFT_EXTEND_V2`
    CUDA-graph replay coverage.
- Probed additional backend candidates after FlashMLA: `cutlass_mla` decode is
  unavailable on local SM90 because it requires compute capability 10.0;
  `trtllm_mla` decode is unavailable because FlashInfer's XQA MLA path requires
  SM120a/SM121a; `tokenspeed_mla` construction requires FP8 KV cache; and
  `dual_chunk_flash_attn` requires a method-specific packed-query fixture.
- `python -m py_compile test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/mla/test_flashmla.py`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
- `python test/manual/attention/unittest/mla/test_flashinfer.py -v`
- `python test/manual/attention/unittest/mla/test_flashmla.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 54 tests in 22.393s after adding FlashMLA coverage and switching the MLA
    actual fixture to pass full `[latent, rope]` query tensors.
- `python -m py_compile test/manual/attention/unittest/common/attention_methods/dense_attention.py test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/common/attention_methods/gdn_attention.py test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/dense/test_triton.py`
- `python test/manual/attention/unittest/dense/test_triton.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 48 tests in 22.022s after adding dense Triton `DRAFT_EXTEND_V2`
    CUDA-graph replay coverage.
- `python -m py_compile test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/common/runner_modes/cuda_graph_decode_runner.py test/manual/attention/unittest/common/runner_modes/split_op_runner.py test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/mla/test_flashinfer.py`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
- `python test/manual/attention/unittest/mla/test_flashinfer.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 47 tests in 22.115s after adding FlashInfer MLA coverage and nonzero
    MLA rope-dimension fixture support.
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 41 tests in 21.629s after adding GDN torch-native coverage.
- `python -m py_compile test/manual/attention/unittest/gdn/test_torch_native.py`
- `python test/manual/attention/unittest/gdn/test_torch_native.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 39 tests in 21.608s after adding GDN FlashInfer coverage.
- `python -m py_compile test/manual/attention/unittest/gdn/test_flashinfer.py`
- `python test/manual/attention/unittest/gdn/test_flashinfer.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 36 tests in 21.417s after adding FlashInfer draft-extend CUDA-graph-style replay.
- `python -m py_compile test/manual/attention/unittest/common/attention_methods/dense_attention.py test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/dense/test_flashinfer.py`
- `python test/manual/attention/unittest/dense/test_flashinfer.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 35 tests in 21.392s after adding Phase 4 target-verify CUDA-graph-style replay.
- `python -m py_compile test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/common/attention_methods/dense_attention.py test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/dense/test_triton.py test/manual/attention/unittest/dense/test_flashinfer.py test/manual/attention/unittest/swa/test_triton.py test/manual/attention/unittest/mla/test_triton.py`
- `python test/manual/attention/unittest/dense/test_triton.py -v`
- `python test/manual/attention/unittest/dense/test_flashinfer.py -v`
- `python test/manual/attention/unittest/swa/test_triton.py -v`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
  - Ran 31 tests in 21.303s after adding FA3, FA4, and Flex dense coverage.
- `python -m py_compile python/sglang/srt/layers/attention/torch_flex_backend.py test/manual/attention/unittest/dense/test_flex_attention.py`
- `python test/manual/attention/unittest/dense/test_flex_attention.py -v`
- `python -m py_compile test/manual/attention/unittest/common/attention_methods/dense_attention.py test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/dense/test_fa3.py test/manual/attention/unittest/dense/test_fa4.py`
- `python test/manual/attention/unittest/dense/test_fa3.py -v`
- `python test/manual/attention/unittest/dense/test_fa4.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`
- `python -m py_compile test/manual/attention/unittest/swa/test_triton.py test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py`
- `python test/manual/attention/unittest/swa/test_triton.py -v`
- `python -m py_compile test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/dense/test_triton.py test/manual/attention/unittest/dense/test_flashinfer.py`
- `python test/manual/attention/unittest/dense/test_triton.py -v`
- `python test/manual/attention/unittest/dense/test_flashinfer.py -v`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
- `python -m py_compile test/manual/attention/unittest/common/runner_modes/speculative_target_verify_runner.py test/manual/attention/unittest/common/runner_modes/speculative_draft_extend_runner.py test/manual/attention/unittest/common/attention_methods/dense_attention.py test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/common/attention_methods/gdn_attention.py test/manual/attention/unittest/dense/test_triton.py test/manual/attention/unittest/dense/test_flashinfer.py test/manual/attention/unittest/mla/test_triton.py`
- `python test/manual/attention/unittest/dense/test_triton.py -v`
- `python test/manual/attention/unittest/dense/test_flashinfer.py -v`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
- `python test/manual/attention/unittest/gdn/test_triton.py -v`
- `python test/manual/attention/unittest/swa/test_triton.py -v`
- `python test/manual/attention/unittest/swa/test_flashinfer.py -v`
- `python -m py_compile test/manual/attention/unittest/common/runner_modes/cuda_graph_decode_runner.py test/manual/attention/unittest/common/attention_methods/dense_attention.py test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/common/attention_methods/gdn_attention.py test/manual/attention/unittest/dense/test_triton.py test/manual/attention/unittest/dense/test_flashinfer.py test/manual/attention/unittest/mla/test_triton.py test/manual/attention/unittest/gdn/test_triton.py test/manual/attention/unittest/swa/test_triton.py test/manual/attention/unittest/swa/test_flashinfer.py`
- `python test/manual/attention/unittest/dense/test_triton.py -v`
- `python test/manual/attention/unittest/dense/test_flashinfer.py -v`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
- `python test/manual/attention/unittest/gdn/test_triton.py -v`
- `python test/manual/attention/unittest/swa/test_triton.py -v`
- `python test/manual/attention/unittest/swa/test_flashinfer.py -v`
- `python -m py_compile python/sglang/srt/layers/radix_attention.py test/manual/attention/unittest/common/runner_modes/split_op_runner.py test/manual/attention/unittest/common/attention_methods/dense_attention.py test/manual/attention/unittest/common/attention_methods/mla_attention.py test/manual/attention/unittest/common/attention_methods/gdn_attention.py test/manual/attention/unittest/dense/test_triton.py test/manual/attention/unittest/dense/test_flashinfer.py test/manual/attention/unittest/mla/test_triton.py test/manual/attention/unittest/gdn/test_triton.py test/manual/attention/unittest/swa/test_triton.py test/manual/attention/unittest/swa/test_flashinfer.py`
- `python test/manual/attention/unittest/dense/test_triton.py -v`
- `python test/manual/attention/unittest/dense/test_flashinfer.py -v`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
- `python test/manual/attention/unittest/gdn/test_triton.py -v`
- `python test/manual/attention/unittest/swa/test_triton.py -v`
- `python test/manual/attention/unittest/swa/test_flashinfer.py -v`
- Previous broad sweep before the latest runner refactor:
  `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`

---

## Test File Layout

Tests are organized by attention method first and attention backend second:

```text
test/manual/attention/unittest/
  common/
    attention_methods/
      dense_attention.py
      gdn_attention.py
      mla_attention.py
    runner_modes/
      cuda_graph_decode_runner.py
      split_op_runner.py
      speculative_cuda_graph_runner.py
      speculative_draft_extend_runner.py
      speculative_target_verify_runner.py
  dense/
    test_fa3.py
    test_fa4.py
    test_flex_attention.py
    test_torch_native.py
    test_triton.py
    test_flashinfer.py
  swa/
    test_triton.py
    test_flashinfer.py
  mla/
    test_triton.py
    test_flashinfer.py
    test_flashmla.py
    test_trtllm_mla.py
  gdn/
    test_torch_native.py
    test_triton.py
    test_flashinfer.py
```

Each `test_<attn_backend>.py` file verifies one attention method against one
attention backend across supported runner modes, forward modes, and input configs.
Runner implementations such as CUDA graph capture/replay live in isolated
`common/*_runner.py` files and are imported by the attention-method tests; do not
duplicate runner orchestration inside each attention method helper.
When adding a new runner mode or production runner integration, first factor the
runner lifecycle into `common/runner_modes/` with an adapter contract. Backend or
attention-method tests should call that helper and pass callbacks, not reimplement
capture/replay or eager-vs-graph comparison logic locally.
Speculative decoding is not an attention-method folder; it is represented as
`ForwardMode`, runner mode, and synthetic spec metadata cases inside the affected
attention-method folder.

## Problem

### Issue 1 — Coverage gaps

Current tests do not cover representative combinations of:
- Runners: eager, CUDA graph, BCG/PCG, EAGLE and MTP runners with variants.
- Attention backends: many exist, but most are not covered by fast unit-level
  correctness tests.
- Input cases: batch size, prefix length, extend length, page size, ragged
  sequence lengths, sliding-window boundaries, and speculative tree layouts.

### Issue 2 — Test expense

Most coverage today is e2e: full server launch, model load, and eval. That is too
slow and too coarse for validating attention backend math and metadata.

---

## Approach

### Primary test target: attention-module boundary

The primary correctness tests enter through the smallest real attention module that
represents the model family under test. Do not call backend methods directly, and do
not force every case through `RadixAttention`.

Reason:
- For standard MHA/GQA/SWA, the natural boundary can be a small projected
  attention module before calling `RadixAttention`.
- For MLA, DSA, DSV4, linear attention, Mamba, and speculative verify paths, the
  real behavior includes projections, compression, sparse index metadata, state
  updates, tree masks, or other side effects before backend dispatch.
- Backend-only tests miss auxiliary calls such as
  `get_attn_backend().init_mha_chunk_metadata(forward_batch)` in the DeepSeek
  chunked-KV path.

Keep a tiny `RadixAttention` smoke-test suite for the leaf backend contract
(`q/k/v + ForwardBatch -> output`), but keep it outside the main correctness matrix.

### Unified module target adapter

Every attention family implements the same adapter contract:

```python
class AttentionModuleTarget:
    def build_runner(self, case) -> MockModelRunner: ...
    def build_backend(self, runner, case) -> AttentionBackend: ...
    def build_module(self, runner, case) -> torch.nn.Module: ...
    def init_shared_random_weights(self, module, reference, seed) -> None: ...
    def make_inputs(self, batch, case) -> dict[str, torch.Tensor]: ...
    def run_sglang(self, module, inputs, batch) -> torch.Tensor: ...
    def run_reference(self, reference, inputs, batch, dense_kv) -> torch.Tensor: ...
```

This gives one test harness, one capability system, and one input-preparation
pipeline while preserving model-specific behavior.

Recommended targets:

| Attention family | Primary test target |
|---|---|
| Standard MHA | Small GPT/LLaMA-style attention module |
| GQA | Small LLaMA/Qwen-style attention module |
| SWA | Small Mistral/Gemma-style sliding-window attention module |
| MLA | Small DeepSeek-style MLA contract module; direct `DeepseekV2AttentionMLA` tests are optional dispatch/performance-path coverage |
| DSA | DeepSeek sparse attention module |
| DSV4 | DeepSeek V4 attention module |
| Linear KDA/Lightning/GDN | Model-specific linear attention module, falling back to `RadixLinearAttention` only if no smaller model module is exposed |
| Mamba | Mamba/Mamba2 mixer module |
| Spec verify/draft | Module target plus synthetic `SpecInput` metadata |

### Reference implementations

Use an independent reference at the same module boundary whenever practical. No
checkpoint download is required: configs are hardcoded, and weights are random but
copied from the SGLang module into the reference.

Rule: correctness tests must not compare one SGLang/backend implementation against
another SGLang/backend implementation. The expected path may share tensors by
explicit copy, but it must not call `RadixAttention`, `RadixLinearAttention`,
attention backend wrappers, Triton/FlashInfer/FLA kernels, or SGLang helper methods
that encode backend-specific attention behavior.

Reference strategy by family:
- MHA/GQA/SWA: explicit PyTorch reference using dense Q/K/V after the same random
  projections and RoPE as the SGLang module. Use `F.scaled_dot_product_attention`
  with causal or sliding-window masks.
- MLA: explicit DeepSeek MLA math or a minimal HF-compatible attention module with
  copied random projection/compression weights. The actual path should exercise the
  MLA backend-facing contract: `q_nope -> w_kc`, latent KV cache writes through
  `get_token_to_kv_pool()`, `attn_mqa`, and `w_vc`. It does not need to reproduce
  every `DeepseekV2AttentionMLA` forward-method dispatch choice because those
  choices are performance paths that should be mathematically equivalent.
- DSA/DSV4: model-family reference that builds the equivalent sparse or compressed
  attention mask/index result, then compares the final module output.
- Linear KDA/Lightning/GDN and Mamba: compact PyTorch implementations whenever
  feasible. Kernel-to-kernel comparisons are only smoke tests and must be labeled
  as such.
- Speculative verify: explicit causal/tree masks built from synthetic `SpecInput`
  objects, then compared at the same module boundary.

### Real configs, random weights

Model configs (head counts, KV heads, head dim, window size, compression rank, etc.)
are copied from representative HuggingFace configs and hardcoded in the test file.
Attention weights are randomly initialized. For modules with learnable projections,
compression matrices, gates, or RoPE buffers, copy the same random tensors into the
reference before comparison.

### RoPE handling

RoPE is deliberately out of scope for these runner x attention unit tests. Use
post-RoPE-equivalent Q/K tensors or set the model-specific RoPE dimension to zero
when the backend supports that shape. RoPE-specific coverage should live in focused
model/rotary tests, not in every runner/backend compatibility case.

### Forward context

All module-level tests that dispatch through `get_attn_backend()` must publish the
active backend:

```python
with forward_context(ForwardContext(attn_backend=backend)):
    backend.init_forward_metadata(forward_batch)
    output = target.run_sglang(module, inputs, forward_batch)
```

### Capability-first enumeration

Before enumerating parametrized tests, define a capability table/helper:

```python
def supports(case) -> tuple[bool, str]:
    """Return (supported, skip_reason)."""
```

The helper gates invalid combinations by attention family, backend, forward mode,
hardware, graph mode, dtype, page size, and speculative mode. Invalid combinations
are skipped with explicit reasons; they are not silently omitted and not discovered
by crashing deep inside backend initialization.

---

## Attention Backends

### Registered backends from `attention_registry.py`

| Name | Class | Notes |
|---|---|---|
| `flashinfer` | `FlashInferAttnBackend` / `FlashInferMLAAttnBackend` | Primary CUDA backend; dispatches to MLA variant for MLA models |
| `triton` | `TritonAttnBackend` | Pure Triton kernels |
| `torch_native` | `TorchNativeAttnBackend` | PyTorch SDPA |
| `flex_attention` | `TorchFlexAttnBackend` | PyTorch flex attention |
| `fa3` | `FlashAttentionBackend` v3 | Hardware-gated |
| `fa4` | `FlashAttentionBackend` v4 | Hardware-gated |
| `flashmla` | `FlashMLABackend` | MLA-only |
| `cutlass_mla` | `CutlassMLABackend` | MLA-only |
| `trtllm_mha` | `TRTLLMHAAttnBackend` | TensorRT-LLM MHA |
| `trtllm_mla` | `TRTLLMMLABackend` | TensorRT-LLM MLA |
| `tokenspeed_mla` | `TokenspeedMLABackend` | MLA variant |
| `aiter` | `AiterAttnBackend` | AMD ROCm |
| `wave` | `WaveAttnBackend` | Wave attention |
| `ascend` | `AscendAttnBackend` | Ascend NPU |
| `dsa` | `DeepseekSparseAttnBackend` | DeepSeek sparse attention |
| `nsa` | Alias for `dsa` | Deprecated compatibility alias |
| `dsv4` | `DeepseekV4AttnBackend` / HIP radix variant | DeepSeek V4 compressed attention |
| `dual_chunk_flash_attn` | `DualChunkFlashAttentionBackend` | Long-context chunked attention |
| `intel_amx` | `IntelAMXAttnBackend` | Intel CPU AMX |
| `intel_xpu` | `XPUAttentionBackend` | Intel XPU |

### Wrapper / hybrid backends

| Name | Class | Notes |
|---|---|---|
| `hybrid_attn` | `HybridAttnBackend` | Composed when prefill and decode backends differ |
| `tbo` | `TboAttnBackend` | Two-batch overlap wrapper around child backends |
| `hybrid_linear_attn` | `HybridLinearAttnBackend` | Wraps full attention with Mamba/linear-attention layers |

### Linear / state-space backends

These are selected by model config through `attn_backend_wrapper`, not by direct
`--attention-backend` names in the same way as full attention backends.

| Family | Class | Notes |
|---|---|---|
| KDA | `KDAAttnBackend` | Kimi-style linear attention |
| Lightning | `LightningAttentionBackend` | Lightning attention |
| GDN | `GDNAttnBackend` | GDN linear attention |
| Mamba2 | `Mamba2AttnBackend` | Mamba/state-space path |

---

## Speculative Decoding Workers And Runners

### Workers

| Worker | Description |
|---|---|
| `EAGLEWorker` | EAGLE v1 draft worker |
| `EAGLEWorkerV2` | EAGLE v2 draft worker, tree/spec-v2 path |
| `StandaloneWorker` / `StandaloneWorkerV2` | Self-speculative, draft equals target |
| `MultiLayerEagleWorker` / `MultiLayerEagleWorkerV2` | Multi-layer EAGLE |
| `MultiLayerEagleDraftWorker` | Draft worker used by multi-layer EAGLE v2 |
| `FrozenKVMTPWorker` / `FrozenKVMTPWorkerV2` | Frozen-KV MTP |
| `DFlashWorker` | DFlash speculative worker |
| `NGRAMWorker` | N-gram draft worker |

### CUDA graph runners

| Runner | Description |
|---|---|
| `EAGLEDraftCudaGraphRunner` | EAGLE decode draft |
| `EAGLEDraftExtendCudaGraphRunner` | EAGLE draft-extend, including v2 mode |
| `MultiLayerEagleDraftExtendCudaGraphRunner` | Multi-layer EAGLE extend |
| `MultiLayerEagleMultiStepDraftExtendCudaGraphRunner` | Multi-layer, multi-step extend |
| `FrozenKVMTPCudaGraphRunner` | Frozen-KV MTP decode |

---

## Dimensions To Enumerate

### Capability matrix

Represent each test case as:

```python
@dataclass(frozen=True)
class AttentionCase:
    attention_method: str
    config_variant: str
    attention_backend: str
    forward_mode: ForwardMode
    runner_mode: str  # eager, cuda_graph, bcg, pcg, worker-specific graph path
    input_shape: str
    page_size: int
    dtype: torch.dtype
    hardware: str
```

Minimum capability dimensions:
- Backend supports attention family: MHA/GQA/SWA/MLA/DSA/DSV4/linear/Mamba.
- Backend supports attention config variant: MHA head layout, GQA head grouping,
  MQA, SWA, MLA ranks, sparse-index config, linear-attention config, etc.
- Backend supports mode: `DECODE`, `EXTEND`, `MIXED`, `TARGET_VERIFY`,
  `DRAFT_EXTEND`, `DRAFT_EXTEND_V2`.
- Backend supports hardware and dtype.
- Backend supports graph path: eager, CUDA graph, BCG, PCG.
- Model family supports page size and KV pool type.
- Spec worker supports `topk`, `num_draft_steps`, draft runner, and forward mode.

### Phase 2 matrix: module-level backend correctness

Execute the matrix in two passes:
1. **Representative-first pass**: finish comprehensive Phase 2 coverage for
   `torch_native`, `triton`, and `flashinfer`, plus method-specific representative
   paths already in scope such as Triton MLA and Triton GDN.
2. **Backend-expansion pass**: after Phase 2/3/4 are stable for the representative
   set, add more backend files such as `flashmla`, `cutlass_mla`, `trtllm_mha`,
   `trtllm_mla` and `dual_chunk_flash_attn`. `fa3`, `fa4`, and `flex_attention`
   dense backend files are now implemented.

| Dimension | Values |
|---|---|
| **Attention family** | Standard MHA, GQA, SWA, MLA, DSA, DSV4, linear KDA/Lightning/GDN, Mamba |
| **Attention config** | MHA, GQA with `num_heads / num_kv_heads > 1`, MQA with `num_kv_heads=1`, finite-window SWA, MLA rank variants, sparse-index variants |
| **Backend** | Representative-first subset, then capability-gated expansion subset |
| **Forward mode** | `DECODE`, `EXTEND`, `MIXED` |
| **Input shape** | Small ragged batch, exact-page batch, page-boundary batch, long prefix batch, bsz=1 decode |
| **Page size** | 1, representative paged value such as 32 or 64 |

Representative configs, hardcoded with no network access:

| Attention family | Representative model | Key config |
|---|---|---|
| Standard MHA | GPT-2/LLaMA-style tiny config | `num_heads=12, head_dim=64` |
| GQA | LLaMA-3/Qwen-style tiny config | `num_heads=32, num_kv_heads=8, head_dim=128` |
| SWA | Mistral/Gemma-style tiny config | `num_heads=8, num_kv_heads=4, head_dim=256, window_size < seq_len` |
| MLA | DeepSeek-V2-Lite | `qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512` |
| DSA | DeepSeek-V3 sparse config | DSA-specific head and index config |
| DSV4 | DeepSeek-V4 | DSV4 compressed attention config |
| Linear KDA | Kimi-style config | KDA linear config |
| Linear Lightning | Bailing-style config | Lightning config |
| Linear GDN | Hybrid GDN config | GDN config |
| Mamba | Mamba-2 | SSM config |

DeepSeek `MHA_CHUNKED_KV`, absorbed MLA, one-shot, ragged, and paged variants are
model forward-method choices, not `ForwardMode` values. They are intended to be
mathematically equivalent performance paths. Runner x attention compatibility tests do
not need to strictly reproduce every DeepSeek dispatch choice; add focused
dispatch-path tests only when validating those production dispatch decisions.

### Phase 3 matrix: runner and graph integration

| Dimension | Values |
|---|---|
| **Runner mode** | eager baseline, CUDA graph, BCG, PCG |
| **Backend** | Representative-first subset: `torch_native`, `triton`, `flashinfer`; add wrappers such as `hybrid_attn` and TBO after base runner coverage is stable |
| **Forward mode** | `DECODE`, `EXTEND`; selected speculative graph modes in Phase 4 |
| **Attention family** | Small representative subset, not the full Phase 2 matrix |

Two-step verification:
1. Eager module output matches the family reference.
2. Graph replay output matches the eager module output.

Graph consistency does not replace the reference comparison; it validates runner and
metadata bookkeeping after correctness has been established by the eager path.

Representative Phase 3 status:
- `torch_native` is covered in eager mode for decode, extend, GQA, and MQA because it
  is the PyTorch baseline backend and does not exercise CUDA graph replay.
- `triton` and `flashinfer` are covered in CUDA graph decode replay for MHA, GQA,
  and MQA, using distinct capture and replay batches.
- SWA is covered in CUDA graph decode replay for `triton` and `flashinfer` with
  decode lengths inside the configured window.
- GDN is covered in CUDA graph decode replay for the representative `triton`
  hybrid-linear backend path, including recurrent cache restore between capture and
  replay.
- PCG and BCG are covered at unit-test scope through split-op replay paths with
  active forward context and reference/eager comparison. This now includes dense
  `fa3`, `fa4`, and `flex_attention`; full server-level PCG/BCG capture can remain
  in registered integration tests.
- `hybrid_attn` and TBO composition should be added as focused Phase 3 cases after
  the base backend graph paths are stable.

### Phase 4 matrix: speculative decoding attention

Split Phase 4 into two layers.

#### Layer A: synthetic spec metadata attention tests

These tests construct synthetic `SpecInput` objects and validate attention metadata and
module output directly.

| Dimension | Values |
|---|---|
| **Spec info** | `EagleVerifyInput`, EAGLE v2 verify input, `FrozenKVMTPInfo`, `DFlashVerifyInput`, `NGRAM` verify input |
| **Forward mode** | `TARGET_VERIFY`, `DRAFT_EXTEND`, `DRAFT_EXTEND_V2` |
| **topk** | 1 and representative `>1` tree layouts (currently 2 in unit tests; add 4 for worker-shaped fixtures) |
| **num_draft_steps** | 1 and representative multi-step layouts |
| **Backend** | Representative valid backends first, usually `triton` and `flashinfer`; expand only after synthetic spec metadata and graph replay are stable |
| **Execution mode** | eager and CUDA graph where supported |

Reference strategy:
- `DRAFT_EXTEND`: causal prefill reference.
- `DRAFT_EXTEND_V2`: fixed-shape draft-extend reference that accounts for all
  speculative tokens, not only accepted tokens.
- `TARGET_VERIFY`: explicit tree mask built from `spec_info`, then reference
  attention against dense KV.

Also test:
- `get_verify_buffers_to_fill_after_draft()` shape and dtype.
- `update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)` contents.
- Tree-mask equivalence for `topk=1` chain draft and `topk=4` tree draft.

Current Layer A status:
- Dense `triton` and `flashinfer`: EAGLE `TARGET_VERIFY` chain/tree custom-mask
  coverage, plus Frozen-KV MTP, DFlash, and NGRAM chain verify metadata coverage.
- Dense `triton` and `flashinfer`: CUDA-graph-style target-verify replay with a
  fixed padded capture batch and distinct replay input tensors. Triton covers
  EAGLE tree, DFlash chain, and NGRAM chain; FlashInfer covers EAGLE tree,
  Frozen-KV MTP chain, and DFlash chain.
- Dense `flashinfer`: EAGLE and Frozen-KV MTP `DRAFT_EXTEND` ragged accepted-token
  coverage, including CUDA-graph-style replay with a fixed max accepted-token
  capture batch and ragged replay metadata.
- SWA `triton`: EAGLE `TARGET_VERIFY` chain/tree custom-mask coverage with sliding
  window enabled, including CUDA-graph-style target-verify replay for a tree mask.
- MLA `triton`: `TARGET_VERIFY` chain custom-mask coverage plus
  CUDA-graph-style target-verify replay for a tree mask.
- Deferred: Triton `DRAFT_EXTEND` until the fixture/reference semantics are
  clarified; production `DRAFT_EXTEND_V2` / multi-layer draft-runner graph
  coverage until the draft-runner buffer lifecycle is represented faithfully;
  FlashInfer SWA target verify until prefix-lens metadata is available in its
  sliding-window updater; GDN speculative verify until recurrent speculative-state
  setup is represented faithfully.

#### Layer B: worker and draft-runner integration tests

These tests use a small subset of workers/runners to prove they produce compatible
`ForwardBatch` and `SpecInput` metadata. They are not a full Cartesian product.

| Dimension | Values |
|---|---|
| **Spec worker** | `EAGLEWorker`, `EAGLEWorkerV2`, `StandaloneWorker`, `StandaloneWorkerV2`, `MultiLayerEagleWorker`, `MultiLayerEagleWorkerV2`, `FrozenKVMTPWorker`, `FrozenKVMTPWorkerV2`, `DFlashWorker`, `NGRAMWorker` |
| **Draft runner** | `EAGLEDraftCudaGraphRunner`, `EAGLEDraftExtendCudaGraphRunner`, `MultiLayerEagleDraftExtendCudaGraphRunner`, `MultiLayerEagleMultiStepDraftExtendCudaGraphRunner`, `FrozenKVMTPCudaGraphRunner` |
| **Execution mode** | eager worker path, CUDA graph worker path where supported |
| **Backend** | One or two representative valid backends per worker family |

DFlash is a speculative worker/path, not an attention backend. Phase 4 covers
speculative-only attention modes and metadata that Phase 2 cannot exercise.

---

## Implementation Phases

### Phase 1 — Infrastructure (`test/manual/attention/unittest/common/`)

#### 1a. Module target adapters

Create one adapter per attention family:

```python
class MHAAttentionTarget(AttentionModuleTarget): ...
class GQAAttentionTarget(AttentionModuleTarget): ...
class SWAAttentionTarget(AttentionModuleTarget): ...
class DeepSeekMLATarget(AttentionModuleTarget): ...
class DeepSeekDSATarget(AttentionModuleTarget): ...
class DeepSeekV4Target(AttentionModuleTarget): ...
class LinearAttentionTarget(AttentionModuleTarget): ...
class MambaAttentionTarget(AttentionModuleTarget): ...
class SpecVerifyTarget(AttentionModuleTarget): ...
```

Each adapter owns module construction, input preparation, reference execution, and
shape-specific assertions. The outer test harness should not need attention-family
branches except capability filtering.

#### 1b. Mock model runner

Subclass the real `ModelRunner`, overriding `__init__` to skip server startup:

```python
class MockModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        model_config,
        server_args,
        page_size,
        max_batch_size,
        max_context_len,
        token_to_kv_pool,
        dtype,
        device,
        **extra_fields,
    ):
        # Skip ModelRunner.__init__; assign fields directly.
        self.model_config = model_config
        self.server_args = server_args
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.req_to_token_pool = ReqToTokenPool(...)
        self.token_to_kv_pool = token_to_kv_pool
        self.token_to_kv_pool_allocator = ...
        self.sliding_window_size = getattr(model_config, "sliding_window_size", None)
        self.use_mla_backend = model_config.attention_arch == AttentionArch.MLA
        for k, v in extra_fields.items():
            setattr(self, k, v)
```

Use real `ReqToTokenPool` and real KV pools (`MHATokenToKVPool`, `MLATokenToKVPool`,
DSA/DSV4-specific pools) instead of ad hoc tensors. A real subclass means
`isinstance` checks pass and missing fields surface as `AttributeError`.

Unit fixtures run without distributed initialization. For single-rank unit tests,
patch attention tensor-parallel helpers such as `get_attention_tp_size()` to return
`1`, or initialize an equivalent local parallel-state fixture before constructing
backends that size buffers from attention TP metadata.

#### 1c. Forward context helper

```python
@contextmanager
def attention_test_context(backend):
    with forward_context(ForwardContext(attn_backend=backend)):
        yield
```

All module-level tests use this helper before calling `backend.init_forward_metadata`
or module `forward`.

#### 1d. Forward batch factories

One function per mode, making required fields explicit:

```python
def make_decode_batch(bsz, seq_lens, model_runner) -> ForwardBatch:
    """req_pool_indices, out_cache_loc, seq_lens, positions, forward_mode=DECODE."""

def make_extend_batch(bsz, prefix_lens, extend_lens, model_runner) -> ForwardBatch:
    """extend_prefix_lens, extend_seq_lens, extend_num_tokens, positions."""

def make_mixed_batch(decode_bsz, extend_bsz, ..., model_runner) -> ForwardBatch:
    """Combined decode and extend portions; forward_mode=MIXED."""

def make_target_verify_batch(spec_info, model_runner) -> ForwardBatch: ...
def make_draft_extend_batch(spec_info, model_runner, *, v2: bool = False) -> ForwardBatch: ...
def make_forward_batch(case, model_runner) -> ForwardBatch: ...
```

Factories must populate `req_to_token_pool.req_to_token` consistently with
`out_cache_loc`, prefix lengths, page size, and sequence lengths. Add assertions that
the dense reconstruction sees the expected token order.

#### 1e. KV setup and reconstruction

```python
def populate_prefix_kv(
    module,
    forward_batch: ForwardBatch,
    model_runner: MockModelRunner,
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
) -> None: ...

def reconstruct_dense_kv(
    module,
    forward_batch: ForwardBatch,
    model_runner: MockModelRunner,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense K/V in batch-major sequence order after SGLang forward."""
```

For MLA/DSA/DSV4 pools, reconstruction may return family-specific compressed or
expanded tensors as required by the reference.

#### 1f. Reference helpers

```python
def sdpa_mha_reference(...): ...
def sdpa_gqa_reference(...): ...
def sdpa_swa_reference(...): ...
def deepseek_mla_reference(...): ...
def dsa_reference(...): ...
def dsv4_reference(...): ...
def linear_attention_reference(...): ...
def mamba_reference(...): ...
def spec_target_verify_reference(...): ...
```

`make_tree_attn_mask(spec_info)` remains a standalone helper used by speculative
references.

#### 1g. Assertion helper

```python
def assert_close(case, ref, out):
    """Use dtype/family/backend-specific tolerances and print first mismatch."""
```

---

### Phase 2 — Backend correctness tests (`test/manual/attention/unittest/<attention_method>/test_<attention_backend>.py`)

For each supported `AttentionCase`:

1. Select the `AttentionModuleTarget`.
2. Build `MockModelRunner` with the hardcoded representative config.
3. Instantiate the backend through the same registry path used by `ModelRunner`.
4. Build the module and reference module/kernel.
5. Initialize shared random weights and deterministic input tensors.
6. Build `ForwardBatch` through the mode-specific factory.
7. Populate prefix KV for decode or prefix-extend cases.
8. Enter `attention_test_context(backend)`.
9. Call `backend.init_forward_metadata(forward_batch)`.
10. Call `target.run_sglang(module, inputs, forward_batch)`.
11. Reconstruct dense or family-specific KV after the forward pass.
12. Run `target.run_reference(...)`.
13. Assert output closeness with case-specific tolerances.

Required attention config cases:
- MHA where `num_heads == num_kv_heads`.
- GQA where `num_heads / num_kv_heads > 1`.
- MQA where `num_kv_heads == 1`, if supported by the target/backend.
- SWA with a finite window size.
- MLA, DSA, DSV4, linear, and Mamba family-specific configs.

Required input cases:
- Page size 1.
- Paged KV with representative page sizes such as 32 or 64.
- Sequence length exactly equal to one page.
- Sequence length one token below and one token above a page boundary.
- Prefix length exactly equal to one page.
- Prefix plus extend length exactly equal to one page.
- Prefix plus extend length crossing a page boundary.
- Ragged batch with requests below, exactly at, and above a page boundary.
- Decode with nonzero prefix.
- Extend with zero prefix and with nonzero prefix.
- SWA with `seq_len < window_size`, `seq_len == window_size`, and
  `seq_len > window_size`.

Invalid combinations are `skipIf`-guarded through the capability helper.

Optional dispatch-path cases:
- DeepSeek chunked-KV and other `attn_forward_method` choices can be covered by
  focused tests that intentionally instantiate the production DeepSeek module. These
  are not required for the main runner x attention compatibility matrix because the
  forward methods are performance paths with equivalent math.

Initial implementation slice:
- `common/attention_methods/dense_attention.py` contains a projected attention target that owns
  Q/K/V/O projections and dispatches through `RadixAttention`.
- `common/runner_modes/cuda_graph_decode_runner.py` owns CUDA graph capture/replay helpers for dense/SWA,
  MLA, and GDN. Attention-method files only enumerate cases and call runner helpers.
  Dense/SWA and MLA share one projected-attention graph helper; GDN uses a dedicated
  graph helper because capture/replay mutates recurrent cache state.
- `dense/test_torch_native.py`, `dense/test_triton.py`, and
  `dense/test_flashinfer.py` cover representative MHA, GQA, and MQA cases.
- Dense tests exercise page size 1, exact-page extend, page-boundary decode
  (`seq_len = page_size - 1`, `page_size`, `page_size + 1`), zero/nonzero prefix,
  and GQA head grouping as an attention-config case.
- `swa/test_triton.py` covers no-prefix and prefix SWA extend with
  `seq_len < window_size`, `seq_len == window_size`, and
  `seq_len > window_size`.
- `swa/test_flashinfer.py` covers no-prefix SWA extend across the same window
  boundary cases. Prefix+SWA for FlashInfer is intentionally not enabled yet
  because the current synthetic metadata fixture does not faithfully match that
  production path.
- FlashInfer cases use `head_dim=64` to match FlashInfer kernel constraints.
- `mla/test_triton.py` and `gdn/test_triton.py` cover the representative dense-style
  input edge cases for method-specific Triton paths.
- Future representative-first slices should complete remaining Phase 3 graph paths
  and then speculative modes for `triton` and `flashinfer` before adding additional
  Phase 2 backend files.

---

### Phase 3 — Runner integration tests (`test/manual/attention/unittest/common/*_runner.py`)

Use a smaller representative subset from Phase 2. The goal is runner bookkeeping, not
another full correctness matrix.

For each supported runner/backend/family case:

1. Run the eager module path and compare against the reference.
2. Capture/warm up the graph path:
   - CUDA graph: `init_cuda_graph_state`, capture metadata, warmup,
     `on_after_cuda_graph_warmup`, replay metadata.
   - BCG/PCG: exercise the split-op path with active piecewise/breakable forward
     context at unit scope; full graph-capture runner coverage belongs in focused
     registered integration tests.
3. Replay the graph path.
4. Assert graph replay output matches the eager output.

This phase should include SWA and GDN CUDA graph coverage before Phase 4. It should
also include `hybrid_attn` and TBO composition in a focused way instead of as part
of the full Cartesian product.

---

### Phase 4 — Speculative decoding attention (`test/manual/attention/unittest/test_spec_decoding_attention.py`)

#### 4a. Synthetic spec metadata correctness

For each supported synthetic spec case:

1. Construct synthetic `SpecInput` / EAGLE / Frozen-KV MTP / DFlash / NGRAM metadata
   matching `topk` and `num_draft_steps`.
2. Build `ForwardBatch` for `TARGET_VERIFY`, `DRAFT_EXTEND`, or `DRAFT_EXTEND_V2`.
3. Build module inputs and populate prefix/draft KV as required.
4. Enter `attention_test_context(backend)`.
5. Run `backend.init_forward_metadata(batch)` and module forward.
6. Build the explicit reference mask from `spec_info`.
7. Compare module output against the speculative reference.
8. Validate verify-buffer shape/content helpers.

#### 4b. Worker and draft-runner integration

For each selected worker family:

1. Run the worker path with tiny synthetic requests or minimal local fixtures.
2. Inspect the produced `ForwardBatch` and `SpecInput`.
3. Verify required fields, shapes, cache locations, positions, and mode.
4. For CUDA graph runners, compare graph replay output against eager output for the
   same worker/backend family.

Do not run a full Cartesian product of workers, draft runners, backends, `topk`, and
`num_draft_steps`. Use the capability matrix to select representative cases.

---

## Resolved Decisions

1. **Primary target boundary**: The main matrix uses model-level attention modules via
   `AttentionModuleTarget` adapters. Direct `RadixAttention` coverage is retained only
   as a small leaf-backend smoke suite.

2. **References**: References are family-specific. SDPA is only the reference for
   MHA/GQA/SWA-style dense attention and speculative masks after dense KV
   reconstruction; linear, Mamba, MLA, DSA, and DSV4 need their own references.

3. **Forward context**: All module-level tests publish `ForwardContext(attn_backend)`
   before backend metadata initialization and module forward.

4. **Chunked-KV**: `MHA_CHUNKED_KV` is a DeepSeek attention forward method, not a
   `ForwardMode`. Trigger it through the DeepSeek attention module by patching the
   module threshold or environment to a small value.

5. **Capability matrix**: Every parametrized case is filtered through an explicit
   support helper that returns a skip reason.

6. **Speculative decoding**: Phase 4 is split into synthetic metadata correctness and
   worker/draft-runner integration. It includes `DRAFT_EXTEND_V2`.

7. **DFlash**: DFlash is a speculative worker/path, not an attention backend.

8. **Mock model runner**: Use a `ModelRunner` subclass plus real request-token and KV
   pools so missing backend fields fail loudly and pool layout matches production.

9. **Graph verification**: Eager-vs-reference establishes correctness; graph-vs-eager
   establishes graph metadata and replay consistency.

---

## CI Registration

Use tiers instead of trying to fit the full matrix into one fast job.

- Fast CUDA PR tier: `register_cuda_ci()`, target under 60s. Cover representative
  MHA/GQA/SWA plus one MLA case across `torch_native`, `triton`, and `flashinfer`;
  include decode, extend, ragged lengths, and one paged-KV case.
- Medium CUDA tier: graph integration for selected backends and families, including
  CUDA graph/BCG/PCG where supported.
- Nightly CUDA tier: FA3/FA4/FlashMLA/Cutlass/TRTLLM/dual-chunk/hybrid/TBO and larger
  speculative matrices.
- Speculative tier: synthetic metadata correctness in fast/medium, worker integration
  in medium/nightly depending on cost.
- AMD/NPU/CPU/XPU tiers: register with the appropriate backend-specific CI helpers and
  skip CUDA-only runner modes.
