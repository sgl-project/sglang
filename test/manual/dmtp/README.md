# DMTP regression checks

The optional `SGLANG_DMTP_FUSED_T1_STATE=1` path requires
`SGLANG_MTP_PREFIX_TWO_DRAFT_GRAPHS=1`. It fuses request-slot T1 state
transport and validation; the default remains off. Slot reuse invalidates
cached proposals. The shared state module has no QSA attention dependency.
Use `test_turbo_fused_state.py` for byte-exact transport, request reuse,
sampling invalidation, and CPU/CUDA coverage.

Run from the repository root with a matching SGLang runtime:

```bash
python -m unittest discover -s test/manual/dmtp -p 'test_*.py' -v
```

The common dense path supports the configured tensor-parallel size; it is not
restricted to TP4. PP, attention DP/CP, and other unsupported recipes still fail
explicitly rather than silently changing the execution path.

`SGLANG_MTP_PREFIX_TWO_DRAFT_GRAPHS=1` records the MTP sampling/conditioning tail
and the DFlash sampling tail in their parent graphs. Request-indexed prefetched
proposals must be invalidated when slots are recycled or sampling state changes.
Graph weight/layout changes require a restart. The regression tests also cover
embedding/conditioner replacement and sampling-row refresh.

These operator and lifecycle tests do not replace real-model CUDA Graph, KV
parity, or tensor-parallel generation checks. QSA-specific optimizations are
maintained on `main`; this PR contains only common runtime changes.
