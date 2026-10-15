# MUSA SDK 5.3 Migration

## Scope

- Upgrade `docker/musa.Dockerfile` from the SDK 5.2 package family to 5.3.
- Build AOT MUSA kernels with C++20 through MCC.
- Keep the current JIT changes and the existing `musa43` wheel path unchanged.

## Current Changes

- Docker package names use the `5-3` package family.
- `python/sglang/kernels/aot/setup_musa.py` passes `-std=c++20` and keeps
  `force_mcc` as the compiler selection for the C++ extension.

## Remaining Work

1. Validate the SDK 5.3 packages and `mcc --version` in the candidate runner.
2. Build the candidate image and record its image digest and environment
   fingerprint in private validation notes.
3. Compile a minimal AOT kernel and run the MUSA AOT/JIT smoke tests.
4. Verify old kernel builds and caches do not mix with the SDK 5.3 build.
5. Adapt the MUSA CI and wheel release workflows only after the candidate image
   passes; preserve the existing `musa43` entry point.

## Status

- Dockerfile and AOT source changes: prepared for the formal PR.
- Candidate image: not built.
- SDK 5.3 package availability: pending candidate-runner validation.
- CI runner registration: pending.
