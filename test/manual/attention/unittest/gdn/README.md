# GDN Attention Capability Matrix

This folder covers GDN hybrid-linear attention with a full-attention backend plus
the Triton GDN linear-attention kernel. Expected outputs use a separate pure
PyTorch gated-delta recurrence reference, not Triton/FLA GDN kernels.

## Current Matrix

| Full-attention backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `torch_native` | Full representative GDN input-shape sweep | PCG/BCG extend | Not enabled | No CUDA graph or speculative coverage. |
| `triton` | Full representative GDN input-shape sweep | CUDA graph decode; PCG/BCG extend | EAGLE chain/tree verify; EAGLE chain/tree CUDA graph replay | Tree verify uses a scoped `5e-2` absolute tolerance for bf16 recurrent accumulation. |
| `flashinfer` | Full representative GDN sweep with 64-dim heads for SM90 prefill constraints | CUDA graph decode; PCG/BCG extend | EAGLE chain/tree verify; EAGLE chain/tree CUDA graph replay | Same scoped tree tolerance as Triton. |

## Input And Config Coverage

- Page size 1, exact-page, crossing-page, ragged page-boundary, page-size-32 crossing, decode boundary, and batch-size-1 decode cases.
- GDN uses speculative Mamba state buffers for target verify coverage.
- The split-op tests verify live-token slicing with a larger static token buffer.

## Next Work

- Add additional linear-attention kernel backend variants when available.
- Consider broader speculative worker tags only after EAGLE chain/tree remains stable across kernels.
