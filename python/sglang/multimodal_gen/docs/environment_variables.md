## Cache-DiT Acceleration

*These variables configure cache-dit caching acceleration for Diffusion Transformer (DiT) models.
See [cache-dit documentation](cache_dit.md) for details.*

| Environment Variable                | Description                                                                                                        | Default Value |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------|---------------|
| `SGLANG_CACHE_DIT_ENABLED`          | Enable cache-dit acceleration for DiT models                                                                       | `false`       |
| `SGLANG_CACHE_DIT_FN`               | Number of first transformer blocks to always compute (DBCache Fn parameter)                                        | `1`           |
| `SGLANG_CACHE_DIT_BN`               | Number of last transformer blocks to always compute (DBCache Bn parameter)                                         | `0`           |
| `SGLANG_CACHE_DIT_WARMUP`           | Warmup steps before caching starts (DBCache W parameter)                                                           | `8`           |
| `SGLANG_CACHE_DIT_RDT`              | Residual difference threshold for caching decisions (DBCache R parameter). Lower = better quality, higher = faster | `0.35`        |
| `SGLANG_CACHE_DIT_MC`               | Maximum continuous cached steps (DBCache MC parameter)                                                             | `3`           |
| `SGLANG_CACHE_DIT_TAYLORSEER`       | Enable TaylorSeer calibrator for improved caching accuracy                                                         | `true`        |
| `SGLANG_CACHE_DIT_TS_ORDER`         | TaylorSeer Taylor expansion order (1 or 2)                                                                         | `1`           |
| `SGLANG_CACHE_DIT_SCM_PRESET`       | SCM (Step Computation Masking) preset: `none`, `slow`, `medium`, `fast`, `ultra`                                   | `none`        |
| `SGLANG_CACHE_DIT_SCM_POLICY`       | SCM caching policy: `dynamic` (adaptive) or `static` (fixed pattern)                                               | `dynamic`     |
| `SGLANG_CACHE_DIT_SCM_COMPUTE_BINS` | Custom SCM compute bins (comma-separated, e.g., "8,3,3,2,2")                                                       | Not set       |
| `SGLANG_CACHE_DIT_SCM_CACHE_BINS`   | Custom SCM cache bins (comma-separated, e.g., "1,2,2,2,3")                                                         | Not set       |

