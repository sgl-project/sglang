## Cache-DiT Acceleration

These variables configure cache-dit caching acceleration for Diffusion Transformer (DiT) models.
See [cache-dit documentation](cache_dit.md) for details.

| Environment Variable                | Default | Description                              |
|-------------------------------------|---------|------------------------------------------|
| `SGLANG_CACHE_DIT_ENABLED`          | false   | Enable Cache-DiT acceleration            |
| `SGLANG_CACHE_DIT_FN`               | 1       | First N blocks to always compute         |
| `SGLANG_CACHE_DIT_BN`               | 0       | Last N blocks to always compute          |
| `SGLANG_CACHE_DIT_WARMUP`           | 4       | Warmup steps before caching              |
| `SGLANG_CACHE_DIT_RDT`              | 0.24    | Residual difference threshold            |
| `SGLANG_CACHE_DIT_MC`               | 3       | Max continuous cached steps              |
| `SGLANG_CACHE_DIT_TAYLORSEER`       | false   | Enable TaylorSeer calibrator             |
| `SGLANG_CACHE_DIT_TS_ORDER`         | 1       | TaylorSeer order (1 or 2)                |
| `SGLANG_CACHE_DIT_SCM_PRESET`       | none    | SCM preset (none/slow/medium/fast/ultra) |
| `SGLANG_CACHE_DIT_SCM_POLICY`       | dynamic | SCM caching policy                       |
| `SGLANG_CACHE_DIT_SCM_COMPUTE_BINS` | not set | Custom SCM compute bins                  |
| `SGLANG_CACHE_DIT_SCM_CACHE_BINS`   | not set | Custom SCM cache bins                    |
