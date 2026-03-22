## Apple MPS

| Environment Variable | Default | Description                                                  |
|----------------------|---------|--------------------------------------------------------------|
| `SGLANG_USE_MLX`     | not set | Set to `1` to enable MLX fused Metal kernels for norm ops on MPS |

## Caching Acceleration

These variables configure caching acceleration for Diffusion Transformer (DiT) models.
SGLang supports multiple caching strategies - see [caching documentation](performance/cache/index.md) for an overview.

### Cache-DiT Configuration

See [cache-dit documentation](performance/cache/cache_dit.md) for detailed configuration.

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

## Cloud Storage

These variables configure S3-compatible cloud storage for automatically uploading generated images and videos.

| Environment Variable            | Default | Description                                            |
|---------------------------------|---------|--------------------------------------------------------|
| `SGLANG_CLOUD_STORAGE_TYPE`     | not set | Set to `s3` to enable cloud storage                    |
| `SGLANG_S3_BUCKET_NAME`         | not set | The name of the S3 bucket                              |
| `SGLANG_S3_ENDPOINT_URL`        | not set | Custom endpoint URL (for MinIO, OSS, etc.)             |
| `SGLANG_S3_REGION_NAME`         | us-east-1 | AWS region name                                      |
| `SGLANG_S3_ACCESS_KEY_ID`       | not set | AWS Access Key ID                                      |
| `SGLANG_S3_SECRET_ACCESS_KEY`   | not set | AWS Secret Access Key                                  |

## CUDA Crash Debugging

These variables enable kernel API logging and optional input/output dumps around diffusion CUDA kernel call boundaries. They are useful when tracking down CUDA crashes such as illegal memory access, device-side assert, or shape mismatches in custom kernels.

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `SGLANG_KERNEL_API_LOGLEVEL` | `0` | Controls crash-debug kernel API logging. `1` logs API names, `3` logs tensor metadata, `5` adds tensor statistics, and `10` also writes dump snapshots. |
| `SGLANG_KERNEL_API_LOGDEST` | `stdout` | Destination for crash-debug kernel API logs. Use `stdout`, `stderr`, or a file path. `%i` is replaced with the process PID. |
| `SGLANG_KERNEL_API_DUMP_DIR` | `sglang_kernel_api_dumps` | Output directory for level-10 kernel API dumps. `%i` is replaced with the process PID. |
| `SGLANG_KERNEL_API_DUMP_INCLUDE` | not set | Comma-separated wildcard patterns for kernel API names to include in level-10 dumps. |
| `SGLANG_KERNEL_API_DUMP_EXCLUDE` | not set | Comma-separated wildcard patterns for kernel API names to exclude from level-10 dumps. |
