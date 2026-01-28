## Caching Acceleration

These variables configure caching acceleration for Diffusion Transformer (DiT) models.
SGLang supports multiple caching strategies - see [caching documentation](cache/caching.md) for an overview.

### Cache-DiT Configuration

See [cache-dit documentation](cache/cache_dit.md) for detailed configuration.

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
