# SeaweedFS as the HiCache L3 backend

This backend stores HiCache KV pages in [SeaweedFS](https://github.com/seaweedfs/seaweedfs)
through its S3 gateway. Each KV page is one object. Every SGLang instance pointed at the same
bucket shares the same L3 cache, so a prefix computed by one instance can be prefetched by
another instance, including one on a different node.

## Requirements

- A SeaweedFS cluster with the S3 gateway enabled. For a quick single-node trial:

  ```bash
  weed server -dir=/data/seaweedfs -s3 -s3.port=8333
  ```

- `boto3` in the SGLang environment:

  ```bash
  pip install boto3
  ```

## Launch

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --enable-hierarchical-cache \
  --hicache-storage-backend seaweedfs \
  --hicache-storage-backend-extra-config '{"endpoint": "http://seaweedfs-s3:8333", "bucket": "sglang-hicache"}'
```

`--hicache-storage-backend-extra-config` also accepts `@path/to/config.json` (or YAML/TOML).

| Key | Default | Meaning |
|---|---|---|
| `endpoint` | required | URL of the SeaweedFS S3 gateway |
| `bucket` | `sglang-hicache` | Bucket for KV pages; created if it does not exist |
| `prefix` | empty | Extra key prefix, for example to separate deployments in one bucket |
| `region` | `us-east-1` | Region sent with S3 requests |
| `access_key`, `secret_key` | unset | S3 credentials; when unset, the standard AWS credential chain is used (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, shared credentials file, and so on) |
| `max_workers` | `16` | Concurrent S3 requests per rank |

Prefer the AWS credential chain over putting secrets in the command line.

## Tuning

- **Page size.** Each KV page is one object, so `--page-size` sets the object size and the number
  of S3 requests per prefix. Small pages turn a long prefix into many small GETs, where per-request
  overhead dominates. Start with `--page-size 64` or larger.
- **Prefetch threshold.** A prefetch from SeaweedFS pays off only when loading the prefix is faster
  than recomputing it, which depends on the model, the GPU, the prefix length and the bandwidth to
  SeaweedFS. The default `prefetch_threshold` of 256 tokens is low for a remote store; measure
  time-to-first-token with and without the cache for a few prefix lengths and set the threshold at
  the break-even point, for example:

  ```bash
  --hicache-storage-backend-extra-config '{"endpoint": "http://seaweedfs-s3:8333", "prefetch_threshold": 8192}'
  ```

  `prefetch_threshold`, `prefetch_timeout_base` and `prefetch_timeout_per_ki_token` are read by
  SGLang's cache controller, not by this backend.

## Object layout

Objects are named `<prefix>/<model>/<rank scope>/<page hash>`:

- MHA models include `tp<rank>of<size>`, because tensor-parallel ranks hold different heads.
- MLA models omit the tp rank, because every rank holds identical KV and only one rank backs up.
- Pipeline-parallel (`pp<rank>of<size>`) and context-parallel (`cp<rank>of<size>`) components are
  added when those degrees are greater than one.

Instances share cache entries only when the model name and parallel layout match.

## Behaviour and limits

- The backend uses the zero-copy HiCache interface: pages move directly between SeaweedFS and the
  host KV pool's own buffers, without a staging page.
- A prefetch stops at the first missing page. Pages after a miss are not loaded.
- An object whose size does not match the host page is treated as a miss and never copied.
- `clear()` deletes only the calling rank's objects. Other models and ranks sharing the bucket are
  left alone.
- The backend does not evict. Size the SeaweedFS cluster for the cache, or apply a retention
  policy on the SeaweedFS side.
