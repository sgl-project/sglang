# FTKA vs current DS microbench results

## Environment

- torch: `2.11.0+cu130`
- device: `NVIDIA H200`
- platform: `Linux-5.15.0-94-generic-x86_64-with-glibc2.39`
- flashinfer: OK `0.6.11.post1`
- ftka: OK `<unknown>` (installed_commit=`<unknown>`, target_commit=`d8803b29961c44d77a747636ad4282bd7a9094af`)

## Per-shape per-path results

| bs | h_kv | ctx | top_k | path | status | mean µs | score µs | parity | graph | extra |
|---:|-----:|----:|------:|------|--------|--------:|---------:|--------|-------|-------|
| 1 | 1 | 32768 | 512 | `torch` | ok | 92.1 | 19.1 | ok | ok |  |
| 1 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 75.0 | 19.1 | ok | ok |  |
| 1 | 1 | 32768 | 512 | `ftka_raft_topk` | ok | 30.0 | 19.1 | ok | ok |  |
| 1 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 32768 | 1024 | `torch` | ok | 91.6 | 18.6 | ok | ok |  |
| 1 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 79.1 | 18.6 | ok | ok |  |
| 1 | 1 | 32768 | 1024 | `ftka_raft_topk` | ok | 34.0 | 18.6 | ok | ok |  |
| 1 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 32768 | 2048 | `torch` | ok | 88.4 | 18.8 | ok | ok |  |
| 1 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 73.7 | 18.8 | ok | ok |  |
| 1 | 1 | 32768 | 2048 | `ftka_raft_topk` | ok | 37.5 | 18.8 | ok | ok |  |
| 1 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 32768 | 4096 | `torch` | ok | 89.1 | 18.7 | ok | ok |  |
| 1 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.7 | - | - | top_k>2048 ceiling |
| 1 | 1 | 32768 | 4096 | `ftka_raft_topk` | ok | 39.8 | 18.7 | ok | ok |  |
| 1 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 32768 | 8192 | `torch` | ok | 88.4 | 18.4 | ok | ok |  |
| 1 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.4 | - | - | top_k>2048 ceiling |
| 1 | 1 | 32768 | 8192 | `ftka_raft_topk` | ok | 43.4 | 18.4 | ok | ok |  |
| 1 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 32768 | 512 | `torch` | ok | 89.0 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 73.7 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 512 | `ftka_raft_topk` | ok | 31.4 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 32768 | 1024 | `torch` | ok | 88.9 | 18.5 | ok | ok |  |
| 4 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 74.9 | 18.5 | ok | ok |  |
| 4 | 1 | 32768 | 1024 | `ftka_raft_topk` | ok | 33.8 | 18.5 | ok | ok |  |
| 4 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 32768 | 2048 | `torch` | ok | 89.0 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 74.0 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 2048 | `ftka_raft_topk` | ok | 37.5 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 32768 | 4096 | `torch` | ok | 88.2 | 18.5 | ok | ok |  |
| 4 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.5 | - | - | top_k>2048 ceiling |
| 4 | 1 | 32768 | 4096 | `ftka_raft_topk` | ok | 40.0 | 18.5 | ok | ok |  |
| 4 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 32768 | 8192 | `torch` | ok | 88.2 | 18.4 | ok | ok |  |
| 4 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.4 | - | - | top_k>2048 ceiling |
| 4 | 1 | 32768 | 8192 | `ftka_raft_topk` | ok | 43.0 | 18.4 | ok | ok |  |
| 4 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 32768 | 512 | `torch` | ok | 89.2 | 18.6 | ok | ok |  |
| 8 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 73.9 | 18.6 | ok | ok |  |
| 8 | 1 | 32768 | 512 | `ftka_raft_topk` | ok | 31.2 | 18.6 | ok | ok |  |
| 8 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 32768 | 1024 | `torch` | ok | 89.5 | 18.4 | ok | ok |  |
| 8 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 73.7 | 18.4 | ok | ok |  |
| 8 | 1 | 32768 | 1024 | `ftka_raft_topk` | ok | 33.7 | 18.4 | ok | ok |  |
| 8 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 32768 | 2048 | `torch` | ok | 89.4 | 18.7 | ok | ok |  |
| 8 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 74.1 | 18.7 | ok | ok |  |
| 8 | 1 | 32768 | 2048 | `ftka_raft_topk` | ok | 37.8 | 18.7 | ok | ok |  |
| 8 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 32768 | 4096 | `torch` | ok | 89.1 | 33.7 | ok | ok |  |
| 8 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 33.7 | - | - | top_k>2048 ceiling |
| 8 | 1 | 32768 | 4096 | `ftka_raft_topk` | ok | 40.6 | 33.7 | ok | ok |  |
| 8 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 32768 | 8192 | `torch` | ok | 89.3 | 18.5 | ok | ok |  |
| 8 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.5 | - | - | top_k>2048 ceiling |
| 8 | 1 | 32768 | 8192 | `ftka_raft_topk` | ok | 43.0 | 18.5 | ok | ok |  |
| 8 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 32768 | 512 | `torch` | ok | 89.0 | 18.7 | ok | ok |  |
| 16 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 74.2 | 18.7 | ok | ok |  |
| 16 | 1 | 32768 | 512 | `ftka_raft_topk` | ok | 31.6 | 18.7 | ok | ok |  |
| 16 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 32768 | 1024 | `torch` | ok | 88.6 | 18.4 | ok | ok |  |
| 16 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 73.5 | 18.4 | ok | ok |  |
| 16 | 1 | 32768 | 1024 | `ftka_raft_topk` | ok | 34.2 | 18.4 | ok | ok |  |
| 16 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 32768 | 2048 | `torch` | ok | 89.1 | 18.8 | ok | ok |  |
| 16 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 73.4 | 18.8 | ok | ok |  |
| 16 | 1 | 32768 | 2048 | `ftka_raft_topk` | ok | 37.8 | 18.8 | ok | ok |  |
| 16 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 32768 | 4096 | `torch` | ok | 88.9 | 18.3 | ok | ok |  |
| 16 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 16 | 1 | 32768 | 4096 | `ftka_raft_topk` | ok | 40.6 | 18.3 | ok | ok |  |
| 16 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 32768 | 8192 | `torch` | ok | 88.7 | 18.5 | ok | ok |  |
| 16 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.5 | - | - | top_k>2048 ceiling |
| 16 | 1 | 32768 | 8192 | `ftka_raft_topk` | ok | 44.5 | 18.5 | ok | ok |  |
| 16 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 32768 | 512 | `torch` | ok | 89.2 | 18.7 | ok | ok |  |
| 32 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 74.1 | 18.7 | ok | ok |  |
| 32 | 1 | 32768 | 512 | `ftka_raft_topk` | ok | 31.5 | 18.7 | ok | ok |  |
| 32 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 32768 | 1024 | `torch` | ok | 88.7 | 18.9 | ok | ok |  |
| 32 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 73.6 | 18.9 | ok | ok |  |
| 32 | 1 | 32768 | 1024 | `ftka_raft_topk` | ok | 34.7 | 18.9 | ok | ok |  |
| 32 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 32768 | 2048 | `torch` | ok | 89.1 | 18.7 | ok | ok |  |
| 32 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 73.8 | 18.7 | ok | ok |  |
| 32 | 1 | 32768 | 2048 | `ftka_raft_topk` | ok | 38.2 | 18.7 | ok | ok |  |
| 32 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 32768 | 4096 | `torch` | ok | 88.6 | 18.7 | ok | ok |  |
| 32 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.7 | - | - | top_k>2048 ceiling |
| 32 | 1 | 32768 | 4096 | `ftka_raft_topk` | ok | 41.3 | 18.7 | ok | ok |  |
| 32 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 32768 | 8192 | `torch` | ok | 88.9 | 20.1 | ok | ok |  |
| 32 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 20.1 | - | - | top_k>2048 ceiling |
| 32 | 1 | 32768 | 8192 | `ftka_raft_topk` | ok | 45.1 | 20.1 | ok | ok |  |
| 32 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 65536 | 512 | `torch` | ok | 87.6 | 18.4 | ok | ok |  |
| 1 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 73.0 | 18.4 | ok | ok |  |
| 1 | 1 | 65536 | 512 | `ftka_raft_topk` | ok | 74.0 | 18.4 | ok | ok |  |
| 1 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 65536 | 1024 | `torch` | ok | 88.4 | 18.4 | ok | ok |  |
| 1 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 73.0 | 18.4 | ok | ok |  |
| 1 | 1 | 65536 | 1024 | `ftka_raft_topk` | ok | 80.1 | 18.4 | ok | ok |  |
| 1 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 65536 | 2048 | `torch` | ok | 88.7 | 18.7 | ok | ok |  |
| 1 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 73.2 | 18.7 | ok | ok |  |
| 1 | 1 | 65536 | 2048 | `ftka_raft_topk` | ok | 88.7 | 18.7 | ok | ok |  |
| 1 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 65536 | 4096 | `torch` | ok | 88.5 | 18.7 | ok | ok |  |
| 1 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.7 | - | - | top_k>2048 ceiling |
| 1 | 1 | 65536 | 4096 | `ftka_raft_topk` | ok | 96.9 | 18.7 | ok | ok |  |
| 1 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 65536 | 8192 | `torch` | ok | 88.1 | 18.4 | ok | ok |  |
| 1 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.4 | - | - | top_k>2048 ceiling |
| 1 | 1 | 65536 | 8192 | `ftka_raft_topk` | ok | 102.3 | 18.4 | ok | ok |  |
| 1 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 65536 | 512 | `torch` | ok | 87.7 | 18.3 | ok | ok |  |
| 4 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 73.7 | 18.3 | ok | ok |  |
| 4 | 1 | 65536 | 512 | `ftka_raft_topk` | ok | 73.5 | 18.3 | ok | ok |  |
| 4 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 65536 | 1024 | `torch` | ok | 88.6 | 18.6 | ok | ok |  |
| 4 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 74.4 | 18.6 | ok | ok |  |
| 4 | 1 | 65536 | 1024 | `ftka_raft_topk` | ok | 80.4 | 18.6 | ok | ok |  |
| 4 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 65536 | 2048 | `torch` | ok | 88.3 | 18.8 | ok | ok |  |
| 4 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 73.7 | 18.8 | ok | ok |  |
| 4 | 1 | 65536 | 2048 | `ftka_raft_topk` | ok | 88.4 | 18.8 | ok | ok |  |
| 4 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 65536 | 4096 | `torch` | ok | 88.6 | 18.1 | ok | ok |  |
| 4 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.1 | - | - | top_k>2048 ceiling |
| 4 | 1 | 65536 | 4096 | `ftka_raft_topk` | ok | 96.8 | 18.1 | ok | ok |  |
| 4 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 65536 | 8192 | `torch` | ok | 88.2 | 18.6 | ok | ok |  |
| 4 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.6 | - | - | top_k>2048 ceiling |
| 4 | 1 | 65536 | 8192 | `ftka_raft_topk` | ok | 102.4 | 18.6 | ok | ok |  |
| 4 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 65536 | 512 | `torch` | ok | 88.9 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 73.2 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 512 | `ftka_raft_topk` | ok | 74.9 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 65536 | 1024 | `torch` | ok | 88.7 | 18.7 | ok | ok |  |
| 8 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 73.7 | 18.7 | ok | ok |  |
| 8 | 1 | 65536 | 1024 | `ftka_raft_topk` | ok | 82.0 | 18.7 | ok | ok |  |
| 8 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 65536 | 2048 | `torch` | ok | 89.2 | 19.0 | ok | ok |  |
| 8 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 73.8 | 19.0 | ok | ok |  |
| 8 | 1 | 65536 | 2048 | `ftka_raft_topk` | ok | 90.0 | 19.0 | ok | ok |  |
| 8 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 65536 | 4096 | `torch` | ok | 88.8 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 8 | 1 | 65536 | 4096 | `ftka_raft_topk` | ok | 98.8 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 65536 | 8192 | `torch` | ok | 89.0 | 18.2 | ok | ok |  |
| 8 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.2 | - | - | top_k>2048 ceiling |
| 8 | 1 | 65536 | 8192 | `ftka_raft_topk` | ok | 102.3 | 18.2 | ok | ok |  |
| 8 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 65536 | 512 | `torch` | ok | 88.4 | 18.8 | ok | ok |  |
| 16 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 73.7 | 18.8 | ok | ok |  |
| 16 | 1 | 65536 | 512 | `ftka_raft_topk` | ok | 76.0 | 18.8 | ok | ok |  |
| 16 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 65536 | 1024 | `torch` | ok | 88.1 | 18.7 | ok | ok |  |
| 16 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 74.1 | 18.7 | ok | ok |  |
| 16 | 1 | 65536 | 1024 | `ftka_raft_topk` | ok | 81.9 | 18.7 | ok | ok |  |
| 16 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 65536 | 2048 | `torch` | ok | 88.4 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 73.6 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 2048 | `ftka_raft_topk` | ok | 90.3 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 65536 | 4096 | `torch` | ok | 88.5 | 18.4 | ok | ok |  |
| 16 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.4 | - | - | top_k>2048 ceiling |
| 16 | 1 | 65536 | 4096 | `ftka_raft_topk` | ok | 99.2 | 18.4 | ok | ok |  |
| 16 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 65536 | 8192 | `torch` | ok | 89.0 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.5 | - | - | top_k>2048 ceiling |
| 16 | 1 | 65536 | 8192 | `ftka_raft_topk` | ok | 104.1 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 65536 | 512 | `torch` | ok | 90.4 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 74.2 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 512 | `ftka_raft_topk` | ok | 78.0 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 65536 | 1024 | `torch` | ok | 90.7 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 74.0 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 1024 | `ftka_raft_topk` | ok | 84.9 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 65536 | 2048 | `torch` | ok | 92.4 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 74.5 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 2048 | `ftka_raft_topk` | ok | 91.6 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 65536 | 4096 | `torch` | ok | 93.4 | 32.1 | ok | ok |  |
| 32 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 32.1 | - | - | top_k>2048 ceiling |
| 32 | 1 | 65536 | 4096 | `ftka_raft_topk` | ok | 99.3 | 32.1 | ok | ok |  |
| 32 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 65536 | 8192 | `torch` | ok | 95.4 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 32.0 | - | - | top_k>2048 ceiling |
| 32 | 1 | 65536 | 8192 | `ftka_raft_topk` | ok | 106.7 | 32.0 | ok | ok |  |
| 32 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 131072 | 512 | `torch` | ok | 87.9 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 76.9 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 512 | `ftka_raft_topk` | ok | 133.0 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 131072 | 1024 | `torch` | ok | 88.4 | 18.6 | ok | ok |  |
| 1 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 79.8 | 18.6 | ok | ok |  |
| 1 | 1 | 131072 | 1024 | `ftka_raft_topk` | ok | 139.9 | 18.6 | ok | ok |  |
| 1 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 131072 | 2048 | `torch` | ok | 88.2 | 18.4 | ok | ok |  |
| 1 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 85.3 | 18.4 | ok | ok |  |
| 1 | 1 | 131072 | 2048 | `ftka_raft_topk` | ok | 152.1 | 18.4 | ok | ok |  |
| 1 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 131072 | 4096 | `torch` | ok | 88.3 | 18.3 | ok | ok |  |
| 1 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 1 | 1 | 131072 | 4096 | `ftka_raft_topk` | ok | 168.3 | 18.3 | ok | ok |  |
| 1 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 1 | 1 | 131072 | 8192 | `torch` | ok | 88.5 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.2 | - | - | top_k>2048 ceiling |
| 1 | 1 | 131072 | 8192 | `ftka_raft_topk` | ok | 186.1 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 131072 | 512 | `torch` | ok | 87.5 | 18.7 | ok | ok |  |
| 4 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 78.9 | 18.7 | ok | ok |  |
| 4 | 1 | 131072 | 512 | `ftka_raft_topk` | ok | 132.9 | 18.7 | ok | ok |  |
| 4 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 131072 | 1024 | `torch` | ok | 88.1 | 18.4 | ok | ok |  |
| 4 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 82.0 | 18.4 | ok | ok |  |
| 4 | 1 | 131072 | 1024 | `ftka_raft_topk` | ok | 140.0 | 18.4 | ok | ok |  |
| 4 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 131072 | 2048 | `torch` | ok | 87.4 | 18.5 | ok | ok |  |
| 4 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 89.5 | 18.5 | ok | ok |  |
| 4 | 1 | 131072 | 2048 | `ftka_raft_topk` | ok | 151.7 | 18.5 | ok | ok |  |
| 4 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 131072 | 4096 | `torch` | ok | 87.6 | 18.4 | ok | ok |  |
| 4 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.4 | - | - | top_k>2048 ceiling |
| 4 | 1 | 131072 | 4096 | `ftka_raft_topk` | ok | 168.8 | 18.4 | ok | ok |  |
| 4 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 1 | 131072 | 8192 | `torch` | ok | 87.4 | 18.4 | ok | ok |  |
| 4 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.4 | - | - | top_k>2048 ceiling |
| 4 | 1 | 131072 | 8192 | `ftka_raft_topk` | ok | 186.4 | 18.4 | ok | ok |  |
| 4 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 131072 | 512 | `torch` | ok | 89.1 | 18.8 | ok | ok |  |
| 8 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 80.9 | 18.8 | ok | ok |  |
| 8 | 1 | 131072 | 512 | `ftka_raft_topk` | ok | 133.2 | 18.8 | ok | ok |  |
| 8 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 131072 | 1024 | `torch` | ok | 88.7 | 18.9 | ok | ok |  |
| 8 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 83.4 | 18.9 | ok | ok |  |
| 8 | 1 | 131072 | 1024 | `ftka_raft_topk` | ok | 141.3 | 18.9 | ok | ok |  |
| 8 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 131072 | 2048 | `torch` | ok | 91.0 | 18.8 | ok | ok |  |
| 8 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 89.3 | 18.8 | ok | ok |  |
| 8 | 1 | 131072 | 2048 | `ftka_raft_topk` | ok | 153.7 | 18.8 | ok | ok |  |
| 8 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 131072 | 4096 | `torch` | ok | 88.7 | 18.6 | ok | ok |  |
| 8 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.6 | - | - | top_k>2048 ceiling |
| 8 | 1 | 131072 | 4096 | `ftka_raft_topk` | ok | 169.7 | 18.6 | ok | ok |  |
| 8 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 8 | 1 | 131072 | 8192 | `torch` | ok | 91.0 | 18.6 | ok | ok |  |
| 8 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.6 | - | - | top_k>2048 ceiling |
| 8 | 1 | 131072 | 8192 | `ftka_raft_topk` | ok | 187.2 | 18.6 | ok | ok |  |
| 8 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 131072 | 512 | `torch` | ok | 95.0 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 81.8 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 512 | `ftka_raft_topk` | ok | 134.0 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 131072 | 1024 | `torch` | ok | 94.6 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 85.4 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 1024 | `ftka_raft_topk` | ok | 141.8 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 131072 | 2048 | `torch` | ok | 96.9 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 93.5 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 2048 | `ftka_raft_topk` | ok | 153.8 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 131072 | 4096 | `torch` | ok | 99.1 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 32.0 | - | - | top_k>2048 ceiling |
| 16 | 1 | 131072 | 4096 | `ftka_raft_topk` | ok | 171.2 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 16 | 1 | 131072 | 8192 | `torch` | ok | 102.6 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 32.0 | - | - | top_k>2048 ceiling |
| 16 | 1 | 131072 | 8192 | `ftka_raft_topk` | ok | 189.2 | 32.0 | ok | ok |  |
| 16 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 131072 | 512 | `torch` | ok | 115.1 | 87.1 | ok | ok |  |
| 32 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 84.0 | 87.1 | ok | ok |  |
| 32 | 1 | 131072 | 512 | `ftka_raft_topk` | ok | 138.4 | 87.1 | ok | ok |  |
| 32 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 131072 | 1024 | `torch` | ok | 114.8 | 88.3 | ok | ok |  |
| 32 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 89.1 | 88.3 | ok | ok |  |
| 32 | 1 | 131072 | 1024 | `ftka_raft_topk` | ok | 145.6 | 88.3 | ok | ok |  |
| 32 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 131072 | 2048 | `torch` | ok | 116.2 | 86.7 | ok | ok |  |
| 32 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 93.0 | 86.7 | ok | ok |  |
| 32 | 1 | 131072 | 2048 | `ftka_raft_topk` | ok | 157.0 | 86.7 | ok | ok |  |
| 32 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 131072 | 4096 | `torch` | ok | 120.3 | 86.9 | ok | ok |  |
| 32 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 86.9 | - | - | top_k>2048 ceiling |
| 32 | 1 | 131072 | 4096 | `ftka_raft_topk` | ok | 181.3 | 86.9 | ok | ok |  |
| 32 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 32 | 1 | 131072 | 8192 | `torch` | ok | 123.6 | 89.3 | ok | ok |  |
| 32 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 89.3 | - | - | top_k>2048 ceiling |
| 32 | 1 | 131072 | 8192 | `ftka_raft_topk` | ok | 243.6 | 89.3 | ok | ok |  |
| 32 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
| 4 | 8 | 65536 | 2048 | `torch` | ok | 91.7 | 60.9 | ok | ok |  |
| 4 | 8 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 87.7 | 60.9 | ok | ok |  |
| 4 | 8 | 65536 | 2048 | `ftka_raft_topk` | ok | 91.8 | 60.9 | ok | ok |  |
| 4 | 8 | 65536 | 2048 | `ftka_gemv+ftka_topk` | error | - | - | - | - | setup: RuntimeError: P4 (ftka_gemv+ftka_topk) is structurall |
