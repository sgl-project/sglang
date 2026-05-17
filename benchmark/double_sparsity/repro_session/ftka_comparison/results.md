# FTKA vs current DS microbench results

## Environment

- torch: `2.11.0+cu130`
- device: `NVIDIA H200`
- platform: `Linux-5.15.0-94-generic-x86_64-with-glibc2.39`
- flashinfer: OK `0.6.11.post1`
- ftka: SKIP — `ftka package not installed` (target_commit=`d8803b29961c44d77a747636ad4282bd7a9094af`)

## Per-shape per-path results

| bs | h_kv | ctx | top_k | path | status | mean µs | score µs | parity | graph | extra |
|---:|-----:|----:|------:|------|--------|--------:|---------:|--------|-------|-------|
| 1 | 1 | 32768 | 512 | `torch` | ok | 93.4 | 18.5 | ok | ok |  |
| 1 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 74.2 | 18.5 | ok | ok |  |
| 1 | 1 | 32768 | 512 | `ftka_raft_topk` | skipped | - | 18.5 | - | - | ftka package not installed |
| 1 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 32768 | 1024 | `torch` | ok | 90.5 | 18.8 | ok | ok |  |
| 1 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 74.1 | 18.8 | ok | ok |  |
| 1 | 1 | 32768 | 1024 | `ftka_raft_topk` | skipped | - | 18.8 | - | - | ftka package not installed |
| 1 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 32768 | 2048 | `torch` | ok | 89.1 | 18.1 | ok | ok |  |
| 1 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 74.2 | 18.1 | ok | ok |  |
| 1 | 1 | 32768 | 2048 | `ftka_raft_topk` | skipped | - | 18.1 | - | - | ftka package not installed |
| 1 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 32768 | 4096 | `torch` | ok | 93.6 | 18.6 | ok | ok |  |
| 1 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.6 | - | - | top_k>2048 ceiling |
| 1 | 1 | 32768 | 4096 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 1 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 32768 | 8192 | `torch` | ok | 92.5 | 18.8 | ok | ok |  |
| 1 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.8 | - | - | top_k>2048 ceiling |
| 1 | 1 | 32768 | 8192 | `ftka_raft_topk` | skipped | - | 18.8 | - | - | ftka package not installed |
| 1 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 32768 | 512 | `torch` | ok | 89.6 | 18.4 | ok | ok |  |
| 4 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 73.8 | 18.4 | ok | ok |  |
| 4 | 1 | 32768 | 512 | `ftka_raft_topk` | skipped | - | 18.4 | - | - | ftka package not installed |
| 4 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 32768 | 1024 | `torch` | ok | 89.3 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 73.5 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 1024 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 4 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 32768 | 2048 | `torch` | ok | 89.7 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 73.8 | 18.6 | ok | ok |  |
| 4 | 1 | 32768 | 2048 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 4 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 32768 | 4096 | `torch` | ok | 89.4 | 18.5 | ok | ok |  |
| 4 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.5 | - | - | top_k>2048 ceiling |
| 4 | 1 | 32768 | 4096 | `ftka_raft_topk` | skipped | - | 18.5 | - | - | ftka package not installed |
| 4 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 32768 | 8192 | `torch` | ok | 89.6 | 18.2 | ok | ok |  |
| 4 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.2 | - | - | top_k>2048 ceiling |
| 4 | 1 | 32768 | 8192 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 4 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 32768 | 512 | `torch` | ok | 90.3 | 18.1 | ok | ok |  |
| 8 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 103.4 | 18.1 | ok | ok |  |
| 8 | 1 | 32768 | 512 | `ftka_raft_topk` | skipped | - | 18.1 | - | - | ftka package not installed |
| 8 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 32768 | 1024 | `torch` | ok | 90.3 | 18.3 | ok | ok |  |
| 8 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 73.9 | 18.3 | ok | ok |  |
| 8 | 1 | 32768 | 1024 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 8 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 32768 | 2048 | `torch` | ok | 90.2 | 18.1 | ok | ok |  |
| 8 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 73.6 | 18.1 | ok | ok |  |
| 8 | 1 | 32768 | 2048 | `ftka_raft_topk` | skipped | - | 18.1 | - | - | ftka package not installed |
| 8 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 32768 | 4096 | `torch` | ok | 90.6 | 18.3 | ok | ok |  |
| 8 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 8 | 1 | 32768 | 4096 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 8 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 32768 | 8192 | `torch` | ok | 90.2 | 18.1 | ok | ok |  |
| 8 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.1 | - | - | top_k>2048 ceiling |
| 8 | 1 | 32768 | 8192 | `ftka_raft_topk` | skipped | - | 18.1 | - | - | ftka package not installed |
| 8 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 32768 | 512 | `torch` | ok | 91.1 | 18.4 | ok | ok |  |
| 16 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 73.1 | 18.4 | ok | ok |  |
| 16 | 1 | 32768 | 512 | `ftka_raft_topk` | skipped | - | 18.4 | - | - | ftka package not installed |
| 16 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 32768 | 1024 | `torch` | ok | 89.7 | 18.4 | ok | ok |  |
| 16 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 74.1 | 18.4 | ok | ok |  |
| 16 | 1 | 32768 | 1024 | `ftka_raft_topk` | skipped | - | 18.4 | - | - | ftka package not installed |
| 16 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 32768 | 2048 | `torch` | ok | 89.7 | 18.3 | ok | ok |  |
| 16 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 74.4 | 18.3 | ok | ok |  |
| 16 | 1 | 32768 | 2048 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 16 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 32768 | 4096 | `torch` | ok | 90.3 | 18.2 | ok | ok |  |
| 16 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.2 | - | - | top_k>2048 ceiling |
| 16 | 1 | 32768 | 4096 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 16 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 32768 | 8192 | `torch` | ok | 90.6 | 18.3 | ok | ok |  |
| 16 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 16 | 1 | 32768 | 8192 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 16 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 32768 | 512 | `torch` | ok | 89.9 | 18.5 | ok | ok |  |
| 32 | 1 | 32768 | 512 | `flashinfer_topk_page_table` | ok | 73.7 | 18.5 | ok | ok |  |
| 32 | 1 | 32768 | 512 | `ftka_raft_topk` | skipped | - | 18.5 | - | - | ftka package not installed |
| 32 | 1 | 32768 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 32768 | 1024 | `torch` | ok | 89.7 | 18.6 | ok | ok |  |
| 32 | 1 | 32768 | 1024 | `flashinfer_topk_page_table` | ok | 73.8 | 18.6 | ok | ok |  |
| 32 | 1 | 32768 | 1024 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 32 | 1 | 32768 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 32768 | 2048 | `torch` | ok | 90.3 | 18.5 | ok | ok |  |
| 32 | 1 | 32768 | 2048 | `flashinfer_topk_page_table` | ok | 74.0 | 18.5 | ok | ok |  |
| 32 | 1 | 32768 | 2048 | `ftka_raft_topk` | skipped | - | 18.5 | - | - | ftka package not installed |
| 32 | 1 | 32768 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 32768 | 4096 | `torch` | ok | 90.5 | 18.6 | ok | ok |  |
| 32 | 1 | 32768 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.6 | - | - | top_k>2048 ceiling |
| 32 | 1 | 32768 | 4096 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 32 | 1 | 32768 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 32768 | 8192 | `torch` | ok | 89.9 | 18.6 | ok | ok |  |
| 32 | 1 | 32768 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.6 | - | - | top_k>2048 ceiling |
| 32 | 1 | 32768 | 8192 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 32 | 1 | 32768 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 65536 | 512 | `torch` | ok | 88.4 | 18.2 | ok | ok |  |
| 1 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 73.6 | 18.2 | ok | ok |  |
| 1 | 1 | 65536 | 512 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 1 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 65536 | 1024 | `torch` | ok | 89.1 | 18.2 | ok | ok |  |
| 1 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 73.0 | 18.2 | ok | ok |  |
| 1 | 1 | 65536 | 1024 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 1 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 65536 | 2048 | `torch` | ok | 89.5 | 18.3 | ok | ok |  |
| 1 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 73.0 | 18.3 | ok | ok |  |
| 1 | 1 | 65536 | 2048 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 1 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 65536 | 4096 | `torch` | ok | 89.0 | 18.2 | ok | ok |  |
| 1 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.2 | - | - | top_k>2048 ceiling |
| 1 | 1 | 65536 | 4096 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 1 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 65536 | 8192 | `torch` | ok | 88.8 | 18.3 | ok | ok |  |
| 1 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 1 | 1 | 65536 | 8192 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 1 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 65536 | 512 | `torch` | ok | 88.6 | 18.0 | ok | ok |  |
| 4 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 74.1 | 18.0 | ok | ok |  |
| 4 | 1 | 65536 | 512 | `ftka_raft_topk` | skipped | - | 18.0 | - | - | ftka package not installed |
| 4 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 65536 | 1024 | `torch` | ok | 89.4 | 18.1 | ok | ok |  |
| 4 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 73.9 | 18.1 | ok | ok |  |
| 4 | 1 | 65536 | 1024 | `ftka_raft_topk` | skipped | - | 18.1 | - | - | ftka package not installed |
| 4 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 65536 | 2048 | `torch` | ok | 89.2 | 18.1 | ok | ok |  |
| 4 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 72.8 | 18.1 | ok | ok |  |
| 4 | 1 | 65536 | 2048 | `ftka_raft_topk` | skipped | - | 18.1 | - | - | ftka package not installed |
| 4 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 65536 | 4096 | `torch` | ok | 89.2 | 18.3 | ok | ok |  |
| 4 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 4 | 1 | 65536 | 4096 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 4 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 65536 | 8192 | `torch` | ok | 89.4 | 18.3 | ok | ok |  |
| 4 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 4 | 1 | 65536 | 8192 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 4 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 65536 | 512 | `torch` | ok | 89.7 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 73.8 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 512 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 8 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 65536 | 1024 | `torch` | ok | 89.9 | 18.1 | ok | ok |  |
| 8 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 73.9 | 18.1 | ok | ok |  |
| 8 | 1 | 65536 | 1024 | `ftka_raft_topk` | skipped | - | 18.1 | - | - | ftka package not installed |
| 8 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 65536 | 2048 | `torch` | ok | 89.8 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 73.4 | 18.3 | ok | ok |  |
| 8 | 1 | 65536 | 2048 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 8 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 65536 | 4096 | `torch` | ok | 90.1 | 18.6 | ok | ok |  |
| 8 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.6 | - | - | top_k>2048 ceiling |
| 8 | 1 | 65536 | 4096 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 8 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 65536 | 8192 | `torch` | ok | 90.1 | 18.6 | ok | ok |  |
| 8 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.6 | - | - | top_k>2048 ceiling |
| 8 | 1 | 65536 | 8192 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 8 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 65536 | 512 | `torch` | ok | 89.6 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 73.2 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 512 | `ftka_raft_topk` | skipped | - | 18.5 | - | - | ftka package not installed |
| 16 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 65536 | 1024 | `torch` | ok | 89.6 | 18.6 | ok | ok |  |
| 16 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 73.8 | 18.6 | ok | ok |  |
| 16 | 1 | 65536 | 1024 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 16 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 65536 | 2048 | `torch` | ok | 89.4 | 18.6 | ok | ok |  |
| 16 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 73.8 | 18.6 | ok | ok |  |
| 16 | 1 | 65536 | 2048 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 16 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 65536 | 4096 | `torch` | ok | 89.7 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.5 | - | - | top_k>2048 ceiling |
| 16 | 1 | 65536 | 4096 | `ftka_raft_topk` | skipped | - | 18.5 | - | - | ftka package not installed |
| 16 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 65536 | 8192 | `torch` | ok | 91.7 | 18.5 | ok | ok |  |
| 16 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.5 | - | - | top_k>2048 ceiling |
| 16 | 1 | 65536 | 8192 | `ftka_raft_topk` | skipped | - | 18.5 | - | - | ftka package not installed |
| 16 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 65536 | 512 | `torch` | ok | 92.8 | 32.2 | ok | ok |  |
| 32 | 1 | 65536 | 512 | `flashinfer_topk_page_table` | ok | 73.8 | 32.2 | ok | ok |  |
| 32 | 1 | 65536 | 512 | `ftka_raft_topk` | skipped | - | 32.2 | - | - | ftka package not installed |
| 32 | 1 | 65536 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 65536 | 1024 | `torch` | ok | 93.7 | 32.1 | ok | ok |  |
| 32 | 1 | 65536 | 1024 | `flashinfer_topk_page_table` | ok | 73.9 | 32.1 | ok | ok |  |
| 32 | 1 | 65536 | 1024 | `ftka_raft_topk` | skipped | - | 32.1 | - | - | ftka package not installed |
| 32 | 1 | 65536 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 65536 | 2048 | `torch` | ok | 94.8 | 32.2 | ok | ok |  |
| 32 | 1 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 73.7 | 32.2 | ok | ok |  |
| 32 | 1 | 65536 | 2048 | `ftka_raft_topk` | skipped | - | 32.2 | - | - | ftka package not installed |
| 32 | 1 | 65536 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 65536 | 4096 | `torch` | ok | 95.9 | 32.2 | ok | ok |  |
| 32 | 1 | 65536 | 4096 | `flashinfer_topk_page_table` | skipped | - | 32.2 | - | - | top_k>2048 ceiling |
| 32 | 1 | 65536 | 4096 | `ftka_raft_topk` | skipped | - | 32.2 | - | - | ftka package not installed |
| 32 | 1 | 65536 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 65536 | 8192 | `torch` | ok | 98.2 | 32.1 | ok | ok |  |
| 32 | 1 | 65536 | 8192 | `flashinfer_topk_page_table` | skipped | - | 32.1 | - | - | top_k>2048 ceiling |
| 32 | 1 | 65536 | 8192 | `ftka_raft_topk` | skipped | - | 32.1 | - | - | ftka package not installed |
| 32 | 1 | 65536 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 131072 | 512 | `torch` | ok | 89.3 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 77.6 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 512 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 1 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 131072 | 1024 | `torch` | ok | 88.8 | 18.3 | ok | ok |  |
| 1 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 80.5 | 18.3 | ok | ok |  |
| 1 | 1 | 131072 | 1024 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 1 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 131072 | 2048 | `torch` | ok | 89.2 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 86.0 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 2048 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 1 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 131072 | 4096 | `torch` | ok | 89.5 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.2 | - | - | top_k>2048 ceiling |
| 1 | 1 | 131072 | 4096 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 1 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 1 | 1 | 131072 | 8192 | `torch` | ok | 89.0 | 18.2 | ok | ok |  |
| 1 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.2 | - | - | top_k>2048 ceiling |
| 1 | 1 | 131072 | 8192 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 1 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 131072 | 512 | `torch` | ok | 90.3 | 18.2 | ok | ok |  |
| 4 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 79.7 | 18.2 | ok | ok |  |
| 4 | 1 | 131072 | 512 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 4 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 131072 | 1024 | `torch` | ok | 88.8 | 18.2 | ok | ok |  |
| 4 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 82.9 | 18.2 | ok | ok |  |
| 4 | 1 | 131072 | 1024 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 4 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 131072 | 2048 | `torch` | ok | 88.8 | 18.2 | ok | ok |  |
| 4 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 90.3 | 18.2 | ok | ok |  |
| 4 | 1 | 131072 | 2048 | `ftka_raft_topk` | skipped | - | 18.2 | - | - | ftka package not installed |
| 4 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 131072 | 4096 | `torch` | ok | 88.3 | 18.3 | ok | ok |  |
| 4 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 4 | 1 | 131072 | 4096 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 4 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 1 | 131072 | 8192 | `torch` | ok | 89.2 | 18.3 | ok | ok |  |
| 4 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.3 | - | - | top_k>2048 ceiling |
| 4 | 1 | 131072 | 8192 | `ftka_raft_topk` | skipped | - | 18.3 | - | - | ftka package not installed |
| 4 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 131072 | 512 | `torch` | ok | 90.1 | 18.6 | ok | ok |  |
| 8 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 81.7 | 18.6 | ok | ok |  |
| 8 | 1 | 131072 | 512 | `ftka_raft_topk` | skipped | - | 18.6 | - | - | ftka package not installed |
| 8 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 131072 | 1024 | `torch` | ok | 89.6 | 18.7 | ok | ok |  |
| 8 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 84.2 | 18.7 | ok | ok |  |
| 8 | 1 | 131072 | 1024 | `ftka_raft_topk` | skipped | - | 18.7 | - | - | ftka package not installed |
| 8 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 131072 | 2048 | `torch` | ok | 90.3 | 18.5 | ok | ok |  |
| 8 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 90.0 | 18.5 | ok | ok |  |
| 8 | 1 | 131072 | 2048 | `ftka_raft_topk` | skipped | - | 18.5 | - | - | ftka package not installed |
| 8 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 131072 | 4096 | `torch` | ok | 90.6 | 18.8 | ok | ok |  |
| 8 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 18.8 | - | - | top_k>2048 ceiling |
| 8 | 1 | 131072 | 4096 | `ftka_raft_topk` | skipped | - | 18.8 | - | - | ftka package not installed |
| 8 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 8 | 1 | 131072 | 8192 | `torch` | ok | 93.6 | 18.4 | ok | ok |  |
| 8 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 18.4 | - | - | top_k>2048 ceiling |
| 8 | 1 | 131072 | 8192 | `ftka_raft_topk` | skipped | - | 18.4 | - | - | ftka package not installed |
| 8 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 131072 | 512 | `torch` | ok | 97.3 | 32.1 | ok | ok |  |
| 16 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 82.7 | 32.1 | ok | ok |  |
| 16 | 1 | 131072 | 512 | `ftka_raft_topk` | skipped | - | 32.1 | - | - | ftka package not installed |
| 16 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 131072 | 1024 | `torch` | ok | 97.4 | 32.1 | ok | ok |  |
| 16 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 86.2 | 32.1 | ok | ok |  |
| 16 | 1 | 131072 | 1024 | `ftka_raft_topk` | skipped | - | 32.1 | - | - | ftka package not installed |
| 16 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 131072 | 2048 | `torch` | ok | 99.7 | 32.1 | ok | ok |  |
| 16 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 93.8 | 32.1 | ok | ok |  |
| 16 | 1 | 131072 | 2048 | `ftka_raft_topk` | skipped | - | 32.1 | - | - | ftka package not installed |
| 16 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 131072 | 4096 | `torch` | ok | 101.4 | 32.1 | ok | ok |  |
| 16 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 32.1 | - | - | top_k>2048 ceiling |
| 16 | 1 | 131072 | 4096 | `ftka_raft_topk` | skipped | - | 32.1 | - | - | ftka package not installed |
| 16 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 16 | 1 | 131072 | 8192 | `torch` | ok | 105.3 | 32.1 | ok | ok |  |
| 16 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 32.1 | - | - | top_k>2048 ceiling |
| 16 | 1 | 131072 | 8192 | `ftka_raft_topk` | skipped | - | 32.1 | - | - | ftka package not installed |
| 16 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 131072 | 512 | `torch` | ok | 118.0 | 87.1 | ok | ok |  |
| 32 | 1 | 131072 | 512 | `flashinfer_topk_page_table` | ok | 85.4 | 87.1 | ok | ok |  |
| 32 | 1 | 131072 | 512 | `ftka_raft_topk` | skipped | - | 87.1 | - | - | ftka package not installed |
| 32 | 1 | 131072 | 512 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 131072 | 1024 | `torch` | ok | 117.5 | 87.2 | ok | ok |  |
| 32 | 1 | 131072 | 1024 | `flashinfer_topk_page_table` | ok | 90.0 | 87.2 | ok | ok |  |
| 32 | 1 | 131072 | 1024 | `ftka_raft_topk` | skipped | - | 87.2 | - | - | ftka package not installed |
| 32 | 1 | 131072 | 1024 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 131072 | 2048 | `torch` | ok | 119.0 | 87.1 | ok | ok |  |
| 32 | 1 | 131072 | 2048 | `flashinfer_topk_page_table` | ok | 93.9 | 87.1 | ok | ok |  |
| 32 | 1 | 131072 | 2048 | `ftka_raft_topk` | skipped | - | 87.1 | - | - | ftka package not installed |
| 32 | 1 | 131072 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 131072 | 4096 | `torch` | ok | 121.0 | 87.3 | ok | ok |  |
| 32 | 1 | 131072 | 4096 | `flashinfer_topk_page_table` | skipped | - | 87.3 | - | - | top_k>2048 ceiling |
| 32 | 1 | 131072 | 4096 | `ftka_raft_topk` | skipped | - | 87.3 | - | - | ftka package not installed |
| 32 | 1 | 131072 | 4096 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 32 | 1 | 131072 | 8192 | `torch` | ok | 126.0 | 86.0 | ok | ok |  |
| 32 | 1 | 131072 | 8192 | `flashinfer_topk_page_table` | skipped | - | 86.0 | - | - | top_k>2048 ceiling |
| 32 | 1 | 131072 | 8192 | `ftka_raft_topk` | skipped | - | 86.0 | - | - | ftka package not installed |
| 32 | 1 | 131072 | 8192 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
| 4 | 8 | 65536 | 2048 | `torch` | ok | 95.2 | 60.5 | ok | ok |  |
| 4 | 8 | 65536 | 2048 | `flashinfer_topk_page_table` | ok | 89.3 | 60.5 | ok | ok |  |
| 4 | 8 | 65536 | 2048 | `ftka_raft_topk` | skipped | - | 60.5 | - | - | ftka package not installed |
| 4 | 8 | 65536 | 2048 | `ftka_gemv+ftka_topk` | skipped | - | - | - | - | ftka package not installed |
