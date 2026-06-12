| Metric | Latest main | PR | Speedup |
|---|---:|---:|---:|
| Full request wall | 9.867 +/- 0.180 s | 8.701 +/- 0.192 s | 1.134x |
| Denoise wall | 8.510 +/- 0.152 s | 7.385 +/- 0.195 s | 1.152x |
| Denoise step2+ sum | 8.166 +/- 0.073 s | 7.111 +/- 0.205 s | 1.148x |
| Denoise median step | 257.30 +/- 8.22 ms | 223.32 +/- 7.51 ms | 1.152x |
| Peak memory | 120.1 +/- 0.0 GB | 120.1 +/- 0.0 GB | 1.000x |

| Run | Main E2E | PR E2E | Main denoise | PR denoise |
|---:|---:|---:|---:|---:|
| 1 | 9.970 s | 10.864 s | 8.575 s | 9.321 s |
| 2 | 9.994 s | 8.837 s | 8.618 s | 7.523 s |
| 3 | 9.739 s | 8.565 s | 8.402 s | 7.247 s |
