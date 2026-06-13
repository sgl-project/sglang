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

Supplemental LTX-2.3 extra preset, final PR commit `696b8b240b871aa98b4ce9c44a2f3bbe557a5055`:

| Preset | Mode | E2E | Denoise | Peak mem | Notes |
|---|---|---:|---:|---:|---|
| `ltx23-one-stage` | `default` | 26.965 s | 25.178 s | 56.6 GB | Native SGLang backend, production helper command, 2x B200 |
| `ltx23-one-stage` | `max-autotune-no-cudagraphs` | n/a | n/a | n/a | Attempted forced mode; stopped after ~14 min in inductor autotune without a valid perf dump |
