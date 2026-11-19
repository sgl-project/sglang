sgl-kernel branch / base, each run on its own (suite geomean of per-shape median ratios):

| run (process order) | paged | paged16 | sparse | blocks |
|---|---:|---:|---:|---:|
| 1 (base, branch, FI small det, mode 2, FI small, FI) | 0.9999x | 1.0018x | 1.0032x | 1.0015x |
| 2 (branch, mode 2, base, FI, FI small det, FI small) | 0.9958x | 0.9970x | 0.9966x | 0.9972x |
| 3 (mode 2, FI, branch, FI small, base, FI small det) | 0.9995x | 0.9992x | 0.9985x | 1.0000x |
| 4 (FI, FI small, mode 2, FI small det, branch, base) | 1.0003x | 1.0019x | 1.0058x | 1.0039x |
| 5 (FI small, FI small det, FI, base, mode 2, branch) | 0.9992x | 0.9994x | 0.9947x | 1.0001x |
| 6 (FI small det, base, FI small, branch, FI, mode 2) | 1.0025x | 0.9981x | 0.9956x | 0.9973x |

Each variant's run geomean relative to its six-run mean (all 184 shapes), and its slot:

| variant | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 |
|---|---:|---:|---:|---:|---:|---:|
| base | +0.07% (slot 1) | +0.01% (slot 3) | +0.21% (slot 5) | -0.39% (slot 6) | +0.15% (slot 4) | -0.06% (slot 2) |
| branch | +0.18% (slot 2) | -0.32% (slot 1) | +0.20% (slot 3) | -0.22% (slot 5) | +0.11% (slot 6) | +0.05% (slot 4) |
| mode 2 | +0.13% (slot 4) | +0.34% (slot 2) | -0.06% (slot 1) | -0.44% (slot 3) | -0.05% (slot 5) | +0.09% (slot 6) |
| FI | +0.01% (slot 6) | +0.02% (slot 4) | -0.07% (slot 2) | +0.12% (slot 1) | -0.13% (slot 3) | +0.05% (slot 5) |
| FI small | -0.03% (slot 5) | +0.09% (slot 6) | -0.25% (slot 4) | +0.16% (slot 2) | +0.11% (slot 1) | -0.07% (slot 3) |
| FI small det | -0.11% (slot 3) | +0.05% (slot 5) | +0.01% (slot 6) | +0.04% (slot 4) | +0.01% (slot 2) | +0.00% (slot 1) |

Fixed per-replay cost c (median over the cells with R > 8: the fit is singular at R = 8 and unreliable below, where 3 ms / R is too coarse a stand-in for the probe) and median replay length per process:

| variant | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 |
|---|---:|---:|---:|---:|---:|---:|
| base | 35.3 us, 1.90 ms | 35.0 us, 1.92 ms | 37.9 us, 1.86 ms | 29.5 us, 2.03 ms | 38.0 us, 1.88 ms | 33.8 us, 1.94 ms |
| branch | 36.7 us, 1.90 ms | 30.2 us, 2.01 ms | 36.1 us, 1.90 ms | 30.1 us, 2.00 ms | 38.3 us, 1.88 ms | 37.4 us, 1.91 ms |
| mode 2 | 37.3 us, 1.88 ms | 39.5 us, 1.84 ms | 35.1 us, 1.92 ms | 30.0 us, 2.02 ms | 37.3 us, 1.93 ms | 37.8 us, 1.88 ms |
| FI | 35.2 us, 2.39 ms | 37.1 us, 2.36 ms | 36.3 us, 2.37 ms | 37.9 us, 2.35 ms | 35.3 us, 2.41 ms | 38.1 us, 2.35 ms |
| FI small | 34.5 us, 2.47 ms | 37.0 us, 2.44 ms | 30.3 us, 2.53 ms | 38.2 us, 2.43 ms | 36.8 us, 2.44 ms | 33.8 us, 2.48 ms |
| FI small det | 33.2 us, 2.73 ms | 36.2 us, 2.67 ms | 35.9 us, 2.67 ms | 35.8 us, 2.67 ms | 36.3 us, 2.70 ms | 34.3 us, 2.71 ms |

sgl-kernel branch / base of the per-call time t (fixed per-replay cost removed):

| run | paged | paged16 | sparse | blocks |
|---|---:|---:|---:|---:|
| 1 | 0.9998x | 1.0001x | 1.0009x | 1.0002x |
| 2 | 0.9996x | 0.9998x | 1.0003x | 0.9996x |
| 3 | 1.0007x | 1.0006x | 0.9997x | 1.0007x |
| 4 | 0.9999x | 0.9997x | 0.9999x | 1.0001x |
| 5 | 0.9997x | 0.9997x | 0.9988x | 0.9998x |
| 6 | 0.9998x | 0.9994x | 0.9980x | 0.9993x |
| geomean of runs 1-6 | 0.9999x | 0.9999x | 0.9996x | 1.0000x |
