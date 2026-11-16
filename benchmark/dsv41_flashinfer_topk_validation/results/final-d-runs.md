sgl-kernel branch / base, each run on its own (suite geomean of per-shape median ratios):

| run (process order) | paged | paged16 | sparse | blocks |
|---|---:|---:|---:|---:|
| 1 (base, branch, FI small det, mode 2, FI small, FI) | 1.0004x | 0.9961x | 0.9972x | 0.9955x |
| 2 (branch, mode 2, base, FI, FI small det, FI small) | 0.9945x | 1.0027x | 0.9998x | 1.0004x |
| 3 (mode 2, FI, branch, FI small, base, FI small det) | 0.9960x | 1.0006x | 0.9992x | 1.0001x |
| 4 (FI, FI small, mode 2, FI small det, branch, base) | 1.0014x | 0.9996x | 0.9992x | 0.9992x |
| 5 (FI small, FI small det, FI, base, mode 2, branch) | 0.9963x | 0.9966x | 0.9952x | 0.9965x |
| 6 (FI small det, base, FI small, branch, FI, mode 2) | 0.9999x | 0.9945x | 0.9930x | 0.9948x |

Each variant's run geomean relative to its six-run mean (all 184 shapes), and its slot:

| variant | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 |
|---|---:|---:|---:|---:|---:|---:|
| base | +0.17% (slot 1) | +0.04% (slot 3) | +0.04% (slot 5) | -0.43% (slot 6) | +0.08% (slot 4) | +0.11% (slot 2) |
| branch | +0.25% (slot 2) | -0.06% (slot 1) | -0.01% (slot 3) | -0.17% (slot 5) | -0.09% (slot 6) | +0.09% (slot 4) |
| mode 2 | +0.02% (slot 4) | +0.32% (slot 2) | -0.33% (slot 1) | -0.14% (slot 3) | -0.02% (slot 5) | +0.16% (slot 6) |
| FI | -0.30% (slot 6) | +0.15% (slot 4) | -0.08% (slot 2) | +0.12% (slot 1) | -0.09% (slot 3) | +0.21% (slot 5) |
| FI small | -0.11% (slot 5) | +0.03% (slot 6) | +0.02% (slot 4) | +0.04% (slot 2) | +0.07% (slot 1) | -0.06% (slot 3) |
| FI small det | +0.17% (slot 3) | +0.06% (slot 5) | -0.22% (slot 6) | -0.00% (slot 4) | -0.09% (slot 2) | +0.08% (slot 1) |

Fixed per-replay cost c (median over the cells with R > 8: the fit is singular at R = 8 and unreliable below, where 3 ms / R is too coarse a stand-in for the probe) and median replay length per process:

| variant | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 |
|---|---:|---:|---:|---:|---:|---:|
| base | 38.6 us, 1.86 ms | 38.0 us, 1.93 ms | 36.0 us, 1.91 ms | 30.2 us, 2.01 ms | 35.3 us, 1.91 ms | 35.7 us, 1.91 ms |
| branch | 37.5 us, 1.92 ms | 30.8 us, 2.01 ms | 35.5 us, 1.98 ms | 28.8 us, 2.02 ms | 31.3 us, 1.99 ms | 35.0 us, 1.96 ms |
| mode 2 | 30.4 us, 2.01 ms | 36.8 us, 1.91 ms | 36.1 us, 2.17 ms | 31.6 us, 2.00 ms | 35.5 us, 1.99 ms | 37.1 us, 1.94 ms |
| FI | 33.9 us, 2.56 ms | 37.0 us, 2.40 ms | 28.3 us, 2.47 ms | 35.1 us, 2.43 ms | 29.0 us, 2.45 ms | 38.7 us, 2.39 ms |
| FI small | 33.9 us, 2.48 ms | 37.6 us, 2.45 ms | 36.5 us, 2.46 ms | 37.1 us, 2.45 ms | 37.0 us, 2.45 ms | 38.3 us, 2.46 ms |
| FI small det | 37.5 us, 2.69 ms | 36.0 us, 2.68 ms | 28.2 us, 2.74 ms | 36.4 us, 2.72 ms | 31.6 us, 2.71 ms | 36.5 us, 2.68 ms |

sgl-kernel branch / base of the per-call time t (fixed per-replay cost removed):

| run | paged | paged16 | sparse | blocks |
|---|---:|---:|---:|---:|
| 1 | 1.0009x | 1.0007x | 0.9996x | 1.0009x |
| 2 | 1.0001x | 0.9994x | 0.9989x | 0.9997x |
| 3 | 0.9985x | 0.9998x | 0.9996x | 0.9996x |
| 4 | 1.0006x | 1.0004x | 1.0000x | 1.0006x |
| 5 | 0.9994x | 0.9991x | 0.9976x | 0.9983x |
| 6 | 0.9997x | 0.9993x | 0.9983x | 0.9993x |
| geomean of runs 1-6 | 0.9999x | 0.9998x | 0.9990x | 0.9997x |
