sgl-kernel branch / base, each run on its own (suite geomean of per-shape median ratios):

| run (process order) | paged | paged16 | sparse | blocks |
|---|---:|---:|---:|---:|
| 1 (base, branch, FI small det, mode 2, FI small, FI) | 0.9970x | 1.0013x | 1.0027x | 0.9989x |
| 2 (branch, mode 2, base, FI, FI small det, FI small) | 1.0015x | 1.0023x | 1.0030x | 1.0019x |
| 3 (mode 2, FI, branch, FI small, base, FI small det) | 0.9969x | 0.9986x | 1.0058x | 1.0000x |
| 4 (FI, FI small, mode 2, FI small det, branch, base) | 0.9991x | 1.0002x | 1.0008x | 1.0007x |
| 5 (FI small, FI small det, FI, base, mode 2, branch) | 1.0033x | 1.0006x | 1.0010x | 1.0016x |
| 6 (FI small det, base, FI small, branch, FI, mode 2) | 1.0018x | 0.9948x | 0.9939x | 0.9947x |

Each variant's run geomean relative to its six-run mean (all 184 shapes), and its slot:

| variant | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 |
|---|---:|---:|---:|---:|---:|---:|
| base | +0.06% (slot 1) | +0.03% (slot 3) | +0.12% (slot 5) | -0.00% (slot 6) | -0.07% (slot 4) | -0.14% (slot 2) |
| branch | -0.09% (slot 2) | +0.22% (slot 1) | -0.07% (slot 3) | -0.03% (slot 5) | +0.18% (slot 6) | -0.21% (slot 4) |
| mode 2 | -0.03% (slot 4) | +0.17% (slot 2) | -0.11% (slot 1) | -0.09% (slot 3) | +0.06% (slot 5) | -0.01% (slot 6) |
| FI | -0.09% (slot 6) | -0.04% (slot 4) | -0.10% (slot 2) | +0.02% (slot 1) | +0.30% (slot 3) | -0.09% (slot 5) |
| FI small | -0.01% (slot 5) | -0.08% (slot 6) | -0.03% (slot 4) | +0.04% (slot 2) | +0.06% (slot 1) | +0.02% (slot 3) |
| FI small det | +0.07% (slot 3) | +0.03% (slot 5) | -0.10% (slot 6) | +0.11% (slot 4) | +0.09% (slot 2) | -0.21% (slot 1) |

Fixed per-replay cost c (median over the cells with R > 8: the fit is singular at R = 8 and unreliable below, where 3 ms / R is too coarse a stand-in for the probe) and median replay length per process:

| variant | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 |
|---|---:|---:|---:|---:|---:|---:|
| base | 37.3 us, 1.92 ms | 34.3 us, 1.93 ms | 37.4 us, 1.93 ms | 35.6 us, 1.91 ms | 34.4 us, 1.93 ms | 30.8 us, 1.99 ms |
| branch | 33.6 us, 1.94 ms | 38.3 us, 1.86 ms | 35.9 us, 1.96 ms | 34.2 us, 1.94 ms | 38.3 us, 1.87 ms | 30.3 us, 1.98 ms |
| mode 2 | 36.7 us, 1.95 ms | 38.2 us, 1.87 ms | 33.5 us, 1.96 ms | 32.0 us, 1.97 ms | 37.9 us, 1.89 ms | 32.5 us, 1.98 ms |
| FI | 35.6 us, 2.38 ms | 35.1 us, 2.38 ms | 35.1 us, 2.39 ms | 38.5 us, 2.34 ms | 44.8 us, 2.31 ms | 34.8 us, 2.39 ms |
| FI small | 35.2 us, 2.46 ms | 33.6 us, 2.49 ms | 35.5 us, 2.46 ms | 38.1 us, 2.44 ms | 37.1 us, 2.48 ms | 34.8 us, 2.46 ms |
| FI small det | 35.9 us, 2.71 ms | 37.6 us, 2.67 ms | 32.3 us, 2.68 ms | 37.8 us, 2.66 ms | 37.8 us, 2.67 ms | 29.7 us, 2.72 ms |

sgl-kernel branch / base of the per-call time t (fixed per-replay cost removed):

| run | paged | paged16 | sparse | blocks |
|---|---:|---:|---:|---:|
| 1 | 0.9998x | 0.9994x | 0.9987x | 0.9997x |
| 2 | 0.9991x | 0.9993x | 0.9992x | 0.9988x |
| 3 | 0.9991x | 0.9990x | 0.9997x | 0.9989x |
| 4 | 1.0002x | 1.0004x | 1.0002x | 1.0003x |
| 5 | 0.9997x | 0.9998x | 1.0002x | 0.9999x |
| 6 | 1.0002x | 1.0002x | 1.0006x | 1.0004x |
| geomean of runs 1-6 | 0.9997x | 0.9997x | 0.9997x | 0.9997x |
