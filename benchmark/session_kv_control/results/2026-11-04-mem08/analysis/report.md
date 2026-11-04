# Sustained KV stress measurements

All completed requests remain in latency and throughput accounting, including invalid control paths. Failed requests remain in planned denominators. Source publication is separate. Negative deltas mean lower latency/cost; every direction is retained.

| Cell | mode | group | verified / valid | planned / completed / invalid | TTFT mean / p95 ms | arm seconds | requests/s | output tokens/s | host / storage tokens |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| l2-natural-baseline | natural | on_demand | True / True | 183 / 183 / 0 | 4552.012 / 7880.944 | 513.087 | 0.357 | 1.082 | 0 / 196608 |
| l2-natural-baseline | natural | pin | True / True | 183 / 183 / 0 | 586.192 / 767.835 | 463.465 | 0.395 | 1.198 | 196608 / 0 |
| l2-pressure-baseline | demote | on_demand | True / True | 183 / 183 / 0 | 3630.534 / 6154.894 | 421.734 | 0.434 | 1.316 | 0 / 196608 |
| l2-pressure-baseline | demote | pin | True / True | 183 / 183 / 0 | 573.820 / 871.750 | 422.327 | 0.433 | 1.314 | 196608 / 0 |
| l3-c16-natural-baseline | natural | on_demand | True / True | 384 / 384 / 0 | 14749.863 / 20518.328 | 519.972 | 0.739 | 23.632 | 0 / 6291456 |
| l3-c16-natural-baseline | natural | prefetch | True / True | 384 / 384 / 0 | 11963.659 / 20605.290 | 582.297 | 0.659 | 21.103 | 49152 / 6242304 |
| l3-c16-natural-optimized | natural | on_demand | True / True | 384 / 384 / 0 | 7920.359 / 15689.088 | 321.928 | 1.193 | 38.170 | 0 / 6291456 |
| l3-c16-natural-optimized | natural | prefetch | True / True | 384 / 384 / 0 | 7133.039 / 14109.678 | 325.332 | 1.180 | 37.771 | 163840 / 6127616 |
| l3-c32-natural-baseline | natural | on_demand | True / False | 384 / 384 / 0 | 21562.747 / 38196.241 | 435.796 | 0.881 | 28.197 | 114688 / 4665856 |
| l3-c32-natural-baseline | natural | prefetch | True / False | 384 / 384 / 1 | 15697.403 / 30541.329 | 483.671 | 0.794 | 25.406 | 311296 / 5219648 |
| l3-c32-natural-optimized | natural | on_demand | True / False | 384 / 384 / 0 | 14734.846 / 25954.812 | 295.999 | 1.297 | 41.514 | 16384 / 5011712 |
| l3-c32-natural-optimized | natural | prefetch | True / False | 384 / 384 / 4 | 10583.513 / 30153.746 | 259.837 | 1.478 | 47.291 | 665920 / 5329984 |
| l3-c8-natural-baseline | natural | on_demand | True / True | 384 / 384 / 0 | 3650.831 / 6164.592 | 471.394 | 0.815 | 26.067 | 0 / 6291456 |
| l3-c8-natural-baseline | natural | prefetch | True / True | 384 / 384 / 0 | 4630.237 / 8123.189 | 559.416 | 0.686 | 21.966 | 245760 / 6045696 |
| l3-c8-natural-optimized | natural | on_demand | True / True | 384 / 384 / 0 | 1343.191 / 2506.447 | 293.137 | 1.310 | 41.919 | 0 / 6291456 |
| l3-c8-natural-optimized | natural | prefetch | True / True | 384 / 384 / 0 | 699.029 / 1796.825 | 262.992 | 1.460 | 46.724 | 2572288 / 3719168 |

Only the initial laboratory cache-flush RPC is excluded from work rates below. Capability refresh, priming/publication, pressure, waits, inference, control drain and cleanup remain included. Inclusive arm rates above remain available.

| Cell / group | flush timing arms | flush mean seconds | work seconds | work requests/s | work output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| l2-natural-baseline / on_demand | 3 / 3 | 19.790 | 453.716 | 0.403 | 1.223 |
| l2-natural-baseline / pin | 3 / 3 | 3.985 | 451.510 | 0.405 | 1.229 |
| l2-pressure-baseline / on_demand | 3 / 3 | 0.024 | 421.662 | 0.434 | 1.316 |
| l2-pressure-baseline / pin | 3 / 3 | 0.036 | 422.218 | 0.433 | 1.314 |
| l3-c16-natural-baseline / on_demand | 3 / 3 | 0.050 | 519.821 | 0.739 | 23.639 |
| l3-c16-natural-baseline / prefetch | 3 / 3 | 0.044 | 582.164 | 0.660 | 21.107 |
| l3-c16-natural-optimized / on_demand | 3 / 3 | 0.028 | 321.845 | 1.193 | 38.180 |
| l3-c16-natural-optimized / prefetch | 3 / 3 | 0.023 | 325.263 | 1.181 | 37.779 |
| l3-c32-natural-baseline / on_demand | 3 / 3 | 0.064 | 435.605 | 0.882 | 28.209 |
| l3-c32-natural-baseline / prefetch | 3 / 3 | 0.057 | 483.500 | 0.794 | 25.415 |
| l3-c32-natural-optimized / on_demand | 3 / 3 | 0.026 | 295.921 | 1.298 | 41.525 |
| l3-c32-natural-optimized / prefetch | 3 / 3 | 0.026 | 259.759 | 1.478 | 47.305 |
| l3-c8-natural-baseline / on_demand | 3 / 3 | 0.053 | 471.236 | 0.815 | 26.076 |
| l3-c8-natural-baseline / prefetch | 3 / 3 | 0.039 | 559.300 | 0.687 | 21.970 |
| l3-c8-natural-optimized / on_demand | 3 / 3 | 0.023 | 293.067 | 1.310 | 41.929 |
| l3-c8-natural-optimized / prefetch | 3 / 3 | 0.030 | 262.903 | 1.461 | 46.740 |

| Cell / paired block | candidate | valid | mean TTFT delta ms | inclusive arm delta ms | work delta ms | work requests/s delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| l2-natural-baseline / 0 | pin | True | -2387.609 | -46171.580 | 1251.282 | -0.003 |
| l2-natural-baseline / 1 | pin | True | -4943.761 | -7872.374 | -7882.208 | 0.023 |
| l2-natural-baseline / 2 | pin | True | -4566.090 | 4421.567 | 4425.339 | -0.011 |
| l2-pressure-baseline / 0 | pin | True | -3457.268 | -5701.092 | -5709.845 | 0.018 |
| l2-pressure-baseline / 1 | pin | True | -2230.006 | -3511.310 | -3538.241 | 0.012 |
| l2-pressure-baseline / 2 | pin | True | -3482.869 | 9805.306 | 9804.040 | -0.028 |
| l3-c16-natural-baseline / 0 | prefetch | True | -4654.375 | 7428.384 | 7441.948 | -0.029 |
| l3-c16-natural-baseline / 1 | prefetch | True | -1640.102 | 30673.189 | 30671.075 | -0.091 |
| l3-c16-natural-baseline / 2 | prefetch | True | -2064.134 | 24223.001 | 24229.973 | -0.121 |
| l3-c16-natural-optimized / 0 | prefetch | True | -834.405 | -5794.442 | -5788.349 | 0.057 |
| l3-c16-natural-optimized / 1 | prefetch | True | -649.715 | 5196.788 | 5200.062 | -0.061 |
| l3-c16-natural-optimized / 2 | prefetch | True | -877.842 | 4001.179 | 4006.379 | -0.047 |
| l3-c32-natural-baseline / 0 | prefetch | True | -7683.363 | -3702.277 | -3662.487 | 0.024 |
| l3-c32-natural-baseline / 1 | prefetch | True | -7497.817 | 13053.238 | 13032.104 | -0.064 |
| l3-c32-natural-baseline / 2 | prefetch | False | -2414.853 | 38523.702 | 38525.546 | -0.203 |
| l3-c32-natural-optimized / 0 | prefetch | True | -6210.809 | -28951.661 | -28946.831 | 0.507 |
| l3-c32-natural-optimized / 1 | prefetch | False | -2481.144 | -706.991 | -713.462 | 0.009 |
| l3-c32-natural-optimized / 2 | prefetch | True | -3762.048 | -6503.700 | -6501.965 | 0.108 |
| l3-c8-natural-baseline / 0 | prefetch | True | 1487.564 | 39487.606 | 39504.916 | -0.165 |
| l3-c8-natural-baseline / 1 | prefetch | True | 1118.453 | 30582.709 | 30606.493 | -0.147 |
| l3-c8-natural-baseline / 2 | prefetch | True | 332.201 | 17951.622 | 17952.479 | -0.075 |
| l3-c8-natural-optimized / 0 | prefetch | True | -330.546 | -5949.254 | -5936.426 | 0.089 |
| l3-c8-natural-optimized / 1 | prefetch | True | -841.068 | -13224.480 | -13230.577 | 0.192 |
| l3-c8-natural-optimized / 2 | prefetch | True | -760.869 | -10971.011 | -10997.134 | 0.170 |

GPU and host peaks in analysis.json are sampled maxima. Arm attribution uses raw arm-result logging time minus arm duration when available, with an unknown output-validation/logging gap; row-start bounds are a marked fallback. GPU allocated memory does not establish occupied KV capacity. Active token_usage/kv_used exclude evictable radix KV; resident/reserved tokens use boot capacity minus kv_available and can include allocation/transfer work. Three blocks do not establish independent request-level confidence intervals or stable p99.

L2 foreground, pressure, complete arm metrics and L3 tool-wait client cycles are separate in analysis.json. The natural profile preserves native GPU residency; the demote profile deliberately removes GPU prefixes. Compare profiles separately.


| Cell / arm | initial flush seconds | capability refresh seconds (included) | inclusive arm seconds | work seconds |
| --- | ---: | ---: | ---: | ---: |
| l2-natural-baseline / b0-pin | 11.875 | 0.025 | 163.341 | 151.466 |
| l2-natural-baseline / b0-on_demand | 59.298 | 0.006 | 209.512 | 150.215 |
| l2-natural-baseline / b1-on_demand | 0.028 | 0.003 | 148.235 | 148.207 |
| l2-natural-baseline / b1-pin | 0.038 | 0.010 | 140.363 | 140.325 |
| l2-natural-baseline / b2-on_demand | 0.045 | 0.004 | 155.339 | 155.294 |
| l2-natural-baseline / b2-pin | 0.042 | 0.004 | 159.761 | 159.719 |
| l2-pressure-baseline / b0-pin | 0.028 | 0.006 | 135.037 | 135.009 |
| l2-pressure-baseline / b0-on_demand | 0.020 | 0.003 | 140.738 | 140.719 |
| l2-pressure-baseline / b1-on_demand | 0.022 | 0.003 | 138.767 | 138.745 |
| l2-pressure-baseline / b1-pin | 0.049 | 0.004 | 135.255 | 135.207 |
| l2-pressure-baseline / b2-on_demand | 0.031 | 0.003 | 142.229 | 142.198 |
| l2-pressure-baseline / b2-pin | 0.032 | 0.003 | 152.034 | 152.002 |
| l3-c16-natural-baseline / b0-prefetch | 0.030 | 0.004 | 186.241 | 186.212 |
| l3-c16-natural-baseline / b0-on_demand | 0.043 | 0.005 | 178.813 | 178.770 |
| l3-c16-natural-baseline / b1-on_demand | 0.058 | 0.005 | 192.500 | 192.441 |
| l3-c16-natural-baseline / b1-prefetch | 0.061 | 0.005 | 223.173 | 223.112 |
| l3-c16-natural-baseline / b2-on_demand | 0.049 | 0.004 | 148.659 | 148.610 |
| l3-c16-natural-baseline / b2-prefetch | 0.042 | 0.004 | 172.882 | 172.840 |
| l3-c16-natural-optimized / b0-prefetch | 0.019 | 0.003 | 111.355 | 111.335 |
| l3-c16-natural-optimized / b0-on_demand | 0.025 | 0.003 | 117.149 | 117.124 |
| l3-c16-natural-optimized / b1-on_demand | 0.031 | 0.003 | 102.092 | 102.061 |
| l3-c16-natural-optimized / b1-prefetch | 0.028 | 0.004 | 107.289 | 107.261 |
| l3-c16-natural-optimized / b2-on_demand | 0.027 | 0.004 | 102.687 | 102.660 |
| l3-c16-natural-optimized / b2-prefetch | 0.021 | 0.003 | 106.688 | 106.667 |
| l3-c32-natural-baseline / b0-prefetch | 0.044 | 0.004 | 139.289 | 139.245 |
| l3-c32-natural-baseline / b0-on_demand | 0.084 | 0.005 | 142.991 | 142.907 |
| l3-c32-natural-baseline / b1-on_demand | 0.060 | 0.004 | 155.090 | 155.031 |
| l3-c32-natural-baseline / b1-prefetch | 0.081 | 0.006 | 168.143 | 168.063 |
| l3-c32-natural-baseline / b2-on_demand | 0.048 | 0.003 | 137.715 | 137.667 |
| l3-c32-natural-baseline / b2-prefetch | 0.046 | 0.003 | 176.238 | 176.193 |
| l3-c32-natural-optimized / b0-prefetch | 0.022 | 0.003 | 72.282 | 72.260 |
| l3-c32-natural-optimized / b0-on_demand | 0.027 | 0.003 | 101.233 | 101.207 |
| l3-c32-natural-optimized / b1-on_demand | 0.030 | 0.003 | 103.694 | 103.664 |
| l3-c32-natural-optimized / b1-prefetch | 0.037 | 0.006 | 102.987 | 102.950 |
| l3-c32-natural-optimized / b2-on_demand | 0.021 | 0.003 | 91.072 | 91.050 |
| l3-c32-natural-optimized / b2-prefetch | 0.020 | 0.003 | 84.568 | 84.548 |
| l3-c8-natural-baseline / b0-prefetch | 0.040 | 0.005 | 195.996 | 195.956 |
| l3-c8-natural-baseline / b0-on_demand | 0.057 | 0.003 | 156.508 | 156.451 |
| l3-c8-natural-baseline / b1-on_demand | 0.044 | 0.003 | 148.668 | 148.624 |
| l3-c8-natural-baseline / b1-prefetch | 0.021 | 0.003 | 179.251 | 179.231 |
| l3-c8-natural-baseline / b2-on_demand | 0.057 | 0.006 | 166.217 | 166.161 |
| l3-c8-natural-baseline / b2-prefetch | 0.056 | 0.004 | 184.169 | 184.113 |
| l3-c8-natural-optimized / b0-prefetch | 0.007 | 0.002 | 89.603 | 89.596 |
| l3-c8-natural-optimized / b0-on_demand | 0.020 | 0.003 | 95.553 | 95.533 |
| l3-c8-natural-optimized / b1-on_demand | 0.026 | 0.004 | 100.906 | 100.880 |
| l3-c8-natural-optimized / b1-prefetch | 0.033 | 0.004 | 87.682 | 87.649 |
| l3-c8-natural-optimized / b2-on_demand | 0.024 | 0.003 | 96.678 | 96.654 |
| l3-c8-natural-optimized / b2-prefetch | 0.050 | 0.008 | 85.707 | 85.657 |

- l3-c16-natural-baseline source publication: 64/64 generations; 148.845 seconds, excluded from arm throughput.
- l3-c16-natural-optimized source publication: 64/64 generations; 146.870 seconds, excluded from arm throughput.
- l3-c32-natural-baseline source publication: 64/64 generations; 147.160 seconds, excluded from arm throughput.
- l3-c32-natural-baseline: completed; "invalid measured arms"
- l3-c32-natural-optimized source publication: 64/64 generations; 145.593 seconds, excluded from arm throughput.
- l3-c32-natural-optimized: completed; "invalid measured arms"
- l3-c8-natural-baseline source publication: 64/64 generations; 143.544 seconds, excluded from arm throughput.
- l3-c8-natural-optimized source publication: 64/64 generations; 146.030 seconds, excluded from arm throughput.
- calibration-0: incomplete; "manifest/checksum closure missing; live partial cells are excluded"
- l2-smoke: functional_only; "invalid measured arms"
- l3-smoke: failed_setup_or_run; {"error": "source publication failed; comparison cannot begin", "traceback": "Traceback (most recent call last):\n  File \"/tmp/sglang-rfc-build/mem08-harness-worktree/benchmark/session_kv_control/stress_bench.py\", line 491, in run\n    raise BenchError(\"source publication failed; comparison cannot begin\")\nsession_kv_bench.BenchError: source publication failed; comparison cannot begin\n"}
- l3-smoke-v2: functional_only; "invalid measured arms"
