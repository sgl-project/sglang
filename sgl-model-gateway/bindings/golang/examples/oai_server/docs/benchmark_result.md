/tmp/ShareGPT_V3_unfiltered_cleaned_split.json: 100%|████████████████████| 642M/642M [10:02<00:00, 1.12MB/s]
#Input tokens: 50561
#Output tokens: 25883
Starting warmup with 5 sequences...
Warmup completed with 5 sequences. Starting main benchmark run...

============ Serving Benchmark Result ============
Backend:                                 sglang-oai-chat
Traffic request rate:                    20.0
Max request concurrency:                 20
Successful requests:                     100
Benchmark duration (s):                  107.24
Total input tokens:                      50561
Total input text tokens:                 50561
Total input vision tokens:               0
Total generated tokens:                  25883
Total generated tokens (retokenized):    129591
Request throughput (req/s):              0.93
Input token throughput (tok/s):          471.48
Output token throughput (tok/s):         241.36
Total token throughput (tok/s):          712.84
Concurrency:                             16.42
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   17609.46
Median E2E Latency (ms):                 12343.82
---------------Time to First Token----------------
Mean TTFT (ms):                          190.71
Median TTFT (ms):                        164.86
P99 TTFT (ms):                           397.72
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          162.55
Median TPOT (ms):                        63.51
P99 TPOT (ms):                           1337.20
---------------Inter-Token Latency----------------
Mean ITL (ms):                           25.85
Median ITL (ms):                         24.26
P95 ITL (ms):                            48.26
P99 ITL (ms):                            119.04
Max ITL (ms):                            194.58
==================================================

✓ E2E test completed


## Rust
============ Serving Benchmark Result ============
Backend:                                 sglang-oai-chat
Traffic request rate:                    20.0
Max request concurrency:                 20
Successful requests:                     100
Benchmark duration (s):                  37.71
Total input tokens:                      50561
Total input text tokens:                 50561
Total input vision tokens:               0
Total generated tokens:                  25883
Total generated tokens (retokenized):    25599
Request throughput (req/s):              2.65
Input token throughput (tok/s):          1340.75
Output token throughput (tok/s):         686.35
Total token throughput (tok/s):          2027.10
Concurrency:                             18.58
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   7008.05
Median E2E Latency (ms):                 7061.24
---------------Time to First Token----------------
Mean TTFT (ms):                          156.09
Median TTFT (ms):                        133.81
P99 TTFT (ms):                           318.53
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.59
Median TPOT (ms):                        26.75
P99 TPOT (ms):                           29.18
---------------Inter-Token Latency----------------
Mean ITL (ms):                           26.71
Median ITL (ms):                         23.61
P95 ITL (ms):                            66.11
P99 ITL (ms):                            115.30
Max ITL (ms):                            201.08
==================================================


## golang
#Input tokens: 50561
#Output tokens: 25883
Starting warmup with 5 sequences...
Warmup completed with 5 sequences. Starting main benchmark run...

============ Serving Benchmark Result ============
Backend:                                 sglang-oai-chat
Traffic request rate:                    20.0
Max request concurrency:                 20
Successful requests:                     100
Benchmark duration (s):                  34.22
Total input tokens:                      50561
Total input text tokens:                 50561
Total input vision tokens:               0
Total generated tokens:                  22970
Total generated tokens (retokenized):    31740
Request throughput (req/s):              2.92
Input token throughput (tok/s):          1477.70
Output token throughput (tok/s):         671.32
Total token throughput (tok/s):          2149.03
Concurrency:                             18.42
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   6303.33
Median E2E Latency (ms):                 6294.46
---------------Time to First Token----------------
Mean TTFT (ms):                          157.10
Median TTFT (ms):                        149.16
P99 TTFT (ms):                           251.98
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.49
Median TPOT (ms):                        27.15
P99 TPOT (ms):                           28.73
---------------Inter-Token Latency----------------
Mean ITL (ms):                           26.97
Median ITL (ms):                         24.61
P95 ITL (ms):                            52.39
P99 ITL (ms):                            86.52
Max ITL (ms):                            194.55
==================================================
