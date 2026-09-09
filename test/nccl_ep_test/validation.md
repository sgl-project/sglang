# Recorded NCCL EP validation

The two-rank execution snapshot is based on PR #32329 commit
`efd30456466549fd2be6131eaba0edd8972506a4`. The validated implementation patch has
SHA256 `7227531565825e203c3fdc71fbdf14e013c2ec01a08f03c831aa280d46a67c1e`.
PR preparation relocates the shared harness, registers the unit tests, provides
an in-tree CLI, and adds installation metadata/documentation. Those changes must
not be confused with a fresh two-GPU run of the final submission commit.

## Environment

Recorded on 2026-09-09: two RTX PRO 6000 Blackwell Server Edition GPUs (SM120),
SYS topology with bidirectional direct CUDA peer access, Ubuntu 22.04.5,
Python 3.12.12, driver 595.71.05, CUDA Toolkit 13.0.88. The driver advertises CUDA
compatibility 13.2; Torch's CUDA build is 13.0. The official Torch
`2.11.0+cu130` wheel was used with `nccl4py==0.4.1`,
`nccl-extensions==0.1.0`, and the explicit NCCL 2.30.7 override. Torch's NCCL
compile-time query returns 2.28.9; the actual mapped library returns 23007.

## Correctness

All counts below are per rank unless stated otherwise. All comparisons use
`rtol=0, atol=0` with exactly representable fixtures.

| Test | Result |
| --- | --- |
| CPU oracle | 180 fixtures passed |
| Native eager, weighted and identity experts | 180 checks per mode |
| SGLang eager including FP8 conversion, weighted and identity experts | 360 checks per mode |
| Native / SGLang single-bucket Graph | 4 checks each |
| Native / SGLang dynamic Graph matrix | 4118 checks each |
| Actual runner input loading, streams, padding and recapture | 12 checks |
| Controlled host capture failure, cleanup and fresh capture | 6 checks |
| Separate native true T=0 / capacity-controlled duplicate-expert probes | Both passed |

Dynamic tests use buckets 8/16/32, two layers, two capture generations and 1000
replays per generation. The native continuation descriptor-lifetime failure found
on hardware is covered by the corrected implementation and a weak-reference mock
regression. All required graph, handle and group closures were observed. The
matrix verifier checked 27 required reports and the 2942-file source inventory.

The submitted registered tests also passed locally on RTX 4060 Laptop (SM89):
88 tests across the six NCCL EP files, including real CUDA Graphs with fake EP.
That local environment uses the historical Torch `2.11.0+cu130.nccl2307` build;
only the target-hardware results above use the official Torch wheel.

Full-model serving, model accuracy, real expert GEMMs, concurrent replay and
cross-node execution were not evaluated. True T=0 and duplicate-expert probes
are native eager input checks, not a SGLang Graph input contract.

## Unprofiled synthetic-step latency

Four rounds of 200 samples per mode/rank, 20 warmups per block, alternating
mode order. Take the maximum across aligned rank samples before median/p95.
The measured step includes GPU input copies, two synthetic expert layers,
receive snapshots, retained combine results and host launch gaps.

| Tokens/rank | Routing | Eager median / p95 (ms) | Graph median / p95 (ms) | Median ratio |
| --- | --- | --- | --- | --- |
| 8 | balanced | 0.7774 / 1.1629 | 0.1413 / 0.1671 | 5.50x |
| 8 | hotspot | 0.7676 / 0.8485 | 0.1440 / 0.1561 | 5.33x |
| 16 | balanced | 0.7708 / 0.9042 | 0.1526 / 0.1783 | 5.05x |
| 16 | hotspot | 0.7755 / 0.8583 | 0.1582 / 0.1833 | 4.90x |
| 32 | balanced | 0.7790 / 0.9509 | 0.1756 / 0.2011 | 4.44x |
| 32 | hotspot | 0.7747 / 0.8736 | 0.1884 / 0.2143 | 4.11x |

These ratios measure the complete serial synthetic step, not isolated EP kernels
or model throughput. They do not establish equivalence to a PIX/PXB/NVLink pair.

## Nsight Systems

Nsight Systems 2025.3.1, separate run with CUDA Graph node tracing:
4000 NVTX replay ranges correspond to 4000 Graph launches; each GPU records
8000 dispatch and 8000 combine kernel instances as Graph nodes. Recorded replay
host ranges contain no CUDA device allocation/free, Graph instantiation/destruction,
or module/library loads. Harness copies and event operations remain present.
Both rank oracles pass under profiling. Profiling and rank-arrival skew were not
isolated from communication cost; the profiled dispatch tail is not a SYS-link
latency measurement.

## Evidence identifiers

The raw matrix, rank JSON, benchmark samples and Nsight trace are retained outside
the source tree to avoid committing generated binary evidence. The identifiers
below allow an attached archive to be verified; hashes are not download links.

- Matrix JSON SHA256: `55d6706e00958bb6f1b3c18bd01459dacdab662df7f9af7d7b5b9c41bd57bc57`
- `server-final-evidence.tar.gz` SHA256: `dc2f588b808ab3375ccc026c0105eee0b16e85c7fb5d31c1a3cafbb415253793`
- `target-jit-evidence.tar.gz` SHA256: `b8a1b38934712f50d58ff465c127bdb0cf24f94d7c7ae045a8b6a9820e893d82`
