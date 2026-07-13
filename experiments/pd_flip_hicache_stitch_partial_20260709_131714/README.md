# PD flip HiCache stitch partial run - 2026-07-09 13:18:29 +08:00

Status: implementation deployed and unit-tested earlier; cluster experiment did not reach request/replay phase.

Blocking condition:
- cloud-099 was unreachable from local SSH and from cloud-100/101/102 private network.
- 192.168.0.42 did not answer ping, Mooncake master 50051, metadata 8080, or SGLang 32000.
- Workers on cloud-100/101/102 started with HiCache/Mooncake args and then waited for Mooncake master, repeatedly logging Client not available / Failed to create client.

Important evidence:
- Each worker log shows --enable-hierarchical-cache --hicache-storage-backend mooncake --disaggregation-decode-enable-radix-cache.
- Each worker allocated about 90.09 GB host memory for hierarchical KV cache per TP rank under default hicache_ratio=2.0.
- No TTFT/TPOT/SLO raw data was produced in this run because the distributed service never became ready.

Files:
- cloud100/101/102.worker.waiting_queue_full_link.log: raw worker logs.
- cloud100/101/102.status.txt: post-stop status snapshots.
- cloud099.status_attempt.txt: failed SSH attempt evidence.
- implementation.diff.patch: source diff for the minimal gated implementation.
- source_snapshot/: copies of touched source/scripts.

Rollback note:
- Runtime behavior is gated by SGLANG_PD_FLIP_HICACHE_STITCH=1 and HiCache args. Without those, the old path remains the default.
