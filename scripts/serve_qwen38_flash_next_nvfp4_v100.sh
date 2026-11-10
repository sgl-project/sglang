#!/usr/bin/env bash
# Serve RadixArk/Qwen3.8-Flash-Next-NVFP4 (Qwen4-arch, NVFP4 W4A16, E5M2 KV)
# on a 4x V100 NVLink domain. No conda, no docker — plain venv.
#
# Before the FIRST serve, run the (GPU-touching) validation once:
#   bash scripts/smoke_v100.sh
#
# Usage (from the repo root):
#   bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh           # target-only
#   bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp       # + built-in MTP-3/4
#
# Env overrides:
#   FLASH_NEXT_MODEL=/path/to/Qwen3.8-Flash-Next-NVFP4   (required)
#   SGLANG_V100_VENV=/path/to/venv                       (default: $HOME/sglang-v100-venv)
#   FLASH_NEXT_GPUS=0,1,2,3    (one 4-GPU NVLink domain)
#   SGLANG_V100_HOST=0.0.0.0          (shared with DSV41)
#   SGLANG_V100_PORT=11435            (shared with DSV41; not 30000)
#   FLASH_NEXT_HOST / FLASH_NEXT_PORT  (legacy aliases for the same pair)
#   FLASH_NEXT_DP=1            (data-parallel replicas; needs 4 GPUs each)
set -euo pipefail

MODE="${1:-target}"
case "$MODE" in target|mtp) ;; *) echo "mode must be 'target' or 'mtp'" >&2; exit 1;; esac

# Override with SGLANG_V100_VENV; defaults to a sibling venv or an already
# activated environment.
VENV="${SGLANG_V100_VENV:-$HOME/sglang-v100-venv}"
[[ -x "$VENV/bin/python" ]] || VENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-$VENV}}"
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV; set SGLANG_V100_VENV" >&2; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${FLASH_NEXT_MODEL:-}"
[[ -n "$MODEL" ]] || { echo "set FLASH_NEXT_MODEL to the Qwen3.8-Flash-Next-NVFP4 checkout" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${FLASH_NEXT_GPUS:-0,1,2,3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# The venv's bin must be on PATH, not just its python: the SM70 JIT kernels
# shell out to `ninja`, which is a venv-local binary with no system copy.
# Production gets away without it only because its JIT cache is warm.
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
# Runtime JIT kernels shell out to a bare `nvcc`, which defaults to the host
# `gcc` -- version 15 here, which ships no cc1plus (and CUDA 12.9 caps host
# support at GCC 14 anyway). Without this the SM70 JIT modules and the
# custom all-reduce fail to build, the latter silently falling back to NCCL.
# CUDA 12.x caps host-compiler support at GCC 14, and some distros default
# `gcc` to 15 -- sometimes without a cc1plus at all, so every JIT compile
# dies. Pick the newest complete GCC <= 14. NVCC_PREPEND_FLAGS is needed
# separately from CUDAHOSTCXX: the runtime JIT invokes a bare nvcc, which
# reads the former and not the latter.
for _v in 14 13 12; do
  if [[ -x "/usr/bin/g++-$_v" ]] && ls /usr/libexec/gcc/*/$_v/cc1plus >/dev/null 2>&1; then
    export CC="/usr/bin/gcc-$_v" CXX="/usr/bin/g++-$_v" CUDAHOSTCXX="/usr/bin/g++-$_v"
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-$_v"
    break
  fi
done
export TORCH_CUDA_ARCH_LIST=7.0
# V100 runtime env (same set as the README's Flash-Next commands)
export FLASHINFER_DISABLE_VERSION_CHECK=1
# NCCL_P2P_LEVEL stays unset: NCCL picks NVLink where it exists and PCIe P2P
# behind a PCIe switch; forcing NVL routes PCIe-only cards through host memory.
export SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
# 4x V100 PCIe-only (no NVLink, P2P via one PLX): SGLANG_CUSTOM_AR_ALLOW_PCIE=1
# (default off; one-shot push, 128 KiB cap). Do not set it on an NVLink mesh.
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float16
export SGLANG_SM70_FORCE_FP16=1
# Measured batch-one Volta projections and Qwen fusions. Set either to 0
# before launch to select the previous implementation for comparisons.
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_SM70_QWEN_FUSIONS=1
export SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_NUMA_BIND_V2=0
# V100 runtime pin is sglang-kernel 0.4.6.post1, not upstream 0.4.7.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export PYTHONPATH="$REPO/python"
# hicache_storage.py defaults to /tmp/hicache. On this box /tmp is tmpfs, so an
# unset dir is a RAM allocation. Keep the file tier on disk.
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-$HOME/hicache_storage}"
# Without the metadata cache, every disk-tier lookup lists the whole storage
# directory, so requests wait longer as it fills: a 600-token tool result
# waited 92-119 ms before prefill at 50-68k files, and 20 ms with the cache.
# The cache only remembers files that exist, and eviction drops them from it.
# A file removed behind its back ends that prefetch early (the rest is
# recomputed); it never yields wrong data.
export SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE="${SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE:-1}"

# Docker-v2 tested sizing (README: 4 live requests, full 262K context)
args=(
  --trust-remote-code
  --model-path "$MODEL"
  --served-model-name qwen38next-nvfp4
  --dtype float16
  --quantization modelopt_fp4
  --reasoning-parser auto
  --tool-call-parser auto
  --attention-backend tilelang_fa_v100
  --linear-attn-prefill-backend tilelang
  --linear-attn-decode-backend triton
  # FP16 KV is the decision of record: on this sparse-attention model a fast
  # sm70 decode kernel reads only the selected top-k K/V, so fp16's 2 bytes/
  # element costs no decode speed and gives higher MTP acceptance than the
  # 1-byte e5m2 KV. `auto` resolves to the model's float16 dtype on sm70.
  # To serve the 1-byte e5m2 KV instead (roughly doubles the pool token count)
  # set FLASH_NEXT_EXTRA_ARGS='--kv-cache-dtype fp8_e5m2' AND lower
  # --mem-fraction-static, since the larger pool leaves less prefill headroom.
  --kv-cache-dtype auto
  --tensor-parallel-size 4
  # Data-parallel replicas (default 1 = the current single TP4 runner on GPUs
  # 0-3). With FLASH_NEXT_DP=2 and FLASH_NEXT_GPUS=0..7: replica 0 -> GPUs 0-3,
  # replica 1 -> GPUs 4-7, both under ONE front-end (built-in load balancing).
  # dp-size 1 is the no-op default, so this is inert unless FLASH_NEXT_DP>1.
  --dp-size "${FLASH_NEXT_DP:-1}"
  --host "${SGLANG_V100_HOST:-${FLASH_NEXT_HOST:-0.0.0.0}}"
  --port "${SGLANG_V100_PORT:-${FLASH_NEXT_PORT:-11435}}"
  # 0.86 (was 0.88): fp16 QSA KV is the decision of record (fast decode kernel).
  # The 2026-09-11 reliability hammer OOM-crashed the engine at 0.88 under
  # beyond-spec load (32k-token contexts, np=8): free device memory fell to
  # ~0.6 GiB, the prefill activation did not fit, every TP rank raised
  # "Prefill out of memory", the scheduler died, and systemd's auto-restart hung
  # on the crashed GPU state. 0.86 gives the transient prefill/mamba path more
  # headroom: available_gpu_mem 3.89 -> 4.54 GB. The cost is the KV pool,
  # 402912 -> 352864 tokens (-12%); a 0.02 fraction moves ~0.64 GB because the
  # pool is a small slice of the 32 GiB. QSA's sparsity makes each extra pool
  # token cheap to serve (decode attention is O(top-k), not O(context)), and
  # 352k tokens still holds many long agentic contexts at max_running_requests=3,
  # so the pool shrink does not hurt typical use. The same 32k x np8 stress ran
  # 5 min clean at 0.86 (0 OOM, no restart) vs the 0.88 crash at ~2 min. The
  # "max KV" for multi-agent still comes from the host+disk tiers below.
  --mem-fraction-static "${FLASH_NEXT_MEM_FRACTION:-0.86}"
  --context-length 262144
  # 3, not 4: measured on this hardware, aggregate decode PEAKS at three
  # concurrent requests (116.7 tok/s) and falls at four (108.8) while TTFT
  # jumps 4x, 656 ms -> 2.5 s. The fourth slot buys no throughput and costs
  # every stream its latency. Per-stream decode is 67/53/43/38 tok/s at
  # np=1/2/3/4. Measure with realistic text before changing this: a
  # random-token benchmark reports aggregate still climbing at four, because
  # random continuations are degenerate and the MTP draft model predicts them
  # almost perfectly (accept 3.98 of 4, against 2.6-2.7 on real text).
  --max-running-requests 3
  --max-mamba-cache-size 20
  --chunked-prefill-size 4096
  # Cap total prefill tokens per forward pass at 4096 (was 8192/16384). The
  # GDN linear-attention prefill buffer grows with the chunk; at the ~2.5G
  # headroom left after the KV pool + hierarchical-cache GPU overhead, an 8k
  # chunk OOMed (134 MiB alloc failed). 4096 halves that buffer so it fits;
  # long prefills still work, just in 4k chunks. This is the OOM safety lever.
  --max-prefill-tokens 4096
  # Multi-tier (hierarchical) KV cache: GPU (above) + host RAM + unbounded disk
  # spill (file backend -> SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR). Shared
  # prefixes across agents/requests become a host/disk hit instead of a full
  # recompute. Mamba-aware via HiMambaRadixCache (host KV + host mamba pools).
  # NOTE: --hicache-size N is PER-RANK and applies to BOTH the host KV pool AND
  # the host Mamba pool, so total = N * 2 * 4 ranks = 8N GB. N=16 -> 128G total
  # left only ~2G free (thrash-prone); N=8 -> 64G total, ~60G+ headroom. Safe
  # ONLY with SGLANG_NUMA_BIND_V2=0 (host memory interleaves across both nodes;
  # with the default node1 membind this swap-stormed). Disk tier = catch-all.
  --enable-hierarchical-cache
  # Host-tier pool size is PER-RANK (GB): total = N * 2 pools (KV+Mamba) * 4
  # ranks * dp replicas. SGLang asserts the host pool > the device (GPU KV)
  # pool in TOKENS (memory_pool_host.py:254/:1302). The device KV pool is
  # 694k tokens/rank; N=5 clears it by only ~23% (the secondary KV pool in
  # build_hybrid_mamba_stack) and measured RACY (asserted once, passed on
  # retry -> latent crash-loop), so default to the proven-good N=8 (~2x
  # margin, the single-replica value) -> no flap.
  #
  # FILE TIER WRITE POLICY = write_back (2026-08-31, after reboot): the
  # default write_through was measured catastrophically wrong for this
  # storage. Cold array reads are healthy (4x Intenso SATA SSDs behind the
  # 3108: SMART clean, 2.2-3.2 GB/s quiescent, BMC no degraded/rebuilding)
  # but collapse to 20-50 MB/s while the file tier's write_through dirties
  # page cache (45k-token prefill: concurrent cold dd 30 MB/s; 678 MB/s 5s
  # after; writeback storm drains for minutes -- DRAM-less consumer SSDs
  # eat the SLC cache). That is what made the disk READBACK path measure
  # 344 tok/s (3 tok/s at 55k tokens) -- it was reading from a
  # write-saturated array. write_back (hi_mamba_radix_cache.py): no
  # device->host->disk writes on hits; disk writes ONLY on tier eviction
  # (memory pressure) and shutdown flush. Steady state: zero disk writes;
  # the 128Gi host tier holds hot sessions (a pinned 230k session ~35GB
  # fits one replica's 64Gi pool). Disk tier keeps its designed job:
  # cross-replica sharing + restart durability. Do NOT switch back to
  # write_through on this box without solving the writeback QoS collapse.
  # (write_through_selective is NOT an alternative: agentic sessions
  # re-hit the same prefix every turn, so the hit_count>=2 threshold fires
  # within a turn -> effectively full write volume.)
  #
  # Host-tier pool size is PER-RANK (GB): total = N * 2 pools (KV+Mamba) * 4
  # ranks * dp replicas. SGLang asserts the host pool > the device (GPU KV)
  # pool in TOKENS (memory_pool_host.py:254/:1302). The device KV pool is
  # 694k tokens/rank; N=5 clears it by only ~23% (the secondary KV pool in
  # build_hybrid_mamba_stack) and measured RACY (asserted once, passed on
  # retry -> latent crash-loop), so default to the proven-good N=8 (~2x
  # margin, the single-replica value) -> no flap. Unbounded prefix reuse still
  # comes from the SHARED disk tier (content-addressed, identical across
  # replicas, write_back since 2026-08-31), so the per-replica host mirror
  # stays thin.
  --hicache-size "${FLASH_NEXT_HICACHE:-8}"
  # Upstream flipped this default layer_first -> page_first (b5bcd76a4 /
  # #21631). Pinned explicitly so a future flip cannot move it silently --
  # but it must be page_first: MambaPoolHost accepts only page_first /
  # page_first_direct (pool_host/mamba.py), so layer_first is not an option
  # for a mamba model with the hierarchical cache at all.
  # NOTE: the --hicache-size 8 sizing above was measured under layer_first;
  # re-validate host-tier memory once this run is stable.
  --hicache-mem-layout page_first
  --hicache-storage-backend file
  --hicache-write-policy write_back
  # Idle-sleep (2026-08-31): without this, each TP rank's scheduler loop
  # busy-spins at ~100% CPU when idle -- every idle iteration calls
  # check_hicache_events() -> drain_storage_control_queues(), which runs an
  # NCCL all_reduce (tp-group) even when all storage queues are empty, so the
  # loop iterates at the all_reduce round-trip rate and pins a core per rank
  # (plus the NCCL proxy threads: ~7.5 cores total with DP=2, GPUs at 0%).
  # With the flag, rank 0 of each replica blocks on a zmq poller and the
  # other ranks sleep at the all_reduce barrier; a new request wakes the
  # poller immediately, so no TTFT penalty. (Upstream IdleSleeper exists for
  # exactly this: "each GPU would otherwise pin one thread at 100% CPU".)
  --sleep-on-idle
  # Track --max-running-requests above: with three admitted requests the decode
  # batch is only ever 1-3, so capturing a bs=4 graph costs capture time and
  # memory for a shape that can no longer occur.
  --cuda-graph-max-bs-decode 3
  --cuda-graph-bs-decode 1 2 3
  # Upstream turns breakable prefill graphs on for this model (#41729). Its
  # capture (target 3.8 GB, then the draft) does not fit next to the KV pool
  # that --mem-fraction-static above was tuned for: the draft capture ran out
  # of memory at 4096 tokens. Off, as before the 10-05 sync, until the TTFT
  # gain is measured against a smaller pool.
  --disable-prefill-cuda-graph
  --mamba-radix-cache-strategy extra_buffer
  --mamba-full-memory-ratio 0.2
  # Report prefix-cache hits as usage.prompt_tokens_details.cached_tokens
  # (OpenAI) / usage.cache_read_input_tokens (Anthropic /v1/messages). Off by
  # default, so without this Claude Code's cost readout shows 0 cache reads
  # even though the hierarchical cache is serving them. Pure reporting: the
  # cached_tokens count is already tracked in meta_info regardless of this
  # flag; it only gates surfacing it. (Upstream sgl-project/sglang#29703.)
  --enable-cache-report
  # Prometheus /metrics. Staged 2026-09-01; activates on the NEXT restart
  # (operator triggers it -- a live session may be running on this engine, so
  # do not auto-restart to pick it up). Surfaces what the Decode-batch log
  # lines already print (accept len/rate, gen throughput, queue-req, full-token
  # usage) as continuous histograms, PLUS prefix-cache hit/miss counters the
  # log does not expose. Low overhead. NOTE: the one-off accept-length question
  # is already answered from logs (mean 2.47 over 1015 decode samples, never
  # net-negative, throughput rises monotonically 47->74 tok/s with it), so
  # metrics are for ONGOING cache-hit-rate + queue-depth watching, not a single
  # read. Deliberately NOT adding --enable-metrics-for-all-schedulers: per-rank
  # instrumentation adds per-step overhead and we run DP1 (one replica).
  --enable-metrics
  # Explicit on V100 (fp16): the 51 GB PLE n-gram table must sit in host
  # memory. Without offload it is created on GPU and OOMs at ~31.5 GiB/rank
  # during create_weights.
  --ple-offload-embedding
  # Builds FlashInfer's sampling kernels before serving (JIT, ~90 s when cold).
  --warmups sampling
)
if [[ "$MODE" == mtp ]]; then
  # Built-in MTP-3/4 loads the MTP module from the same checkpoint.
  args+=(
    --speculative-algorithm EAGLE
    --speculative-draft-model-path "$MODEL"
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

# Extra launch args appended at the end (space-separated); empty by default.
# Appended last, so they override the defaults above (last-wins). Use it to
# serve the 1-byte e5m2 KV instead of the fp16 default:
#   FLASH_NEXT_EXTRA_ARGS='--kv-cache-dtype fp8_e5m2'
# (pair with a lower --mem-fraction-static; see the --kv-cache-dtype comment).
if [[ -n "${FLASH_NEXT_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"$FLASH_NEXT_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

cd "$REPO"
exec "$VENV/bin/python" -m sglang.launch_server "${args[@]}"
