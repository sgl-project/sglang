#!/bin/bash
# Runner-performance snapshot for the AMD CI.
#
# Background: in https://github.com/sgl-project/sglang/actions/runs/26452774262
# every MI300 job timed out (sgl-kernel 14m, stage-a 30m, stage-b 45-60m, ...)
# while the matching scheduled MI325 run
# (https://github.com/sgl-project/sglang/actions/runs/26468467018) on the same
# commit / image finished stage-a in 11m36s and most stage-b shards in <40m.
# Same workflow, same docker image, different runner pool, very different
# wall-clock -- so we need host-level evidence (network, disk, CPU, GPU clocks)
# captured on BOTH pools to triage where the gap actually is.
#
# This script prints a compact, grep-friendly snapshot before any test step.
# It is wired into ensure_vram_clear.sh so every AMD job emits it without
# editing 20+ workflow YAML steps. Every section is best-effort and bounded
# by `timeout`; the script never fails the CI step.

set +e

# CONTEXT distinguishes whether this snapshot is from the host (default,
# invoked from ensure_vram_clear.sh before any container exists) or from
# inside the CI container (invoked from amd_ci_start_container.sh via
# `docker exec -e CONTEXT=container ci_sglang ...`). Same script, two
# vantage points, so we can diff bridge-network / mount / TCP-stack
# behaviour without duplicating the metric set.
CONTEXT="${CONTEXT:-host}"
DIAG_TAG="[runner-perf-diag:${CONTEXT}]"

section() {
    printf '\n%s === %s ===\n' "$DIAG_TAG" "$*"
}

kv() {
    printf '%s %s=%s\n' "$DIAG_TAG" "$1" "$2"
}

section "identity"
kv hostname "$(hostname 2>/dev/null || echo unknown)"
kv kernel "$(uname -rsm 2>/dev/null || echo unknown)"
kv date_utc "$(date -u +%FT%TZ)"
kv runner_name "${RUNNER_NAME:-unset}"
kv runner_arch "${RUNNER_ARCH:-unset}"
kv github_job "${GITHUB_JOB:-unset}"
kv github_run_id "${GITHUB_RUN_ID:-unset}"

section "cpu"
LANG=C lscpu 2>/dev/null \
    | awk -F: '/^Model name|^CPU\(s\)|^Thread.*per core|^Core.*per socket|^Socket|^CPU MHz|^CPU max MHz|^BogoMIPS/ {gsub(/^ +| +$/, "", $2); printf "%s %s=%s\n", "'"$DIAG_TAG"'", $1, $2}' \
    || true
kv loadavg "$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown)"

section "memory"
# Fall back to /proc/meminfo when `free` is not installed (some slim
# container images strip procps-ng).
if command -v free >/dev/null 2>&1; then
    free -m 2>/dev/null \
        | awk 'NR==2 {printf "%s mem_total_mb=%s mem_used_mb=%s mem_free_mb=%s mem_available_mb=%s\n", "'"$DIAG_TAG"'", $2, $3, $4, $7}' \
        || true
elif [ -r /proc/meminfo ]; then
    awk -v tag="$DIAG_TAG" '
        /^MemTotal:/     {total=$2}
        /^MemFree:/      {free=$2}
        /^MemAvailable:/ {avail=$2}
        END {printf "%s mem_total_mb=%d mem_free_mb=%d mem_available_mb=%d\n", tag, total/1024, free/1024, avail/1024}
    ' /proc/meminfo
fi

section "disk capacity"
for mp in /home/runner /home/runner/sglang-data /tmp /var/lib/docker; do
    [ -e "$mp" ] || continue
    df -BM --output=target,size,used,avail,pcent "$mp" 2>/dev/null \
        | awk -v tag="$DIAG_TAG" 'NR==2 {printf "%s df target=%s size=%s used=%s avail=%s use=%s\n", tag, $1, $2, $3, $4, $5}'
done

section "disk write throughput (256 MiB, fsync)"
# Two locations because /home/runner/sglang-data is the bind-mount that holds
# the HF cache / pip cache / model weights (the hot path for AMD CI), and /tmp
# is the baseline (usually tmpfs/local-ssd). If they diverge, the bind-mount's
# backing store is the bottleneck.
for target_dir in /home/runner/sglang-data /tmp; do
    [ -d "$target_dir" ] || { kv "dd_$target_dir" "skip (missing)"; continue; }
    [ -w "$target_dir" ] || { kv "dd_$target_dir" "skip (not writable by $(id -un))"; continue; }
    tmpfile="$target_dir/_runner_perf_diag.tmp"
    result=$(timeout 25 dd if=/dev/zero of="$tmpfile" bs=1M count=256 conv=fsync 2>&1 | tail -1)
    rm -f "$tmpfile" 2>/dev/null
    kv "dd_${target_dir}" "$result"
done

section "network latency (3 probes)"
for host in registry-1.docker.io 10.245.143.50 huggingface.co cdn-lfs.huggingface.co pypi.org github.com objects.githubusercontent.com; do
    if out=$(timeout 6 ping -c 3 -W 1 -q "$host" 2>/dev/null); then
        # ping output line shapes:
        #   rtt min/avg/max/mdev = 36.018/36.425/36.515/0.213 ms
        #   3 packets transmitted, 3 received, 0% packet loss, time 2003ms
        rtt=$(printf '%s\n' "$out" | sed -nE 's|^(rtt\|round-trip) [^=]* = ([^ ]+) ms.*|\2|p')
        loss=$(printf '%s' "$out" | grep -oE '[0-9.]+% packet loss' | head -n1)
        kv "ping_$host" "rtt_ms=${rtt:-none} ${loss:-no_loss}"
    else
        kv "ping_$host" "unreachable"
    fi
done

section "network http (curl, small endpoints)"
# Use HEAD-ish requests so we measure handshake/TTFB without pulling MB. The
# Docker Hub manifest endpoint and PyPI simple index are exactly what the
# install step hits later; failing them here predicts failures there. The
# 10.245.143.50:5000 entry probes the AMD CI's LAN docker registry via TCP
# (ping is ICMP-blocked on both pools, masking the fact that the registry
# IS reachable from MI300 — confirmed in sgl-project/sglang#26260).
fmt='%{http_code} dnstime=%{time_namelookup}s connect=%{time_connect}s tls=%{time_appconnect}s ttfb=%{time_starttransfer}s total=%{time_total}s\n'
for url in \
    "https://registry-1.docker.io/v2/" \
    "https://auth.docker.io/token?service=registry.docker.io" \
    "http://10.245.143.50:5000/v2/" \
    "https://pypi.org/simple/pip/" \
    "https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct/tree/main" \
    "https://github.com/"; do
    out=$(timeout 15 curl -sS -o /dev/null -w "$fmt" "$url" 2>&1 || echo "curl_err")
    kv "http_${url}" "$out"
done

section "tcp tunables"
# The MI300 vs MI325 host gap is dominated by these (see #26260): MI325
# uses bbr + 16 MiB rmem/wmem, MI300 ships stock cubic + 6 MiB/4 MiB
# rmem/wmem, which caps single-flow throughput to high-RTT CDNs (HF /
# Docker Hub / S3) at ~rmem_max/RTT and is exactly the pattern visible in
# the http_sample numbers above.
for k in \
    net.ipv4.tcp_congestion_control \
    net.core.default_qdisc \
    net.core.rmem_max \
    net.core.wmem_max \
    net.ipv4.tcp_rmem \
    net.ipv4.tcp_wmem \
    net.ipv4.tcp_window_scaling \
    net.ipv4.tcp_mtu_probing \
    net.ipv4.tcp_timestamps; do
    # Prefer reading /proc/sys directly so this works in container images
    # that strip the `sysctl` binary (procps-ng), and skip the binary
    # fork-exec when we can.
    proc_path="/proc/sys/$(echo "$k" | tr . /)"
    if [ -r "$proc_path" ]; then
        v=$(cat "$proc_path" 2>/dev/null)
    elif command -v sysctl >/dev/null 2>&1; then
        v=$(sysctl -n "$k" 2>/dev/null || echo unavailable)
    else
        v=unavailable
    fi
    # Collapse tabs/multiple spaces so the kv= block stays single-line.
    v=$(printf '%s' "$v" | tr '\t' ' ' | tr -s ' ')
    kv "$k" "$v"
done

section "network http throughput (~10 MiB sample, follows redirects)"
# Public small artifact on GitHub releases CDN; representative of HF LFS path
# the test step uses to fetch model snapshots. -L follows the 302 to the
# CDN so we measure real bytes/sec, not redirect latency.
sample_url="https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz"
sample_out=$(timeout 30 curl -sSL -o /dev/null \
    -w "http=%{http_code} ttfb=%{time_starttransfer}s total=%{time_total}s size=%{size_download}B speed_Bps=%{speed_download}\n" \
    "$sample_url" 2>&1 || echo "curl_err")
kv "http_sample" "$sample_out"

section "model download throughput (HF LFS, 1 GiB range, time-capped)"
# Sustained large-transfer probe. The ~10 MiB sample above is too small to
# expose the single-flow high-BDP throughput ceiling (cubic vs bbr) that
# dominates multi-GB safetensors loads: DeepSeek-V3 load_weight measured
# ~3650s on this MI300 pool vs ~646s on MI325 (~5.6x). This pulls a range
# from the same public DeepSeek-V3-0324 shard the stage-c suite loads, so
# the number is directly comparable to the real model-fetch path. Range-
# capped to 1 GiB and curl --max-time capped so it can never run long;
# speed_Bps is the sustained bytes/sec regardless of whether the cap is hit.
model_url="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/resolve/main/model-00001-of-000163.safetensors"
model_out=$(timeout 150 curl -sSL --max-time 120 -r 0-1073741823 -o /dev/null \
    -w "http=%{http_code} ttfb=%{time_starttransfer}s total=%{time_total}s size=%{size_download}B speed_Bps=%{speed_download}\n" \
    "$model_url" 2>&1) || true
[ -z "$model_out" ] && model_out="curl_err"
kv "model_download_sample" "$model_out"

section "rocm gpu state"
# Clocks + thermal + perf level. If MI300 GPUs are clock-locked or thermally
# throttled (vs MI325), this will show it directly. -1 timeout safety because
# rocm-smi can hang when the driver is unhealthy.
if command -v rocm-smi >/dev/null 2>&1; then
    timeout 15 rocm-smi --showclocks --showtemp --showpower --showperflevel 2>&1 \
        | sed "s/^/$DIAG_TAG  /" || kv rocm_smi "timed out"
else
    kv rocm_smi "not installed"
fi

section "docker"
if command -v docker >/dev/null 2>&1; then
    docker info --format 'server={{.ServerVersion}} driver={{.Driver}} root={{.DockerRootDir}} containers={{.Containers}} images={{.Images}} cpus={{.NCPU}} mem_mb={{div .MemTotal 1048576}}' 2>/dev/null \
        | awk -v tag="$DIAG_TAG" '{print tag" docker_info "$0}' || true
fi

section "hf cache"
# On the host the HF cache lives at /home/runner/sglang-data/hf-cache; inside
# the container the same backing store is bind-mounted at /sgl-data/hf-cache
# (see -e HF_HOME=/sgl-data/hf-cache in amd_ci_start_container.sh). Probe
# both so the host snapshot tells us if the runner has the bind-mount
# populated, and the container snapshot tells us if the mount actually
# made it inside (PR #26260 saw it MISSING in-container — that's a
# separate failure mode from the TCP-tuning gap).
if [ "$CONTEXT" = "container" ]; then
    HF_CACHE_PATH=/sgl-data/hf-cache
else
    HF_CACHE_PATH=/home/runner/sglang-data/hf-cache
fi
if [ -d "$HF_CACHE_PATH" ]; then
    # `du` over a multi-TB HF cache is expensive; use --max-depth=0 and bound
    # with timeout so a slow disk doesn't blow the diagnostic budget.
    size=$(timeout 15 du -sBM --max-depth=0 "$HF_CACHE_PATH" 2>/dev/null | awk '{print $1}')
    kv hf_cache_path "$HF_CACHE_PATH"
    kv hf_cache_size "${size:-timeout}"
else
    kv hf_cache_path "$HF_CACHE_PATH (missing)"
fi
# Also check the /sgl-data bind-mount root from inside the container so we
# notice if amd_ci_start_container.sh's CACHE_VOLUME logic dropped it
# (which #26260 observed: "WARNING: /sgl-data does NOT exist inside
# container ⇒ HF / MIOPEN cache paths will fail or be created in
# container layer").
if [ "$CONTEXT" = "container" ]; then
    if [ -d /sgl-data ]; then
        kv sgl_data_mount "present"
    else
        kv sgl_data_mount "MISSING (cache writes will hit container layer, not bind-mount)"
    fi
fi

printf '\n%s === diagnostics done ===\n' "$DIAG_TAG"
exit 0
