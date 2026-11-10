#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MAX_JOBS:-}" ]]; then
  printf '%s\n' "$MAX_JOBS"
  exit 0
fi

cpu_jobs="$(nproc)"
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"

# Respect a cgroup memory limit when Docker has one. memory.max contains "max"
# for an unlimited container.
if [[ -r /sys/fs/cgroup/memory.max ]]; then
  cgroup_max="$(</sys/fs/cgroup/memory.max)"
  cgroup_used="$(</sys/fs/cgroup/memory.current)"
  if [[ "$cgroup_max" =~ ^[0-9]+$ ]] && (( cgroup_max > cgroup_used )); then
    cgroup_available_kib=$(( (cgroup_max - cgroup_used) / 1024 ))
    (( cgroup_available_kib < available_kib )) && \
      available_kib="$cgroup_available_kib"
  fi
fi

# Keep 16 GiB for the OS/linker and budget 4 GiB for each compiler process.
memory_jobs=$(( (available_kib - 16 * 1024 * 1024) / (4 * 1024 * 1024) ))
(( memory_jobs < 1 )) && memory_jobs=1
(( memory_jobs < cpu_jobs )) && cpu_jobs="$memory_jobs"
printf '%s\n' "$cpu_jobs"
