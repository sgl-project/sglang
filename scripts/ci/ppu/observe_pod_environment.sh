#!/bin/bash
# 编排侧只读观察当前作业的计算 Pod；不保存 Pod spec、环境变量或凭证。
# runner 无 Python/jq，故用白名单 JSONPath 输出 TSV，由计算侧转换为结构化证据。
# 调用：bash observe_pod_environment.sh start；所有失败均不影响测试门禁。
set -u

mode="${1:-start}"
output="${2:-/wl_nas/devops/${GITHUB_RUN_ID:-0}-${GITHUB_RUN_ATTEMPT:-1}/${PPU_ENV_JOB:-unknown}/environment-pods.tsv}"
limit="${PPU_ENV_OBSERVER_LIMIT:-21600}"
interval="${PPU_ENV_OBSERVER_INTERVAL:-10}"
expected="${PPU_ENV_EXPECTED_PODS:-1}"
status_file="${output%.tsv}.status"

if [ "$mode" = start ]; then
  # timeout 同时覆盖 kubectl 和 NAS 写入，nohup 不继承流水线日志管道。
  if command -v timeout >/dev/null 2>&1; then
    nohup timeout --kill-after=2 "$((limit + 5))" bash "$0" watch "$output" </dev/null >/dev/null 2>&1 &
  else
    echo '::warning::环境观察器未启动：缺少 timeout'
  fi
  exit 0
fi
if [ "$mode" != watch ]; then
  exit 0
fi

mkdir -p "$(dirname "$output")" 2>/dev/null || exit 0
write_status() {
  printf '%s\n' "$1" > "${status_file}.partial" 2>/dev/null &&
    mv "${status_file}.partial" "$status_file" 2>/dev/null
  return 0
}
sanitize_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g; s/^-*//; s/-*$//'
}
# 与固定 SHA 的命名契约一致；matrix 调用显式传相同 job_suffix，避免依赖自动后缀。
owner=$(sanitize_name "${GITHUB_REPOSITORY_OWNER:-unknown}" | cut -c1-20 | sed 's/-*$//')
suffix=$(sanitize_name "${PPU_ENV_JOB:-unknown}" | cut -c1-20 | sed 's/-*$//')
job=$(printf 'ppu-%s-%s-%s-%s' "$owner" "${GITHUB_RUN_ID:-0}" "${GITHUB_RUN_ATTEMPT:-1}" "$suffix" | cut -c1-52 | sed 's/-*$//')
# 字段均来自 API 身份/状态；不能改为完整 -o json，以免归档 Secret 环境变量。
query='{range .items[*]}{.metadata.name}{"\t"}{.metadata.uid}{"\t"}{.spec.nodeName}{"\t"}{.spec.containers[?(@.name=="worker")].image}{"\t"}{.status.containerStatuses[?(@.name=="worker")].imageID}{"\n"}{end}'
end=$((SECONDS + limit))
last_status=waiting
write_status waiting
while [ "$SECONDS" -lt "$end" ]; do
  kubectl_bin=$(command -v kubectl 2>/dev/null || true)
  if [ -z "$kubectl_bin" ] && [ -x /opt/bin/kubectl ]; then
    kubectl_bin=/opt/bin/kubectl
  fi
  if [ -z "$kubectl_bin" ] && [ -x "${HOME:-/nonexistent}/.local/bin/kubectl" ]; then
    kubectl_bin="${HOME}/.local/bin/kubectl"
  fi
  if [ -z "$kubectl_bin" ]; then
    last_status=kubectl_unavailable
  elif (
    set -o pipefail
    ulimit -f 128
    "$kubectl_bin" get pods -n "${PPU_ENV_NAMESPACE:-default}" -l "ppu-job=$job" \
      --request-timeout=5s -o "jsonpath=$query" |
      awk -F '\t' -v job="$job" -v expected="$expected" '
        NF == 5 && index($1, job "-worker-") == 1 {
          rank = substr($1, length(job) + 9)
          if (rank ~ /^[0-9]+$/ && rank + 0 < expected) print
        }'
  ) > "${output}.partial" 2>/dev/null; then
    # 空查询不能覆盖已经得到的实际身份；多个 Pod 全部具有 imageID 后即可停止。
    if [ -s "${output}.partial" ]; then
      mv "${output}.partial" "$output" 2>/dev/null || exit 0
      ready=$(awk -F '\t' 'NF == 5 && $5 ~ /sha256:/ && !seen[$1]++ {count++} END {print count+0}' "$output")
      total=$(awk 'END {print NR+0}' "$output")
      if [ "$ready" -eq "$expected" ] && [ "$total" -eq "$expected" ]; then
        write_status ok
        exit 0
      fi
    fi
    last_status=waiting_for_image_ids
  else
    last_status=query_failed
  fi
  write_status "$last_status"
  sleep "$interval"
done
write_status "timeout:$last_status"
exit 0
