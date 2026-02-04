#!/bin/bash

# --- 脚本说明 ---
# 功能：根据指定的环境（线上/线下）向 SGLang Model Gateway 注册 Worker。
# 用法：
#   ./register_workers.sh online    # 注册线上环境的 Worker
#   ./register_workers.sh offline   # 注册线下环境的 Worker

# --- 配置区域 ---
# 在这里修改不同环境的 IP 地址和端口

# 线上环境配置 (Online)
ONLINE_GATEWAY_HOST="33.184.122.82"
ONLINE_GATEWAY_PORT="30000"
# 在这个环境中，Gateway 和 Worker 都在同一台机器上
ONLINE_REGULAR_URL1="http://${ONLINE_GATEWAY_HOST}:9201"
ONLINE_PREFILL_URL1="http://${ONLINE_GATEWAY_HOST}:9101"
ONLINE_DECODE_URL1="http://${ONLINE_GATEWAY_HOST}:9103"
ONLINE_BOOTSTRAP_PORT1=30001
ONLINE_REGULAR_URL2="http://${ONLINE_GATEWAY_HOST}:9202"
ONLINE_PREFILL_URL2="http://${ONLINE_GATEWAY_HOST}:9102"
ONLINE_DECODE_URL2="http://${ONLINE_GATEWAY_HOST}:9104"
ONLINE_BOOTSTRAP_PORT2=30002

# 线下环境配置 (Offline)
OFFLINE_GATEWAY_HOST="11.160.41.175"
OFFLINE_GATEWAY_PORT="30000"
# 在这个环境中，Worker 在另一台机器上
OFFLINE_WORKER_HOST="11.167.5.90"
OFFLINE_REGULAR_URL="http://${OFFLINE_WORKER_HOST}:9001"
OFFLINE_PREFILL_URL="http://${OFFLINE_WORKER_HOST}:9002"
OFFLINE_DECODE_URL="http://${OFFLINE_WORKER_HOST}:9003"
OFFLINE_BOOTSTRAP_PORT=30001


# 模型 ID (通用)
MODEL_ID="Qwen3/32B"

# --- 函数定义 ---

# 注册 Worker 的函数
# 参数:
# 1. Gateway 地址 (e.g., http://host:port)
# 2. Worker URL
# 3. Worker 类型 (regular, prefill, decode)
# 4. Bootstrap 端口 (可选, 仅 prefill 需要)
register_worker() {
    local gateway_address=$1
    local worker_url=$2
    local worker_type=$3
    local bootstrap_port=$4

    echo "--------------------------------------------------"
    echo "Registering ${worker_type} worker:"
    echo "  Gateway: ${gateway_address}"
    echo "  Worker URL: ${worker_url}"

    # 构建 JSON payload
    local payload
    if [[ "$worker_type" == "prefill" && -n "$bootstrap_port" ]]; then
        # Prefill worker 需要 bootstrap_port
        payload=$(cat <<EOF
{
  "url": "${worker_url}",
  "worker_type": "${worker_type}",
  "model_id": "${MODEL_ID}",
  "bootstrap_port": ${bootstrap_port}
}
EOF
)
    else
        # Regular 和 Decode worker
        payload=$(cat <<EOF
{
  "url": "${worker_url}",
  "worker_type": "${worker_type}",
  "model_id": "${MODEL_ID}"
}
EOF
)
    fi

    # 发送 curl 请求
    curl -X POST "${gateway_address}/workers" \
      -H "Content-Type: application/json" \
      -d "${payload}" \
      --silent --show-error --fail # 增加一些 curl 参数使其更健壮

    # 检查 curl 的退出码
    if [ $? -eq 0 ]; then
        echo -e "\n✅ Registration command for ${worker_type} sent successfully."
    else
        echo -e "\n❌ ERROR: Failed to send registration command for ${worker_type}."
    fi
    echo "--------------------------------------------------"
    # 在两次请求之间稍作停顿，给网关处理时间
    sleep 1
}

# --- 主逻辑 ---

# 检查是否提供了环境参数
if [ -z "$1" ]; then
    echo "错误: 请提供环境参数."
    echo "用法: $0 [online|offline]"
    exit 1
fi

# 获取环境参数并转换为小写
environment=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# 根据环境参数选择配置并执行
case $environment in
    online)
        echo "🚀 Starting registration for ONLINE environment..."
        gateway="http://${ONLINE_GATEWAY_HOST}:${ONLINE_GATEWAY_PORT}"
        register_worker "${gateway}" "${ONLINE_REGULAR_URL1}" "regular"
        register_worker "${gateway}" "${ONLINE_PREFILL_URL1}" "prefill" "${ONLINE_BOOTSTRAP_PORT1}"
        register_worker "${gateway}" "${ONLINE_DECODE_URL1}" "decode"
        register_worker "${gateway}" "${ONLINE_REGULAR_URL2}" "regular"
        register_worker "${gateway}" "${ONLINE_PREFILL_URL2}" "prefill" "${ONLINE_BOOTSTRAP_PORT2}"
        register_worker "${gateway}" "${ONLINE_DECODE_URL2}" "decode"
        echo "🎉 ONLINE environment registration finished."
        ;;

    offline)
        echo "🚀 Starting registration for OFFLINE environment..."
        gateway="http://${OFFLINE_GATEWAY_HOST}:${OFFLINE_GATEWAY_PORT}"
        register_worker "${gateway}" "${OFFLINE_REGULAR_URL}" "regular"
        register_worker "${gateway}" "${OFFLINE_PREFILL_URL}" "prefill" "${OFFLINE_BOOTSTRAP_PORT}"
        register_worker "${gateway}" "${OFFLINE_DECODE_URL}" "decode"
        echo "🎉 OFFLINE environment registration finished."
        ;;

    *)
        echo "错误: 无效的环境参数 '$1'."
        echo "请使用 'online' 或 'offline'."
        exit 1
        ;;
esac
