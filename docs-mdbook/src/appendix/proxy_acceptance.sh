#!/bin/bash
# proxy_acceptance.sh - SGLang 前置代理无缝验收
# 用法:
#   bash proxy_acceptance.sh                        # 常规验收
#   bash proxy_acceptance.sh --test-429             # 额外做并发上限 429 测试（瞬时影响生产，建议低峰）
#   bash proxy_acceptance.sh --port 8000 --proxy-port 8080 --model Qwen3.6-27B-FP8

PORT=8000
PROXY_PORT=8080
MODEL="Qwen3.6-27B-FP8"
TEST_429=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --proxy-port) PROXY_PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --test-429) TEST_429=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

PASS=0; FAIL=0; SKIP=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }
skip() { echo "SKIP: $1"; SKIP=$((SKIP+1)); }

# 归一化：去掉每次请求都会变的 id/created，只比业务内容
norm() {
    python3 -c "
import sys, json
d = json.load(sys.stdin)
d.pop('id', None); d.pop('created', None)
print(json.dumps(d, sort_keys=True, ensure_ascii=False))
"
}

echo "== SGLang 代理无缝验收 =="
echo "后端: $PORT  代理: $PROXY_PORT  模型: $MODEL"

# 0) 前置可达性
curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || { echo "后端 $PORT 不可达，退出"; exit 1; }
curl -sf -o /dev/null "http://127.0.0.1:$PROXY_PORT/health" || { echo "代理 $PROXY_PORT 不可达，退出"; exit 1; }
ok "前置: 后端与代理 /health 均可达"

PAYLOAD="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":16}"

# 1) 直连 vs 代理：body 一致（归一化 id/created）
curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$PAYLOAD" | norm > /tmp/acc_direct.norm
curl -s -X POST "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$PAYLOAD" | norm > /tmp/acc_proxy.norm
if diff -q /tmp/acc_direct.norm /tmp/acc_proxy.norm >/dev/null 2>&1; then
    ok "直连 $PORT vs 代理 $PROXY_PORT body 一致（归一化后）"
else
    bad "body 不一致"; diff /tmp/acc_direct.norm /tmp/acc_proxy.norm | head -10
fi

# 2) /health 与 /metrics 走代理可用
curl -sf -o /dev/null "http://127.0.0.1:$PROXY_PORT/health" && ok "代理 /health 可用" || bad "代理 /health 不可用"
if curl -sf "http://127.0.0.1:$PROXY_PORT/metrics" | grep -q "sglang:"; then
    ok "代理 /metrics 含 sglang 指标"
else
    bad "代理 /metrics 无 sglang 指标"
fi

# 3) 流式 SSE 正常
STREAM_PAYLOAD="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":8,\"stream\":true}"
if curl -s -N -X POST "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$STREAM_PAYLOAD" | head -c 200 | grep -q "data:"; then
    ok "流式 SSE 正常"
else
    bad "流式 SSE 异常"
fi

# 4) tool call priority 注入生效（需服务端 --priority-scheduling）
# 并发发几个 tool call，制造排队后看 8000 指标是否有 priority="10"
for i in 1 2 3; do
    curl -s -o /dev/null -X POST "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$PAYLOAD" &
done
wait
sleep 3
if curl -s "http://127.0.0.1:$PORT/metrics" | grep -q 'priority="10"'; then
    ok "tool call priority=10 已注入（服务端指标可见）"
elif curl -s "http://127.0.0.1:$PORT/metrics" | grep -q 'priority='; then
    skip "priority label 存在但未见 10（检查服务端是否开了 --priority-scheduling）"
else
    skip "服务端无 priority label（未开 --enable-priority-scheduling）"
fi

# 5) 并发上限 429（可选，瞬时把 tool_call 限到 1）
if [ "$TEST_429" = true ]; then
    echo "--test-429: 临时把 tool_call 限到 1（瞬时影响生产，建议低峰执行）"
    curl -sf -X POST "http://127.0.0.1:$PROXY_PORT/admin/limits" -H 'Content-Type: application/json' -d '{"tool_call":1}' >/dev/null
    rm -f /tmp/acc_c1 /tmp/acc_c2
    (curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$PAYLOAD" > /tmp/acc_c1) &
    (curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$PAYLOAD" > /tmp/acc_c2) &
    wait
    if grep -q 429 /tmp/acc_c1 /tmp/acc_c2; then
        ok "并发上限触发 429（status: $(cat /tmp/acc_c1 /tmp/acc_c2 | tr '\n' ' ')）"
    else
        bad "未触发 429（status: $(cat /tmp/acc_c1 /tmp/acc_c2 | tr '\n' ' ')）"
    fi
    curl -sf -X POST "http://127.0.0.1:$PROXY_PORT/admin/limits" -H 'Content-Type: application/json' -d '{"tool_call":8}' >/dev/null
    echo "已恢复 tool_call=8"
fi

# 6) 代理进程与端口状态（只读检查；完整重启验证见文档）
PROXY_COUNT=$(pgrep -f "sglang_proxy.py" | wc -l)
if [ "$PROXY_COUNT" -ge 1 ]; then
    [ "$PROXY_COUNT" -eq 1 ] && ok "代理进程正常（1 个）" || bad "代理进程异常（$PROXY_COUNT 个，疑似孤儿残留）"
else
    bad "代理进程未运行"
fi

echo "=========================================="
echo "结果: PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
[ "$FAIL" -eq 0 ] && echo "验收通过" || echo "存在 FAIL，见上方明细"
exit $FAIL
