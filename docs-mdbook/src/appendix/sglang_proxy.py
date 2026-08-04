#!/usr/bin/env python3
"""SGLang 前置代理：按请求类型限并发 + 注入 priority。

配合 sglang_start.sh --proxy-port 使用：
客户端连本代理端口，代理限并发/注入 priority 后转发到本机 SGLang。
"""

import argparse
import asyncio

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()

LIMITS = {"tool_call": 8, "thinking": 12}
sems = {k: asyncio.Semaphore(v) for k, v in LIMITS.items()}
BACKEND = "http://127.0.0.1:8000"
TIMEOUT = 10.0


def classify(body: dict) -> str:
    # 方法1：按 chat_template_kwargs 区分
    if body.get("chat_template_kwargs", {}).get("enable_thinking") is False:
        return "tool_call"
    # 方法2：带 tools 字段也是 tool call
    if body.get("tools"):
        return "tool_call"
    # 注意：不带 enable_thinking/tools 的请求默认归入 thinking 桶（12 并发、无 priority）
    return "thinking"


async def _forward(
    kind: str,
    path: str,
    body: dict,
    default_media: str,
    too_many_body: dict,
    headers: dict | None = None,
):
    """限并发 + 转发：超时 429、后端不可达 502（释放槽位，不泄漏）、上游错误原样透传。
    请求头（含 Authorization）与响应头均透传，不丢东西。"""
    try:
        await asyncio.wait_for(sems[kind].acquire(), TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(too_many_body, status_code=429)

    fwd_headers = (
        {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("host", "content-length", "connection")
        }
        if headers
        else None
    )
    client = httpx.AsyncClient(timeout=None)
    try:
        req = client.build_request("POST", BACKEND + path, json=body, headers=fwd_headers)
        resp = await client.send(req, stream=True)
    except Exception as e:  # noqa: BLE001 后端不可达：释放槽位，避免泄漏导致全量 429
        sems[kind].release()
        await client.aclose()
        return JSONResponse({"error": "backend unreachable", "detail": str(e)}, status_code=502)

    resp_headers = {
        k: v
        for k, v in resp.headers.items()
        if k.lower() not in ("content-type", "content-length", "transfer-encoding", "connection")
    }
    if resp.status_code >= 400:
        sems[kind].release()
        err = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return Response(
            content=err,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type") or "application/json",
            headers=resp_headers,
        )

    async def stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            sems[kind].release()
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        media_type=resp.headers.get("content-type") or default_media,
        headers=resp_headers,
    )


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    kind = classify(body)
    forwarded = dict(body)
    if kind == "tool_call":
        forwarded["priority"] = 10  # 注入 priority，配合服务端 --priority-scheduling
    return await _forward(
        kind,
        "/v1/chat/completions",
        forwarded,
        "text/event-stream",
        {"error": "too many requests", "type": "concurrency_limit"},
        headers=request.headers,
    )


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic 风格接口：按 thinking 字段分类限并发。
    注意：Anthropic 协议无 priority 字段，无法注入（SGLang anthropic 入口会拒绝未知字段）。"""
    body = await request.json()
    kind = "thinking" if body.get("thinking", {}).get("type") == "enabled" else "tool_call"
    return await _forward(
        kind,
        "/v1/messages",
        body,
        "application/json",
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "too many requests"},
        },
        headers=request.headers,
    )


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(BACKEND + "/v1/models")
        return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/admin/limits")
async def set_limits(payload: dict):
    """运行时限并发调整，无需重启。示例:
    curl -X POST http://127.0.0.1:8080/admin/limits \\
         -H 'Content-Type: application/json' \\
         -d '{"tool_call": 12, "thinking": 16}'
    """
    updated = {}
    for k, v in payload.items():
        if k in LIMITS:
            v = int(v)
            if v > 0:
                LIMITS[k] = v
                sems[k] = asyncio.Semaphore(v)  # 重建信号量，在途请求不受影响
                updated[k] = v
    return {"limits": LIMITS, "updated": updated}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, request: Request):
    """兜底透传：/health /metrics /v1/completions /v1/embeddings 等其它接口，
    不参与限并发/priority，原样转发给 SGLang。"""
    target = f"{BACKEND}/{path}"
    if request.url.query:
        target += "?" + request.url.query
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "connection")
    }
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(request.method, target, content=body, headers=headers)
    resp = await client.send(req, stream=True)

    async def stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        media_type=resp.headers.get("content-type") or "application/json",
        status_code=resp.status_code,
        headers={
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("content-type", "content-length", "transfer-encoding", "connection")
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--listen", type=int, default=8080)
    ap.add_argument("--tool-call-limit", type=int, default=8)
    ap.add_argument("--thinking-limit", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()
    BACKEND = args.backend
    TIMEOUT = args.timeout
    LIMITS["tool_call"] = args.tool_call_limit
    LIMITS["thinking"] = args.thinking_limit
    sems = {k: asyncio.Semaphore(v) for k, v in LIMITS.items()}
    uvicorn.run(app, host="0.0.0.0", port=args.listen)
