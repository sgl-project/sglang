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
from fastapi.responses import JSONResponse, StreamingResponse

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
    return "thinking"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    kind = classify(body)
    forwarded = dict(body)
    if kind == "tool_call":
        forwarded["priority"] = 10  # 注入 priority，配合服务端 --priority-scheduling

    try:
        await asyncio.wait_for(sems[kind].acquire(), TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": "too many requests", "type": "concurrency_limit"},
            status_code=429,
        )

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request("POST", BACKEND + "/v1/chat/completions", json=forwarded)
    resp = await client.send(req, stream=True)
    if resp.status_code >= 400:
        sems[kind].release()
        body_err = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return JSONResponse(body_err, status_code=resp.status_code)

    async def stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            # 流结束才释放槽位，长输出期间持续占用
            sems[kind].release()
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(), media_type=resp.headers.get("content-type", "text/event-stream")
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
