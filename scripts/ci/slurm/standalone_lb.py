"""Standalone PD load balancer (no sglang_router rust dep).
Adds bootstrap fields to each request, fans it out to the prefill and decode
servers, and returns the decode response.
Usage: python3 standalone_lb.py --prefill URL BOOTSTRAP_PORT --decode URL --host H --port P
"""

import argparse
import asyncio
import random
import urllib.parse

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

CHUNK = 1024 * 64


class LB:
    def __init__(
        self, prefill_url, bootstrap_port, decode_url, host, port, timeout=3600
    ):
        self.prefill_url = prefill_url
        self.bootstrap_port = bootstrap_port
        self.decode_url = decode_url
        self.host = host
        self.port = port
        self.timeout = timeout

    async def generate(self, req, endpoint):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            tasks = [
                session.post(f"{self.prefill_url}/{endpoint}", json=req),
                session.post(f"{self.decode_url}/{endpoint}", json=req),
            ]
            prefill_response, decode_response = await asyncio.gather(*tasks)
            ret_json = await decode_response.json()
            return ORJSONResponse(content=ret_json, status_code=decode_response.status)

    async def generate_stream(self, req, endpoint):
        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                tasks = [
                    session.post(f"{self.prefill_url}/{endpoint}", json=req),
                    session.post(f"{self.decode_url}/{endpoint}", json=req),
                ]
                _, decode_response = await asyncio.gather(*tasks)
                async for chunk in decode_response.content.iter_chunked(CHUNK):
                    yield chunk

        return StreamingResponse(stream_results(), media_type="text/event-stream")


app = FastAPI()
lb: LB = None


@app.get("/health")
async def health():
    return Response(status_code=200)


@app.get("/get_model_info")
@app.get("/model_info")
async def model_info():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{lb.prefill_url}/model_info") as r:
            return ORJSONResponse(content=await r.json())


@app.get("/v1/models")
async def v1_models():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{lb.prefill_url}/v1/models") as r:
            return ORJSONResponse(content=await r.json(), status_code=r.status)


def _bootstrap(req):
    parsed = urllib.parse.urlparse(lb.prefill_url)
    out = req.copy()
    out.update(
        {
            "bootstrap_host": parsed.hostname,
            "bootstrap_port": lb.bootstrap_port,
            "bootstrap_room": random.randint(0, 2**63 - 1),
        }
    )
    return out


@app.post("/generate")
async def handle_generate(request_data: dict):
    req = _bootstrap(request_data)
    if request_data.get("stream", False):
        return await lb.generate_stream(req, "generate")
    return await lb.generate(req, "generate")


async def _forward(request_data, endpoint):
    req = _bootstrap(request_data)
    if request_data.get("stream", False):
        return await lb.generate_stream(req, endpoint)
    return await lb.generate(req, endpoint)


@app.post("/v1/chat/completions")
async def chat(request_data: dict):
    return await _forward(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def comp(request_data: dict):
    return await _forward(request_data, "v1/completions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefill", nargs=2, metavar=("URL", "BOOTSTRAP_PORT"), required=True
    )
    ap.add_argument("--decode", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    lb = LB(a.prefill[0], int(a.prefill[1]), a.decode, a.host, a.port)
    uvicorn.run(app, host=a.host, port=a.port)
