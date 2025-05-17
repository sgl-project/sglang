"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


def setup_logger():
    logger = logging.getLogger("tiny_lb")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[tiny_lb] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


class DownstreamServer:
    def __init__(self, url: str):
        self.url = url
        self._ongoing_request_num = 0

    def is_full(self):
        return self._ongoing_request_num >= TODO

    @asynccontextmanager
    async def around_request(self):
        TODO
        self._ongoing_request_num += 1
        try:
            yield
        finally:
            self._ongoing_request_num -= 1


class MiniLoadBalancer:
    def __init__(self, downstream_servers: List[DownstreamServer]):
        self.downstream_servers = downstream_servers

    async def select_server(self):
        for downstream_server in downstream_servers:
            if not downstream_server.is_full():
                return downstream_server
        raise Exception("no available server")

    async def generate_stream(self, req, downstream_server: DownstreamServer, endpoint="generate"):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=3600
                )  # Add timeout for request reliability
            ) as session:
                async with downstream_server.around_request():
                    try:
                        response = await session.post(f"{downstream_server.url}/{endpoint}", json=req)
                        async for chunk in response.content:
                            yield chunk
                    except Exception as e:
                        error_msg = {
                            "error": {"message": f"Stream processing error: {str(e)}"}
                        }
                        yield b"data: " + orjson.dumps(
                            error_msg, option=orjson.OPT_NON_STR_KEYS
                        ) + b"\n\n"
                    finally:
                        if response is not None:
                            await response.release()

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
load_balancer: Optional[MiniLoadBalancer] = None


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    server = await load_balancer.select_server()
    if request_data.get("stream", False):
        return await load_balancer.generate_stream(request_data, server)
    else:
        raise NotImplementedError


def run(downstream_urls, host, port):
    global load_balancer
    load_balancer = MiniLoadBalancer(downstream_urls)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini Load Balancer Server")
    parser.add_argument(
        "--downstream", type=str, default=[], nargs="+", help="URLs for downstream servers"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server (default: 8000)"
    )
    args = parser.parse_args()

    downstream_servers = [DownstreamServer(url) for url in args.downstream]
    run(downstream_servers, args.host, args.port)
