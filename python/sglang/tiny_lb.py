"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import dataclasses
import logging
from typing import List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sglang.srt.disaggregation.utils import PDRegistryRequest


def setup_logger():
    logger = logging.getLogger("pdlb")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[PDLB (Python)] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


@dataclasses.dataclass
class PrefillConfig:
    url: str
    bootstrap_port: Optional[int] = None


class MiniLoadBalancer:
    def __init__(self, downstream_urls: List[str]):
        self.downstream_urls = downstream_urls

    async def select_server(self):
        assert len(self.downstream_urls) > 0
        return TODO

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=3600
                )  # Add timeout for request reliability
            ) as session:
                try:
                    # Create the tasks for both prefill and decode requests
                    tasks = [
                        session.post(
                            f"{prefill_server}/{endpoint}", json=modified_request
                        ),
                        session.post(
                            f"{decode_server}/{endpoint}", json=modified_request
                        ),
                    ]
                    # Wait for both responses to complete. Since this is streaming, they return immediately.
                    prefill_response, decode_response = await asyncio.gather(*tasks)
                    async for chunk in decode_response.content:
                        yield chunk
                except Exception as e:
                    error_msg = {
                        "error": {"message": f"Stream processing error: {str(e)}"}
                    }
                    yield b"data: " + orjson.dumps(
                        error_msg, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
                finally:
                    if prefill_response is not None:
                        await prefill_response.release()

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

    run(args.downstream, args.host, args.port)
