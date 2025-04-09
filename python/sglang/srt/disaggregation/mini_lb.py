"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import random
import urllib
from itertools import chain

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse


class MiniLoadBalancer:
    def __init__(self, prefill_servers, decode_servers):
        self.prefill_servers = prefill_servers
        self.decode_servers = decode_servers

    def select_pair(self):
        return random.choice(self.prefill_servers), random.choice(self.decode_servers)

    async def generate(
        self, modified_request, prefill_server, decode_server
    ) -> ORJSONResponse:

        async with aiohttp.ClientSession() as session:
            tasks = [
                session.post(f"{prefill_server}/generate", json=modified_request),
                session.post(f"{decode_server}/generate", json=modified_request),
            ]
            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await asyncio.gather(*tasks)

            return ORJSONResponse(
                content=await decode_response.json(),
                status_code=decode_response.status,
            )

    async def generate_stream(self, modified_request, prefill_server, decode_server):
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
                            f"{prefill_server}/generate", json=modified_request
                        ),
                        session.post(
                            f"{decode_server}/generate", json=modified_request
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
load_balancer = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_check():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    prefill_infos = []
    decode_infos = []
    async with aiohttp.ClientSession() as session:
        for server in chain(prefill_servers):
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in chain(decode_servers):
            server_info = await session.get(f"{server}/get_server_info")
            decode_infos.append(await server_info.json())

    return {"prefill": prefill_infos, "decode": decode_infos}


@app.get("/get_model_info")
async def get_model_info():
    # Dummy model information
    model_info = {
        "model_path": "/path/to/dummy/model",
        "tokenizer_path": "/path/to/dummy/tokenizer",
        "is_generation": True,
        "preferred_sampling_params": {"temperature": 0.7, "max_new_tokens": 128},
    }
    return ORJSONResponse(content=model_info)


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    prefill_server, decode_server = load_balancer.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_room": random.randint(0, 2**63 - 1),
        }
    )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request, prefill_server, decode_server
        )
    else:
        return await load_balancer.generate(
            modified_request, prefill_server, decode_server
        )


@app.get("/v1/models")
async def get_models():
    prefill_server = load_balancer.prefill_servers[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def run(prefill_addrs, decode_addrs, host, port):
    global load_balancer
    load_balancer = MiniLoadBalancer(prefill_addrs, decode_addrs)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini Load Balancer Server")
    parser.add_argument(
        "--prefill", required=True, help="Comma-separated URLs for prefill servers"
    )
    parser.add_argument(
        "--decode", required=True, help="Comma-separated URLs for decode servers"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server (default: 8000)"
    )
    args = parser.parse_args()
    run(args.prefill.split(","), args.decode.split(","), args.host, args.port)
