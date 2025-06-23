"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import dataclasses
import logging
import random
import urllib
from itertools import chain
from typing import List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.disaggregation.utils import PDRegistryRequest

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error


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
    def __init__(self, prefill_configs: List[PrefillConfig], decode_servers: List[str]):
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = decode_servers

    def add_prefill_server(self, new_prefill_config: PrefillConfig):
        self.prefill_configs.append(new_prefill_config)
        self.prefill_servers.append(new_prefill_config.url)

    def add_decode_server(self, new_decode_server: str):
        self.decode_servers.append(new_decode_server)

    def select_pair(self):
        # TODO: return some message instead of panic
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        prefill_config = random.choice(self.prefill_configs)
        decode_server = random.choice(self.decode_servers)
        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=3600
            )  # Add timeout for request reliability
        ) as session:
            tasks = [
                session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                session.post(f"{decode_server}/{endpoint}", json=modified_request),
            ]

            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await asyncio.gather(*tasks)

            if "return_logprob" in modified_request:

                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()

                # merge `meta_info.input_token_logprobs` from prefill to decode
                if "meta_info" in ret_json:
                    if "input_token_logprobs" in ret_json["meta_info"]:
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
            else:
                ret_json = await decode_response.json()

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

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
                # Create the tasks for both prefill and decode requests
                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                    session.post(f"{decode_server}/{endpoint}", json=modified_request),
                ]
                # Wait for both responses to complete. Since this is streaming, they return immediately.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if modified_request.get("return_logprob", False):
                    prefill_chunks = []
                    async for chunk in prefill_response.content:
                        prefill_chunks.append(chunk)

                    first_prefill_chunk = (
                        prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                    )
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                    async for chunk in decode_response.content:
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"][
                                    "input_token_logprobs"
                                ]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )

                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
load_balancer: Optional[MiniLoadBalancer] = None


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
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in chain(prefill_servers):
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in chain(decode_servers):
            server_info = await session.get(f"{server}/get_server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }


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
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [
                    _generate_bootstrap_room() for _ in range(batch_size)
                ],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
            }
        )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request, prefill_server, decode_server, "generate"
        )
    else:
        return await load_balancer.generate(
            modified_request, prefill_server, decode_server, "generate"
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = parsed_url.hostname
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": _generate_bootstrap_room(),
        }
    )

    if request_data.get("stream", False):
        return await load_balancer.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await load_balancer.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


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


@app.post("/register")
async def register(obj: PDRegistryRequest):
    if obj.mode == "prefill":
        load_balancer.add_prefill_server(
            PrefillConfig(obj.registry_url, obj.bootstrap_port)
        )
        logger.info(
            f"Registered prefill server: {obj.registry_url} with bootstrap port: {obj.bootstrap_port}"
        )
    elif obj.mode == "decode":
        load_balancer.add_decode_server(obj.registry_url)
        logger.info(f"Registered decode server: {obj.registry_url}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Must be either PREFILL or DECODE.",
        )

    logger.info(
        f"#Prefill servers: {len(load_balancer.prefill_configs)}, "
        f"#Decode servers: {len(load_balancer.decode_servers)}"
    )

    return Response(status_code=200)


def run(prefill_configs, decode_addrs, host, port):
    global load_balancer
    load_balancer = MiniLoadBalancer(prefill_configs, decode_addrs)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # FIXME: remove this, use the unified entry point: sglang.srt.disaggregation.launch_lb
    from sglang.srt.disaggregation.launch_lb import main

    main()
