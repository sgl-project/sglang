"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import copy
import dataclasses
import logging
import random
import urllib
from http import HTTPStatus
from itertools import chain, cycle
from typing import List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.utils import maybe_wrap_ipv6_address

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


@dataclasses.dataclass
class VisionConfig:
    url: str
    bootstrap_port: Optional[int] = None


class MiniLoadBalancer:
    def __init__(
        self,
        vision_configs: List[VisionConfig],
        prefill_configs: List[PrefillConfig],
        decode_servers: List[str],
        timeout: int,
    ):
        self.vision_configs = vision_configs
        self.vision_servers = [v.url for v in vision_configs]
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = decode_servers
        self.enable_multimodal_disagg = bool(vision_configs)

        # round robin selection of vision server and prefill server
        self.vision_server_index = 0
        self.prefill_server_index = 0
        self.timeout = timeout
        self.vision_configs_cycle = cycle(vision_configs)
        self.prefill_configs_cycle = cycle(prefill_configs)

    def add_vision_server(self, new_vision_config: VisionConfig):
        self.vision_configs.append(new_vision_config)
        self.vision_servers.append(new_vision_config.url)

    def add_prefill_server(self, new_prefill_config: PrefillConfig):
        self.prefill_configs.append(new_prefill_config)
        self.prefill_servers.append(new_prefill_config.url)

    def add_decode_server(self, new_decode_server: str):
        self.decode_servers.append(new_decode_server)

    def select_pair(self):
        # TODO: return some message instead of panic
        if self.enable_multimodal_disagg:
            # support random selection of vision server and prefill server
            if len(self.vision_configs) == 0 or len(self.prefill_servers) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="No vision servers or prefill servers available",
                )
            vision_config = self.vision_configs[self.vision_server_index]
            prefill_server = self.prefill_servers[self.prefill_server_index]
            self.vision_server_index = (self.vision_server_index + 1) % len(
                self.vision_configs
            )
            self.prefill_server_index = (self.prefill_server_index + 1) % len(
                self.prefill_servers
            )
            return vision_config.url, vision_config.bootstrap_port, prefill_server
        else:
            if len(self.prefill_configs) == 0 or len(self.decode_servers) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="No prefill servers or decode servers available",
                )
            prefill_config = self.prefill_configs_cycle.__next__()
            decode_server = random.choice(self.decode_servers)
            return prefill_config.url, prefill_config.bootstrap_port, decode_server

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout
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

    async def _check_single_response(self, response: aiohttp.ClientResponse):
        try:
            response_json = await response.json()
            if response.status != 200:
                return False, response_json
            return True, None
        except Exception as e:
            return False, f"Response check failed: {e}"

    async def multimodal_generate(
        self,
        vision_modified_request,
        prefill_modified_request,
        vision_server,
        prefill_server,
        endpoint="v1/chat/completions",
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=3600
            )  # Add timeout for request reliability
        ) as session:
            vision_task = asyncio.create_task(
                session.post(
                    f"{vision_server}/{endpoint}", json=vision_modified_request
                )
            )
            ret_task = asyncio.create_task(
                session.post(
                    f"{prefill_server}/{endpoint}", json=prefill_modified_request
                )
            )

            # check vision response first, to avoid prefill request being blocked by vision server
            vision_response = await vision_task
            is_success, error_message = await self._check_single_response(
                vision_response
            )
            if not is_success:
                # abort prefill request
                await session.post(
                    f"{prefill_server}/abort_request",
                    json={"rid": prefill_modified_request["rid"]},
                )
                logger.info(f"Abort prefill request: {prefill_modified_request['rid']}")
                raise HTTPException(
                    status_code=vision_response.status,
                    detail=error_message,
                )

            ret_response = await ret_task
            return ORJSONResponse(
                content=await ret_response.json(),
                status_code=ret_response.status,
            )

    async def multimodal_generate_stream(
        self,
        vision_modified_request,
        prefill_modified_request,
        vision_server,
        prefill_server,
        endpoint="v1/chat/completions",
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=3600
                )  # Add timeout for request reliability
            ) as session:
                vision_task = asyncio.create_task(
                    session.post(
                        f"{vision_server}/{endpoint}", json=vision_modified_request
                    )
                )
                ret_task = asyncio.create_task(
                    session.post(
                        f"{prefill_server}/{endpoint}", json=prefill_modified_request
                    )
                )

                vision_response = await vision_task
                is_success, error_message = await self._check_single_response(
                    vision_response
                )
                if not is_success:
                    # abort prefill request
                    await session.post(
                        f"{prefill_server}/abort_request",
                        json={"rid": prefill_modified_request["rid"]},
                    )
                    logger.info(
                        f"Abort prefill request: {prefill_modified_request['rid']}"
                    )
                    raise HTTPException(
                        status_code=vision_response.status,
                        detail=error_message,
                    )

                ret_response = await ret_task

                if prefill_modified_request.get("return_logprob", False):
                    async for chunk in ret_response.content:
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in ret_response.content:
                        yield chunk

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )

    async def _check_single_response(self, response: aiohttp.ClientResponse):
        try:
            response_json = await response.json()
            if response.status != 200:
                return False, response_json
            return True, None
        except Exception as e:
            return False, f"Response check failed: {e}"

    async def multimodal_generate(
        self,
        vision_modified_request,
        prefill_modified_request,
        vision_server,
        prefill_server,
        endpoint="v1/chat/completions",
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=3600
            )  # Add timeout for request reliability
        ) as session:
            vision_task = asyncio.create_task(
                session.post(
                    f"{vision_server}/{endpoint}", json=vision_modified_request
                )
            )
            ret_task = asyncio.create_task(
                session.post(
                    f"{prefill_server}/{endpoint}", json=prefill_modified_request
                )
            )

            # check vision response first, to avoid prefill request being blocked by vision server
            vision_response = await vision_task
            is_success, error_message = await self._check_single_response(
                vision_response
            )
            if not is_success:
                # abort prefill request
                await session.post(
                    f"{prefill_server}/abort_request",
                    json={"rid": prefill_modified_request["rid"]},
                )
                logger.info(f"Abort prefill request: {prefill_modified_request['rid']}")
                raise HTTPException(
                    status_code=vision_response.status,
                    detail=error_message,
                )

            ret_response = await ret_task
            return ORJSONResponse(
                content=await ret_response.json(),
                status_code=ret_response.status,
            )

    async def multimodal_generate_stream(
        self,
        vision_modified_request,
        prefill_modified_request,
        vision_server,
        prefill_server,
        endpoint="v1/chat/completions",
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout
                )  # Add timeout for request reliability
            ) as session:
                vision_task = asyncio.create_task(
                    session.post(
                        f"{vision_server}/{endpoint}", json=vision_modified_request
                    )
                )
                ret_task = asyncio.create_task(
                    session.post(
                        f"{prefill_server}/{endpoint}", json=prefill_modified_request
                    )
                )

                vision_response = await vision_task
                is_success, error_message = await self._check_single_response(
                    vision_response
                )
                if not is_success:
                    # abort prefill request
                    await session.post(
                        f"{prefill_server}/abort_request",
                        json={"rid": prefill_modified_request["rid"]},
                    )
                    logger.info(
                        f"Abort prefill request: {prefill_modified_request['rid']}"
                    )
                    raise HTTPException(
                        status_code=vision_response.status,
                        detail=error_message,
                    )

                ret_response = await ret_task

                if prefill_modified_request.get("return_logprob", False):
                    async for chunk in ret_response.content:
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in ret_response.content:
                        yield chunk

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout
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
    result = {}

    if all_internal_states:
        result["internal_states"] = all_internal_states
    else:
        # Fallback with dummy data if no internal states found
        result["internal_states"] = [
            {
                "last_gen_throughput": 0.0,
                "avg_spec_accept_length": None,
            }
        ]

    if prefill_infos:
        result["prefill"] = prefill_infos
    if decode_infos:
        result["decode"] = decode_infos

    return result


@app.get("/get_model_info")
async def get_model_info():
    global load_balancer

    if not load_balancer or not load_balancer.prefill_servers:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = load_balancer.prefill_servers[0]
    endpoint_url = f"{target_server_url}/get_model_info"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server_url}"
                            f"Status: {response.status}, Response: {error_text}"
                        ),
                    )

                model_info_json = await response.json()
                return ORJSONResponse(content=model_info_json)

        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Failed to get model info from backend",
            )


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
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
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
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


async def _forward_to_backend_multimodal(request_data: dict, endpoint_name: str):
    if endpoint_name != "v1/chat/completions":
        raise HTTPException(
            status_code=400,
            detail=f"Endpoint name should be 'v1/chat/completions', but got {endpoint_name}",
        )

    vision_server, bootstrap_port, prefill_server = load_balancer.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(vision_server)
    hostname = parsed_url.hostname
    vision_modified_request = copy.deepcopy(request_data)
    language_modified_request = copy.deepcopy(request_data)

    bootstrap_room = (
        _generate_bootstrap_room()
        if request_data.get("bootstrap_room", None) is None
        else request_data["bootstrap_room"]
    )
    vision_modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
            "rid": (
                request_data["rid"] if "rid" in request_data else str(bootstrap_room)
            ),
            "stream": False,
        }
    )
    language_modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
            "rid": (
                request_data["rid"] if "rid" in request_data else str(bootstrap_room)
            ),
        }
    )
    # only keep text input for language request
    for message in language_modified_request["messages"]:
        if isinstance(message["content"], list):
            text_content = []
            for content in message["content"]:
                if content["type"] == "text":
                    text_content.append(content)
            message["content"] = text_content

    if request_data.get("stream", False):
        return await load_balancer.multimodal_generate_stream(
            vision_modified_request,
            language_modified_request,
            vision_server,
            prefill_server,
            endpoint=endpoint_name,
        )
    else:
        return await load_balancer.multimodal_generate(
            vision_modified_request,
            language_modified_request,
            vision_server,
            prefill_server,
            endpoint=endpoint_name,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    if load_balancer.enable_multimodal_disagg:
        return await _forward_to_backend_multimodal(request_data, "v1/chat/completions")
    else:
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


def run(vision_configs, prefill_configs, decode_addrs, host, port, timeout):
    global load_balancer
    load_balancer = MiniLoadBalancer(
        vision_configs, prefill_configs, decode_addrs, timeout=timeout
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # FIXME: remove this, use the unified entry point: sglang.srt.disaggregation.launch_lb
    from sglang.srt.disaggregation.launch_lb import main

    main()
