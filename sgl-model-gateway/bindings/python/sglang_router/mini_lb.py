"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import ipaddress
import logging
import random
import urllib
from http import HTTPStatus
from itertools import chain
from typing import Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang_router.router_args import RouterArgs

try:
    from sglang.srt.tracing.trace import (
        process_tracing_init,
        trace_get_remote_propagate_context,
        trace_req_finish,
        trace_req_start,
        trace_set_thread_info,
        trace_slice_end,
        trace_slice_start,
    )

    trace_package_imported = True
except ImportError:
    trace_package_imported = False

logger = logging.getLogger(__name__)

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


class MiniLoadBalancer:
    def __init__(
        self,
        router_args: RouterArgs,
    ):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.otlp_traces_endpoint = router_args.otlp_traces_endpoint
        self.enable_trace = router_args.enable_trace
        if self.enable_trace and not trace_package_imported:
            logger.warning(
                "Tracing is not supported in this environment. Please install sglang."
            )
            self.enable_trace = False
        self.encode_urls = router_args.encode_urls

        self.encode_idx = list(range(len(self.encode_urls)))

    def _validate_router_args(self, router_args: RouterArgs):
        logger.warning(
            "\x1b[33mMiniLB is only for debugging purposes, it only supports random policy!\033[0m"
        )

        # NOTE: too many arguments unsupported, just validate some important ones
        if router_args.policy != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not router_args.pd_disaggregation and not router_args.e_disaggregation:
            raise ValueError("MiniLB only supports PD/E disaggregation mode")

        if router_args.pd_disaggregation and router_args.e_disaggregation:
            raise ValueError(
                "MiniLB does not support PD and E disaggregation modes at the same time."
            )

        if len(router_args.prefill_urls) == 0:
            raise ValueError("MiniLB requires at least one prefill server")

        if router_args.pd_disaggregation and len(router_args.decode_urls) == 0:
            raise ValueError(
                "The PD disaggregation mode requires at least one decode server."
            )

        if router_args.e_disaggregation and len(router_args.decode_urls) != 0:
            raise ValueError("The E disaggregation mode doesn't require decode server")

    def start(self):
        global lb
        lb = self
        if self.enable_trace:
            process_tracing_init(self.otlp_traces_endpoint, "sglang")
            trace_set_thread_info("Mini lb")
        uvicorn.run(app, host=self.host, port=self.port)

    def select_pair(self):
        pidx = random.randint(0, len(self.prefill_urls) - 1)
        if len(self.decode_urls) != 0:
            didx = random.randint(0, len(self.decode_urls) - 1)
            decode_url = self.decode_urls[didx]
        else:
            decode_url = None
        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            decode_url,
        )

    async def embedding_bootstrap(
        self, session, prefill_url, req_id, embedding_length, embedding_dim
    ):
        response = await session.post(
            f"{prefill_url}/embedding_bootstrap",
            json={
                "req_id": req_id,
                "embedding_length": embedding_length,
                "embedding_dim": embedding_dim,
            },
        )
        response_json = await response.json()
        session_id = response_json["session_id"]
        buffer_address = response_json["buffer_address"]
        return session_id, buffer_address

    async def encode(
        self, request_data, encode_urls, endpoint_encode, endpoint_send, prefill_url
    ):
        messages = request_data.get("messages")
        if messages is None or len(encode_urls) == 0:
            return

        # Extract mm_items
        img_list = []
        for message in messages:
            for item in message.get("content"):
                if item.get("type") == "image_url":
                    img_url = item.get("image_url").get("url")
                    img_list.append(img_url)

        if len(img_list) == 0:
            return

        req_id = request_data.get("bootstrap_room")
        prefill_host = request_data["bootstrap_host"]

        # Split mm_items
        encode_requests = []
        random.shuffle(self.encode_idx)
        num_items_assigned = [
            (idx + len(img_list)) // len(self.encode_urls) for idx in self.encode_idx
        ]
        num_parts = sum(1 for x in num_items_assigned if x != 0)
        cum_num_items = 0
        cum_idx = 0
        for idx, assigned_num in enumerate(num_items_assigned):
            if assigned_num == 0:
                continue
            encode_requests.append(
                {
                    "encoder_idx": idx,
                    "mm_items": img_list[cum_num_items : cum_num_items + assigned_num],
                    "num_parts": num_parts,
                    "part_idx": cum_idx,
                    "req_id": req_id,
                    "prefill_url": prefill_url,
                    "bootstrap_host": prefill_host,
                }
            )
            cum_idx += 1
            cum_num_items += assigned_num

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout
            )  # Add timeout for request reliability
        ) as session:
            # Send encode requests

            tasks = [
                session.post(
                    f"{encode_urls[encode_request['encoder_idx']]}/{endpoint_encode}",
                    json=encode_request,
                )
                for encode_request in encode_requests
            ]

            responses = await asyncio.gather(*tasks)
            response_json_list_unsort = [
                await response.json() for response in responses
            ]

            # zmq backend: return is None
            if None in response_json_list_unsort:
                return

            # mooncake backend: send bootstrap info

            embedding_size_list_sort = [None for _ in range(num_parts)]
            embedding_length_tot = 0
            response_json_list_sort = [None for _ in range(num_parts)]
            for response_json in response_json_list_unsort:
                idx = response_json["part_idx"]
                embedding_size_list_sort[idx] = response_json["embedding_size"]
                embedding_length_tot += response_json["embedding_len"]
                response_json_list_sort[idx] = response_json

            offset = 0
            metadata_tasks = []
            session_id, buffer_address = await self.embedding_bootstrap(
                session,
                prefill_url,
                req_id,
                embedding_length_tot,
                response_json_list_sort[0]["embedding_dim"],
            )
            for idx in range(len(tasks)):
                response_json = response_json_list_sort[idx]
                buffer_address_adjust = offset + buffer_address
                response_json.update(
                    {
                        "session_id": session_id,
                        "buffer_address": buffer_address_adjust,
                        "bootstrap_host": prefill_host,
                        "prefill_url": prefill_url,
                    }
                )
                metadata_tasks.append(
                    session.post(
                        f"{encode_urls[response_json['encoder_idx']]}/{endpoint_send}",
                        json=response_json,
                    )
                )
                offset += embedding_size_list_sort[idx]
            await asyncio.gather(*metadata_tasks)

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout
            )  # Add timeout for request reliability
        ) as session:
            headers = {}
            bootstrap_room_list = []
            if self.enable_trace:
                bootstrap_room_list = (
                    modified_request["bootstrap_room"]
                    if isinstance(modified_request["bootstrap_room"], list)
                    else [modified_request["bootstrap_room"]]
                )
                trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
                headers = {"trace_context": trace_context}

            tasks = [
                session.post(f"{prefill_server}/{endpoint}", json=modified_request)
            ]
            if decode_server is not None:
                tasks.append(
                    session.post(f"{decode_server}/{endpoint}", json=modified_request)
                )

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end("mini_lb_launch", bootstrap_room, auto_next_anon=True)

            # Wait for both responses to complete. Prefill should end first.
            responses = await asyncio.gather(*tasks)
            prefill_response = responses[0]
            decode_response = (
                responses[1] if decode_server is not None else prefill_response
            )
            if "return_logprob" in modified_request and decode_server is not None:

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

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

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
                    total=self.timeout
                )  # Add timeout for request reliability
            ) as session:
                # Create the tasks for both prefill and decode requests
                headers = {}
                bootstrap_room_list = []
                if self.enable_trace:
                    bootstrap_room_list = (
                        modified_request["bootstrap_room"]
                        if isinstance(modified_request["bootstrap_room"], list)
                        else [modified_request["bootstrap_room"]]
                    )
                    trace_context = trace_get_remote_propagate_context(
                        bootstrap_room_list
                    )
                    headers = {"trace_context": trace_context}

                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=modified_request)
                ]
                if decode_server is not None:
                    tasks.append(
                        session.post(
                            f"{decode_server}/{endpoint}", json=modified_request
                        )
                    )

                for bootstrap_room in bootstrap_room_list:
                    trace_slice_end(
                        "mini_lb_launch", bootstrap_room, auto_next_anon=True
                    )
                # Wait for both responses to complete. Since this is streaming, they return immediately.
                responses = await asyncio.gather(*tasks)
                prefill_response = responses[0]
                decode_response = (
                    responses[1] if decode_server is not None else prefill_response
                )

                if (
                    modified_request.get("return_logprob", False)
                    and decode_server is not None
                ):
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

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
lb: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.get(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in lb.prefill_urls:
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in lb.decode_urls:
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


async def _get_model_info_impl():
    if not lb or not lb.prefill_urls:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = lb.prefill_urls[0]
    endpoint_url = f"{target_server_url}/model_info"

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


@app.get("/model_info")
async def model_info():
    return await _get_model_info_impl()


@app.get("/get_model_info")
async def get_model_info():
    return await _get_model_info_impl()


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    prefill_server, bootstrap_port, decode_server = lb.select_pair()

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
        return await lb.generate_stream(
            modified_request, prefill_server, decode_server, "generate"
        )
    else:
        return await lb.generate(
            modified_request, prefill_server, decode_server, "generate"
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    prefill_server, bootstrap_port, decode_server = lb.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    bootstrap_room = _generate_bootstrap_room()

    # Send requests to encode server
    encode_request = request_data.copy()
    encode_request.update(
        {
            "bootstrap_room": bootstrap_room,
            "bootstrap_host": hostname,
        }
    )
    asyncio.create_task(
        lb.encode(encode_request, lb.encode_urls, "encode", "send", prefill_server)
    )

    modified_request = encode_request.copy()
    modified_request.update(
        {"bootstrap_port": bootstrap_port, "encode_urls": lb.encode_urls}
    )

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await lb.generate(
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
    bootstrap_room = random.randint(0, 2**63 - 1)
    if lb.enable_trace:
        trace_req_start(bootstrap_room, bootstrap_room, role="router")
        trace_slice_start("mini_lb_launch", bootstrap_room)
    return bootstrap_room


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = lb.prefill_urls[0]  # Get the first prefill server
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
