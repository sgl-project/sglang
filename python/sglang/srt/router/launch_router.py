"""
SGLang Router manages DP workers and routes requests to them based on a routing policy.

Glossary:

- Worker (DP Worker): An SGLang instance hosting an SGLang backend, consisting of a server,
  tokenizer, scheduler, and detokenizer. It can have Tensor Parallelism (TP) > 1.
  Also referred to as a DP Worker because the routing performs Data Parallelism.
- Worker Server: The server component of an SGLang Worker instance.
- Worker Client: The client used to interact with a Worker Server.
- Router: A service that routes requests to Worker Servers, manages scaling,
  and ensures fault tolerance.
- Routing Policy: The strategy used to distribute requests among Workers.
- Router Server: The server hosting the Router service.

Notes:

- We assume all workers are configured identically. We don't support heterogeneous workers yet.
- The Router performs data parallelism, distributing requests across multiple DP Workers.


Router APIs:

1. /generate: Selects a Worker based on the routing policy and forwards the request.
2. /health: Returns the Router's health status.
3. /health_generate: Checks Router health by generating one token.
4. /get_server_args: Returns the first Worker's server arguments.
5. /get_model_info: Returns the model information from the first Worker.
6. /flush_cache: Clears the cache of all Workers.
7. /add_worker: Adds a new Worker to the Router.
8. /remove_worker: Removes a Worker from the Router.
...more

Features:

1. Fault Tolerance: Periodically checks Worker health and removes unhealthy instances.
2. Dynamic Scaling: Supports adding or removing Workers dynamically.
3. Flexible Routing: Implements multiple routing policies (e.g., round-robin, random).


Example:
python launch_router.py --host 127.0.0.1 --port 8080 --policy round_robin --worker-urls http://127.0.0.1:9000 http://127.0.0.1:9002
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, Request, Response

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.router.router import get_router_class
from sglang.srt.router.utils import configure_logger
from sglang.srt.router.worker import WorkerUpdateReq

logger = logging.getLogger(__name__)

router = None

##############
# Worker Health Check
##############


async def is_healthy_or_remove(worker):
    """
    Check if the worker is healthy or remove it from the router
    """

    # If worker is not in the list, no need to perform the check
    if router.if_exist(worker.server_url) is False:
        return
    try:
        res = await worker.client.get("/health")
        res.raise_for_status()
    except Exception as e:
        # either disconnected or unhealthy
        router.remove_worker(worker.server_url)
        logger.warning(
            f"Worker {worker.server_url} is unhealthy and was removed. Error: {e}"
        )


async def health_check_loop():
    """
    a non-blocking event loop to check the health of the workers and remove the unhealthy one
    """
    while True:
        logger.info("Checking worker health...")
        tasks = []
        for server_url, worker in router.server_url_to_worker.items():
            tasks.append(is_healthy_or_remove(worker))

        await asyncio.gather(*tasks)
        await asyncio.sleep(60)


async def wait_until_ready():
    """
    Before the http server starts, we wait until all workers are ready
    """
    timeout = 120  # seconds
    start_time = time.time()

    while True:
        success = True
        for server_url, worker in router.server_url_to_worker.items():
            try:
                res = await worker.client.get("/health")
                res.raise_for_status()
            except Exception:
                # if any exception happens, we will break and retry
                success = False
                logger.warning(f"Worker {server_url} is not ready yet")
                break

        if success is False:
            if time.time() - start_time > timeout:
                raise Exception(
                    f"Timeout {timeout} seconds waiting for workers to be ready"
                )
        else:
            break

        await asyncio.sleep(10)

    logger.info("All workers are ready and the router is ready to serve!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    https://fastapi.tiangolo.com/advanced/events/#lifespan-function

    1. The code before "yield" is executed before the server starts.
    2. The code after "yield" is executed after the server ends.
    """

    # to hack around hanging issue on SIGINT: https://github.com/encode/uvicorn/issues/1301
    signal.signal(signal.SIGINT, lambda signalNumber, frame: sys.exit())
    # wait until all init workers are ready
    await wait_until_ready()
    # kick off a non-blocking event loop to monitor the health of the workers
    task = asyncio.create_task(health_check_loop())

    # separator
    yield

    # shutdown event
    task.cancel()


app = FastAPI(lifespan=lifespan)


################################
## SGLang instance compatible routes
################################

# TODO: `/flush_cache`, `/health_generate``


@app.get("/health")
async def health():
    """
    Returns the health status of the router.
    """
    return Response(status_code=200, content="OK")


@app.get("/get_server_args")
async def get_server_args():
    """
    Returns the server arguments of the first worker.
    """
    first_worker = router.worker_list[0]

    max_retries = 5

    for _ in range(max_retries):
        try:
            res = await first_worker.client.get("/get_server_args")
            res.raise_for_status()
        except Exception as e:
            logger.warning(f"Error getting server args: {e}")
            await is_healthy_or_remove(first_worker)
            continue
        return Response(content=res.content, media_type="application/json")
    return Response(
        status_code=500,
        content=f"Failed to get server args after {max_retries} retries",
    )


@app.get("/get_model_info")
async def get_model_info():
    """
    Returns the model info of the first worker.
    """
    first_worker = router.worker_list[0]

    max_retries = 5

    for _ in range(max_retries):
        try:
            res = await first_worker.client.get("/get_model_info")
            res.raise_for_status()
        except Exception as e:
            logger.warning(f"Error getting server args: {e}")
            await is_healthy_or_remove(first_worker)
            continue
        return Response(content=res.content, media_type="application/json")

    return Response(
        status_code=500, content=f"Failed to generate token after {max_retries} retries"
    )


@app.api_route("/generate", methods=["POST", "PUT"])
async def generate(obj: GenerateReqInput, request: Request):
    """
    Generates response from the selected worker.
    """

    max_retries = 5

    for _ in range(max_retries):
        try:
            selected_worker = router.calc_priority()
            res = await selected_worker.client.post("/generate", json=asdict(obj))
            res.raise_for_status()
        except Exception as e:
            logger.warning(f"Error generating response: {e}")
            await is_healthy_or_remove(selected_worker)
            continue

        return Response(content=res.content, media_type="application/json")

    return Response(
        status_code=500,
        content=f"Failed to generate response after {max_retries} retries",
    )


####################
## Worker Management
####################


@app.post("/add_worker")
async def add_worker(worker_info: WorkerUpdateReq):
    server_url = worker_info.server_url
    try:
        router.add_worker(server_url)
        return Response(
            status_code=200, content=f"Worker {server_url} was added succesfully"
        )
    except Exception as e:
        return Response(status_code=500, content=str(e))


@app.post("/remove_worker")
async def remove_worker(worker_info: WorkerUpdateReq):
    server_url = worker_info.server_url
    try:
        router.remove_worker(server_url)
        return Response(
            status_code=200, content=f"Worker {server_url} was removed succesfully"
        )
    except Exception as e:
        return Response(status_code=500, content=str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="SGLang Router")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Router host")
    parser.add_argument("--port", type=int, default=8080, help="Router port")
    parser.add_argument(
        "--policy", type=str, default="round_robin", help="Routing policy"
    )
    # + means the args are gathered to a list, and #args must >= 1
    parser.add_argument(
        "--worker-urls",
        required=True,
        nargs="+",
        type=str,
        help="Space-separated list of DP workers' URLs",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser.parse_args()


def launch_router():
    global router
    args = parse_args()

    configure_logger(args.log_level)

    router_class = get_router_class(args.policy)
    router = router_class(args.worker_urls)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    launch_router()
