"""
SGLang Router manages DP workers and routes requests to them based on a routing policy.

Glossary:

- Worker (DP Worker): A server hosting an SGLang backend, consisting of a server,
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

Key Features:
1. Fault Tolerance: Periodically checks Worker health and removes unhealthy instances.
2. Dynamic Scaling: Supports adding or removing Workers dynamically.
3. Flexible Routing: Implements multiple routing policies (e.g., round-robin, random).

Usage:
python -m sglang.launch_router --host <router_host> --port <router_port> --policy <policy_name> --server-urls <url_1> <url_2> ... <url_n>

Example:
python -m sglang.launch_router --host 127.0.0.1 --port 8080 --policy round-robin --server-urls 127.0.0.1:8081 127.0.0.1:8082 127.0.0.1:8083
"""

from fastapi import FastAPI
import uvicorn
import argparse
from fastapi.responses import Response
from typing import List, Dict
import httpx
import asyncio
from sglang.srt.router.router import get_router_class, BaseRouter
from contextlib import asynccontextmanager
import logging
from sglang.srt.router.worker import Worker


def configure_logger(log_level, prefix: str = ""):
    # add level to the format
    format = f"[%(asctime)s{prefix}] %(levelname)s: %(message)s"

    # format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    logging.basicConfig(
        level=log_level,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

logger = logging.getLogger(__name__)
configure_logger(logging.INFO, " [Router]")

router = None

# run a non-blocking event loop to check the healthiness of the workers and kick out the unhealthy one
async def health_check():

    async def handler(worker: Worker):
        try:
            res = await worker.client.get("/health")
            if res.status_code != 200:
                raise Exception(f"Worker {worker.server_url} is unhealthy")
        # if disconnected or not status 200
        except Exception as e:
            router.remove_worker(worker.server_url)
            logger.warning(f"Worker {worker.server_url} is unhealthy and was removed: {e}")


    while True:
        logger.info("Checking worker health...")

        tasks = []
        for server_url, worker in router.server_url_to_worker.items():
            tasks.append(handler(worker))
        
        responses = await asyncio.gather(*tasks)

        await asyncio.sleep(10)

# https://fastapi.tiangolo.com/advanced/events/#lifespan-function
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup event
    # kick off a non-blocking event loop to monitor the healthiness of the workers
    task = asyncio.create_task(health_check())
    
    yield
    # shutdown event
    task.cancel()

app = FastAPI(lifespan=lifespan)

from dataclasses import dataclass

@dataclass
class WorkerInfo:
    server_url: str

@app.post("/add_worker")
async def add_worker(worker_info: WorkerInfo):
    server_url = worker_info.server_url
    try:
        # TODO: this should be async and wait until the worker is healthy
        router.add_worker(server_url)
        return Response(status_code=200, content=f"Worker {server_url} was added succesfully")
    except Exception as e:
        return Response(status_code=500, content=str(e))

@app.post("/remove_worker")
async def remove_worker(worker_info: WorkerInfo):
    server_url = worker_info.server_url
    try:
        router.remove_worker(server_url)
        return Response(status_code=200, content=f"Worker {server_url} was removed succesfully")
    except Exception as e:
        return Response(status_code=500, content=str(e))


@app.get("/get_server_args")
async def get_server_args():
    """
    Returns the server arguments of the first worker.
    If run into errors, check whether the worker is healthy.
    If not, kick the worker out, and rerun the get_server_args function
    """
    tasks = []
    first_worker = router.worker_list[0]

    try:
        ret = await first_worker.client.get("/get_server_args")
        return ret.content
    except Exception as e:
        ret = await first_worker.client.get("/health")
        if ret.status_code != 200:
            # the error is because the worker is unhealthy
            # kick the worker out, and select the worker again
            router.remove_worker(first_worker.server_url)
            return await get_server_args()
        else:
            # the error is due to other reasons, just return the error
            return Response(status_code=500, content=str(e))

@app.post("/generate")
async def generate():
    """
    Generates a token from the selected worker.
    If run into errors, check whether the worker is healthy.
    If not, kick the worker out, and rerun the generate function
    """
    selected_worker = router.calc_priority()

    try:
        ret = await selected_worker.client.post("/generate")
        return ret.content
    except Exception as e:
        ret = await selected_worker.client.get("/health")
        if ret.status_code != 200:
            # the error is because the worker is unhealthy
            # kick the worker out, and select the worker again
            router.remove_worker(selected_worker.server_url)
            return await generate()
        else:
            # the error is due to other reasons, just return the error
            return Response(status_code=500, content=str(e))


# add arg parser
def parse_args():
    parser = argparse.ArgumentParser(description="SGLang Router")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Router host")
    parser.add_argument("--port", type=int, default=8080, help="Router port")
    parser.add_argument("--policy", type=str, default="round_robin", help="Routing policy")
    # + means the args are gathered to a list, and #args must >= 1
    parser.add_argument("--worker-urls", required=True, nargs="+", type=str, help="Space-separated list of DP workers' URLs")
    return parser.parse_args()

def launch_router():
    global router
    args = parse_args()
    router_class = get_router_class(args.policy)
    router = router_class(args.worker_urls)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    launch_router()


"""
Test

1. Basic

Run 4 workers, each on a different port. 
Run a router with the round-robin policy and add all 4 workers to the router.

Send /generate 8 times, the output should be [0, 1, 2, 3, 0, 1, 2, 3]
Send /get_server_args, the output should be the arg from 0

2. Fault Tolerance

Run 4 workers, each on a different port.
Tear down one worker.
Send /generate 4 times, ensure the evicted worker is not selected.


Run 4 workers, each on a different port.
Add one worker.
Send /generate 5 times, ensure the new worker is included
"""
