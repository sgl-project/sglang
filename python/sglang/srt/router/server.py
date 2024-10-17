
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
- The Router performs data parallelism, distributing requests across multiple DP Workers.
- Ensure that all Workers are configured identically for consistent performance.

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

router = None

# run a non-blocking event loop to check the healthiness of the workers and kick out the unhealthy one
async def health_check():
    while True:
        tasks = []
        for server_url, client in router.url_to_client.items():
            tasks.append(client.get("/health"))
        
        responses = await asyncio.gather(*tasks)
        for res in responses:
            if res.status_code != 200:
                router.remove_worker(res.url)
        
        await asyncio.sleep(10)

# https://fastapi.tiangolo.com/advanced/events/#lifespan-function
@asynccontextmanager
async def lifespan():
    # startup event
    # kick off a non-blocking event loop to check the healthiness of the workers
    task = asyncio.create_task(health_check())
    
    yield
    # shutdown event
    task.cancel()

app = FastAPI(lifespan=lifespan)


@app.post("/add_worker")
async def add_worker(server_url: str):
    try:
        router.add_worker(server_url)
        return Response(status_code=200, content=f"Worker {server_url} was added succesfully")
    except Exception as e:
        return Response(status_code=500, content=str(e))

@app.post("/remove_worker")
async def remove_worker(server_url: str):
    try:
        router.remove_worker(server_url)
        return Response(status_code=200, content=f"Worker {server_url} was removed succesfully")
    except Exception as e:
        return Response(status_code=500, content=str(e))

@app.get("/get_server_args")
async def get_server_args():
    tasks = []
    router: BaseRouter
    # assume all of them are healthy
    # forward the request to all servers and aggregate the response to a list
    for worker in router.worker_list:
        tasks.append(worker.client.get("/get_server_args"))
    
    responses = await asyncio.gather(*tasks)
    ret = [res.content for res in responses]
    return ret

@app.get("/generate")
async def generate():
    selected_worker = router.calc_priority()
    return await selected_worker.client.get("/generate")


# add arg parser
def parse_args():
    parser = argparse.ArgumentParser(description="SGLang Router")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Router host")
    parser.add_argument("--port", type=int, default=8080, help="Router port")
    parser.add_argument("--policy", type=str, default="round-robin", help="Routing policy")
    # + means the args are gathered to a list, and #args must >= 1
    parser.add_argument("--worker-urls", required=True, nargs="+", type=str, help="Space-separated list of DP workers' URLs")
    return parser.parse_args()

def launch_router():
    global router
    args = parse_args()
    router_class = get_router_class(args.policy)
    router = router_class(args.server_urls)
    uvicorn.run(app)

if __name__ == "__main__":
    launch_router()