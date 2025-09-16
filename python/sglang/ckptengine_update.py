import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Literal

import requests
import torch
import torch.distributed as dist
import zmq
from checkpoint_engine.ps import ParameterServer, request_inference_to_update
from loguru import logger
from safetensors import safe_open

CKPTENGINE_PORT = 33001


@contextmanager
def timer(msg: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{msg} duration: {end - start:.2f} seconds")


def request_inference_to_update(
    port, socket_paths: dict[str, str], host="localhost", timeout: float = 300.0
):
    socket = zmq.Context().socket(zmq.PUSH)
    socket.connect(f"tcp://{host}:{port}")
    message = json.dumps(socket_paths).encode("utf-8")
    socket.send(message)


def split_checkpoint_files(
    checkpoint_path: str, rank: int, world_size: int
) -> list[str]:
    checkpoint_files = [
        os.path.join(checkpoint_path, f)
        for f in filter(
            lambda x: x.endswith(".safetensors"), os.listdir(checkpoint_path)
        )
    ]
    files_per_rank = (len(checkpoint_files) + world_size - 1) // world_size
    return checkpoint_files[rank * files_per_rank : (rank + 1) * files_per_rank]


def split_tensors(
    checkpoint_path: str, rank: int, world_size: int
) -> dict[str, torch.Tensor]:
    index_fn = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_fn, "r") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    weights_per_rank = (len(weight_map) + world_size - 1) // world_size
    fn_tensors: dict[str, list[str]] = defaultdict(list)
    weight_keys = list(weight_map.items())
    for name, file in weight_keys[
        rank * weights_per_rank : (rank + 1) * weights_per_rank
    ]:
        fn_tensors[file].append(name)
    named_tensors = {}
    for file, names in fn_tensors.items():
        with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
            for name in names:
                named_tensors[name] = f.get_tensor(name)
    return named_tensors


def req_inference(endpoint: str, inference_parallel_size: int):
    rank = int(os.getenv("RANK", None))
    src = rank // inference_parallel_size * inference_parallel_size

    def req_func(socket_paths: list[tuple[str, str]]):
        if rank == src:
            request_inference_to_update(
                CKPTENGINE_PORT + rank,
                dict(socket_paths[src : src + inference_parallel_size]),
            )

    return req_func


def update_weights(
    ps: ParameterServer,
    checkpoint_name: str,
    checkpoint_files: list[str],
    named_tensors: dict[str, torch.Tensor],
    req_func: Callable[[list[tuple[str, str]]], None],
    inference_parallel_size: int,
    save_metas_file: str | None = None,
    update_method: Literal["broadcast", "p2p", "all"] = "broadcast",
):
    ps.register_checkpoint(
        checkpoint_name, files=checkpoint_files, named_tensors=named_tensors
    )
    ps.init_process_group()
    dist.barrier()
    with timer("Gather metas"):
        ps.gather_metas(checkpoint_name)
    if save_metas_file and int(os.getenv("RANK")) == 0:
        with open(save_metas_file, "wb") as f:
            pickle.dump(ps.get_metas(), f)

    if update_method == "broadcast" or update_method == "all":
        with timer("Update weights without setting ranks"):
            ps.update(checkpoint_name, req_func)

    if update_method == "p2p" or update_method == "all":
        if update_method:
            # sleep 2s to wait destroy process group
            time.sleep(2)
        with timer("Update weights with setting ranks"):
            ps.update(
                checkpoint_name, req_func, ranks=list(range(inference_parallel_size))
            )


def join(
    ps: ParameterServer,
    checkpoint_name: str,
    save_metas_file: str,
    req_func: Callable[[list[tuple[str, str]]], None],
    inference_parallel_size: int,
):
    assert save_metas_file, "save_metas_file is required"
    with open(save_metas_file, "rb") as f:
        metas = pickle.load(f)
    ps.init_process_group()
    dist.barrier()
    with timer("Gather metas before join"):
        ps.gather_metas(checkpoint_name)
    ps.load_metas(metas)
    with timer(
        f"Update weights with setting ranks as range(0, {inference_parallel_size}) by using p2p"
    ):
        ps.update(checkpoint_name, req_func, ranks=list(range(inference_parallel_size)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update weights example")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--save-metas-file", type=str, default=None)
    parser.add_argument("--load-metas-file", type=str, default=None)
    parser.add_argument("--sleep-time", type=int, default=0)
    parser.add_argument("--inference-parallel-size", type=int, default=8)
    parser.add_argument("--checkpoint-name", type=str, default="my-checkpoint-iter-0")
    parser.add_argument("--update-method", type=str, default="broadcast")
    args = parser.parse_args()
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    req_func = req_inference(args.endpoint, args.inference_parallel_size)
    ps = ParameterServer(auto_pg=True)
    if args.load_metas_file:
        join(
            ps,
            args.checkpoint_name,
            args.load_metas_file,
            req_func,
            args.inference_parallel_size,
        )
    else:
        if os.path.exists(
            os.path.join(args.checkpoint_path, "model.safetensors.index.json")
        ):
            named_tensors = split_tensors(args.checkpoint_path, rank, world_size)
            checkpoint_files = []
        else:
            checkpoint_files = split_checkpoint_files(
                args.checkpoint_path, rank, world_size
            )
            named_tensors = {}
        update_weights(
            ps,
            args.checkpoint_name,
            checkpoint_files,
            named_tensors,
            req_func,
            args.inference_parallel_size,
            args.save_metas_file,
            args.update_method,
        )
    time.sleep(args.sleep_time)
