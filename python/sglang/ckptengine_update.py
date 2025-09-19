import argparse
import json
import os
import pickle
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Literal

import requests
import torch
import torch.distributed as dist
import zmq
from checkpoint_engine.ps import (
    ParameterServer,
    _gen_h2d_buckets,
    _to_named_tensor,
    request_inference_to_update,
)
from loguru import logger
from safetensors import safe_open
from torch.multiprocessing.reductions import reduce_tensor

CKPTENGINE_PORT = 33001


def my_custom_update_p2p(
    self,
    checkpoint_name: str,
    req_func: Callable[[list[tuple[str, str]]], None],
    ranks: list[int],
):
    assert self._p2p_store is not None, "p2p store is not initialized"
    assert ranks, "ranks should be set"
    if len(self._current_global_parameter_metas) == 0:
        raise ValueError("parameter metas is empty")
    assert (
        dist.is_initialized()
    ), "process group is not initialized when update model per bucket p2p"

    need_update = self._rank in ranks
    logger.info(
        f"[rank{self._rank}] update checkpoint {checkpoint_name} p2p, {need_update=} with {ranks=}, "
        f"gpu_count {self._gpu_count}, world_size {self._world_size}"
    )

    if not need_update:
        return

    # first execute a barrier to avoid subsequent cuda oom
    dist.barrier()

    bucket_size, _ = self._detect_bucket_size(disable_h2d_buffer=True)
    buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device="cuda")
    ipc_buffer_name = "__ipc_buffer___"
    self._p2p_store.register_named_tensors({ipc_buffer_name: buffer})
    logger.info(
        f"[rank{self._rank}] register buffer, shape={buffer.shape}, dtype={buffer.dtype}, data_ptr={buffer.data_ptr()}, nbytes={buffer.nbytes}"
    )
    handle = reduce_tensor(buffer)

    buckets = _gen_h2d_buckets(self._current_global_parameter_metas, bucket_size)
    socket, socket_paths = self._bind_zmq_socket()
    req_thread = threading.Thread(
        target=req_func,
        args=(socket_paths,),
    )
    req_thread.start()
    socket.send_pyobj(handle)
    last_owner_rank = self._rank
    for gidx, (owner_rank, bucket) in enumerate(buckets):
        self._logger_rank0(
            f"[rank{self._rank}] begin to update bucket {gidx + 1}/{len(buckets)} owner_rank {owner_rank} in checkpoint {checkpoint_name}, bucket_size: {bucket.size / 1024 / 1024:.2f}MiB, length: {len(bucket.items)}. "
        )
        _buffer = buffer[gidx % 2 * bucket_size : gidx % 2 * bucket_size + bucket.size]
        if dist.get_rank() == 0:
            self._copy_to_buffer(checkpoint_name, bucket, _buffer, owner_rank)
        # broadcast the collected data to all ranks
        dist.broadcast(_buffer, src=0)
        if last_owner_rank == self._rank:
            socket.recv()
        dist.barrier()
        if owner_rank == self._rank:
            socket.send_pyobj(_to_named_tensor(bucket.items, gidx % 2 * bucket_size))
        last_owner_rank = owner_rank

    if last_owner_rank == self._rank:
        socket.recv()
    socket.send_pyobj(None)
    socket.recv()
    req_thread.join()
    dist.barrier()
    socket.close()
    self._p2p_store.unregister_named_tensors([ipc_buffer_name])
    torch.cuda.empty_cache()


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


def req_inference(inference_parallel_size: int):
    rank = int(os.getenv("RANK", None))
    src = rank // inference_parallel_size * inference_parallel_size

    def req_func(socket_paths: list[tuple[str, str]]):
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
    parser.add_argument("--checkpoint-name", type=str, default="sglang-ckpt-iter-0")
    parser.add_argument("--update-method", type=str, default="broadcast")
    args = parser.parse_args()
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    req_func = req_inference(args.inference_parallel_size)
    ParameterServer._update_per_bucket_p2p = my_custom_update_p2p
    ps = ParameterServer(auto_pg=True)
    ps._gpu_count = args.inference_parallel_size
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
