import multiprocessing
import os
import time
import traceback
import unittest
from multiprocessing import Process
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM

from sglang.srt.entrypoints.engine import Engine as SglangEngine
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    CustomTestCase,
    find_available_port,
)

TEST_SUITE = dict(
    model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    mem_fraction_static=0.83,
    dp_size=2,
    tp_size=2,
)


class EngineWrapper:
    """
    A wrapper around Sglang engine to mock multi instance cases such as RL traing.

    """

    def __init__(
        self, model_path, random_seed, mem_fraction_static, device_mesh_cpu, base_gpu_id
    ):
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._rank = device_mesh_cpu.get_rank()
        self._tp_size = device_mesh_cpu.size()
        tp_size_per_node = self._tp_size
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0
        engine_kwargs = dict(
            model_path=model_path,
            random_seed=random_seed,
            mem_fraction_static=mem_fraction_static,
            base_gpu_id=base_gpu_id,
            enable_memory_saver=True,
            tp_size=self._tp_size,
            node_rank=node_rank,
            nnodes=1,
        )
        self._engine = None
        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = SglangEngine(**engine_kwargs)

        dist.barrier(group=self._device_mesh_cpu.get_group())

    def update_weights_from_tensor(
        self, named_tensors: Iterable[Tuple[str, torch.Tensor]]
    ):
        if self._tp_rank == 0:
            self._engine.update_weights_from_tensor(list(named_tensors))
            self._engine.flush_cache()
        dist.barrier(group=self._device_mesh_cpu.get_group())

    def release_memory_occupation(self, tags):
        if self._tp_rank == 0:
            self._engine.release_memory_occupation(tags)
        dist.barrier(group=self._device_mesh_cpu.get_group())

    def resume_memory_occupation(self, tags):
        if self._tp_rank == 0:
            self._engine.resume_memory_occupation(tags)
        dist.barrier(group=self._device_mesh_cpu.get_group())

    def shutdown(self):
        if self._tp_rank == 0:
            self._engine.shutdown()
        dist.barrier(group=self._device_mesh_cpu.get_group())


def get_gpu_memory_gb(gpu_id=0):
    return torch.cuda.device_memory_used() / 1024**3


class TestMultiInstanceReleaseMemoryOccupation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        multiprocessing.set_start_method("spawn")

    def test_multi_instance_release_memory_occupation(self):
        master_port = find_available_port(23456)

        dp_size = TEST_SUITE["dp_size"]
        tp_size = TEST_SUITE["tp_size"]
        world_size = dp_size * tp_size
        processes = []
        output_reader, output_writer = multiprocessing.Pipe(duplex=False)
        for rank in range(world_size):
            p = Process(
                target=_run_sglang_subprocess,
                kwargs=dict(
                    rank=rank,
                    dp_size=dp_size,
                    tp_size=tp_size,
                    model_path=TEST_SUITE["model_path"],
                    master_port=master_port,
                    output_writer=output_writer,
                    mem_fraction_static=TEST_SUITE["mem_fraction_static"],
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(world_size):
            self.assertTrue(
                output_reader.recv(), f"Subprocess fail. Check the logs above."
            )
        for p in processes:
            p.join()


def _run_sglang_subprocess(
    rank: int,
    dp_size: int,
    tp_size: int,
    model_path: str,
    master_port: int,
    output_writer,
    mem_fraction_static: float,
):
    engine = None
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
            world_size=dp_size * tp_size,
        )
        torch.cuda.set_device(rank)

        base_gpu_id = rank // tp_size * tp_size
        mesh_kwargs = dict(
            mesh_shape=(dp_size, tp_size, 1), mesh_dim_names=["dp", "tp", "pp"]
        )
        inference_device_mesh_device = init_device_mesh("cuda", **mesh_kwargs)
        inference_device_mesh_cpu = init_device_mesh("cpu", **mesh_kwargs)
        print(
            f"subprocess[{rank=},{base_gpu_id=},{rank=},{tp_size=}] {inference_device_mesh_device=} {inference_device_mesh_cpu=}"
        )

        _mem_usage = get_gpu_memory_gb(rank)
        print(f"GPU{rank} Memory usage before starting Engine: {_mem_usage}")

        engine = EngineWrapper(
            model_path=model_path,
            random_seed=42,
            mem_fraction_static=mem_fraction_static,
            device_mesh_cpu=inference_device_mesh_cpu["tp"],
            base_gpu_id=base_gpu_id,
        )
        print(f"subprocess[{rank=}] {engine=}", flush=True)

        # 1 - release kv cache
        _mem_usage = get_gpu_memory_gb(rank)
        print(f"GPU{rank} Memory usage before releasing Sgl KV cache: {_mem_usage}")
        engine.release_memory_occupation(tags=["kv_cache"])
        _curr_usage = get_gpu_memory_gb(rank)
        assert (
            _curr_usage < _mem_usage
        ), f"Memory usage after releasing KV cache must be reduced! before: {_mem_usage} vs after: {_curr_usage}"

        # 2 - release sglang weights
        _mem_usage = get_gpu_memory_gb(rank)
        print(f"GPU{rank} Memory usage before releasing Sgl weights: {_mem_usage}")
        engine.release_memory_occupation(tags=["weights"])

        _curr_usage = get_gpu_memory_gb(rank)
        assert (
            _curr_usage < _mem_usage
        ), f"Memory usage after releasing weights must be reduced! before: {_mem_usage} vs after: {_curr_usage}"

        # 3 - load hf model
        _mem_usage = get_gpu_memory_gb(rank)
        print(
            f"GPU{rank} Memory usage after releasing Sgl weights and kv cache: {_mem_usage}"
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
            torch_dtype="bfloat16",
            device_map=f"cuda:{rank}",
            trust_remote_code=True,
        ).cuda()
        _curr_usage = get_gpu_memory_gb(rank)
        assert (
            _curr_usage > _mem_usage
        ), f"Memory usage after loading hf model must be increased! before: {_mem_usage} vs after: {_curr_usage}"

        # 4 - resume sglang weights and update the weights
        _mem_usage = get_gpu_memory_gb(rank)
        print(f"GPU{rank} Memory usage after loading hf model: {_mem_usage}")
        engine.resume_memory_occupation(tags=["weights"])
        engine.update_weights_from_tensor(
            named_tensors=list(hf_model.named_parameters())
        )

        # 5 - release hf model
        _mem_usage = get_gpu_memory_gb(rank)
        print(f"GPU{rank} Memory usage after resuming Sgl weights: {_mem_usage}")
        del hf_model
        hf_model = None
        torch.cuda.empty_cache()
        time.sleep(3)
        torch.cuda.empty_cache()
        _curr_usage = get_gpu_memory_gb(rank)
        assert (
            _curr_usage < _mem_usage
        ), f"Memory usage after releasing hf model must be reduced! before: {_mem_usage} vs after: {_curr_usage}"

        # 6 - resume slgang kv cache
        _mem_usage = get_gpu_memory_gb(rank)
        print(f"GPU{rank} Memory usage after releasing hf model: {_mem_usage}")
        engine.resume_memory_occupation(tags=["kv_cache"])
        _curr_usage = get_gpu_memory_gb(rank)
        assert (
            _curr_usage > _mem_usage
        ), f"Memory usage after resuming kv cache must be increased! before: {_mem_usage} vs after: {_curr_usage}"

        # 7 - Final checking!
        _mem_usage = get_gpu_memory_gb(rank)
        print(f"GPU{rank} Memory usage after resuming Sgl KV cache: {_mem_usage}")

        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()

    if engine:
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
