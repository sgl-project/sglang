import gc
import multiprocessing
import os
import traceback
import unittest
from multiprocessing import Process
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM

from sglang.srt.entrypoints.engine import Engine as SglangEngine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    CustomTestCase,
    find_available_port,
)

register_cuda_ci(est_time=64, suite="stage-c-test-4-gpu-h100")

TEST_SUITE = dict(
    model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    mem_fraction_static=0.83,
    dp_size=2,
    tp_size=2,
)

# Minimum expected memory change in MB for each operation.
# Llama-3.2-1B bf16 is ~2GB total, ~1GB per TP rank.
# KV cache with mem_fraction_static=0.83 is much larger.
MIN_DELTA_MB = 200


class EngineWrapper:
    """
    A wrapper around Sglang engine to mock multi instance cases such as RL training.

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


def get_gpu_memory_mb(device_id: int) -> float:
    """Return device-level GPU memory used in MB."""
    free, total = torch.cuda.mem_get_info(device_id)
    return (total - free) / (1024**2)


def assert_memory_decreased(before_mb, after_mb, step_name):
    delta = before_mb - after_mb
    assert delta > MIN_DELTA_MB, (
        f"[{step_name}] Expected memory decrease > {MIN_DELTA_MB} MB, "
        f"got delta={delta:.0f} MB (before={before_mb:.0f}, after={after_mb:.0f})"
    )


def assert_memory_increased(before_mb, after_mb, step_name):
    delta = after_mb - before_mb
    assert delta > MIN_DELTA_MB, (
        f"[{step_name}] Expected memory increase > {MIN_DELTA_MB} MB, "
        f"got delta={delta:.0f} MB (before={before_mb:.0f}, after={after_mb:.0f})"
    )


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
        inference_device_mesh_cpu = init_device_mesh("cpu", **mesh_kwargs)

        # Only TP master ranks (rank % tp_size == 0) create the Engine and
        # measure memory. Non-master ranks share the same GPU and would see
        # device-level memory from the master's Engine workers, causing
        # unpredictable assertion results.
        is_tp_master = rank % tp_size == 0

        engine = EngineWrapper(
            model_path=model_path,
            random_seed=42,
            mem_fraction_static=mem_fraction_static,
            device_mesh_cpu=inference_device_mesh_cpu["tp"],
            base_gpu_id=base_gpu_id,
        )
        print(f"subprocess[{rank=}] engine created, {is_tp_master=}", flush=True)

        # 1 - release kv cache
        if is_tp_master:
            mem_before = get_gpu_memory_mb(rank)
            print(f"GPU{rank} before releasing KV cache: {mem_before:.0f} MB")
        engine.release_memory_occupation(tags=["kv_cache"])
        if is_tp_master:
            mem_after = get_gpu_memory_mb(rank)
            assert_memory_decreased(mem_before, mem_after, "release KV cache")

        # 2 - release sglang weights
        if is_tp_master:
            mem_before = get_gpu_memory_mb(rank)
            print(f"GPU{rank} before releasing weights: {mem_before:.0f} MB")
        engine.release_memory_occupation(tags=["weights"])
        if is_tp_master:
            mem_after = get_gpu_memory_mb(rank)
            assert_memory_decreased(mem_before, mem_after, "release weights")

        # 3 - load hf model (TP master only)
        hf_model = None
        if is_tp_master:
            mem_before = get_gpu_memory_mb(rank)
            print(f"GPU{rank} before loading HF model: {mem_before:.0f} MB")
            # Avoid device_map= which triggers accelerate dispatch hooks in
            # transformers v5, preventing clean memory release on del.
            hf_model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
                torch_dtype="bfloat16",
            ).to(f"cuda:{rank}")
            mem_after = get_gpu_memory_mb(rank)
            assert_memory_increased(mem_before, mem_after, "load HF model")
        dist.barrier(group=inference_device_mesh_cpu["tp"].get_group())

        # 4 - resume sglang weights and update from hf model
        engine.resume_memory_occupation(tags=["weights"])
        engine.update_weights_from_tensor(
            named_tensors=list(hf_model.named_parameters()) if hf_model else []
        )

        # 5 - release hf model (TP master only)
        if is_tp_master:
            mem_before = get_gpu_memory_mb(rank)
            print(f"GPU{rank} before releasing HF model: {mem_before:.0f} MB")
            del hf_model
            gc.collect()
            torch.cuda.empty_cache()
            mem_after = get_gpu_memory_mb(rank)
            assert_memory_decreased(mem_before, mem_after, "release HF model")
        dist.barrier(group=inference_device_mesh_cpu["tp"].get_group())

        # 6 - resume kv cache
        if is_tp_master:
            mem_before = get_gpu_memory_mb(rank)
            print(f"GPU{rank} before resuming KV cache: {mem_before:.0f} MB")
        engine.resume_memory_occupation(tags=["kv_cache"])
        if is_tp_master:
            mem_after = get_gpu_memory_mb(rank)
            assert_memory_increased(mem_before, mem_after, "resume KV cache")
            print(f"GPU{rank} final memory: {mem_after:.0f} MB")

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
