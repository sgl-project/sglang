import gc
import os
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.srt.utils import init_custom_process_group
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)

mp.set_start_method("spawn", force=True)


class TestParameterUpdateLatency(unittest.TestCase):
    @classmethod
    def init_process(
        cls, rank, world_size, param_queue, state_dict_key_to_shape, tp_size, model_name
    ):
        torch.cuda.set_device(rank)
        print(f"Testing model: {model_name}")
        sync_group = init_custom_process_group(
            backend="nccl",
            init_method="tcp://localhost:65501",
            world_size=world_size,
            rank=rank,
            group_name="sync_group",
        )

        if rank == 0:
            os.environ["NCCL_CUMEM_ENABLE"] = "0"
            os.environ["NCCL_NVLS_ENABLE"] = "0"

            # 初始化进程组
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_parameter_update_group",
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"Rank {rank} init process group time: {time_end - time_begin:.3f}s")

            # 广播参数
            print(f"Rank {rank} before barrier")
            dist.barrier(group=sync_group)
            print(f"Rank {rank} after barrier")
            torch.cuda.synchronize()
            time_begin_broadcast = time.time()
            for name, shape in state_dict_key_to_shape.items():
                torch.cuda.synchronize()
                time_begin = time.time()
                weights = torch.ones(shape, dtype=torch.bfloat16, device=f"cuda:{rank}")
                torch.distributed.broadcast(weights, src=0, group=cls.group)
                torch.cuda.synchronize()
                time_end = time.time()
                print(
                    f"Rank {rank} broadcast {name} {shape} time: {time_end - time_begin:.3f}s"
                )
            torch.cuda.synchronize()
            time_end_broadcast = time.time()
            print(
                f"Rank {rank} broadcast all parameters time: {time_end_broadcast - time_begin_broadcast:.3f}s"
            )

            param_queue.put(("rank0_done", True))

        elif rank == 1:
            # 初始化引擎
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.engine = sgl.Engine(
                model_path=model_name,
                random_seed=42,
                base_gpu_id=rank,
                tp_size=tp_size,
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"Rank {rank} init engine time: {time_end - time_begin:.3f}s")

            # 初始化参数更新组
            torch.cuda.synchronize()
            time_begin = time.time()
            cls.engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_parameter_update_group",
                backend="nccl",
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"Rank {rank} init parameter update group time: {time_end - time_begin:.3f}s"
            )

            # 更新参数并测量时间
            print(f"Rank {rank} before barrier")
            dist.barrier(group=sync_group)
            print(f"Rank {rank} after barrier")
            torch.cuda.synchronize()
            time_begin_update = time.time()
            for name, shape in state_dict_key_to_shape.items():
                torch.cuda.synchronize()
                time_begin = time.time()
                cls.engine.update_parameter_from_distributed(
                    name, dtype=torch.bfloat16, shape=shape, empty_cache=True
                )
                torch.cuda.synchronize()
                time_end = time.time()
                print(
                    f"Rank {rank} update {name} {shape} time: {time_end - time_begin:.3f}s"
                )
            torch.cuda.synchronize()
            time_end_update = time.time()
            print(
                f"Rank {rank} update all parameters time: {time_end_update - time_begin_update:.3f}s"
            )

            param_queue.put(("rank1_done", True))
            cls.engine.shutdown()

    @classmethod
    def setUpClass(cls):
        assert torch.cuda.device_count() >= 2, "At least 2 GPUs are required"
        cls.test_suits = [1]
        cls.model_names = [
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_MODEL_NAME_FOR_TEST,
        ]

        if torch.cuda.device_count() >= 4:
            cls.test_suits.append(2)

        # 初始化每个模型的 state_dict_key_to_shape
        cls.model_state_dict_shapes = {}
        for model_name in cls.model_names:
            torch.cuda.synchronize()
            time_begin = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="bfloat16"
            ).to("cuda:0")
            state_dict = model.state_dict()
            cls.model_state_dict_shapes[model_name] = {
                key: state_dict[key].shape for key in state_dict.keys()
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time_end = time.time()
            print(
                f"Initialize state dict shapes for {model_name} time: {time_end - time_begin:.3f}s"
            )

    def test_parameter_update_latency(self):
        for model_name in self.model_names:
            print(f"Testing model: {model_name}")
            state_dict_key_to_shape = self.model_state_dict_shapes[model_name]

            for tp_size in self.test_suits:
                print(f"test tp_size: {tp_size}")
                world_size = 1 + tp_size
                param_queue = mp.Queue()
                results = {}

                torch.cuda.synchronize()
                time_begin = time.time()

                context = mp.spawn(
                    self.init_process,
                    args=(
                        world_size,
                        param_queue,
                        state_dict_key_to_shape,
                        tp_size,
                        model_name,
                    ),
                    nprocs=2,
                    join=False,
                )

                while len(results) < 2:
                    try:
                        key, value = param_queue.get(timeout=5)
                        results[key] = value
                    except Exception as e:
                        if all(not p.is_alive() for p in context.processes):
                            break

                context.join()
                torch.cuda.synchronize()
                time_end = time.time()
                print(f"Total time for {model_name}: {time_end - time_begin:.3f}s")

                if len(results) != 2:
                    raise RuntimeError(f"Expected 2 results but got {len(results)}")

                del context
                param_queue.close()
                param_queue.join_thread()
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
