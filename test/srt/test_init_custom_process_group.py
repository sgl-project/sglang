import time
import unittest

import torch
import torch.multiprocessing as mp

from sglang.srt.utils import init_custom_process_group
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)

mp.set_start_method("spawn", force=True)


class TestProcessGroupInit(unittest.TestCase):
    @classmethod
    def init_process(cls, rank, world_size):
        torch.cuda.set_device(rank)

        if rank == 0:
            # 初始化进程组
            print(f"rank {rank} init custom process group")
            torch.cuda.synchronize()
            time_begin = time.time()
            group = init_custom_process_group(
                backend="nccl",
                init_method="tcp://localhost:65500",
                world_size=world_size,
                rank=rank,
                group_name="test_process_group",
            )

            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} init process group time: {time_end - time_begin:.3f}s")

        elif rank == 1:
            # 初始化引擎的进程组
            print(f"rank {rank} init parameter update group")
            torch.cuda.synchronize()
            time_begin = time.time()
            from sglang import Engine

            engine = Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,  # 使用小模型测试
                random_seed=42,
                base_gpu_id=rank,
                tp_size=1,
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} init engine time: {time_end - time_begin:.3f}s")
            torch.cuda.synchronize()
            time_begin = time.time()
            engine.init_parameter_update_group(
                master_address="localhost",
                master_port="65500",
                rank_offset=1,
                world_size=world_size,
                group_name="test_process_group",
                backend="nccl",
            )
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"rank {rank} init process group time: {time_end - time_begin:.3f}s")

            engine.shutdown()

    def test_process_group_init(self):
        assert torch.cuda.device_count() >= 2, "需要至少2个GPU"

        torch.cuda.synchronize()
        time_begin = time.time()

        context = mp.spawn(
            self.init_process, args=(2,), nprocs=2, join=True  # world_size = 2
        )

        torch.cuda.synchronize()
        time_end = time.time()
        print(f"总耗时: {time_end - time_begin:.3f}s")


if __name__ == "__main__":
    unittest.main()
