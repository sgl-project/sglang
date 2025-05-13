import unittest

import torch.multiprocessing as mp
from sglang.test.test_utils import CustomTestCase
from torch.multiprocessing import Process


class TestExpertLocationUpdater(CustomTestCase):
    def _test_core(
        self,
        num_gpus: int,
    ):
        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for rank in range(num_gpus):
            p = Process(
                target=_run_subprocess,
                kwargs=dict(
                    rank=rank,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(num_gpus):
            self.assertTrue(output_reader.recv(), f"Subprocess has error, please see logs above.")

        for p in processes:
            p.join()


if __name__ == "__main__":
    unittest.main()
