import traceback
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
                    output_writer=output_writer,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(num_gpus):
            self.assertTrue(output_reader.recv(), f"Subprocess has error, please see logs above.")

        for p in processes:
            p.join()


def _run_subprocess(
    rank: int,
    output_writer,
):
    try:
        TODO

        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()


if __name__ == "__main__":
    unittest.main()
