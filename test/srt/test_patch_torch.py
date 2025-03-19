import os
import traceback
import unittest
from typing import List

import torch.multiprocessing as mp


class TestReleaseMemoryOccupation(unittest.TestCase):
    def test_monkey_patch_torch_reductions(self):
        self._test_monkey_patch_torch_reductions_core()

    def _test_monkey_patch_torch_reductions_core(self):
        print(f'test_monkey_patch_torch_reductions_core {os.environ.get("CUDA_VISIBLE_DEVICES")=}')
        cuda_visible_devices_list: List[int] = \
            [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')]

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        queue = mp.Queue()
        for rank in range(2):
            os.environ['CUDA_VISIBLE_DEVICES'] = TODO
            p = mp.Process(
                target=_run_subprocess,
                kwargs=dict(rank=rank, queue=queue, output_writer=output_writer),
            )
            p.start()
            processes.append(p)

        for _ in range(len(processes)):
            self.assertTrue(output_reader.recv(), f"Subprocess has error, please see logs above.")

        for p in processes:
            p.join()


def _run_subprocess(rank: int, queue: mp.Queue, output_writer):
    print(f'subprocess[{rank}] start {os.environ.get("CUDA_VISIBLE_DEVICES")=}', flush=True)
    try:
        TODO
        execution_ok = True
    except Exception as e:
        print(f"subprocess[{rank}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()


if __name__ == "__main__":
    unittest.main()
