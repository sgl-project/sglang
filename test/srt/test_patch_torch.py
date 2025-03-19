import traceback
import unittest

import torch.multiprocessing as mp


class TestReleaseMemoryOccupation(unittest.TestCase):
    def test_monkey_patch_torch_reductions(self):
        _test_monkey_patch_torch_reductions_core()


def _test_monkey_patch_torch_reductions_core():
    processes = []
    output_reader, output_writer = mp.Pipe(duplex=False)
    queue = mp.Queue()
    for rank in range(2):
        p = mp.Process(
            target=_run_subprocess,
            kwargs=dict(queue=queue, output_writer=output_writer),
        )
        p.start()
        processes.append(p)

    for _ in range(len(processes)):
        self.assertTrue(output_reader.recv(), f"Subprocess has error, please see logs above.")

    for p in processes:
        p.join()


def _run_subprocess(queue: mp.Queue, output_writer):
    try:
        TODO
        execution_ok = True
    except Exception as e:
        print(f"subprocess has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()


if __name__ == "__main__":
    unittest.main()
