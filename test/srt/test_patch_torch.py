import unittest

import torch.multiprocessing as mp


class TestReleaseMemoryOccupation(unittest.TestCase):
    def test_monkey_patch_torch_reductions(self):
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


if __name__ == "__main__":
    unittest.main()
