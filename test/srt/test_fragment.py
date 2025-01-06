import unittest
from multiprocessing import Process


class TestFragment(unittest.TestCase):
    def test_fragment(self):
        tp_size = 2

        processes = []
        for tp_rank in range(tp_size):
            p = Process(target=_run_subprocess, args=(tp_rank,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def _run_subprocess(tp_rank: int):
    TODO


if __name__ == "__main__":
    unittest.main()
