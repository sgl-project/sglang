import unittest
from multiprocessing import Process

from sglang import Engine
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)

_TP_SIZE = 2


class TestFragment(unittest.TestCase):
    def test_fragment(self):

        processes = []
        for tp_rank in range(_TP_SIZE):
            p = Process(target=_run_subprocess, args=(tp_rank,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def _run_subprocess(tp_rank: int):
    engine = Engine(
        model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        mem_fraction_static=0.1,
        tp_size=_TP_SIZE,
        fragment_tp_rank=tp_rank,
        fragment_nccl_port=23456,
    )

    output = engine.generate(
        prompt="1+1=2, 1+2=3, 1+3=4, 1+4=",
        sampling_params=dict(max_new_tokens=16, temperature=0.0),
    )
    print(f'{tp_rank=} {output=}')

    assert output.text == '5, 1+5=6, 1+6=7,'

    engine.shutdown()


if __name__ == "__main__":
    unittest.main()