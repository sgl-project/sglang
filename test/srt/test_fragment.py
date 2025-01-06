import multiprocessing as mp
import unittest
from multiprocessing import Process

from sglang import Engine
from sglang.srt.server_args import get_random_available_port
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

_TP_SIZE = 2


class TestFragment(unittest.TestCase):
    def test_fragment(self):
        fragment_nccl_port = get_random_available_port()

        processes = []
        readers = []
        for tp_rank in range(_TP_SIZE):
            reader, writer = mp.Pipe(duplex=False)
            p = Process(
                target=_run_subprocess, args=(tp_rank, fragment_nccl_port, writer)
            )
            p.start()
            processes.append(p)
            readers.append(reader)

        for reader in readers:
            output_text = reader.recv()
            self.assertEqual(output_text, "5, 1+5=6, 1+6=7,")

        for p in processes:
            p.join()


def _run_subprocess(tp_rank: int, fragment_nccl_port: int, writer):
    print(f"run_subprocess[{tp_rank=}] Start")

    engine = Engine(
        model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        mem_fraction_static=0.1,
        tp_size=_TP_SIZE,
        fragment_tp_rank=tp_rank,
        fragment_nccl_port=fragment_nccl_port,
    )
    print(f"run_subprocess[{tp_rank=}] Initialized {engine=}")

    output = engine.generate(
        prompt="1+1=2, 1+2=3, 1+3=4, 1+4=",
        sampling_params=dict(max_new_tokens=16, temperature=0.0),
    )
    print(f"{tp_rank=} {output=} {output['text']=}")

    writer.send(output["text"])
    writer.close()

    print(f"run_subprocess[{tp_rank=}] engine.shutdown")
    engine.shutdown()


if __name__ == "__main__":
    unittest.main()
