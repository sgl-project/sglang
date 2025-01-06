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

        outputs = [reader.recv() for reader in readers]
        for output in outputs:
            self.assertEqual(outputs[0], output)

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
        fragment_gpu_id=tp_rank,
    )
    print(f"run_subprocess[{tp_rank=}] Initialized {engine=}")

    ans = []

    for prompt in [
        ['Today is a sunny day and I like', 'I have a very good idea on'],
        ['Hello, I am', 'What is your name?', 'Mathematics is defined as'],
    ]:
        output = engine.generate(
            prompt=prompt,
            sampling_params=[dict(max_new_tokens=16)] * len(prompt),
        )
        print(f"{tp_rank=} {prompt=} {output=} {output['text']=}")
        ans.append(output['text'])

    writer.send(ans)
    writer.close()

    print(f"run_subprocess[{tp_rank=}] engine.shutdown")
    engine.shutdown()


if __name__ == "__main__":
    unittest.main()
