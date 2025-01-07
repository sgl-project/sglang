import multiprocessing
import multiprocessing as mp
import unittest
from multiprocessing import Process

from sglang import Engine
from sglang.srt.engine_fragment import EngineFragment
from sglang.srt.server_args import EngineFragmentArgs
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

_TP_SIZE = 2


class TestFragment(unittest.TestCase):
    def test_fragment(self):
        multiprocessing.set_start_method("spawn")

        fragment_args = EngineFragmentArgs.init_new(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            mem_fraction_static=0.1,
            tp_size=_TP_SIZE,
            random_seed=42,
        )

        processes = []
        reader, writer = mp.Pipe(duplex=False)
        for tp_rank in range(_TP_SIZE):
            p = Process(
                target=_run_subprocess,
                args=(tp_rank, fragment_args, writer),
            )
            p.start()
            processes.append(p)

        output = reader.recv()
        print(output)
        # TODO add assertions

        for p in processes:
            p.join()


def _run_subprocess(tp_rank: int, fragment_args, writer):
    print(f"run_subprocess[{tp_rank=}] Start")

    fragment = EngineFragment(
        tp_rank=tp_rank,
        gpu_id=tp_rank,
        fragment_args=fragment_args,
    )
    print(f"run_subprocess[{tp_rank=}] {fragment=}", flush=True)

    # Engine can be put anywhere, e.g. tp_rank=0, or other places
    if tp_rank == 0:
        engine = Engine(fragment_args=fragment_args)
        print(f"run_subprocess[{tp_rank=}] {engine=}", flush=True)

        ans = []

        for prompt in [
            ["Today is a sunny day and I like", "I have a very good idea on"],
            ["Hello, I am", "What is your name?", "Mathematics is defined as"],
        ]:
            print(f"Start generation", flush=True)
            outputs = engine.generate(
                prompt=prompt,
                sampling_params=[dict(max_new_tokens=16)] * len(prompt),
            )
            print(f"End generation {tp_rank=} {prompt=} {outputs=}", flush=True)
            ans += [o["text"] for o in outputs]

        writer.send(ans)
        writer.close()

        print(f"run_subprocess[{tp_rank=}] engine.shutdown", flush=True)
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
