import multiprocessing
import multiprocessing as mp
import traceback
import unittest
from multiprocessing import Process

from sglang import Engine
from sglang.srt.server.engine_fragment import EngineFragment
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

_TP_SIZE = 2


class TestFragment(unittest.TestCase):
    def test_fragment(self):
        multiprocessing.set_start_method("spawn")

        queue = multiprocessing.Queue()

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for tp_rank in range(_TP_SIZE):
            p = Process(
                target=_run_subprocess,
                args=(tp_rank, queue, output_writer),
            )
            p.start()
            processes.append(p)

        output = output_reader.recv()
        print(output)
        # TODO add assertions

        for p in processes:
            p.join()


def _run_subprocess(tp_rank: int, queue: multiprocessing.Queue, output_writer):
    try:
        print(f"subprocess[{tp_rank=}] Start")

        # Engine can be put anywhere, e.g. tp_rank=0, or other places
        if tp_rank == 0:
            engine = Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                mem_fraction_static=0.1,
                tp_size=_TP_SIZE,
                random_seed=42,
                fragment=True,
            )
            print(f"subprocess[{tp_rank=}] {engine=}", flush=True)

            for _ in range(_TP_SIZE):
                queue.put(engine.fragment_args)

        # can use e.g. torch.distributed to broadcast it; here for simplicity we do not init torch.distributed
        fragment_args = queue.get()

        fragment = EngineFragment(
            fragment_args=fragment_args,
            tp_rank=tp_rank,
            gpu_id=tp_rank,
        )
        print(f"subprocess[{tp_rank=}] {fragment=}", flush=True)

        if tp_rank == 0:
            engine.await_fragments()
            print(f"subprocess[{tp_rank=}] end wait engine launch", flush=True)

            ans = []
            for prompt in [
                ["Today is a sunny day and I like", "I have a very good idea on"],
                ["Hello, I am", "What is your name?", "Mathematics is defined as"],
            ]:
                print(f"subprocess[{tp_rank=}] Start generation", flush=True)
                outputs = engine.generate(
                    prompt=prompt,
                    sampling_params=[dict(max_new_tokens=16)] * len(prompt),
                )
                print(f"subprocess[{tp_rank=}] End generation {tp_rank=} {prompt=} {outputs=}", flush=True)
                ans += [o["text"] for o in outputs]

            output_writer.send(ans)
            output_writer.close()

            print(f"subprocess[{tp_rank=}] engine.shutdown", flush=True)
            engine.shutdown()

            for _ in range(_TP_SIZE):
                queue.put('LEAVE')

        # Again, can use torch barrier
        assert queue.get() == 'LEAVE'

    except Exception as e:
        print(f"subprocess[{tp_rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    unittest.main()
