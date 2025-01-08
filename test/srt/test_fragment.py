import multiprocessing
import multiprocessing as mp
import traceback
import unittest
from multiprocessing import Process

from sglang.srt.server.engine_fragment import EngineFragment
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

_TP_SIZE = 2


class TestFragment(unittest.TestCase):
    def test_fragment(self):
        multiprocessing.set_start_method("spawn")
        nccl_port = 12345

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for tp_rank in range(_TP_SIZE):
            p = Process(
                target=_run_subprocess,
                args=(tp_rank, nccl_port, output_writer),
            )
            p.start()
            processes.append(p)

        outputs = [output_reader.recv() for _ in range(_TP_SIZE)]
        print(outputs)
        for output in outputs:
            self.assertEqual(
                output,
                [
                    " to spend it outdoors. I decided to take a walk in the nearby park.",
                    " how to improve the performance of my website. I've been doing some research and",
                    " a new user of the platform. I am looking for a new laptop to buy",
                    " I'm looking for someone to help me with a project.\nI'm a student",
                    " the science of numbers and their properties. It is a vast and complex field that",
                ],
            )

        for p in processes:
            p.join()


def _run_subprocess(tp_rank: int, nccl_port: int, output_writer):
    try:
        print(f"subprocess[{tp_rank=}] Start")

        fragment = EngineFragment(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            mem_fraction_static=0.1,
            tp_size=_TP_SIZE,
            random_seed=42,
            # fragment args
            tp_rank=tp_rank,
            gpu_id=tp_rank,
            nccl_port=nccl_port,
        )
        print(f"subprocess[{tp_rank=}] {fragment=}", flush=True)

        # NOTE: We deliberately call fragment.generate *twice* to confirm this function can be called multiple times
        # In real batch generation, surely we should only call fragment.generate once
        ans = []
        for prompt in [
            ["Today is a sunny day and I like", "I have a very good idea on"],
            ["Hello, I am", "What is your name?", "Mathematics is defined as"],
        ]:
            print(f"subprocess[{tp_rank=}] Start generation", flush=True)
            outputs = fragment.generate(
                prompt=prompt,
                sampling_params=[dict(max_new_tokens=16, temperature=0.0)] * len(prompt),
            )
            print(f"subprocess[{tp_rank=}] End generation {prompt=} {outputs=}", flush=True)
            ans += [o["text"] for o in outputs]

        output_writer.send(ans)
        output_writer.close()

    except Exception as e:
        print(f"subprocess[{tp_rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        raise

    print(f"subprocess[{tp_rank=}] end", flush=True)
    # TODO Wait for more real tests, if it does not exit there, then we should fix it
    exit(0)


if __name__ == "__main__":
    unittest.main()
