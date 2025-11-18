import os
import traceback
import unittest
from typing import Dict, List

import torch
import torch.multiprocessing as mp

from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions


class TestReleaseMemoryOccupation(unittest.TestCase):
    def test_monkey_patch_torch_reductions(self):
        mp.set_start_method("spawn", force=True)

        for enable_patch in [False, True]:
            for params in [
                # Same visible devices
                dict(
                    sender_info=dict(
                        visible_devices=[0, 1],
                        tensor_device=1,
                    ),
                    receiver_info=dict(
                        visible_devices=[0, 1],
                        tensor_device=1,
                    ),
                ),
                # Different visible devices
                dict(
                    sender_info=dict(
                        visible_devices=[0, 1],
                        tensor_device=1,
                    ),
                    receiver_info=dict(
                        visible_devices=[1, 0],
                        # If enable patch, this should be fixed, and cuda:1 becomes cuda:0
                        tensor_device=0 if enable_patch else 1,
                    ),
                ),
            ]:
                with self.subTest(f"{enable_patch=} {params=}"):
                    self._test_monkey_patch_torch_reductions_core(
                        enable_patch=enable_patch, **params
                    )

    def _test_monkey_patch_torch_reductions_core(
        self,
        sender_info: Dict,
        receiver_info: Dict,
        enable_patch: bool,
    ):
        print(
            f'test_monkey_patch_torch_reductions_core {os.environ.get("CUDA_VISIBLE_DEVICES")=}'
        )
        cuda_visible_devices_list: List[int] = [
            int(x)
            for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(
                ","
            )
        ]

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        queue = mp.Queue()
        for role, info in [
            ("sender", sender_info),
            ("receiver", receiver_info),
        ]:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(cuda_visible_devices_list[device])
                for device in info["visible_devices"]
            )
            p = mp.Process(
                target=_run_subprocess,
                kwargs=dict(
                    role=role,
                    queue=queue,
                    output_writer=output_writer,
                    tensor_device=info["tensor_device"],
                    enable_patch=enable_patch,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(len(processes)):
            self.assertTrue(
                output_reader.recv(), f"Subprocess has error, please see logs above."
            )

        for p in processes:
            p.join()


def _run_subprocess(
    role: str, queue: mp.Queue, output_writer, tensor_device: int, enable_patch: bool
):
    print(
        f'subprocess[{role}] start {os.environ.get("CUDA_VISIBLE_DEVICES")=}',
        flush=True,
    )

    if enable_patch:
        print(f"subprocess[{role}] execute monkey_patch_torch_reductions", flush=True)
        monkey_patch_torch_reductions()

    try:
        if role == "sender":
            tensor = torch.tensor([1.0, 2.0], device=f"cuda:{tensor_device}")
            print(f"sender queue.put {tensor=} {tensor.device=}")
            queue.put(tensor)
            assert queue.get() == "done"
        elif role == "receiver":
            tensor = queue.get()
            print(f"receiver queue.get {tensor=} {tensor.device=}")
            assert str(tensor.device) == f"cuda:{tensor_device}"
            queue.put("done")
        else:
            raise NotImplementedError

        execution_ok = True
    except Exception as e:
        print(f"subprocess[{role}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()


if __name__ == "__main__":
    unittest.main()
