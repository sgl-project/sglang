import os
import traceback
import unittest
from typing import List

import torch
import torch.multiprocessing as mp

from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=38, suite="stage-b-test-2-gpu-large")


class TestReleaseMemoryOccupation(unittest.TestCase):
    def test_monkey_patch_torch_reductions(self):
        mp.set_start_method("spawn", force=True)

        cuda_visible_devices_list: List[int] = [
            int(x)
            for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(
                ","
            )
        ]

        # Sender's cuda:1 and receiver's cuda:0 map to the same physical device.
        # With the patch, the IPC tensor must land on receiver's cuda:0.
        sender_info = dict(visible_devices=[0, 1], tensor_device=1)
        receiver_info = dict(visible_devices=[1, 0], tensor_device=0)

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        # Split into SPSC queues; a single shared mp.Queue lets the sender's
        # get() pop its own put before the receiver wakes (CUDA IPC self-reopen fails).
        tensor_queue = mp.Queue()
        ack_queue = mp.Queue()
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
                    tensor_queue=tensor_queue,
                    ack_queue=ack_queue,
                    output_writer=output_writer,
                    tensor_device=info["tensor_device"],
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
    role: str,
    tensor_queue: mp.Queue,
    ack_queue: mp.Queue,
    output_writer,
    tensor_device: int,
):
    print(
        f'subprocess[{role}] start {os.environ.get("CUDA_VISIBLE_DEVICES")=}',
        flush=True,
    )

    monkey_patch_torch_reductions()

    try:
        if role == "sender":
            tensor = torch.tensor([1.0, 2.0], device=f"cuda:{tensor_device}")
            print(f"sender tensor_queue.put {tensor=} {tensor.device=}")
            tensor_queue.put(tensor)
            assert ack_queue.get() == "done"
        elif role == "receiver":
            tensor = tensor_queue.get()
            print(f"receiver tensor_queue.get {tensor=} {tensor.device=}")
            assert str(tensor.device) == f"cuda:{tensor_device}"
            ack_queue.put("done")
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
