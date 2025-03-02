# Copyright (c) OpenMMLab. All rights reserved.
import json
import threading
from functools import partial
from queue import Queue

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


def prepare_tensor(name, input_tensor):
    """Create grpcclient's InferInput instance according to a given tensor."""
    t = grpcclient.InferInput(
        name, list(input_tensor.shape), np_to_triton_dtype(input_tensor.dtype)
    )
    t.set_data_from_numpy(input_tensor)
    return t


def stream_callback(que, result, error):
    """Callback function invoked by triton."""
    que.put((result, error))


def process_result(que):
    """Process and print results in queue."""
    while True:
        res = que.get()
        if res is not None:
            result, err = res
            if err is not None:
                print(err)
            else:
                response = result.as_numpy("response")
                print(f"generated text: {response}")
        else:
            break


if __name__ == "__main__":
    model_name = "Qwen2-7B"
    prompts = json.dumps([{"role": "user", "content": "please introduce yourself"}])
    max_tokens = 512
    temperature = 1.0
    top_p = 1.0
    top_k = 1
    ignore_eos = False
    stream = True

    res_que = Queue()

    process_thread = threading.Thread(target=process_result, args=(res_que,))
    process_thread.start()
    with grpcclient.InferenceServerClient("0.0.0.0:33337") as client:
        inputs = [
            prepare_tensor("prompt", np.array([prompts.encode()], dtype=np.object_)),
            prepare_tensor("max_tokens", np.array([max_tokens], dtype=np.int32)),
            prepare_tensor("temperature", np.array([temperature], dtype=np.float_)),
            prepare_tensor("top_p", np.array([top_p], dtype=np.float_)),
            prepare_tensor("top_k", np.array([top_k], dtype=np.int32)),
            prepare_tensor("ignore_eos", np.array([ignore_eos], dtype=np.bool_)),
            prepare_tensor("stream", np.array([stream], dtype=np.bool_)),
        ]

        # async_stream
        client.start_stream(partial(stream_callback, res_que))
        client.async_stream_infer(
            model_name, inputs, sequence_start=True, sequence_end=True
        )

    res_que.put(None)
    process_thread.join()
