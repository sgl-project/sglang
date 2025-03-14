import gc
import time
import unittest
import pickle
import base64
import numpy as np
import torch
import requests

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
    is_in_ci,
)

from sglang.utils import terminate_process

def _check_param_server(param_name, expect_values):
    response = requests.get(
        f"{DEFAULT_URL_FOR_TEST}/get_weights_by_name",
        json={"name": param_name, "truncate_size": len(expect_values)},
        timeout=20,
    )
    response.raise_for_status()

    returned_data = response.json()[0]  # The weight values returned by the server
    
    # If the data is 2D, take the first row and the first 5 values for comparison
    if isinstance(returned_data, list):
        actual_values = np.array(returned_data, dtype=np.float32)
        if actual_values.ndim == 2:  # The server returned (N, D)
            actual_values = actual_values[0, :5]  # Take the first row and first 5 values
        else:
            actual_values = actual_values[:5]  # Take the first 5 elements
    else:
        raise ValueError(f"Unexpected data format from server: {returned_data}")

    expected_values = np.array(expect_values, dtype=np.float32)

    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Server check failed for {param_name}. Actual={actual_values}, Expected={expected_values}"
    )


def update_weights_server(param_list, tp_size, load_format=None):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": "Bearer None",
    }
    for idx, (param_name, tensor) in enumerate(param_list):
        gathered_serialized_tensors = [tensor for _ in range(tp_size)]
        send_data = [param_name, gathered_serialized_tensors]
        raw_bytes = pickle.dumps(send_data)
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        post_payload = {
            "serialized_named_tensors": b64_str,
            "load_format": load_format,
            "flush_cache": (idx == len(param_list) - 1),
        }
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/update_weights_from_tensor",
            json=post_payload,
            headers=headers,
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Server update failed for {param_name}: {resp.text}")

class TestUpdateWeightsFromTensorServer(unittest.TestCase):
    def _test_update(self, backend, load_format=None, tp_size=2, dp_size=1, model_name=DEFAULT_SMALL_MODEL_NAME_FOR_TEST):

        param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)]
        new_tensor_shape = (16384, 2048)
        new_tensor = torch.full(new_tensor_shape, 1.5, dtype=torch.float32)
        param_list = [(name, new_tensor.clone()) for name in param_names]
        process = popen_launch_server(
            model=model_name,
            base_url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=("--tp-size", str(tp_size), "--dp-size", str(dp_size)),
        )
        try:
            _check_param_server(param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110])
            update_weights_server(param_list, tp_size=tp_size, load_format=load_format)
            for name in param_names[:3]:
                _check_param_server(name, [1.5] * 5)
        finally:
            terminate_process(process)
            gc.collect()
            torch.cuda.empty_cache()


    def test_server_update_default(self):
        if is_in_ci():
            import random
            test_setting = random.sample([(1,1),
                                          (1,2),
                                          (2,1),
                                          (2,2),
                                          ], 1)
        else:
            test_setting = [(1,1),
                            (2,1),
                            ]
        for tp_size, dp_size in test_setting:
            print(f"test_setting: {tp_size}, {dp_size}")
            self._test_update(backend="Server", tp_size=tp_size, dp_size=dp_size, load_format=None)



if __name__ == "__main__":
    unittest.main()
