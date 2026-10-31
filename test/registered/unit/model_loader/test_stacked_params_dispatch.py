# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-b-test-cpu")

import unittest

import torch
from torch import nn

from sglang.srt.model_loader.auto_loader import (
    STANDARD_GATE_UP_MAPPING,
    STANDARD_QKV_MAPPING,
    StackedParamsDispatch,
    load_with_stacked_dispatch,
)


class _ParamWithLoader(nn.Parameter):
    def __new__(cls, data, weight_loader=None):
        param = super().__new__(cls, data)
        param.weight_loader = weight_loader or (lambda p, t, *a: p.copy_(t))
        return param


class TestStackedParamsDispatch(unittest.TestCase):
    def test_try_load_qkv_shard(self):
        qkv = _ParamWithLoader(torch.zeros(6, 4))
        params = {"qkv_proj.weight": qkv}
        calls = []

        def wl(param, tensor, shard_id):
            calls.append(shard_id)

        qkv.weight_loader = wl

        target = STANDARD_QKV_MAPPING.try_load(
            "q_proj.weight", torch.ones(2, 4), params
        )
        self.assertEqual(target, "qkv_proj.weight")
        self.assertEqual(calls, ["q"])

    def test_try_load_returns_target_when_param_missing(self):
        target = STANDARD_QKV_MAPPING.try_load("q_proj.weight", torch.ones(2, 4), {})
        self.assertEqual(target, "qkv_proj.weight")

    def test_try_load_no_match(self):
        self.assertIsNone(
            STANDARD_QKV_MAPPING.try_load("o_proj.weight", torch.ones(2, 4), {})
        )

    def test_gate_up_mapping(self):
        gate_up = _ParamWithLoader(torch.zeros(8, 4))
        params = {"gate_up_proj.weight": gate_up}
        shard_ids = []

        def wl(param, tensor, shard_id):
            shard_ids.append(shard_id)

        gate_up.weight_loader = wl

        target = STANDARD_GATE_UP_MAPPING.try_load(
            "up_proj.weight", torch.ones(4, 4), params
        )
        self.assertEqual(target, "gate_up_proj.weight")
        self.assertEqual(shard_ids, [1])

    def test_load_with_stacked_dispatch_direct_param(self):
        linear = nn.Linear(3, 2, bias=False)
        linear.weight = nn.Parameter(torch.zeros(3, 2), requires_grad=False)
        linear.weight.weight_loader = lambda p, t: p.data.copy_(t)
        module = nn.Module()
        module.down_proj = linear

        loaded = load_with_stacked_dispatch(
            module,
            [("down_proj.weight", torch.ones(3, 2))],
            StackedParamsDispatch(mappings=()),
        )
        self.assertIn("down_proj.weight", loaded)
        self.assertTrue(torch.allclose(linear.weight.data, torch.ones(3, 2)))

    def test_load_with_stacked_dispatch_stacked_then_direct(self):
        qkv_linear = nn.Linear(2, 6, bias=False)
        o_linear = nn.Linear(2, 3, bias=False)
        qkv_linear.weight = nn.Parameter(torch.zeros(6, 2), requires_grad=False)
        o_linear.weight = nn.Parameter(torch.zeros(3, 2), requires_grad=False)
        qkv_calls = []

        def qkv_wl(param, tensor, shard_id):
            qkv_calls.append(shard_id)

        qkv_linear.weight.weight_loader = qkv_wl
        o_linear.weight.weight_loader = lambda p, t: p.data.copy_(t)

        module = nn.Module()
        module.qkv_proj = qkv_linear
        module.o_proj = o_linear

        loaded = load_with_stacked_dispatch(
            module,
            [
                ("q_proj.weight", torch.ones(2, 2)),
                ("o_proj.weight", torch.full((3, 2), 2.0)),
            ],
            STANDARD_QKV_MAPPING,
        )
        self.assertEqual(loaded, {"qkv_proj.weight", "o_proj.weight"})
        self.assertEqual(qkv_calls, ["q"])


if __name__ == "__main__":
    unittest.main()
