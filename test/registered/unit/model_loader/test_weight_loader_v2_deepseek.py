# Copyright 2026 SGLang Team
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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import Parameter

from sglang.srt.model_loader.auto_loader import (
    ExpertParamsDispatch,
    MultiInputFusion,
    _deepseek_mla_remap,
    maybe_remap_deepseek_mla_kv_scale,
    remap_fused_shared_expert_names,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDeepseekSharedExpertRemap(CustomTestCase):
    def test_remap_fused_shared_expert_names(self):
        weights = [
            (
                "model.layers.3.mlp.shared_experts.gate_proj.weight",
                torch.zeros(2),
            ),
        ]
        out = list(remap_fused_shared_expert_names(weights, n_routed_experts=256))
        self.assertEqual(
            out[0][0],
            "model.layers.3.mlp.experts.256.gate_proj.weight",
        )

    def test_registry_shared_expert_substr(self):
        model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=128))
        mapper = _deepseek_mla_remap(model)
        mapped = list(
            mapper.apply(
                [
                    (
                        "model.layers.0.mlp.shared_experts.down_proj.weight",
                        torch.zeros(1),
                    )
                ]
            )
        )
        self.assertEqual(
            mapped[0][0],
            "model.layers.0.mlp.experts.128.down_proj.weight",
        )


class TestDeepseekMlaKvScaleRemap(CustomTestCase):
    def test_maybe_remap_deepseek_mla_kv_scale(self):
        params = {
            "model.layers.0.self_attn.attn_mqa.k_scale": Parameter(torch.ones(1)),
        }
        name = "model.layers.0.self_attn.kv_a_proj_with_mqa.k_scale"
        self.assertEqual(
            maybe_remap_deepseek_mla_kv_scale(name, params),
            "model.layers.0.self_attn.attn_mqa.k_scale",
        )


class TestMultiInputFusionMla(CustomTestCase):
    def test_fuses_q_and_kv_a_proj(self):
        loaded: list[torch.Tensor] = []

        def _wl(param, weight):
            loaded.append(weight.clone())

        fused = Parameter(torch.zeros(6, 4))
        fused.weight_loader = _wl
        prefix = "model.layers.0.self_attn"
        params = {f"{prefix}.fused_qkv_a_proj_with_mqa.weight": fused}
        fusion = MultiInputFusion(
            source_substrs=("q_a_proj", "kv_a_proj_with_mqa"),
            fused_param_substr="fused_qkv_a_proj_with_mqa",
            cat_dim=0,
        )
        q = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        kv = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)
        self.assertIsNone(fusion.try_load(f"{prefix}.q_a_proj.weight", q, params))
        r2 = fusion.try_load(f"{prefix}.kv_a_proj_with_mqa.weight", kv, params)
        self.assertEqual(r2, f"{prefix}.fused_qkv_a_proj_with_mqa.weight")
        torch.testing.assert_close(loaded[0], torch.cat([q, kv], dim=0))


class TestExpertParamsDispatch(CustomTestCase):
    def test_routes_expert_gate_proj(self):
        calls = []

        def _wl(param, tensor, qualname, shard_id=None, expert_id=None):
            calls.append((qualname, shard_id, expert_id))

        param = Parameter(torch.zeros(1))
        param.weight_loader = _wl
        params = {"model.layers.1.mlp.experts.w13_weight": param}
        dispatch = ExpertParamsDispatch(
            mappings=(("experts.w13_", "experts.3.gate_proj.", 3, "w1"),)
        )
        name = "model.layers.1.mlp.experts.3.gate_proj.weight"
        target = dispatch.try_load(name, torch.ones(1), params)
        self.assertEqual(target, "model.layers.1.mlp.experts.w13_weight")
        self.assertEqual(calls[0][1], "w1")


class TestDeepseekV2WeightLoaderV2Gate(CustomTestCase):
    def test_v2_env_dispatches_to_do_load_weights_v2(self):
        from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
            DeepseekV2WeightLoaderMixin,
        )

        class _Host(DeepseekV2WeightLoaderMixin):
            pass

        host = _Host()
        with patch.object(host, "_do_load_weights_v2", return_value={"x"}) as mock_v2:
            with patch(
                "sglang.srt.models.deepseek_common.deepseek_weight_loader.envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.get",
                return_value=True,
            ):
                result = host.do_load_weights([])
        mock_v2.assert_called_once()
        self.assertEqual(result, {"x"})


if __name__ == "__main__":
    unittest.main()
