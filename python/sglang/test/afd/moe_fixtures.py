from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.afd import config as afd_config

nn = torch.nn


def bare_module(cls):
    value = cls.__new__(cls)
    nn.Module.__init__(value)
    return value


class ModelBoundaries(dict):
    """Patch shared model boundary names with pytest-managed restoration."""

    def __init__(self, monkeypatch, modules, values):
        super().__init__(values)
        self.monkeypatch = monkeypatch
        self.modules = modules

    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        for module in self.modules:
            if hasattr(module, name):
                self.monkeypatch.setattr(module, name, value)

    def update(self, **values):
        for name, value in values.items():
            self[name] = value


@pytest.fixture
def routing(monkeypatch):
    import inspect

    from sglang.srt.afd import model_proxy
    from sglang.srt.layers.moe import topk
    from sglang.srt.models import deepseek_v2 as deepseek
    from sglang.srt.models import glm4_moe as glm

    values = {
        name: getattr(topk, name)
        for name in ("TopKConfig", "TopKOutputFormat", "StandardTopKOutput")
    }
    values["biased_grouped_topk_impl"] = inspect.unwrap(topk.biased_grouped_topk_impl)
    values.update(
        {
            name: getattr(deepseek, name)
            for name in (
                "DeepseekV2DecoderLayer",
                "DeepseekV2MoE",
                "DeepseekV2ForCausalLM",
            )
        }
    )
    values.update(
        {
            name: getattr(glm, name)
            for name in (
                "GlmMoeDsaAFDDecoderLayer",
                "GlmMoeDsaForCausalLM",
                "make_glm_dsa_attention_mlp",
            )
        }
    )
    values.update(
        AFDProxyMLP=model_proxy.AFDProxyMLP,
        AFDProxyAttention=model_proxy.AFDProxyAttention,
    )
    namespace = ModelBoundaries(monkeypatch, (deepseek, glm), values)
    args = SimpleNamespace(
        enable_eplb=False,
        init_expert_location="trivial",
        ep_num_redundant_experts=0,
        enable_waterfill=False,
        afd_config=afd_config.AFDConfig(),
    )
    namespace.update(
        afd_execution_mode=lambda: afd_config.AFDExecutionMode.FFN,
        get_moe_a2a_backend=lambda: SimpleNamespace(
            is_none=lambda: True, is_deepep=lambda: False
        ),
        get_server_args=lambda: args,
        get_moe_runner_backend=lambda: SimpleNamespace(value="triton"),
        use_intel_amx_backend=lambda _: False,
        _use_mnnvl_cutedsl_fusion=lambda: False,
        _is_cuda=True,
        _is_musa=False,
        _is_xpu=False,
        _use_aiter=False,
        maybe_fuse_routed_scale_and_shared_add=lambda experts, routed, shared, scale: (
            routed if shared is None else routed + shared
        ),
        should_skip_post_experts_all_reduce=lambda **kwargs: False,
        tensor_model_parallel_all_reduce=lambda x: x,
        post_experts_all_reduce=lambda x: x,
        should_add_replicated_moe_output=lambda: True,
        ExpertLocationDispatchInfo=SimpleNamespace(init_new=lambda **kwargs: None),
    )
    namespace["get_exec"] = lambda: SimpleNamespace(moe=namespace["get_server_args"]())
    namespace["get_forward"] = lambda: SimpleNamespace(flashinfer_trtllm_bypass=False)
    namespace["get_parallel"] = lambda: SimpleNamespace(enable_prefill_cp=False)
    namespace["get_spec"] = lambda: SimpleNamespace(speculative_algorithm=None)
    mega_moe = ModuleType("sglang.srt.layers.moe.mega_moe")
    mega_moe.should_use_mega_moe = lambda *args: False
    mega_moe.forward_mega_moe = lambda *args, **kwargs: pytest.fail("MegaMoE called")
    monkeypatch.setitem(sys.modules, mega_moe.__name__, mega_moe)
    with torch.inference_mode():
        yield namespace


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(19)
        self.weight = nn.Parameter(torch.randn(8, 4, generator=generator))
        self.e_score_correction_bias = nn.Parameter(
            torch.tensor([0.8, -0.7, 0.6, 0.1, -0.5, 0.3, -0.4, 0.2])
        )
        self.calls = 0
        self.e_score_correction_bias_vl = None

    def forward(self, x, allocator=None, forward_batch=None):
        self.calls += 1
        return torch.nn.functional.linear(x, self.weight)


class SelectedTopK(nn.Module):
    def __init__(self, namespace, bias):
        super().__init__()
        self.namespace = namespace
        self.calls = 0
        self.topk_config = namespace["TopKConfig"](
            top_k=2,
            use_grouped_topk=True,
            num_expert_group=2,
            topk_group=1,
            renormalize=True,
            correction_bias=bias,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
        )

    def forward(self, hidden, logits, **kwargs):
        self.calls += 1
        c = self.topk_config
        weights, ids = self.namespace["biased_grouped_topk_impl"](
            hidden,
            logits,
            c.correction_bias,
            c.top_k,
            c.renormalize,
            c.num_expert_group,
            c.topk_group,
            num_fused_shared_experts=c.num_fused_shared_experts,
            routed_scaling_factor=c.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=c.apply_routed_scaling_factor_on_output,
        )
        return self.namespace["StandardTopKOutput"](weights, ids, logits)

    def empty_topk_output(self, device, layer_id=None):
        return self.namespace["StandardTopKOutput"](
            torch.empty(0, self.topk_config.top_k, dtype=torch.float32, device=device),
            torch.empty(0, self.topk_config.top_k, dtype=torch.int32, device=device),
            None,
        )


def _moe(namespace, *, prescaled=False, inplace=False, tp_size=1, shared_tp1=False):
    gate = Gate()
    topk = SelectedTopK(namespace, gate.e_score_correction_bias)
    topk.topk_config.apply_routed_scaling_factor_on_output = prescaled
    generator = torch.Generator().manual_seed(73)
    expert_weights = torch.randn(8, 4, 4, generator=generator)

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_runner_config = SimpleNamespace(inplace=inplace)
            self.quant_method = object()
            self.seen = []

        def forward(self, x, selected):
            self.seen.append(selected)
            per_expert = torch.einsum("ri,ek i->rek", x, expert_weights)
            chosen = per_expert.gather(
                1, selected.topk_ids.long().unsqueeze(-1).expand(-1, -1, 4)
            )
            output = (chosen * selected.topk_weights.unsqueeze(-1)).sum(1)
            return output if prescaled else output * 2.5

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = gate
            self.topk = topk
            self.experts = Experts()
            self.top_k = 2
            self.layer_id = 0
            self.is_nextn = self.is_hash = self.is_deepseek_v4 = False
            self._fuse_shared_experts_inside_sbo = False
            self._shared_expert_tp1 = shared_tp1
            self.tp_size = tp_size
            self.num_fused_shared_experts = 0
            self.routed_scaling_factor = 2.5

        def _maybe_quant_moe_input_once(self, x):
            return None

        def _forward_shared_experts(self, x, allocator=None, *, pre_quant_input=None):
            return x * 0.125

        def forward(self, x, forward_batch=None):
            return namespace["DeepseekV2MoE"].forward_normal(self, x)

    return MoE(), expert_weights
