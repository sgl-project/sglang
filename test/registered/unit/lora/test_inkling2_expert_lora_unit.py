"""Inkling2 routed-expert LoRA: adapter layout, scaling, EP split, token mask and output form.

The Triton deltas are swapped for their torch references (sglang.test.kernels), so these
cases pin the wrapper and memory-pool contract on CPU; kernel parity is a GPU test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

import sglang.srt.models.inkling_common.inkling2_lora as inkling2_lora
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.models.inkling_common.inkling2_lora import Inkling2ExpertsWithLoRA
from sglang.srt.models.inkling_common.inkling2_moe import Inkling2Experts, RoutedOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels import inkling2_lora_reference as ref

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_D, _F, _E, _R, _TOP_K, _ALPHA = 32, 16, 4, 16, 2, 32  # scaling alpha / r = 2
_CAPS = (4.0, 25.0)
_EXPERTS = "model.llm.layers.0.mlp.experts"


def _miles_tensors(seed=0):
    """Expert factors as miles exports them (lora_names.py: w1/w3 gate/up, w2 down)."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g) * 0.5  # noqa: E731
    return {
        f"{_EXPERTS}.w1.lora_A.weight": rnd(1, _R, _D),
        f"{_EXPERTS}.w1.lora_B.weight": rnd(_E, _F, _R),
        f"{_EXPERTS}.w3.lora_A.weight": rnd(1, _R, _D),
        f"{_EXPERTS}.w3.lora_B.weight": rnd(_E, _F, _R),
        f"{_EXPERTS}.w2.lora_A.weight": rnd(_E, _R, _F),
        f"{_EXPERTS}.w2.lora_B.weight": rnd(1, _D, _R),
    }


def _adapter(tensors):
    text = SimpleNamespace(n_shared_experts=0, architectures=None)
    adapter = LoRAAdapter(
        uid="inkling2",
        config=LoRAConfig.from_dict(
            dict(
                peft_type="LORA",
                r=_R,
                lora_alpha=_ALPHA,
                target_modules=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            )
        ),
        base_hf_config=SimpleNamespace(
            num_hidden_layers=1,
            architectures=["Inkling2LoopForConditionalGeneration"],
            get_text_config=lambda: text,
        ),
        load_config=None,
        lora_backend=None,
    )
    adapter.initialize_weights_from_tensors(dict(tensors))
    return adapter


class _FakeExperts(nn.Module):
    """Inkling2Experts' routing contract with zero base weights: rows sorted by
    expert, this rank's rows [offsets[0], offsets[-1]), output zero before LoRA."""

    def __init__(self, ep_size=1, ep_rank=0):
        super().__init__()
        self.tp_size, self.ep_size, self.ep_rank = 1, ep_size, ep_rank
        self.moe_tp_rank, self.is_shared_fused_moe = 0, False
        self.num_global_experts = _E
        self.w13_weight = torch.empty(_E // ep_size, 1, 1)
        self.gate_cap, self.up_cap = _CAPS

    def prepare_lora_prefill_exchange(self):
        pass

    def forward(self, x, weights, ids, combine=True, lora=None):
        sorted_ids, dst2src = torch.sort(ids.flatten(), stable=True)
        src2dst = torch.empty_like(dst2src)
        src2dst[dst2src] = torch.arange(dst2src.numel())
        counts = torch.bincount(sorted_ids, minlength=_E)
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        num_local = _E // self.ep_size
        offsets = offsets[self.ep_rank * num_local : (self.ep_rank + 1) * num_local + 1]
        acc = torch.zeros(ids.numel(), 2 * _F, dtype=torch.bfloat16)
        lora.gate_up_delta(
            x=x,
            acc=acc,
            dst2src=dst2src,
            src2dst=src2dst,
            top_k=ids.shape[1],
            gammas=weights.flatten()[dst2src],
            offsets=offsets,
        )
        self.last_acc = acc
        return torch.zeros_like(x)


def _pool(scaling_adapter, *, ep_size=1, ep_rank=0, local_ids=True, module=None):
    """Load the adapter into real LoRAMemoryPool buffers (shared-outer layout)."""
    pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
    pool.num_layer, pool.target_modules = 1, {"gate_up_proj", "down_proj"}
    pool.experts_shared_outer_loras, pool.strict_loading = True, True
    pool.moe_use_local_expert_ids = ep_size > 1 and local_ids
    pool.moe_ep_size, pool.moe_ep_rank = ep_size, ep_rank
    pool._num_experts_local = _E // ep_size
    pool.tp_rank, pool.tp_size, pool.moe_tp_rank, pool.moe_tp_size = 0, 1, 0, 1
    pool.lora_added_tokens_size, pool.max_rank_spaced_layout = 0, False
    pool.enable_lora_overlap_loading, pool.base_model = False, None
    pool.max_lora_rank, pool.pin_memory_available = _R, False
    local = _E // ep_size if pool.moe_use_local_expert_ids else _E
    pool.A_buffer = {
        "gate_up_proj_moe": [torch.zeros(1, 1, 2 * _R, _D, dtype=torch.bfloat16)],
        "down_proj_moe": [torch.zeros(1, local, _R, _F, dtype=torch.bfloat16)],
    }
    pool.B_buffer = {
        "gate_up_proj_moe": [torch.zeros(1, local, 2 * _F, _R, dtype=torch.bfloat16)],
        "down_proj_moe": [torch.zeros(1, 1, _D, _R, dtype=torch.bfloat16)],
    }
    pool.embedding_A_buffer, pool.embedding_B_buffer = {}, {}
    pool.lm_head_A_buffer, pool.lm_head_B_buffer, pool.new_embeddings_buffer = (
        {},
        {},
        {},
    )
    pool.load_lora_weight_to_buffer(
        uid="inkling2",
        buffer_id=0,
        lora_adapter=scaling_adapter,
        lora_modules=[{_EXPERTS: module}],
        lora_embed_tokens_module=None,
        lora_lm_head_module=None,
    )
    return pool


def _wrapper(mapping, *, ep_size=1, ep_rank=0):
    batch_info = SimpleNamespace(
        has_active_lora=True,
        moe_lora_info=SimpleNamespace(token_lora_mapping=mapping),
    )
    backend = SimpleNamespace(is_moe_lora=False, get_batch_info=lambda _: batch_info)
    return Inkling2ExpertsWithLoRA(_FakeExperts(ep_size, ep_rank), backend)


def _install(wrapper, pool):
    wrapper.set_lora_info(
        gate_up_lora_a_weights=pool.A_buffer["gate_up_proj_moe"][0],
        gate_up_lora_b_weights=pool.B_buffer["gate_up_proj_moe"][0],
        down_lora_a_weights=pool.A_buffer["down_proj_moe"][0],
        down_lora_b_weights=pool.B_buffer["down_proj_moe"][0],
    )


@pytest.fixture(autouse=True)
def _reference_kernels():
    with (
        mock.patch.multiple(
            inkling2_lora.inkling2_routed,
            routed_gate_up_delta=ref.routed_gate_up_delta,
            routed_down_inner=ref.routed_down_inner,
            routed_token_rank=ref.routed_token_rank,
        ),
        mock.patch.object(
            inkling2_lora, "unweighted_post_reorder", ref.unweighted_post_reorder
        ),
    ):
        yield


def _batch(tokens=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, _D, generator=g).bfloat16()
    ids = torch.stack([torch.randperm(_E, generator=g)[:_TOP_K] for _ in range(tokens)])
    weights = torch.rand(tokens, _TOP_K, generator=g)
    return x, ids, weights


def _reference_lora(tensors, x, ids, weights, scaling):
    """The adapter's expert contribution with zero base weights, from the raw factors."""
    a_gate, a_up = (
        tensors[f"{_EXPERTS}.w1.lora_A.weight"][0],
        tensors[f"{_EXPERTS}.w3.lora_A.weight"][0],
    )
    b_gate, b_up = (
        tensors[f"{_EXPERTS}.w1.lora_B.weight"],
        tensors[f"{_EXPERTS}.w3.lora_B.weight"],
    )
    a_down, b_down = (
        tensors[f"{_EXPERTS}.w2.lora_A.weight"],
        tensors[f"{_EXPERTS}.w2.lora_B.weight"][0],
    )
    bf = lambda t: t.bfloat16().float()  # noqa: E731  the pool stores factors in BF16
    out = torch.zeros(x.shape[0], _D)
    for t in range(x.shape[0]):
        xt = x[t].float()
        ranks = torch.zeros(_R)
        for k in range(_TOP_K):
            e = int(ids[t, k])
            gate = bf(b_gate[e] * scaling) @ (bf(a_gate) @ xt).bfloat16().float()
            up = bf(b_up[e] * scaling) @ (bf(a_up) @ xt).bfloat16().float()
            h = ref.capped_swiglu(
                gate.bfloat16().float(), up.bfloat16().float(), weights[t, k], *_CAPS
            ).bfloat16()
            ranks += bf(a_down[e]) @ h.float()
        out[t] = bf(b_down * scaling) @ ranks.bfloat16().float()
    return out


def test_miles_expert_factors_load_as_shared_outer_buffers():
    """miles' w1/w3/w2 names stack into gate_up_proj (gate rows, then up) and down_proj
    in the 3D shared-outer shapes the expert wrapper indexes."""
    tensors = _miles_tensors()
    weights = _adapter(tensors).layers[0].weights
    gate_up_a = weights[f"{_EXPERTS}.gate_up_proj.lora_A.weight"]
    gate_up_b = weights[f"{_EXPERTS}.gate_up_proj.lora_B.weight"]
    assert gate_up_a.shape == (1, 2 * _R, _D)
    assert gate_up_b.shape == (_E, 2 * _F, _R)
    torch.testing.assert_close(
        gate_up_b[:, :_F], tensors[f"{_EXPERTS}.w1.lora_B.weight"]
    )
    torch.testing.assert_close(
        gate_up_b[:, _F:], tensors[f"{_EXPERTS}.w3.lora_B.weight"]
    )
    assert weights[f"{_EXPERTS}.down_proj.lora_A.weight"].shape == (_E, _R, _F)
    assert weights[f"{_EXPERTS}.down_proj.lora_B.weight"].shape == (1, _D, _R)


def test_expert_delta_is_scaled_exactly_once():
    """alpha / r = 2 multiplies the expert contribution once: the pool folds it into
    both B factors and the wrapper's deltas must not apply it again (or skip it)."""
    tensors = _miles_tensors()
    x, ids, weights = _batch()
    wrapper = _wrapper(torch.zeros(x.shape[0], dtype=torch.int32))
    _install(wrapper, _pool(_adapter(tensors), module=wrapper))
    got = wrapper(x, weights, ids).float()
    want = _reference_lora(tensors, x, ids, weights, scaling=_ALPHA / _R)
    torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)
    assert not torch.allclose(got, 2 * want, rtol=3e-2, atol=3e-2)


def test_tokens_without_an_adapter_get_no_delta():
    tensors = _miles_tensors()
    x, ids, weights = _batch()
    mapping = torch.tensor([0, -1, 0, -1, 0], dtype=torch.int32)
    wrapper = _wrapper(mapping)
    _install(wrapper, _pool(_adapter(tensors), module=wrapper))
    out = wrapper(x, weights, ids)
    assert not out[1].any() and not out[3].any()
    assert out[0].any() and out[2].any()
    # Their gate/up rows must stay untouched too, or the base SwiGLU would see the delta.
    rows_of_token1 = wrapper.base_layer.last_acc[
        (torch.sort(ids.flatten(), stable=True)[1] // _TOP_K) == 1
    ]
    assert not rows_of_token1.any()


@pytest.mark.parametrize("local_ids", [True, False])
def test_expert_parallel_ranks_sum_to_the_single_rank_delta(local_ids):
    """Each EP rank applies its own experts' B13/A2 slice, whether the pool keeps only
    them or all experts; summing the ranks' rank-space partials (the EP all-reduce)
    before the shared B2 gives the EP=1 result."""
    tensors = _miles_tensors()
    x, ids, weights = _batch(tokens=6)
    mapping = torch.zeros(x.shape[0], dtype=torch.int32)
    single = _wrapper(mapping)
    _install(single, _pool(_adapter(tensors), module=single))
    want = single(x, weights, ids).float()

    partials = []

    def all_reduce(ranks):
        partials.append(ranks)
        return ranks

    group = SimpleNamespace(moe_ep_group=SimpleNamespace(all_reduce=all_reduce))
    with mock.patch.object(inkling2_lora, "get_parallel", lambda: group):
        for rank in range(2):
            wrapper = _wrapper(mapping, ep_size=2, ep_rank=rank)
            _install(
                wrapper,
                _pool(
                    _adapter(tensors),
                    ep_size=2,
                    ep_rank=rank,
                    local_ids=local_ids,
                    module=wrapper,
                ),
            )
            wrapper(x, weights, ids)
    assert len(partials) == 2 and not torch.allclose(partials[0], partials[1])
    summed = (partials[0] + partials[1]).bfloat16()
    got = torch.nn.functional.linear(summed, single.down_lora_b_weights[0, 0])
    torch.testing.assert_close(got.float(), want, rtol=1e-2, atol=1e-2)


def test_routed_rows_are_combined_before_the_per_token_delta():
    """combine=False hands back expert rows for the consumer to sum; the per-token down
    delta needs them summed first, so the result is a top_k 0 RoutedOutput."""
    tokens, top_k = 3, 2
    rows = (
        torch.arange(tokens * top_k * 2, dtype=torch.float32).reshape(-1, 2).bfloat16()
    )
    src2dst = torch.tensor([5, 0, 3, 1, 4, 2])
    delta = torch.ones(tokens, 2)
    out = inkling2_lora.add_down_delta(RoutedOutput(rows, src2dst, top_k), delta)
    assert out.top_k == 0 and out.src2dst is None
    want = ref.unweighted_post_reorder(rows, src2dst, top_k).float() + 1
    torch.testing.assert_close(out.rows.float(), want)


def test_unhooked_expert_kernels_reject_lora():
    """SM120 and Marlin expert paths have no LoRA hook; a LoRA batch must not silently
    run them without the adapter."""
    experts = Inkling2Experts.__new__(Inkling2Experts)
    nn.Module.__init__(experts)
    experts.native_sm120, experts._marlin = False, {"w13": None}
    x, ids, weights = _batch()
    with pytest.raises(NotImplementedError, match="no LoRA hook"):
        experts.forward(x, weights, ids, lora=object())


def test_rank_below_16_is_rejected():
    """The routed LoRA kernels' tl.dot needs rank >= 16; a smaller rank must fail at
    setup, not at the first adapter batch's kernel compile."""
    wrapper = Inkling2ExpertsWithLoRA.__new__(Inkling2ExpertsWithLoRA)
    rank = 8
    with pytest.raises(ValueError, match="power of two >= 16"):
        wrapper.set_lora_info(
            torch.zeros(1, 1, 2 * rank, _D),
            torch.zeros(1, _E, 2 * _F, rank),
            torch.zeros(1, _E, rank, _F),
            torch.zeros(1, 1, _D, rank),
        )


def test_graph_routing_buffers_need_no_fused_moe_quant_info():
    """Server startup sizes the MoE LoRA CUDA-graph buffers for every expert LoRA layer; Inkling2's layer
    has no FusedMoE ``_quant_info``, and startup used to fail reading it."""
    from sglang.srt.lora.backend.base_backend import BaseLoRABackend

    experts = nn.Module()
    experts.top_k, experts.num_experts = _TOP_K, _E
    experts.w13_weight = nn.Parameter(torch.empty(_E, 2 * _F, _D))
    backend = SimpleNamespace()
    BaseLoRABackend.init_cuda_graph_moe_buffers(
        backend,
        max_bs=8,
        max_loras=1,
        compute_dtype=torch.bfloat16,
        moe_layer=SimpleNamespace(base_layer=experts),
    )
    assert backend.moe_cg_buffers["token_lora_mapping"].shape == (8,)
