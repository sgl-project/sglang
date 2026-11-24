# Copyright 2023-2026 SGLang Team
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

"""Unit tests for partial fused projection LoRA weight normalization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.lora.layers import (
    FusedMoEWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
)
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_adapter(
    *,
    r: int = 4,
    lora_alpha: int = 8,
    target_modules: list[str] | None = None,
    base_hf_config=None,
    base_model: torch.nn.Module | None = None,
) -> LoRAAdapter:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
    if base_hf_config is None:
        base_hf_config = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
        )
    return LoRAAdapter(
        uid="test-adapter",
        config=LoRAConfig.from_dict(
            {
                "peft_type": "LORA",
                "r": r,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules,
            }
        ),
        base_hf_config=base_hf_config,
        load_config=None,
        lora_backend=None,
        base_model=base_model,
    )


def _rand_ab(
    out_dim: int,
    in_dim: int,
    rank: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn((rank, in_dim), generator=gen, dtype=dtype),
        torch.randn((out_dim, rank), generator=gen, dtype=dtype),
    )


def _ab_tensors(
    prefix: str,
    member_dims: dict[str, int],
    in_dim: int,
    rank: int,
    *,
    seed: int = 10,
    tail: str = ".weight",
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    tensors, branches = {}, {}
    for idx, (member, out_dim) in enumerate(member_dims.items()):
        a, b = _rand_ab(out_dim, in_dim, rank, seed=seed + idx, dtype=dtype)
        tensors[f"{prefix}.{member}.lora_A{tail}"] = a
        tensors[f"{prefix}.{member}.lora_B{tail}"] = b
        branches[member] = (a, b)
    return tensors, branches


def _split_branch_delta(
    x: torch.Tensor,
    a_stacked: torch.Tensor,
    b_stacked: torch.Tensor,
    slice_dims: tuple[int, ...],
    scaling: float,
) -> torch.Tensor:
    rank = b_stacked.shape[-1]
    assert a_stacked.shape[-2] == len(slice_dims) * rank
    assert b_stacked.shape[-2] == sum(slice_dims)
    parts, offset = [], 0
    for idx, dim in enumerate(slice_dims):
        a_i = a_stacked[idx * rank : (idx + 1) * rank]
        b_i = b_stacked[offset : offset + dim]
        parts.append(scaling * ((x @ a_i.T) @ b_i.T))
        offset += dim
    return torch.cat(parts, dim=-1)


def _stub_qkv_module(
    *,
    total_num_heads: int,
    total_num_kv_heads: int,
    head_size: int,
    v_head_size: int | None = None,
):
    mod = QKVParallelLinear.__new__(QKVParallelLinear)
    torch.nn.Module.__init__(mod)
    mod.total_num_heads = total_num_heads
    mod.total_num_kv_heads = total_num_kv_heads
    mod.head_size = head_size
    if v_head_size is not None:
        mod.v_head_size = v_head_size
    return mod


def _stub_merged_col(output_sizes: list[int]):
    mod = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    torch.nn.Module.__init__(mod)
    mod.output_sizes = list(output_sizes)
    return mod


def _stub_fused_moe(*, is_gated: bool = True, is_shared_fused_moe: bool = False):
    mod = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(mod)
    mod.moe_runner_config = SimpleNamespace(is_gated=is_gated)
    mod.is_shared_fused_moe = is_shared_fused_moe
    return mod


def _wrap_lora(wrapper_cls, base_layer: torch.nn.Module) -> torch.nn.Module:
    wrapped = wrapper_cls.__new__(wrapper_cls)
    torch.nn.Module.__init__(wrapped)
    wrapped.base_layer = base_layer
    return wrapped


def _build_layer_model(layer_children: dict[str, torch.nn.Module]) -> torch.nn.Module:
    model, inner, layer = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
    for name, child in layer_children.items():
        setattr(layer, name, child)
    inner.layers = torch.nn.ModuleList([layer])
    model.model = inner
    return model


def _assert_fused_matches_reference(
    test_case: unittest.TestCase,
    *,
    adapter: LoRAAdapter,
    prefix: str,
    fused_name: str,
    split_members: tuple[str, ...],
    member_dims: dict[str, int],
    subset: tuple[str, ...],
    hidden: int,
    rank: int,
    x: torch.Tensor,
    extra_tensors: dict[str, torch.Tensor] | None = None,
    seed: int = 10,
) -> dict[str, torch.Tensor]:
    tensors, branches = _ab_tensors(
        prefix, {m: member_dims[m] for m in subset}, hidden, rank, seed=seed
    )
    if extra_tensors:
        tensors.update(extra_tensors)
    adapter.initialize_weights_from_tensors(tensors)
    weights = adapter.layers[0].weights
    a_key = f"{prefix}.{fused_name}.lora_A.weight"
    b_key = f"{prefix}.{fused_name}.lora_B.weight"
    expected_keys = {a_key, b_key} | set(extra_tensors or ())
    test_case.assertEqual(set(weights), expected_keys)
    slice_dims = tuple(member_dims[m] for m in split_members)
    test_case.assertEqual(
        tuple(weights[a_key].shape), (len(split_members) * rank, hidden)
    )
    test_case.assertEqual(tuple(weights[b_key].shape), (sum(slice_dims), rank))
    actual = _split_branch_delta(
        x, weights[a_key], weights[b_key], slice_dims, adapter.scaling
    )
    expected = torch.cat(
        [
            (
                adapter.scaling * ((x @ branches[m][0].T) @ branches[m][1].T)
                if m in branches
                else torch.zeros((x.shape[0], member_dims[m]))
            )
            for m in split_members
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    return weights


class TestPartialQKVNormalization(CustomTestCase):
    def test_all_seven_gqa_and_gated_qkv_subsets_match_branch_reference(self):
        """Every non-empty subset of {q, k, v} must fuse into qkv_proj with exact
        Q/K/V output widths for both standard GQA and attn_output_gate=True."""
        hidden, rank = 32, 4
        num_heads, num_kv_heads, head_dim = 8, 2, 64
        x = torch.randn((5, hidden), generator=torch.Generator().manual_seed(99))
        subsets = (
            ("q_proj",),
            ("k_proj",),
            ("v_proj",),
            ("q_proj", "k_proj"),
            ("q_proj", "v_proj"),
            ("k_proj", "v_proj"),
            ("q_proj", "k_proj", "v_proj"),
        )
        prefix = "base_model.model.model.layers.0.self_attn"

        for attn_output_gate, use_base_model in (
            (False, True),
            (False, False),
            (True, True),
            (True, False),
        ):
            q_dim = num_heads * head_dim * (2 if attn_output_gate else 1)
            kv_dim = num_kv_heads * head_dim
            dims = {"q_proj": q_dim, "k_proj": kv_dim, "v_proj": kv_dim}
            cfg = SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=hidden,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
            )
            if attn_output_gate and not use_base_model:
                cfg.attn_output_gate = True

            base_model = None
            if use_base_model:
                attn = torch.nn.Module()
                attn.qkv_proj = _stub_qkv_module(
                    total_num_heads=num_heads * (2 if attn_output_gate else 1),
                    total_num_kv_heads=num_kv_heads,
                    head_size=head_dim,
                )
                base_model = _build_layer_model({"self_attn": attn})

            for subset in subsets:
                with self.subTest(
                    attn_output_gate=attn_output_gate,
                    use_base_model=use_base_model,
                    subset=subset,
                ):
                    adapter = _make_adapter(
                        r=rank,
                        target_modules=list(subset),
                        base_hf_config=cfg,
                        base_model=base_model,
                    )
                    _assert_fused_matches_reference(
                        self,
                        adapter=adapter,
                        prefix=prefix,
                        fused_name="qkv_proj",
                        split_members=("q_proj", "k_proj", "v_proj"),
                        member_dims=dims,
                        subset=subset,
                        hidden=hidden,
                        rank=rank,
                        x=x,
                    )

    def test_tp_kv_head_replication_slices_partial_qkv_b_per_rank(self):
        """With tp_size > num_kv_heads, q_proj-only and v_proj-only B weights
        must shard Q by tp_rank and replicate K/V by kv_shard_id."""
        hidden, rank = 16, 2
        total_heads, total_kv_heads, head_size, tp_size = 8, 2, 8, 4
        q_dim, kv_dim = total_heads * head_size, total_kv_heads * head_size
        q_shard, kv_shard = q_dim // tp_size, head_size
        num_kv_replicas = tp_size // total_kv_heads
        cfg = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=hidden,
            num_attention_heads=total_heads,
            num_key_value_heads=total_kv_heads,
            head_dim=head_size,
        )
        prefix = "model.layers.0.self_attn"

        for member in ("q_proj", "v_proj"):
            out_dim = q_dim if member == "q_proj" else kv_dim
            tensors, branches = _ab_tensors(
                prefix, {member: out_dim}, hidden, rank, seed=42
            )
            b = branches[member][1]
            adapter = _make_adapter(r=rank, target_modules=[member], base_hf_config=cfg)
            adapter.initialize_weights_from_tensors(tensors)
            b_qkv = adapter.layers[0].weights[f"{prefix}.qkv_proj.lora_B.weight"]

            for tp_rank in range(tp_size):
                wrapper = QKVParallelLinearWithLoRA.__new__(QKVParallelLinearWithLoRA)
                wrapper.base_layer = SimpleNamespace(
                    q_proj_shard_size=q_shard,
                    kv_proj_shard_size=kv_shard,
                    num_kv_head_replicas=num_kv_replicas,
                    total_num_heads=total_heads,
                    total_num_kv_heads=total_kv_heads,
                    head_size=head_size,
                    tp_group="tp",
                    kv_tp_group="kv_tp",
                )
                with patch(
                    "sglang.srt.lora.layers.get_group_rank_size",
                    return_value=(tp_rank, tp_size),
                ):
                    sharded = wrapper.slice_lora_b_weights(b_qkv)

                kv_id = tp_rank // num_kv_replicas
                exp_q = (
                    b[tp_rank * q_shard : (tp_rank + 1) * q_shard]
                    if member == "q_proj"
                    else torch.zeros((q_shard, rank))
                )
                exp_k = torch.zeros((kv_shard, rank))
                exp_v = (
                    b[kv_id * kv_shard : (kv_id + 1) * kv_shard]
                    if member == "v_proj"
                    else torch.zeros((kv_shard, rank))
                )
                torch.testing.assert_close(
                    sharded, torch.cat((exp_q, exp_k, exp_v), dim=0)
                )

    def test_qkv_config_edge_cases(self):
        """Missing layout must fail loudly; Qwen3.5 default attn_output_gate,
        full per-layer overrides, and NotImplementedError get_hidden_dim must work."""
        hidden, rank = 24, 2
        p = "model.layers.0.self_attn"

        # 1. Missing Q layout without base_model raises ValueError.
        tensors, _ = _ab_tensors(p, {"v_proj": 16}, hidden, rank, seed=7)
        with self.assertRaises(ValueError):
            _make_adapter(
                r=rank,
                target_modules=["v_proj"],
                base_hf_config=SimpleNamespace(num_hidden_layers=1),
            ).initialize_weights_from_tensors(tensors)

        # 2. Qwen3.5 config omitting attn_output_gate defaults to 2x Q width.
        qwen_cfg = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=hidden,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            model_type="qwen3_5_text",
        )
        tensors, _ = _ab_tensors(p, {"v_proj": 32}, hidden, rank, seed=161)
        adapter = _make_adapter(
            r=rank, target_modules=["v_proj"], base_hf_config=qwen_cfg
        )
        adapter.initialize_weights_from_tensors(tensors)
        self.assertEqual(
            tuple(adapter.layers[0].weights[f"{p}.qkv_proj.lora_B.weight"].shape),
            (2 * 4 * 16 + 2 * 32, rank),
        )

        # 3. Full QKV without base_model permits per-layer head overrides.
        std_cfg = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=hidden,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        tensors, branches = _ab_tensors(
            p, {"q_proj": 8 * 16, "k_proj": 32, "v_proj": 32}, hidden, rank, seed=171
        )
        adapter = _make_adapter(r=rank, base_hf_config=std_cfg)
        adapter.initialize_weights_from_tensors(tensors)
        torch.testing.assert_close(
            adapter.layers[0].weights[f"{p}.qkv_proj.lora_B.weight"],
            torch.cat(
                (branches["q_proj"][1], branches["k_proj"][1], branches["v_proj"][1]),
                dim=0,
            ),
        )

        # 4. get_hidden_dim raising NotImplementedError falls back to config.
        stub_model = torch.nn.Module()
        setattr(
            stub_model,
            "get_hidden_dim",
            lambda _n, _l: (_ for _ in ()).throw(NotImplementedError()),
        )
        tensors, _ = _ab_tensors(p, {"v_proj": 32}, hidden, rank, seed=175)
        adapter = _make_adapter(
            r=rank,
            target_modules=["v_proj"],
            base_hf_config=std_cfg,
            base_model=stub_model,
        )
        adapter.initialize_weights_from_tensors(tensors)
        self.assertEqual(
            tuple(adapter.layers[0].weights[f"{p}.qkv_proj.lora_B.weight"].shape),
            (4 * 16 + 2 * 32, rank),
        )

    def test_vlm_v_head_size_and_dynamic_lora_wrapped_base_modules(self):
        """VLM should_apply_lora filtering, distinct v_head_size, and dynamic
        loading against already-wrapped *WithLoRA modules must resolve exact dims."""
        hidden, inter, rank = 24, 32, 2
        p = "model.layers.0"

        # 1. VLM vision tower filtered out by should_apply_lora.
        lang_attn, vis_attn = torch.nn.Module(), torch.nn.Module()
        lang_attn.qkv_proj = _stub_qkv_module(
            total_num_heads=8, total_num_kv_heads=2, head_size=16
        )
        vis_attn.qkv_proj = _stub_qkv_module(
            total_num_heads=12, total_num_kv_heads=12, head_size=32
        )
        vlm = _build_layer_model({"self_attn": lang_attn})
        vis_layer, vis_enc = torch.nn.Module(), torch.nn.Module()
        vis_layer.self_attn = vis_attn
        vis_enc.layers = torch.nn.ModuleList([vis_layer])
        vlm.vision_tower = vis_enc
        setattr(vlm, "should_apply_lora", lambda name: name.startswith("model.layers."))
        tensors, _ = _ab_tensors(f"{p}.self_attn", {"v_proj": 32}, hidden, rank, seed=9)
        adapter = _make_adapter(r=rank, target_modules=["v_proj"], base_model=vlm)
        adapter.initialize_weights_from_tensors(tensors)
        self.assertEqual(
            tuple(
                adapter.layers[0].weights[f"{p}.self_attn.qkv_proj.lora_B.weight"].shape
            ),
            (8 * 16 + 2 * 32, rank),
        )

        # 2. QKVParallelLinear with v_head_size != head_size.
        vh_attn = torch.nn.Module()
        vh_attn.qkv_proj = _stub_qkv_module(
            total_num_heads=4, total_num_kv_heads=2, head_size=16, v_head_size=24
        )
        vh_model = _build_layer_model({"self_attn": vh_attn})
        vh_dims = {"q_proj": 64, "k_proj": 32, "v_proj": 48}
        for subset in (("k_proj",), ("v_proj",), ("q_proj", "k_proj", "v_proj")):
            tensors, _ = _ab_tensors(
                f"{p}.self_attn",
                {m: vh_dims[m] for m in subset},
                hidden,
                rank,
                seed=150,
            )
            adapter = _make_adapter(
                r=rank, target_modules=list(subset), base_model=vh_model
            )
            adapter.initialize_weights_from_tensors(tensors)
            self.assertEqual(
                tuple(
                    adapter.layers[0]
                    .weights[f"{p}.self_attn.qkv_proj.lora_B.weight"]
                    .shape
                ),
                (64 + 32 + 48, rank),
            )

        # 3. Dynamic load against *WithLoRA wrappers unwraps and skips .base_layer.
        dyn_attn, dyn_mlp = torch.nn.Module(), torch.nn.Module()
        dyn_attn.qkv_proj = _wrap_lora(
            QKVParallelLinearWithLoRA,
            _stub_qkv_module(total_num_heads=8, total_num_kv_heads=2, head_size=16),
        )
        dyn_mlp.gate_up_proj = _wrap_lora(
            MergedColumnParallelLinearWithLoRA, _stub_merged_col([inter, inter])
        )
        dyn_mlp.experts = _wrap_lora(FusedMoEWithLoRA, _stub_fused_moe(is_gated=False))
        dyn_model = _build_layer_model({"self_attn": dyn_attn, "mlp": dyn_mlp})
        t_v, _ = _ab_tensors(f"{p}.self_attn", {"v_proj": 32}, hidden, rank, seed=176)
        t_u, _ = _ab_tensors(f"{p}.mlp", {"up_proj": inter}, hidden, rank, seed=177)
        t_e, b_e = _ab_tensors(
            f"{p}.mlp.experts.0", {"gate_proj": inter}, hidden, rank, seed=178
        )
        adapter = _make_adapter(
            r=rank,
            target_modules=["v_proj", "up_proj", "gate_proj"],
            base_hf_config=SimpleNamespace(num_hidden_layers=1),
            base_model=dyn_model,
        )
        adapter.initialize_weights_from_tensors({**t_v, **t_u, **t_e})
        w = adapter.layers[0].weights
        self.assertEqual(
            tuple(w[f"{p}.self_attn.qkv_proj.lora_B.weight"].shape), (192, rank)
        )
        self.assertEqual(
            tuple(w[f"{p}.mlp.gate_up_proj.lora_B.weight"].shape), (2 * inter, rank)
        )
        torch.testing.assert_close(
            w[f"{p}.mlp.experts.0.gate_up_proj.lora_A.weight"], b_e["gate_proj"][0]
        )
        torch.testing.assert_close(
            w[f"{p}.mlp.experts.0.gate_up_proj.lora_B.weight"], b_e["gate_proj"][1]
        )


class TestPartialGateUpAndMambaNormalization(CustomTestCase):
    def test_dense_gated_mlp_subsets_and_preserved_attention_gate(self):
        """Gated dense MLP supports gate_only, up_only, and gate+up while leaving
        self_attn.gate_proj unmerged."""
        hidden, inter, rank = 24, 48, 4
        x = torch.randn((4, hidden), generator=torch.Generator().manual_seed(11))
        mlp = torch.nn.Module()
        mlp.gate_up_proj = _stub_merged_col([inter, inter])
        base_model = _build_layer_model({"mlp": mlp})
        prefix = "base_model.model.model.layers.0"
        attn_tensors, attn_branches = _ab_tensors(
            f"{prefix}.self_attn", {"gate_proj": hidden}, hidden, rank, seed=55
        )

        for subset in (("gate_proj",), ("up_proj",), ("gate_proj", "up_proj")):
            with self.subTest(subset=subset):
                adapter = _make_adapter(
                    r=rank,
                    target_modules=["self_attn.gate_proj", *subset],
                    base_model=base_model,
                )
                weights = _assert_fused_matches_reference(
                    self,
                    adapter=adapter,
                    prefix=f"{prefix}.mlp",
                    fused_name="gate_up_proj",
                    split_members=("gate_proj", "up_proj"),
                    member_dims={"gate_proj": inter, "up_proj": inter},
                    subset=subset,
                    hidden=hidden,
                    rank=rank,
                    x=x,
                    extra_tensors=attn_tensors,
                    seed=20,
                )
                torch.testing.assert_close(
                    weights[f"{prefix}.self_attn.gate_proj.lora_A.weight"],
                    attn_branches["gate_proj"][0],
                )
                torch.testing.assert_close(
                    weights[f"{prefix}.self_attn.gate_proj.lora_B.weight"],
                    attn_branches["gate_proj"][1],
                )

    def test_non_gated_standalone_up_proj_is_preserved(self):
        """A layer exposing standalone up_proj (no gate_up_proj) must keep up_proj
        unmerged rather than synthesizing a zero gate_proj."""
        hidden, inter, rank = 16, 32, 4
        mlp = torch.nn.Module()
        mlp.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        tensors, _ = _ab_tensors(
            "model.layers.0.mlp", {"up_proj": inter}, hidden, rank, seed=31
        )
        adapter = _make_adapter(
            r=rank,
            target_modules=["up_proj"],
            base_model=_build_layer_model({"mlp": mlp}),
        )
        adapter.initialize_weights_from_tensors(tensors)
        self.assertEqual(set(adapter.layers[0].weights), set(tensors))
        for k, v in tensors.items():
            torch.testing.assert_close(adapter.layers[0].weights[k], v)

    def test_gated_and_non_gated_moe_expert_normalization(self):
        """Gated MoE fuses w3-only into gate_up_proj; non-gated MoE keeps c=1
        without stacking and rejects simultaneous gate_proj + up_proj."""
        hidden, inter, rank = 16, 24, 2
        gated_mlp, ng_mlp = torch.nn.Module(), torch.nn.Module()
        gated_mlp.experts = _stub_fused_moe(is_gated=True)
        ng_mlp.experts = _stub_fused_moe(is_gated=False)
        gated_model = _build_layer_model({"mlp": gated_mlp})
        ng_model = _build_layer_model({"mlp": ng_mlp})
        p = "model.layers.0.mlp.experts.0"
        a, b = _rand_ab(inter, hidden, rank, seed=61)
        gu_a, gu_b = (
            f"{p}.gate_up_proj.lora_A.weight",
            f"{p}.gate_up_proj.lora_B.weight",
        )

        # Gated MoE: w3-only zero-pads gate slot.
        adapter = _make_adapter(
            r=rank, target_modules=["up_proj"], base_model=gated_model
        )
        adapter.initialize_weights_from_tensors(
            {f"{p}.w3.lora_A.weight": a, f"{p}.w3.lora_B.weight": b}
        )
        self.assertEqual(set(adapter.layers[0].weights), {gu_a, gu_b})
        torch.testing.assert_close(
            adapter.layers[0].weights[gu_a], torch.cat((torch.zeros_like(a), a), dim=0)
        )
        torch.testing.assert_close(
            adapter.layers[0].weights[gu_b], torch.cat((torch.zeros_like(b), b), dim=0)
        )

        # Non-gated MoE: gate_proj-only and fused gate_up_proj stay c=1.
        for target, tensors in (
            (
                "gate_proj",
                {f"{p}.gate_proj.lora_A.weight": a, f"{p}.gate_proj.lora_B.weight": b},
            ),
            ("gate_up_proj", {gu_a: a, gu_b: b}),
        ):
            ng_adapter = _make_adapter(
                r=rank, target_modules=[target], base_model=ng_model
            )
            ng_adapter.initialize_weights_from_tensors(tensors)
            torch.testing.assert_close(ng_adapter.layers[0].weights[gu_a], a)
            torch.testing.assert_close(ng_adapter.layers[0].weights[gu_b], b)

        # Non-gated MoE receiving both gate_proj and up_proj must fail loudly.
        with self.assertRaises(ValueError):
            _make_adapter(
                r=rank, target_modules=["gate_proj", "up_proj"], base_model=ng_model
            ).initialize_weights_from_tensors(
                {
                    f"{p}.gate_proj.lora_A.weight": a,
                    f"{p}.gate_proj.lora_B.weight": b,
                    f"{p}.up_proj.lora_A.weight": a,
                    f"{p}.up_proj.lora_B.weight": b,
                }
            )

    def test_mamba2_partial_in_proj_and_mamba1_standalone_x_proj(self):
        """Mamba2 fuses gate_proj-only/x_proj-only into in_proj; Mamba1 keeps
        standalone x_proj intact even when coexisting with in_proj."""
        hidden, inter, x_out, rank = 16, 20, 12, 2
        p = "model.layers.0.mixer"

        # Mamba2: 5-partition in_proj, LoRA on first 2 slices [gate, x].
        m2_mixer = torch.nn.Module()
        m2_mixer.in_proj = _stub_merged_col([inter, inter, 8, 8, 4])
        m2_model = _build_layer_model({"mixer": m2_mixer})
        for member in ("gate_proj", "x_proj"):
            with self.subTest(mamba="v2", member=member):
                tensors, branches = _ab_tensors(
                    p, {member: inter}, hidden, rank, seed=71
                )
                a, b = branches[member]
                adapter = _make_adapter(
                    r=rank, target_modules=["in_proj"], base_model=m2_model
                )
                adapter.initialize_weights_from_tensors(tensors)
                in_a, in_b = f"{p}.in_proj.lora_A.weight", f"{p}.in_proj.lora_B.weight"
                self.assertEqual(set(adapter.layers[0].weights), {in_a, in_b})
                order_a = (
                    (a, torch.zeros_like(a))
                    if member == "gate_proj"
                    else (torch.zeros_like(a), a)
                )
                order_b = (
                    (b, torch.zeros_like(b))
                    if member == "gate_proj"
                    else (torch.zeros_like(b), b)
                )
                torch.testing.assert_close(
                    adapter.layers[0].weights[in_a], torch.cat(order_a, dim=0)
                )
                torch.testing.assert_close(
                    adapter.layers[0].weights[in_b], torch.cat(order_b, dim=0)
                )

        # Mamba1: standalone x_proj alongside in_proj.
        m1_mixer = torch.nn.Module()
        m1_mixer.in_proj = _stub_merged_col([inter, inter])
        m1_mixer.x_proj = torch.nn.Linear(inter, x_out, bias=False)
        m1_model = _build_layer_model({"mixer": m1_mixer})

        x_only, _ = _ab_tensors(
            p, {"x_proj": x_out}, hidden, rank, seed=77, tail=".default.weight"
        )
        adapter = _make_adapter(r=rank, target_modules=["x_proj"], base_model=m1_model)
        adapter.initialize_weights_from_tensors(x_only)
        self.assertEqual(set(adapter.layers[0].weights), set(x_only))

        t_in, b_in = _ab_tensors(p, {"in_proj": 2 * inter}, hidden, rank, seed=181)
        t_x, b_x = _ab_tensors(p, {"x_proj": x_out}, inter, rank, seed=182)
        adapter = _make_adapter(
            r=rank, target_modules=["in_proj", "x_proj"], base_model=m1_model
        )
        adapter.initialize_weights_from_tensors({**t_in, **t_x})
        w = adapter.layers[0].weights
        self.assertEqual(set(w), set(t_in) | set(t_x))
        torch.testing.assert_close(
            w[f"{p}.in_proj.lora_A.weight"], b_in["in_proj"][0].repeat(2, 1)
        )
        torch.testing.assert_close(w[f"{p}.x_proj.lora_A.weight"], b_x["x_proj"][0])

    def test_pp_missing_off_stage_layers_are_cleared_during_normalization(self):
        """Off-stage PPMissingLayer weights must be dropped so partial up_proj
        on off-stage layers does not remain unmerged."""
        hidden, inter, rank = 16, 32, 2
        model, inner, local_layer, mlp = (
            torch.nn.Module(),
            torch.nn.Module(),
            torch.nn.Module(),
            torch.nn.Module(),
        )
        mlp.gate_up_proj = _stub_merged_col([inter, inter])
        local_layer.mlp = mlp
        inner.layers = torch.nn.ModuleList([PPMissingLayer(), local_layer])
        model.model = inner

        t0, _ = _ab_tensors(
            "model.layers.0.mlp", {"up_proj": inter}, hidden, rank, seed=191
        )
        t1, _ = _ab_tensors(
            "model.layers.1.mlp", {"up_proj": inter}, hidden, rank, seed=192
        )
        adapter = _make_adapter(
            r=rank,
            target_modules=["up_proj"],
            base_hf_config=SimpleNamespace(num_hidden_layers=2, hidden_size=hidden),
            base_model=model,
        )
        adapter.initialize_weights_from_tensors({**t0, **t1})
        self.assertEqual(adapter.layers[0].weights, {})
        self.assertEqual(
            set(adapter.layers[1].weights),
            {
                "model.layers.1.mlp.gate_up_proj.lora_A.weight",
                "model.layers.1.mlp.gate_up_proj.lora_B.weight",
            },
        )


class TestPartialGDNAndMLANormalization(CustomTestCase):
    def test_gdn_in_proj_qkvz_partial_subsets_use_exact_key_and_value_dims(self):
        """Partial 4-way GDN subsets must zero-pad missing slices with exact
        key_dim (q/k) and value_dim (v/z) widths."""
        hidden, key_dim, value_dim, rank = 24, 32, 48, 3
        x = torch.randn((4, hidden), generator=torch.Generator().manual_seed(101))
        linear_attn = torch.nn.Module()
        linear_attn.in_proj_qkvz = _stub_merged_col(
            [key_dim, key_dim, value_dim, value_dim]
        )
        base_model = _build_layer_model({"linear_attn": linear_attn})
        cfg = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=hidden,
            linear_key_head_dim=16,
            linear_num_key_heads=2,
            linear_value_head_dim=16,
            linear_num_value_heads=3,
        )
        dims = {
            "in_proj_q": key_dim,
            "in_proj_k": key_dim,
            "in_proj_v": value_dim,
            "in_proj_z": value_dim,
        }

        for use_model in (True, False):
            for subset in (
                ("in_proj_q",),
                ("in_proj_v",),
                ("in_proj_q", "in_proj_z"),
                ("in_proj_k", "in_proj_v"),
            ):
                with self.subTest(use_model=use_model, subset=subset):
                    adapter = _make_adapter(
                        r=rank,
                        target_modules=["in_proj_qkvz"],
                        base_hf_config=cfg,
                        base_model=base_model if use_model else None,
                    )
                    _assert_fused_matches_reference(
                        self,
                        adapter=adapter,
                        prefix="model.layers.0.linear_attn",
                        fused_name="in_proj_qkvz",
                        split_members=(
                            "in_proj_q",
                            "in_proj_k",
                            "in_proj_v",
                            "in_proj_z",
                        ),
                        member_dims=dims,
                        subset=subset,
                        hidden=hidden,
                        rank=rank,
                        x=x,
                        seed=80,
                    )

    def test_mla_q_a_and_kv_a_unequal_widths_and_standalone_fallback(self):
        """MLA partial subsets pad unequal q_a/kv_a widths; already-fused repeats
        A 2x; q_lora_rank=None preserves standalone q_proj and kv_a_proj_with_mqa."""
        hidden, rank = 32, 4
        q_lora_rank, kv_lora_rank, qk_rope_head_dim = 96, 32, 16
        kv_a_dim = kv_lora_rank + qk_rope_head_dim
        x = torch.randn((3, hidden), generator=torch.Generator().manual_seed(123))
        cfg = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=hidden,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        dims = {"q_a_proj": q_lora_rank, "kv_a_proj_with_mqa": kv_a_dim}
        p = "model.layers.0.self_attn"

        for subset in (
            ("q_a_proj",),
            ("kv_a_proj_with_mqa",),
            ("q_a_proj", "kv_a_proj_with_mqa"),
        ):
            with self.subTest(subset=subset):
                adapter = _make_adapter(
                    r=rank, target_modules=list(subset), base_hf_config=cfg
                )
                _assert_fused_matches_reference(
                    self,
                    adapter=adapter,
                    prefix=p,
                    fused_name="fused_qkv_a_proj_with_mqa",
                    split_members=("q_a_proj", "kv_a_proj_with_mqa"),
                    member_dims=dims,
                    subset=subset,
                    hidden=hidden,
                    rank=rank,
                    x=x,
                    seed=130,
                )

        # Already-fused MLA repeats unstacked A 2x.
        fused_t, fused_b = _ab_tensors(
            p,
            {"fused_qkv_a_proj_with_mqa": q_lora_rank + kv_a_dim},
            hidden,
            rank,
            seed=140,
        )
        fused_adapter = _make_adapter(
            r=rank, target_modules=["fused_qkv_a_proj_with_mqa"], base_hf_config=cfg
        )
        fused_adapter.initialize_weights_from_tensors(fused_t)
        torch.testing.assert_close(
            fused_adapter.layers[0].weights[
                f"{p}.fused_qkv_a_proj_with_mqa.lora_A.weight"
            ],
            fused_b["fused_qkv_a_proj_with_mqa"][0].repeat(2, 1),
        )

        # DeepSeek-V2-Lite (q_lora_rank=None): standalone q_proj & kv_a_proj_with_mqa stay intact.
        attn = torch.nn.Module()
        attn.q_proj = torch.nn.Linear(hidden, 64, bias=False)
        attn.kv_a_proj_with_mqa = torch.nn.Linear(hidden, kv_a_dim, bias=False)
        lite_cfg = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=hidden,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        lite_t, _ = _ab_tensors(
            p, {"q_proj": 64, "kv_a_proj_with_mqa": kv_a_dim}, hidden, rank, seed=201
        )
        lite_adapter = _make_adapter(
            r=rank,
            target_modules=["q_proj", "kv_a_proj_with_mqa"],
            base_hf_config=lite_cfg,
            base_model=_build_layer_model({"self_attn": attn}),
        )
        lite_adapter.initialize_weights_from_tensors(lite_t)
        self.assertEqual(set(lite_adapter.layers[0].weights), set(lite_t))


class TestFusedGroupConsistencyErrors(CustomTestCase):
    def test_invalid_fused_groups_raise_value_error(self):
        """Split+fused mixes, pre-stacked A, unpaired A/B, mismatched shapes/
        ranks/dtypes, half-empty up_proj, and 2-way in_proj_qkv + in_proj_z
        must raise ValueError."""
        hidden, rank = 16, 4
        q_dim, kv_dim, inter = 64, 16, 32
        cfg = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=hidden,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=8,
            linear_key_head_dim=8,
            linear_num_key_heads=2,
            linear_value_head_dim=8,
            linear_num_value_heads=3,
        )
        qa, qb = _rand_ab(q_dim, hidden, rank, seed=1)
        va_bad_r, vb_bad_r = _rand_ab(kv_dim, hidden, rank * 2, seed=3)
        va_bad_in, vb_bad_in = _rand_ab(kv_dim, hidden + 4, rank, seed=31)
        va_fp16, vb_fp16 = _rand_ab(kv_dim, hidden, rank, seed=32, dtype=torch.float16)
        qkv_a, qkv_b = _rand_ab(q_dim + 2 * kv_dim, hidden, rank, seed=4)
        ga, gb = _rand_ab(inter, hidden, rank, seed=5)
        gu_a, gu_b = _rand_ab(2 * inter, hidden, rank, seed=6)
        gdn_qkv_a, gdn_qkv_b = _rand_ab(56, hidden, rank, seed=221)
        gdn_z_a, gdn_z_b = _rand_ab(24, hidden, rank, seed=222)
        p, m, g = (
            "model.layers.0.self_attn",
            "model.layers.0.mlp",
            "model.layers.0.linear_attn",
        )

        cases = {
            "qkv split+fused conflict": {
                f"{p}.q_proj.lora_A.weight": qa,
                f"{p}.q_proj.lora_B.weight": qb,
                f"{p}.qkv_proj.lora_A.weight": qkv_a,
                f"{p}.qkv_proj.lora_B.weight": qkv_b,
            },
            "qkv pre-stacked A": {
                f"{p}.qkv_proj.lora_A.weight": qkv_a.repeat(3, 1),
                f"{p}.qkv_proj.lora_B.weight": qkv_b,
            },
            "qkv inconsistent rank across members": {
                f"{p}.q_proj.lora_A.weight": qa,
                f"{p}.q_proj.lora_B.weight": qb,
                f"{p}.v_proj.lora_A.weight": va_bad_r,
                f"{p}.v_proj.lora_B.weight": vb_bad_r,
            },
            "qkv inconsistent in_features across members": {
                f"{p}.q_proj.lora_A.weight": qa,
                f"{p}.q_proj.lora_B.weight": qb,
                f"{p}.v_proj.lora_A.weight": va_bad_in,
                f"{p}.v_proj.lora_B.weight": vb_bad_in,
            },
            "qkv inconsistent dtype across members": {
                f"{p}.q_proj.lora_A.weight": qa,
                f"{p}.q_proj.lora_B.weight": qb,
                f"{p}.v_proj.lora_A.weight": va_fp16,
                f"{p}.v_proj.lora_B.weight": vb_fp16,
            },
            "gate_up split+fused conflict": {
                f"{m}.gate_proj.lora_A.weight": ga,
                f"{m}.gate_proj.lora_B.weight": gb,
                f"{m}.gate_up_proj.lora_A.weight": gu_a,
                f"{m}.gate_up_proj.lora_B.weight": gu_b,
            },
            "gate_up pre-stacked A": {
                f"{m}.gate_up_proj.lora_A.weight": gu_a.repeat(2, 1),
                f"{m}.gate_up_proj.lora_B.weight": gu_b,
            },
            "qkv A missing B": {f"{p}.q_proj.lora_A.weight": qa},
            "qkv B missing A": {f"{p}.q_proj.lora_B.weight": qb},
            "qkv partial member missing B": {
                f"{p}.q_proj.lora_A.weight": qa,
                f"{p}.q_proj.lora_B.weight": qb,
                f"{p}.v_proj.lora_A.weight": va_bad_r[:rank],
            },
            "qkv fused A missing B": {f"{p}.qkv_proj.lora_A.weight": qkv_a},
            "qkv same-member A/B rank mismatch": {
                f"{p}.q_proj.lora_A.weight": qa,
                f"{p}.q_proj.lora_B.weight": torch.randn((q_dim, rank * 2)),
            },
            "gate_up A missing B": {f"{m}.gate_proj.lora_A.weight": ga},
            "gate_up half-empty up_proj": {
                f"{m}.gate_proj.lora_A.weight": ga,
                f"{m}.gate_proj.lora_B.weight": gb,
                f"{m}.up_proj.lora_A.weight": torch.empty((0, hidden)),
                f"{m}.up_proj.lora_B.weight": gb,
            },
            "gdn 2-way in_proj_qkv + in_proj_z": {
                f"{g}.in_proj_qkv.lora_A.weight": gdn_qkv_a,
                f"{g}.in_proj_qkv.lora_B.weight": gdn_qkv_b,
                f"{g}.in_proj_z.lora_A.weight": gdn_z_a,
                f"{g}.in_proj_z.lora_B.weight": gdn_z_b,
            },
        }
        for name, tensors in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                _make_adapter(
                    r=rank, base_hf_config=cfg
                ).initialize_weights_from_tensors(tensors)

    def test_failed_normalization_does_not_partially_mutate_weights(self):
        """If lora_B validation fails, lora_A keys must remain untouched."""
        hidden, rank, q_dim = 16, 4, 64
        cfg = SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=hidden,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=8,
        )
        qa, _ = _rand_ab(q_dim, hidden, rank, seed=211)
        p = "model.layers.0.self_attn"
        weights = {
            f"{p}.q_proj.lora_A.weight": qa,
            f"{p}.q_proj.lora_B.weight": torch.randn((q_dim + 4, rank)),
        }
        with self.assertRaises(ValueError):
            _make_adapter(r=rank, base_hf_config=cfg).normalize_qkv_proj(
                list(weights), weights
            )
        self.assertEqual(
            set(weights),
            {f"{p}.q_proj.lora_A.weight", f"{p}.q_proj.lora_B.weight"},
        )


if __name__ == "__main__":
    unittest.main()
