"""Quark packed uint4 weight-only MoE on the Triton WNA16 kernel.

The checkpoint tensors are packed here in AutoAWQ's nibble order (the checkpoint's
pack_method "reorder"): int32 words of eight 4-bit codes along the output dimension,
weights stored [in, out], one scale and one zero point per group of inputs.
"""

import unittest

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.quark.schemes.quark_w4a16_uint4_moe import (
    QuarkW4A16Uint4MoE,
)
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="base-b-test-1-gpu-small-amd")

NUM_EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, GROUP_SIZE = 4, 256, 128, 2, 64
# AutoAWQ's packing, which Quark's pack_method "reorder" follows:
# nibble i of a word holds code _AWQ_NIBBLE_ORDER[i].
_AWQ_NIBBLE_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _pack_words(codes: torch.Tensor) -> torch.Tensor:
    """[rows, cols] 4-bit codes -> [rows, cols // 8] int32, packed as the checkpoint."""
    rows, cols = codes.shape
    nibbles = (codes.long() & 0xF).view(rows, cols // 8, 8)[..., _AWQ_NIBBLE_ORDER]
    words = (nibbles << (4 * torch.arange(8))).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _checkpoint_tensors(weight: torch.Tensor) -> tuple:
    """[out, in] float weight -> its checkpoint tensors and the dequantized weight."""
    out_features, in_features = weight.shape
    groups = weight.float().view(out_features, in_features // GROUP_SIZE, GROUP_SIZE)
    low, high = groups.amin(-1, keepdim=True), groups.amax(-1, keepdim=True)
    scale = ((high - low).clamp(min=1e-5) / 15).to(torch.bfloat16).float()
    zero_point = torch.round(-low / scale).clamp(0, 15)
    codes = (torch.round(groups / scale) + zero_point).clamp(0, 15)
    dequantized = ((codes - zero_point) * scale).view(out_features, in_features)
    codes = codes.view(out_features, in_features).int()
    zero_point = zero_point.squeeze(-1).int()
    tensors = {
        "weight": _pack_words(codes.T.contiguous()),
        "weight_scale": scale.squeeze(-1).T.contiguous().to(torch.bfloat16),
        "weight_zero_point": _pack_words(zero_point.T.contiguous()),
    }
    return tensors, dequantized


class _ExpertLayer(torch.nn.Module):
    """The attributes of FusedMoE the scheme and its loader read, for one rank."""

    def __init__(self):
        super().__init__()
        self.intermediate_size_per_partition = INTERMEDIATE
        self.num_local_experts = NUM_EXPERTS
        self.moe_tp_size, self.moe_tp_rank = 1, 0
        self.use_presharded_weights = False
        self.uses_expert_location = False
        self.layer_id = 0

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        # Gate rows first, then up; zero points hold two rows per byte.
        half = param.shape[1] // 2
        if shard_id == "w2":
            param.data[expert_id] = loaded_weight
        elif shard_id == "w1":
            param.data[expert_id, :half] = loaded_weight
        else:
            param.data[expert_id, half:] = loaded_weight


def _reference(*, hidden_states, w13, w2, topk_weights, topk_ids):
    out = torch.zeros(hidden_states.shape[0], HIDDEN, dtype=torch.float32)
    x = hidden_states.float().cpu()
    for token in range(x.shape[0]):
        for slot in range(TOP_K):
            expert = int(topk_ids[token, slot])
            gate, up = (x[token] @ w13[expert].T).chunk(2)
            hidden = torch.nn.functional.silu(gate) * up
            out[token] += float(topk_weights[token, slot]) * (hidden @ w2[expert].T)
    return out


def _relative_error() -> float:
    """Kernel output against a float MoE on the dequantized weights (relative L2)."""
    g = torch.Generator().manual_seed(1)
    scheme = QuarkW4A16Uint4MoE(weight_config={"group_size": GROUP_SIZE})
    layer = _ExpertLayer()
    scheme.create_weights(
        layer=layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        params_dtype=torch.bfloat16,
        weight_loader=layer.weight_loader,
    )
    layer.cuda()

    w13_ref = torch.empty(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    w2_ref = torch.empty(NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    params = dict(layer.named_parameters())
    for expert in range(NUM_EXPERTS):
        for shard_id, shape in (
            ("w1", (INTERMEDIATE, HIDDEN)),
            ("w3", (INTERMEDIATE, HIDDEN)),
            ("w2", (HIDDEN, INTERMEDIATE)),
        ):
            weight = torch.randn(shape, generator=g) * 0.1
            tensors, dequantized = _checkpoint_tensors(weight)
            prefix = "w2" if shard_id == "w2" else "w13"
            for suffix, tensor in tensors.items():
                name = f"{prefix}_{suffix}"
                params[name].weight_loader(params[name], tensor, name, shard_id, expert)
            if shard_id == "w2":
                w2_ref[expert] = dequantized
            elif shard_id == "w1":
                w13_ref[expert, :INTERMEDIATE] = dequantized
            else:
                w13_ref[expert, INTERMEDIATE:] = dequantized

    scheme.create_moe_runner(layer, MoeRunnerConfig(activation="silu"))
    scheme.process_weights_after_loading(layer)

    num_tokens = 9
    hidden_states = torch.randn(num_tokens, HIDDEN, generator=g).to(
        device="cuda", dtype=torch.bfloat16
    )
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=g)
    topk_weights, topk_ids = torch.topk(logits.softmax(-1), TOP_K, dim=-1)
    expected = _reference(
        hidden_states=hidden_states,
        w13=w13_ref,
        w2=w2_ref,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states.clone(),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=topk_weights.cuda(),
            topk_ids=topk_ids.to(device="cuda", dtype=torch.int32),
            router_logits=None,
        ),
    )
    actual = scheme.apply_weights(layer, dispatch_output).hidden_states.float().cpu()
    return float((actual - expected).norm() / expected.norm())


@unittest.skipUnless(is_hip(), "needs a ROCm GPU")
class TestQuarkUint4MoeMatchesDequantizedWeights(CustomTestCase):
    # bf16 kernel against an fp32 reference: 0.5% on gfx1151.
    TOLERANCE = 0.02

    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="test")

    def test_kernel_output_matches_a_float_moe(self):
        self.assertLess(_relative_error(), self.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
