"""Isolate BF16 auxiliary/attention rounding; never write a serving certificate."""

import argparse
import json
from types import MethodType
from unittest.mock import patch

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models import dflash
from sglang.test import dspark_target_kv_parity as parity


def hf_norm(self, x, residual=None, post_residual_addition=None):
    assert post_residual_addition is None
    if residual is not None:
        x = x + residual
    value = x.float()
    value = value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + self.variance_epsilon
    )
    value = value.to(x.dtype) * self.weight
    return value if residual is None else (value, x)


def hf_qk(q, k, q_norm, k_norm, head_dim, **kwargs):
    return (
        q_norm(q.reshape(-1, head_dim)).view_as(q),
        k_norm(k.reshape(-1, head_dim)).view_as(k),
    )


def hf_rope(self, positions, q, k, *args, **kwargs):
    table = self.cos_sin_cache[positions].to(q.dtype)
    cos, sin = table.chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]

    def rotate(value):
        x, y = value.reshape(len(positions), -1, self.head_size).chunk(2, dim=-1)
        return torch.cat((x * cos - y * sin, y * cos + x * sin), -1).reshape_as(value)

    return rotate(q), rotate(k)


class ProbeDone(Exception):
    pass


@torch.no_grad()
def probe(reference, serving, embed, head, tensors, anchors, **kwargs):
    from specforge.modeling.draft.dflash import eager_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    width = serving.contract.sequence.prediction_count
    previous = torch.zeros(len(anchors), width, dtype=torch.long, device="cuda")
    for row, anchor in enumerate(anchors):
        tokens = tensors["token_ids"][anchor : anchor + width]
        previous[row, : len(tokens)] = tokens.cuda().long()
    expected = [
        reference(
            tensors=tensors,
            anchor=anchor,
            embed=embed,
            head=head,
            previous_tokens=previous[row : row + 1],
        )
        for row, anchor in enumerate(anchors)
    ]

    def compare(mode):
        actual = serving.forward(
            tensors=tensors, anchors=anchors, previous_tokens=previous
        )
        report = {"diagnostic_only": True, "mode": mode, "dtype": "bfloat16"}
        try:
            report.update(parity.compare_parity_outputs(actual, expected, **kwargs))
        except parity.FixedInputParityError as error:
            report.update(error.report)
        print(json.dumps(report), flush=True)

    compare("production")
    model = serving.model
    model._build_fused_kv_write_bundle = lambda pool: None
    model._fused_kv_write_cache = None
    model._stacked_ctx_kv_cache = None
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.forward = MethodType(hf_norm, module)
    for layer in model.layers:
        layer.self_attn.use_table_qk_norm_rope = False
        rotary = layer.self_attn.rotary_emb
        rotary.forward = MethodType(hf_rope, rotary)
        layer.mlp.act_fn.forward = lambda value: (
            torch.nn.functional.silu(value.chunk(2, -1)[0]) * value.chunk(2, -1)[1]
        )

    def replace_attention(module, inputs, output):
        q = inputs[0]
        training_attention = reference.backbone.layers[module.layer_id].self_attn
        backend = training_attention.config._attn_implementation
        attention_fn = (
            eager_attention_forward
            if backend == "eager"
            else ALL_ATTENTION_FUNCTIONS[backend]
        )
        rows = []
        # The real backend has already written block KV. Re-read its paged pool
        # and recompute attention with the reference backend on the same Q/K/V.
        for row, anchor in enumerate(anchors):
            slots = serving.mapping.req_to_token[row + 1, : anchor + width].long()
            k = serving.pool.get_key_buffer(module.layer_id)[slots].transpose(0, 1)[
                None
            ]
            v = serving.pool.get_value_buffer(module.layer_id)[slots].transpose(0, 1)[
                None
            ]
            query = (
                q[row * width : (row + 1) * width]
                .view(width, module.tp_q_head_num, module.head_dim)
                .transpose(0, 1)[None]
            )
            value = attention_fn(
                training_attention,
                query,
                k,
                v,
                None,
                scaling=training_attention.scaling,
                is_causal=False,
            )[0]
            rows.append(value.reshape(width, -1))
        return torch.cat(rows).reshape_as(output)

    with patch.object(dflash, "apply_qk_norm", hf_qk):
        compare("reference_aux_triton_attention")
        handles = [
            layer.self_attn.attn.register_forward_hook(replace_attention)
            for layer in model.layers
        ]
        try:
            compare("reference_aux_and_attention_real_pool")
        finally:
            for handle in handles:
                handle.remove()
    raise ProbeDone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument(
        "--reference-attention", choices=("eager", "sdpa"), default="sdpa"
    )
    args = parser.parse_args()
    with (
        parity.single_gpu_parity_context(),
        patch.object(parity, "check_fixed_input_parity", probe),
        patch(
            "transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward",
            side_effect=AssertionError("target decoder is forbidden"),
        ),
    ):
        try:
            parity._validate_captured_checkpoint(
                args.checkpoint,
                args.target_path,
                attention_backend=args.reference_attention,
            )
        except ProbeDone:
            pass


if __name__ == "__main__":
    main()
