"""Compare attention arithmetic on real paged KV; never certify serving parity."""

import argparse
import json
from functools import cache, partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sglang.test import dspark_target_kv_parity as parity
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class ProbeDone(Exception):
    pass


def errors(value, expected):
    delta = value.double() - expected.double()
    return {
        "max_abs": delta.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "unequal": torch.count_nonzero(delta).item(),
        "elements": delta.numel(),
    }


@cache
def cudnn_workspace():
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def attention(mode, module, q, k, v):
    if mode == "reference_sdpa":
        return ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, q, k, v, None, scaling=module.scaling, is_causal=False
        )[0].transpose(1, 2)
    if mode in ("flashinfer_cudnn", "flashinfer_cudnn_paged"):
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

        query = q[0].transpose(0, 1).contiguous()
        key, value = (item[0].transpose(0, 1).contiguous() for item in (k, v))
        block_tables = None
        if mode == "flashinfer_cudnn_paged":
            key, value = key[:, :, None], value[:, :, None]
            block_tables = torch.arange(k.shape[2], dtype=torch.int32, device=q.device)[
                None
            ]
        output, _ = cudnn_batch_prefill_with_kv_cache(
            query,
            key,
            value,
            module.scaling,
            cudnn_workspace(),
            max_token_per_sequence=q.shape[2],
            max_sequence_kv=k.shape[2],
            actual_seq_lens_q=torch.tensor(
                [q.shape[2]], dtype=torch.int32, device=q.device
            ).reshape(1, 1, 1, 1),
            actual_seq_lens_kv=torch.tensor(
                [k.shape[2]], dtype=torch.int32, device=q.device
            ).reshape(1, 1, 1, 1),
            block_tables=block_tables,
            causal=False,
            return_lse=False,
            is_cuda_graph_compatible=True,
        )
        return output.transpose(0, 1)[None]
    if mode in ("sdpa_math", "sdpa_flash", "sdpa_cudnn", "sdpa_flash_repeated"):
        backend = {
            "sdpa_math": SDPBackend.MATH,
            "sdpa_flash": SDPBackend.FLASH_ATTENTION,
            "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
            "sdpa_flash_repeated": SDPBackend.FLASH_ATTENTION,
        }[mode]
        if mode == "sdpa_flash_repeated":
            groups = q.shape[1] // k.shape[1]
            k, v = (value.repeat_interleave(groups, dim=1) for value in (k, v))
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(
                q, k, v, scale=module.scaling, is_causal=False, enable_gqa=True
            )
    dtype = torch.float64 if mode == "fp64_math" else torch.float32
    groups = q.shape[1] // k.shape[1]
    key = k.to(dtype).repeat_interleave(groups, dim=1)
    value = v.to(dtype).repeat_interleave(groups, dim=1)
    probability = (q.to(dtype) @ key.transpose(-1, -2) * module.scaling).softmax(-1)
    return (probability @ value).to(q.dtype)


@torch.no_grad()
def probe(
    reference, serving, embed, head, tensors, anchors, *, dump_attention, **kwargs
):
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
    modes = (
        "reference_sdpa",
        "sdpa_math",
        "sdpa_flash",
        "sdpa_cudnn",
        "sdpa_flash_repeated",
        "flashinfer_cudnn",
        "flashinfer_cudnn_paged",
        "fp32_math",
        "fp64_math",
    )
    saved = {}

    def inspect_or_replace(module, inputs, output, *, mode):
        training = reference.backbone.layers[module.layer_id].self_attn
        rows = []
        for row, anchor in enumerate(anchors):
            slots = serving.mapping.req_to_token[row + 1, : anchor + width].long()
            k = serving.pool.get_key_buffer(module.layer_id)[slots].transpose(0, 1)[
                None
            ]
            v = serving.pool.get_value_buffer(module.layer_id)[slots].transpose(0, 1)[
                None
            ]
            q = (
                inputs[0][row * width : (row + 1) * width]
                .reshape(width, module.tp_q_head_num, module.head_dim)
                .transpose(0, 1)[None]
            )
            if mode == "production":
                if module.layer_id == 0 and row == 0:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as profile:
                        attention("reference_sdpa", training, q, k, v)
                    print(
                        json.dumps(
                            {
                                "reference_operators": [
                                    event.key
                                    for event in profile.key_averages()
                                    if "scaled_dot_product" in event.key
                                ]
                            }
                        ),
                        flush=True,
                    )
                actual = (
                    output[row * width : (row + 1) * width]
                    .reshape(width, module.tp_q_head_num, module.head_dim)
                    .transpose(0, 1)[None]
                )
                values = {name: attention(name, training, q, k, v) for name in modes}
                values["production"] = actual
                for name, value in values.items():
                    print(
                        json.dumps(
                            {
                                "diagnostic_only": True,
                                "layer": module.layer_id,
                                "anchor": anchor,
                                "mode": name,
                                "vs_reference_sdpa": errors(
                                    value, values["reference_sdpa"]
                                ),
                                "vs_fp64": errors(value, values["fp64_math"]),
                            }
                        ),
                        flush=True,
                    )
                if dump_attention is not None:
                    for name, value in {"q": q, "k": k, "v": v, **values}.items():
                        saved[f"layer{module.layer_id}.anchor{anchor}.{name}"] = (
                            value.cpu().contiguous()
                        )
            else:
                rows.append(
                    attention(mode, training, q, k, v)
                    .transpose(1, 2)
                    .reshape(width, -1)
                )
        return output if mode == "production" else torch.cat(rows).reshape_as(output)

    for mode in ("production", *modes):
        handles = [
            layer.self_attn.attn.register_forward_hook(
                partial(inspect_or_replace, mode=mode)
            )
            for layer in serving.model.layers
        ]
        try:
            actual = serving.forward(
                tensors=tensors, anchors=anchors, previous_tokens=previous
            )
            try:
                report = parity.compare_parity_outputs(actual, expected, **kwargs)
            except parity.FixedInputParityError as error:
                report = error.report
            print(
                json.dumps({"diagnostic_only": True, "full_backbone": mode, **report}),
                flush=True,
            )
        finally:
            for handle in handles:
                handle.remove()
    if dump_attention is not None:
        save_file(saved, str(dump_attention), metadata={"diagnostic_only": "true"})
    raise ProbeDone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--dump-attention", type=Path)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    with (
        parity.single_gpu_parity_context(),
        patch.object(
            parity,
            "check_fixed_input_parity",
            partial(probe, dump_attention=args.dump_attention),
        ),
        patch(
            "transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward",
            side_effect=AssertionError("target decoder is forbidden"),
        ),
    ):
        try:
            parity._validate_captured_checkpoint(
                args.checkpoint, args.target_path, attention_backend="sdpa"
            )
        except ProbeDone:
            pass


if __name__ == "__main__":
    main()
