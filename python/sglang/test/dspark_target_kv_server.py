"""Test-only per-layer projected-KV observer, installed in spawned workers."""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.models.dspark_target_kv import DSparkTargetKVDraftModel
from sglang.srt.speculative.dspark_components.dspark_target_kv_inject import (
    TargetKVInjector,
)
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.test.dspark_capture_observer import install_capture_observer


def record(model, event):
    path = Path(model.config.test_observation_path)
    event["pp_rank"] = get_pipeline_model_parallel_rank()
    if get_pipeline_model_parallel_world_size() > 1:
        path = path.with_name(f"{path.stem}-pp{event['pp_rank']}{path.suffix}")
    event["tp_rank"] = get_tensor_model_parallel_rank()
    if get_tensor_model_parallel_world_size() > 1:
        path = path.with_name(f"{path.stem}-tp{event['tp_rank']}{path.suffix}")
    with path.open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def rotate_split_half(
    value, positions, base, rotary_dim, inverse=False, *, round_intermediates=False
):
    # Match standard target RoPE's FP32 frequency construction. Reciprocal
    # after pow differs from a negative exponent near BF16 rounding boundaries.
    frequency = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2, device=value.device).float() / rotary_dim)
    )
    angle = positions.float()[:, None, None] * frequency
    cosine, sine = angle.cos(), angle.sin()
    if inverse:
        sine = -sine
    first, second = value[..., :rotary_dim].float().chunk(2, dim=-1)
    if round_intermediates:
        first, second = first.to(value.dtype), second.to(value.dtype)
        cosine, sine = cosine.to(value.dtype), sine.to(value.dtype)
    return torch.cat(
        (
            first * cosine - second * sine,
            second * cosine + first * sine,
            value[..., rotary_dim:].float(),
        ),
        dim=-1,
    )


def rms(value, weight, epsilon, *, cast_before_weight=False):
    source = value.float()
    normalized = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + epsilon)
    if cast_before_weight:
        normalized = normalized.to(value.dtype).float()
    return (normalized * weight.float()).to(value.dtype)


original_write = DSparkTargetKVDraftModel.write_target_kv
original_verify = TargetKVInjector.inject_verify
original_context = TargetKVInjector.ensure_context
original_target_verify = TargetVerifyExecutor.run_non_compact


@torch.no_grad()
def observed_write(self, *, target_kv, pool, positions, cache_loc):
    contract = self.target_kv_contract
    rope = contract.kv.rope_config
    assert not rope["interleaved"]
    features = []
    for layer in contract.kv.layers:
        key = target_kv[f"target_k.{layer.layer_id}"]
        if contract.feature_k_stage == "pre_rope":
            key = rotate_split_half(
                key, positions, rope["theta"], rope["rotary_dim"], inverse=True
            )
        features.extend(
            (
                key.float().flatten(1),
                target_kv[f"target_v.{layer.layer_id}"].float().flatten(1),
            )
        )
    encoded = F.linear(
        torch.cat(features, dim=-1).to(self.kv_encoder.projection.weight.dtype),
        self.kv_encoder.projection.weight,
    )
    encoded = rms(encoded, self.kv_encoder.norm_weight, contract.encoder.rms_norm_eps)
    torch.testing.assert_close(
        self.kv_encoder(target_kv, positions), encoded, rtol=1e-4, atol=1e-4
    )
    original_write(
        self, target_kv=target_kv, pool=pool, positions=positions, cache_loc=cache_loc
    )
    errors = {}
    for layer_id, layer in enumerate(self.layers):
        attn = layer.self_attn
        context_weight = attn.qkv_proj.weight[
            attn.q_size : attn.q_size + 2 * attn.kv_size
        ]
        key, value = F.linear(encoded, context_weight).chunk(2, dim=-1)
        key = rms(
            key.reshape(-1, attn.num_kv_heads, attn.head_dim),
            attn.k_norm.weight,
            attn.k_norm.variance_epsilon,
            cast_before_weight=True,
        )
        key = rotate_split_half(
            key,
            positions,
            attn.rotary_emb.base,
            attn.rotary_emb.rotary_dim,
            round_intermediates=True,
        ).to(key.dtype)
        value = value.reshape_as(key)
        for component, expected, buffer in (
            ("k", key, pool.get_key_buffer(layer_id)),
            ("v", value, pool.get_value_buffer(layer_id)),
        ):
            actual = buffer[cache_loc.long()]
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
            errors[f"{component}.{layer_id}"] = (
                (actual.float() - expected.float()).abs().max().item()
            )
    record(
        self,
        {
            "kind": "projection",
            "num_tokens": positions.numel(),
            "start": int(positions[0]),
            "end": int(positions[-1]) + 1,
            "max_abs": errors,
        },
    )


def observed_verify(self, *, batch, verify_window, commit_lens):
    original_verify(
        self, batch=batch, verify_window=verify_window, commit_lens=commit_lens
    )
    record(
        self.draft_model,
        {
            "kind": "verify",
            "num_commit": commit_lens.tolist(),
            "num_reject": verify_window.verify_cache_loc_2d.numel()
            - int(commit_lens.sum()),
        },
    )


def observed_context(self, batch):
    previous = [req.dspark_projected_context for req in batch.reqs]
    original_context(self, batch)
    for req, before, end in zip(
        batch.reqs, previous, batch.seq_lens_cpu.tolist(), strict=True
    ):
        record(
            self.draft_model,
            {
                "kind": "context",
                "rid": req.rid,
                "retraction_ct": req.retraction_count,
                "previous_end": before.end if before is not None else None,
                "projected_end": req.dspark_projected_context.end,
                "prefix_end": end,
            },
        )


def observed_target_verify(self, **kwargs):
    result = original_target_verify(self, **kwargs)
    assert result.logits_output is None or result.logits_output.hidden_states is None
    record(
        self.kv_injector.draft_model,
        {"kind": "target_verify", "cuda_graph": result.can_run_cuda_graph},
    )
    return result


DSparkTargetKVDraftModel.write_target_kv = observed_write
TargetKVInjector.inject_verify = observed_verify
TargetKVInjector.ensure_context = observed_context
TargetVerifyExecutor.run_non_compact = observed_target_verify
install_capture_observer()


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
