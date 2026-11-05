"""Manual CUDA regression for physical expert profiles through the native Engine.

Run from the repository root, with one or four visible CUDA GPUs:
    PYTHONPATH=python python test/manual/models/test_qwen_moe_expert_profile.py \
        --tp-size 4 --work-dir scratch/expert-profile-test

The work directory must be new. Synthetic checkpoints, exported rank-local
weights and results remain there for inspection. No model download is needed.
"""

import argparse
import json
import math
import signal
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3MoeConfig

import sglang as sgl
from sglang.srt.environ import envs

KEEP = {4: [[7, 1, 5, 3], [6, 2, 4, 0]], 2: [[7, 1], [6, 2]]}
PROMPTS = [[1, 17, 43, 97, 211, 307], [1, 233, 29, 51, 13, 19, 487]]


def make_checkpoints(root, top_k):
    generator = torch.Generator().manual_seed(417)
    config = Qwen3MoeConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=top_k,
        max_position_embeddings=512,
        norm_topk_prob=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    config.dtype = "bfloat16"
    weights = {}

    def random_weight(name, shape):
        weights[name] = (torch.randn(shape, generator=generator) * 0.1).to(
            torch.bfloat16
        )

    random_weight("model.embed_tokens.weight", (512, 256))
    random_weight("lm_head.weight", (512, 256))
    weights["model.norm.weight"] = torch.ones(256, dtype=torch.bfloat16)
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            weights[f"{prefix}.{norm}.weight"] = torch.ones(256, dtype=torch.bfloat16)
        for norm in ("q_norm", "k_norm"):
            weights[f"{prefix}.self_attn.{norm}.weight"] = torch.ones(
                64, dtype=torch.bfloat16
            )
        for projection, rows in (
            ("q_proj", 256),
            ("k_proj", 64),
            ("v_proj", 64),
            ("o_proj", 256),
        ):
            random_weight(f"{prefix}.self_attn.{projection}.weight", (rows, 256))
        random_weight(f"{prefix}.mlp.gate.weight", (8, 256))
        for expert in range(8):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                random_weight(
                    f"{prefix}.mlp.experts.{expert}.{projection}.weight", (256, 256)
                )
    full = root / "full"
    full.mkdir()
    config.save_pretrained(full)
    save_file(weights, str(full / "model.safetensors"), metadata={"format": "pt"})
    for count, keep in KEEP.items():
        if count < top_k:
            continue
        (root / f"profile{count}.json").write_text(
            json.dumps({"num_experts": 8, "keep": keep})
        )
        compact = {
            name: value for name, value in weights.items() if ".mlp." not in name
        }
        for layer, ids in enumerate(keep):
            prefix = f"model.layers.{layer}.mlp"
            compact[f"{prefix}.gate.weight"] = weights[f"{prefix}.gate.weight"][
                ids
            ].contiguous()
            for slot, expert in enumerate(ids):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    compact[f"{prefix}.experts.{slot}.{projection}.weight"] = weights[
                        f"{prefix}.experts.{expert}.{projection}.weight"
                    ]
        destination = root / f"reference{count}"
        destination.mkdir()
        config.num_experts = count
        config.save_pretrained(destination)
        save_file(
            compact, str(destination / "model.safetensors"), metadata={"format": "pt"}
        )
    return weights


def check_exported_weights(folder, original, keep, tp_size, compact_router):
    shapes = {}
    for rank in range(tp_size):
        shard = load_file(str(folder / f"rank-{rank}.safetensors"))
        start, end = rank * (256 // tp_size), (rank + 1) * (256 // tp_size)
        for layer, ids in enumerate(keep):
            prefix = f"model.layers.{layer}.mlp"
            gate = original[f"{prefix}.gate.weight"]
            torch.testing.assert_close(
                shard[f"{prefix}.gate.weight"],
                gate[ids] if compact_router else gate,
                rtol=0,
                atol=0,
            )
            gate_up = [
                torch.stack(
                    [
                        original[f"{prefix}.experts.{expert}.{projection}.weight"][
                            start:end
                        ]
                        for expert in ids
                    ]
                )
                for projection in ("gate_proj", "up_proj")
            ]
            if envs.SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.get():
                expected_w13 = torch.stack(gate_up, dim=2).flatten(1, 2)
            else:
                expected_w13 = torch.cat(gate_up, dim=1)
            expected_w2 = torch.stack(
                [
                    original[f"{prefix}.experts.{expert}.down_proj.weight"][
                        :, start:end
                    ]
                    for expert in ids
                ]
            )
            for name, expected in (
                ("w13_weight", expected_w13),
                ("w2_weight", expected_w2),
            ):
                key = f"{prefix}.experts.{name}"
                torch.testing.assert_close(shard[key], expected, rtol=0, atol=0)
                shapes[f"rank{rank}.{key}"] = list(shard[key].shape)
    return shapes


def check_gpu_routing(top_k):
    from sglang.srt.eplb.expert_location import (
        ExpertLocationMetadata,
        _compute_logical_to_all_physical_map,
    )
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts
    from sglang.srt.runtime_context import get_context

    profile = json.dumps({"num_experts": 8, "keep": [KEEP[4][0]]})
    with get_context().override_server_args(
        device="cuda",
        tp_size=1,
        ep_size=1,
        moe_a2a_backend="none",
        ep_dispatch_algorithm="dynamic",
        moe_expert_profile=profile,
    ):
        physical = torch.tensor([KEEP[4][0]], device="cuda")
        logical = _compute_logical_to_all_physical_map(physical, 8, 1, 0)
        metadata = ExpertLocationMetadata._init_raw(1, physical, logical, moe_ep_rank=0)
        info = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="dynamic",
            partial_logical_to_rank_dispatch_physical_map=None,
            partial_logical_to_all_physical_map=metadata.logical_to_all_physical_map[0],
            partial_logical_to_all_physical_map_num_valid=metadata.logical_to_all_physical_map_num_valid[
                0
            ],
            num_physical_experts=4,
            rank_invariant=True,
        )
        logits = torch.tensor(
            [[100.0, 2.0, 99.0, 1.0, 98.0, 3.0, 97.0, 4.0]], device="cuda"
        )
        output = select_experts(
            torch.zeros(1, 256, device="cuda"),
            logits,
            TopKConfig(top_k=top_k),
            expert_location_dispatch_info=info,
        )
        expected = torch.tensor([[0, 2, 1, 3][:top_k]], device="cuda")
        actual_ids, order = output.topk_ids.long().sort(-1)
        expected_ids, expected_order = expected.sort(-1)
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        expected_weights = torch.arange(4.0, 4.0 - top_k, -1, device="cuda").softmax(
            -1
        )[None]
        torch.testing.assert_close(
            output.topk_weights.gather(-1, order),
            expected_weights.gather(-1, expected_order),
            rtol=1e-5,
            atol=1e-6,
        )


def run_engine(args, name, original):
    is_profile = name.startswith("profile")
    model = args.work_dir / ("full" if is_profile else name)
    engine = sgl.Engine(
        model_path=str(model),
        moe_expert_profile=str(args.work_dir / f"{name}.json") if is_profile else None,
        tp_size=args.tp_size,
        port=args.port,
        dtype="bfloat16",
        moe_runner_backend="triton",
        attention_backend=args.attention_backend,
        skip_tokenizer_init=True,
        context_length=512,
        max_total_tokens=1024,
        max_running_requests=2,
        chunked_prefill_size=256,
        mem_fraction_static=0.15,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        random_seed=417,
    )
    try:
        outputs = [
            engine.generate(
                input_ids=prompt,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                return_logprob=True,
                logprob_start_len=0,
            )
            for prompt in PROMPTS
        ]
        assert all(
            isinstance(output, dict) and len(output["output_ids"]) == 8
            for output in outputs
        )
        (args.work_dir / f"{name}-responses.json").write_text(
            json.dumps(outputs, indent=2)
        )
        exported = args.work_dir / f"{name}-weights"
        exported.mkdir()
        engine.save_sharded_model(path=str(exported), pattern="rank-{rank}.safetensors")
        keep = KEEP[int(name[-1])] if name != "full" else [list(range(8))] * 2
        shapes = check_exported_weights(
            exported, original, keep, args.tp_size, name.startswith("reference")
        )
        (args.work_dir / f"{name}-shapes.json").write_text(json.dumps(shapes, indent=2))
        return outputs
    finally:
        engine.shutdown()


def compare(profile, reference, tolerance):
    differences = []
    for left, right in zip(profile, reference, strict=True):
        assert left["output_ids"] == right["output_ids"]
        for key in ("input_token_logprobs", "output_token_logprobs"):
            for a, b in zip(
                left["meta_info"][key], right["meta_info"][key], strict=True
            ):
                assert a[1] == b[1] and (a[0] is None) == (b[0] is None)
                if a[0] is not None:
                    assert math.isfinite(a[0]) and math.isfinite(b[0])
                    differences.append(abs(a[0] - b[0]))
    assert len(differences) == 27
    maximum = max(differences)
    assert maximum <= tolerance, (maximum, tolerance)
    return {
        "max_logprob_difference": maximum,
        "compared_logprobs": len(differences),
        "tolerance": tolerance,
    }


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31060)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--logprob-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    top_k = 4 if args.tp_size == 4 else 2
    original = make_checkpoints(args.work_dir, top_k)
    result = {
        "tp_size": args.tp_size,
        "token_top_k": top_k,
        "status": "running",
        "comparisons": {},
    }
    counts = (4,) if args.tp_size == 4 else (4, 2)
    try:
        run_engine(args, "full", original)
        for count in counts:
            reference = run_engine(args, f"reference{count}", original)
            profile = run_engine(args, f"profile{count}", original)
            result["comparisons"][f"keep{count}"] = compare(
                profile, reference, args.logprob_tolerance
            )
        check_gpu_routing(top_k)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        (args.work_dir / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
