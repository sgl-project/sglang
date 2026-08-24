#!/usr/bin/env python3

"""Check DVR target scores against deterministic target prefill."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DVR_ALGORITHMS = {
    "DECODE_VERIFY_ROLLBACK",
    "DECODE_VERIFY_ROLLBACK_DFLASH",
    "DECODE_VERIFY_ROLLBACK_EAGLE",
}

SMOKE_OUTPUT_TOKENS = 65
RELEASE_OUTPUT_TOKENS = 10_000

TARGET_SERVER_FIELDS = (
    "model_path",
    "revision",
    "load_format",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "tp_size",
    "pp_size",
    "attention_backend",
    "decode_attention_backend",
    "prefill_attention_backend",
    "linear_attn_backend",
    "linear_attn_decode_backend",
    "linear_attn_prefill_backend",
    "page_size",
    "context_length",
)


def post_json(base_url: str, endpoint: str, payload=None):
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={} if payload is None else {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = response.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def read_server_contract(base_url: str, role: str) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/server_info", timeout=30
    ) as response:
        info = json.loads(response.read().decode())

    algorithm = str(info.get("speculative_algorithm") or "").upper()
    if info.get("enable_deterministic_inference") is not True:
        raise AssertionError(f"{role} server must enable deterministic inference")
    if info.get("enable_prefill_only_deterministic_inference") is True:
        raise AssertionError(f"{role} server must use full deterministic inference")
    if role == "dvr" and algorithm not in DVR_ALGORITHMS:
        raise AssertionError(f"DVR server has speculative_algorithm={algorithm!r}")
    if role == "deterministic" and algorithm:
        raise AssertionError("Deterministic reference must not enable speculation")

    return {
        "role": role,
        "speculative_algorithm": algorithm or None,
        "disable_radix_cache": info.get("disable_radix_cache"),
        "target": {field: info.get(field) for field in TARGET_SERVER_FIELDS},
    }


def flush_cache(base_url: str) -> None:
    for attempt in range(100):
        try:
            result = post_json(base_url, "/flush_cache")
        except urllib.error.HTTPError as error:
            if error.code == 400 and attempt < 99:
                time.sleep(0.1)
                continue
            raise
        if isinstance(result, str) and "Cache flushed" not in result:
            raise RuntimeError(f"Cache flush failed: {result}")
        return


def token_logprobs(response, field: str) -> list[tuple[float, int]]:
    return [
        (float(item[0]), int(item[1]))
        for item in response["meta_info"][field]
        if item[0] is not None
    ]


def compare(name: str, expected, actual) -> dict[str, Any]:
    if len(expected) != len(actual):
        raise AssertionError(f"{name}: length {len(actual)} != {len(expected)}")

    diffs = []
    for index, (expected_item, actual_item) in enumerate(
        zip(expected, actual, strict=True)
    ):
        expected_lp, expected_id = expected_item
        actual_lp, actual_id = actual_item
        if expected_id != actual_id:
            raise AssertionError(
                f"{name}: token {index} differs: {actual_id} != {expected_id}"
            )
        diffs.append(abs(actual_lp - expected_lp))

    result = {
        "tokens": len(expected),
        "maxdiff": max(diffs, default=0.0),
    }
    result["exact"] = result["maxdiff"] == 0.0
    print(f"{name}: {json.dumps(result, sort_keys=True)}")
    if not result["exact"]:
        first = next(index for index, diff in enumerate(diffs) if diff)
        raise AssertionError(
            f"{name}: first selected-token logprob mismatch at generated token {first}"
        )
    return result


def load_prompt(args) -> list[int]:
    if args.input_ids_json is None:
        return list(
            range(args.prompt_token_start, args.prompt_token_start + args.prompt_length)
        )
    payload = json.loads(args.input_ids_json.read_text())
    return list(payload["input_ids"] if isinstance(payload, dict) else payload)


def sampling_policy(args) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "sampling_seed": args.sampling_seed,
    }


def check_context_capacity(server_contract, prompt_length, output_length) -> None:
    context_length = server_contract["target"].get("context_length")
    if context_length is None:
        return
    required_length = prompt_length + output_length
    if required_length > int(context_length):
        raise AssertionError(
            f"Requested sequence length {required_length} exceeds server "
            f"context length {context_length}"
        )


def generate(
    args,
    input_ids,
    max_new_tokens,
    policy,
    *,
    return_logprob=True,
    logprob_start_len=0,
):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            **policy,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": return_logprob,
    }
    if return_logprob:
        payload.update(
            return_text_in_logprobs=False,
            logprob_start_len=logprob_start_len,
        )
    response = post_json(args.base_url, "/generate", payload)
    if not isinstance(response, dict):
        raise AssertionError(f"Generate returned a non-object response: {response!r}")
    if max_new_tokens > 0:
        output_ids = response.get("output_ids")
        if not isinstance(output_ids, list):
            raise AssertionError("Generate response is missing output_ids")
        if len(output_ids) != max_new_tokens:
            raise AssertionError(
                f"Generate returned {len(output_ids)} tokens; "
                f"expected {max_new_tokens}. Check context and token capacity."
            )
    return response


def repeated_prefill(name, input_ids, policy, args):
    baseline = None
    maxdiff = 0.0
    for repeat in range(args.prefill_repeats):
        flush_cache(args.base_url)
        response = generate(args, input_ids, 0, policy)
        scores = token_logprobs(response, "input_token_logprobs")
        if baseline is None:
            baseline = scores
        else:
            result = compare(f"{name} repeat {repeat + 1}", baseline, scores)
            maxdiff = max(maxdiff, result["maxdiff"])

    gate = {
        "repeats": args.prefill_repeats,
        "tokens": len(baseline),
        "maxdiff": maxdiff,
        "exact": maxdiff == 0.0,
    }
    print(f"{name}: {json.dumps(gate, sort_keys=True)}")
    return baseline, gate


def generated_prefill_scores(name, forced_ids, prompt_length, policy, args):
    scores, gate = repeated_prefill(name, forced_ids, policy, args)
    generated_scores = scores[prompt_length - 1 :]
    expected_length = len(forced_ids) - prompt_length
    if len(generated_scores) != expected_length:
        raise AssertionError(
            f"{name}: generated score count {len(generated_scores)} "
            f"!= {expected_length}"
        )
    return generated_scores, gate


def hot_update_checkpoint(args, server_contract, prompt_ids, policy):
    source_model_path = server_contract["target"]["model_path"]
    if source_model_path == args.hot_update_checkpoint:
        raise AssertionError(
            "--hot-update-checkpoint must differ from the checkpoint used to "
            "start the DVR server"
        )

    probe_tokens = min(args.max_new_tokens, args.hot_update_probe_tokens)
    flush_cache(args.base_url)
    before_response = generate(
        args,
        prompt_ids,
        probe_tokens,
        policy,
        return_logprob=True,
    )
    payload = {
        "model_path": args.hot_update_checkpoint,
        "recapture_cuda_graph": False,
        "flush_cache": True,
    }
    load_format = (
        args.hot_update_load_format or server_contract["target"]["load_format"]
    )
    if load_format is not None:
        payload["load_format"] = load_format
    result = post_json(
        args.base_url,
        "/update_weights_from_disk",
        payload,
    )
    if not isinstance(result, dict) or result.get("success") is not True:
        raise AssertionError(f"DVR in-place weight update failed: {result!r}")

    updated_contract = read_server_contract(args.base_url, "dvr")
    if updated_contract["target"]["model_path"] != args.hot_update_checkpoint:
        raise AssertionError(
            "Server did not publish the updated model path: "
            f"{updated_contract['target']['model_path']!r} != "
            f"{args.hot_update_checkpoint!r}"
        )
    print(f"DVR in-place weight update: {json.dumps(result, sort_keys=True)}")
    return updated_contract, {
        "source_model_path": source_model_path,
        "target_model_path": args.hot_update_checkpoint,
        "recapture_cuda_graph": False,
        "flush_cache": True,
        "probe_tokens": probe_tokens,
        "before_output_ids": list(before_response["output_ids"]),
        "before_output_scores": token_logprobs(
            before_response, "output_token_logprobs"
        ),
        "update_result": result,
    }


def finish_hot_update_gate(gate, after_response):
    probe_tokens = gate["probe_tokens"]
    before_ids = gate["before_output_ids"]
    after_ids = list(after_response["output_ids"][:probe_tokens])
    if len(after_ids) != probe_tokens:
        raise AssertionError(
            f"Post-update probe returned {len(after_ids)} tokens; "
            f"expected {probe_tokens}"
        )

    first_token_diff = next(
        (
            index
            for index, (before_id, after_id) in enumerate(
                zip(before_ids, after_ids, strict=True)
            )
            if before_id != after_id
        ),
        None,
    )
    score_maxdiff = None
    if first_token_diff is None:
        before_scores = gate["before_output_scores"]
        after_scores = token_logprobs(after_response, "output_token_logprobs")[
            :probe_tokens
        ]
        if len(before_scores) != probe_tokens or len(after_scores) != probe_tokens:
            raise AssertionError("Weight-update probe is missing output token scores")
        score_maxdiff = max(
            (
                abs(before_score[0] - after_score[0])
                for before_score, after_score in zip(
                    before_scores, after_scores, strict=True
                )
            ),
            default=0.0,
        )

    gate.update(
        {
            "first_token_diff": first_token_diff,
            "selected_logprob_maxdiff": score_maxdiff,
            "changed": first_token_diff is not None or score_maxdiff != 0.0,
        }
    )
    if not gate["changed"]:
        raise AssertionError(
            "The pre-update and post-update probes are identical. Use an "
            "observably perturbed source checkpoint."
        )
    print(f"DVR in-place weight update gate: {json.dumps(gate, sort_keys=True)}")


def check_radix_reuse(args, prompt_ids, policy):
    flush_cache(args.base_url)
    seed_response = generate(
        args,
        prompt_ids,
        args.radix_seed_tokens,
        policy,
        return_logprob=False,
    )
    child_prompt = prompt_ids + list(seed_response["output_ids"])
    child_response = generate(
        args,
        child_prompt,
        args.radix_child_tokens,
        policy,
        return_logprob=True,
        # Preserve prefix reuse while requesting generated-token scores.
        logprob_start_len=-1,
    )
    cached_tokens = int(child_response["meta_info"].get("cached_tokens", -1))
    visible_seed_tokens = len(child_prompt) - 1
    expected_cached_tokens = (
        visible_seed_tokens // args.radix_chunk_size * args.radix_chunk_size
    )
    gate = {
        "cached_tokens": cached_tokens,
        "expected_cached_tokens": expected_cached_tokens,
        "exact": cached_tokens == expected_cached_tokens,
    }
    print(f"Generated-prefix Radix reuse: {json.dumps(gate, sort_keys=True)}")
    if not gate["exact"]:
        raise AssertionError(
            "Generated-prefix Radix reuse mismatch: "
            f"{cached_tokens} != {expected_cached_tokens}"
        )

    output_ids = list(child_response["output_ids"])
    forced_ids = child_prompt + output_ids
    prefill_scores, prefill_gate = generated_prefill_scores(
        "DVR Radix-hit repeated prefill",
        forced_ids,
        len(child_prompt),
        policy,
        args,
    )
    decode_scores = token_logprobs(child_response, "output_token_logprobs")
    trajectory = {
        "prompt_length": len(child_prompt),
        "output_ids": output_ids,
        "forced_ids": forced_ids,
        "dvr_prefill_scores": prefill_scores,
        "dvr_prefill_repeat_gate": prefill_gate,
        "dvr_decode_scores": decode_scores,
        "decode_prefill_gate": compare(
            "DVR Radix-hit decode -> DVR prefill",
            decode_scores,
            prefill_scores,
        ),
    }
    return trajectory, gate


def run_dvr(args) -> None:
    server_contract = read_server_contract(args.base_url, "dvr")
    prompt_ids = load_prompt(args)
    check_context_capacity(server_contract, len(prompt_ids), args.max_new_tokens)
    policy = sampling_policy(args)
    trajectories = {}

    hot_update_gate = None
    if args.hot_update_checkpoint is not None:
        server_contract, hot_update_gate = hot_update_checkpoint(
            args, server_contract, prompt_ids, policy
        )

    trajectory_modes = [("logged", True)]
    if not args.logged_only:
        trajectory_modes.append(("no_logprob", False))
    for name, return_logprob in trajectory_modes:
        flush_cache(args.base_url)
        response = generate(
            args,
            prompt_ids,
            args.max_new_tokens,
            policy,
            return_logprob=return_logprob,
        )
        output_ids = list(response["output_ids"])
        forced_ids = prompt_ids + output_ids
        prefill_scores, prefill_gate = generated_prefill_scores(
            f"DVR {name} repeated prefill",
            forced_ids,
            len(prompt_ids),
            policy,
            args,
        )
        trajectory = {
            "prompt_length": len(prompt_ids),
            "output_ids": output_ids,
            "forced_ids": forced_ids,
            "dvr_prefill_scores": prefill_scores,
            "dvr_prefill_repeat_gate": prefill_gate,
        }
        if return_logprob:
            decode_scores = token_logprobs(response, "output_token_logprobs")
            trajectory["dvr_decode_scores"] = decode_scores
            trajectory["decode_prefill_gate"] = compare(
                "DVR decode -> DVR prefill", decode_scores, prefill_scores
            )
            if hot_update_gate is not None:
                finish_hot_update_gate(hot_update_gate, response)
        trajectories[name] = trajectory

    radix_gate = None
    if args.check_radix_reuse:
        if server_contract["disable_radix_cache"] is True:
            raise AssertionError("Radix reuse check requires Radix cache to be enabled")
        trajectories["radix_hit"], radix_gate = check_radix_reuse(
            args, prompt_ids, policy
        )

    artifact = {
        "schema": 2,
        "dvr_server_contract": server_contract,
        "prompt_ids": prompt_ids,
        "sampling_policy": policy,
        "qualification": {
            "profile": args.qualification_profile,
            "requested_output_tokens": args.max_new_tokens,
            "includes_no_logprob_trajectory": not args.logged_only,
        },
        "trajectories": trajectories,
    }
    if hot_update_gate is not None:
        artifact["weight_update_gate"] = hot_update_gate
    if radix_gate is not None:
        artifact["radix_reuse_gate"] = radix_gate
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote {args.artifact}")


def run_det(args) -> None:
    artifact = json.loads(args.artifact.read_text())
    if artifact.get("schema") != 2:
        raise AssertionError("Regenerate the artifact with the current client")

    dvr_contract = artifact["dvr_server_contract"]
    det_contract = read_server_contract(args.base_url, "deterministic")
    if dvr_contract["target"] != det_contract["target"]:
        raise AssertionError(
            "DVR and deterministic target configurations differ: "
            f"dvr={dvr_contract['target']}, det={det_contract['target']}"
        )

    default_prompt_length = len(artifact["prompt_ids"])
    policy = artifact["sampling_policy"]
    for name, trajectory in artifact["trajectories"].items():
        prompt_length = trajectory.get("prompt_length", default_prompt_length)
        det_scores, repeat_gate = generated_prefill_scores(
            f"Deterministic {name} repeated prefill",
            trajectory["forced_ids"],
            prompt_length,
            policy,
            args,
        )
        trajectory["det_prefill_scores"] = det_scores
        trajectory["det_prefill_repeat_gate"] = repeat_gate
        trajectory["prefill_policy_gate"] = compare(
            f"DVR {name} prefill -> deterministic prefill",
            [tuple(item) for item in trajectory["dvr_prefill_scores"]],
            det_scores,
        )
        if "dvr_decode_scores" in trajectory:
            trajectory["deterministic_target_gate"] = compare(
                "DVR decode -> ordinary deterministic prefill",
                [tuple(item) for item in trajectory["dvr_decode_scores"]],
                det_scores,
            )

    artifact["deterministic_server_contract"] = det_contract
    args.artifact.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Updated {args.artifact}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("dvr", "det"))
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input-ids-json", type=Path)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--prompt-token-start", type=int, default=1000)
    parser.add_argument(
        "--qualification-profile",
        choices=("smoke", "release"),
        default="smoke",
        help="Use release to require at least 10,000 generated tokens.",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--logged-only",
        action="store_true",
        help="Skip the independent no-logprob trajectory.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--prefill-repeats", type=int, default=2)
    parser.add_argument(
        "--hot-update-checkpoint",
        help=(
            "Load this correct checkpoint in place after probing a distinct "
            "perturbed startup checkpoint. CUDA graphs are not recaptured."
        ),
    )
    parser.add_argument("--hot-update-load-format")
    parser.add_argument("--hot-update-probe-tokens", type=int, default=65)
    parser.add_argument("--check-radix-reuse", action="store_true")
    parser.add_argument("--radix-seed-tokens", type=int, default=512)
    parser.add_argument("--radix-child-tokens", type=int, default=65)
    parser.add_argument("--radix-chunk-size", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = (
            RELEASE_OUTPUT_TOKENS
            if args.qualification_profile == "release"
            else SMOKE_OUTPUT_TOKENS
        )
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if (
        args.qualification_profile == "release"
        and args.max_new_tokens < RELEASE_OUTPUT_TOKENS
    ):
        parser.error(
            f"release qualification requires at least {RELEASE_OUTPUT_TOKENS} "
            "generated tokens"
        )
    if args.prefill_repeats < 2:
        parser.error("--prefill-repeats must be at least 2")
    if args.hot_update_probe_tokens <= 0:
        parser.error("--hot-update-probe-tokens must be positive")
    if (
        min(
            args.radix_seed_tokens,
            args.radix_child_tokens,
            args.radix_chunk_size,
        )
        <= 0
    ):
        parser.error("Radix token counts and chunk size must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    (run_dvr if arguments.phase == "dvr" else run_det)(arguments)
