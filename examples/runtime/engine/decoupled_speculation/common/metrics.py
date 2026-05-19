from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .runtime import get_decoupled_spec_actor_env_vars
from .types import ModeMetrics, PromptSample


def _get_real_verify_acceptance_stats(
    meta_info: dict[str, Any],
) -> tuple[float | None, float | None, int, int, int]:
    """Extract draft-token acceptance metrics from current SGLang meta_info."""
    verify_ct = meta_info.get("spec_verify_ct")
    accepted_tokens = meta_info.get("spec_accepted_drafts")
    draft_tokens = meta_info.get("spec_proposed_drafts")
    if verify_ct is None or accepted_tokens is None or draft_tokens is None:
        return None, None, 0, 0, 0

    verify_ct = int(verify_ct)
    accepted_tokens = int(accepted_tokens)
    draft_tokens = int(draft_tokens)
    if verify_ct <= 0 or draft_tokens <= 0:
        return None, None, 0, 0, 0

    # Acclen reports accepted draft tokens only; the verifier-sampled bonus
    # token is intentionally excluded.
    accept_length = accepted_tokens / verify_ct
    accept_rate = accepted_tokens / draft_tokens
    return accept_length, accept_rate, accepted_tokens, draft_tokens, verify_ct


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_list_or_empty(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    counts: list[int] = []
    for item in value:
        try:
            counts.append(int(item))
        except (TypeError, ValueError):
            counts.append(0)
    return counts


def _add_position_counts(total: list[int], update: list[int]) -> None:
    if len(update) > len(total):
        total.extend([0] * (len(update) - len(total)))
    for index, value in enumerate(update):
        total[index] += value


def _position_accept_rates(
    valid_draft_tokens_by_position: list[int],
    valid_accepted_tokens_by_position: list[int],
) -> list[float | None]:
    rates: list[float | None] = []
    for index, draft_tokens in enumerate(valid_draft_tokens_by_position):
        accepted_tokens = (
            valid_accepted_tokens_by_position[index]
            if index < len(valid_accepted_tokens_by_position)
            else 0
        )
        rates.append(
            accepted_tokens / draft_tokens if draft_tokens > 0 else None
        )
    return rates


def _get_valid_draft_acceptance_stats(
    meta_info: dict[str, Any],
) -> tuple[int, int, float | None, list[int], list[int], list[float | None]]:
    valid_draft_tokens = meta_info.get("spec_valid_draft_token_num")
    valid_accepted_tokens = meta_info.get("spec_valid_accept_token_num")
    if valid_draft_tokens is None or valid_accepted_tokens is None:
        return 0, 0, None, [], [], []

    valid_draft_tokens = int(valid_draft_tokens)
    valid_accepted_tokens = int(valid_accepted_tokens)
    valid_draft_tokens_by_position = _int_list_or_empty(
        meta_info.get("spec_valid_draft_token_num_by_position")
    )
    valid_accepted_tokens_by_position = _int_list_or_empty(
        meta_info.get("spec_valid_accept_token_num_by_position")
    )
    valid_accept_rate_by_position = _position_accept_rates(
        valid_draft_tokens_by_position,
        valid_accepted_tokens_by_position,
    )
    if valid_draft_tokens <= 0:
        return (
            valid_draft_tokens,
            valid_accepted_tokens,
            None,
            valid_draft_tokens_by_position,
            valid_accepted_tokens_by_position,
            valid_accept_rate_by_position,
        )

    valid_accept_rate = _float_or_none(meta_info.get("spec_valid_accept_rate"))
    if valid_accept_rate is None:
        valid_accept_rate = valid_accepted_tokens / valid_draft_tokens
    return (
        valid_draft_tokens,
        valid_accepted_tokens,
        valid_accept_rate,
        valid_draft_tokens_by_position,
        valid_accepted_tokens_by_position,
        valid_accept_rate_by_position,
    )


def _get_decoupled_verify_acceptance_stats(
    meta_info: dict[str, Any],
) -> tuple[float | None, float | None, int, int, int]:
    """Return (accept_length, accept_rate, accepted, proposed_drafts, verify_ct)."""
    verify_ct = meta_info.get("spec_verify_ct")
    valid_accepted_tokens = meta_info.get("spec_valid_accept_token_num")
    num_proposed_drafts = meta_info.get("spec_num_proposed_drafts")
    if (
        verify_ct is None
        or valid_accepted_tokens is None
        or num_proposed_drafts is None
    ):
        return None, None, 0, 0, 0

    verify_ct = int(verify_ct)
    valid_accepted_tokens = int(valid_accepted_tokens)
    num_proposed_drafts = int(num_proposed_drafts)
    if verify_ct <= 0:
        return None, None, 0, num_proposed_drafts, 0

    accept_length = valid_accepted_tokens / verify_ct
    accept_rate = (
        valid_accepted_tokens / num_proposed_drafts
        if num_proposed_drafts > 0
        else None
    )
    return (
        accept_length,
        accept_rate,
        valid_accepted_tokens,
        num_proposed_drafts,
        verify_ct,
    )


def collect_mode_metrics(
    *,
    mode: str,
    outputs: list[dict[str, Any]],
    prompt_samples: list[PromptSample],
    verifier_assignments: list[int] | None = None,
    include_output_text: bool = True,
) -> ModeMetrics:
    if len(outputs) != len(prompt_samples):
        raise RuntimeError(
            f"{mode} returned {len(outputs)} outputs for {len(prompt_samples)} prompts"
        )
    if verifier_assignments is not None and len(verifier_assignments) != len(
        prompt_samples
    ):
        raise RuntimeError(
            f"{mode} has {len(verifier_assignments)} verifier assignments for "
            f"{len(prompt_samples)} prompts"
        )

    total_generated_tokens = 0
    total_accepted_tokens = 0
    total_draft_tokens = 0
    total_valid_draft_tokens = 0
    total_valid_accepted_tokens = 0
    total_valid_draft_tokens_by_position: list[int] = []
    total_valid_accepted_tokens_by_position: list[int] = []
    total_verify_ct = 0
    per_request = []
    for index, (sample, output) in enumerate(zip(prompt_samples, outputs, strict=True)):
        output_ids = output.get("output_ids", [])
        generated_tokens = len(output_ids) if isinstance(output_ids, list) else 0
        total_generated_tokens += generated_tokens

        meta_info = output.get("meta_info", {}) or {}
        (
            valid_draft_tokens,
            valid_accepted_tokens,
            valid_accept_rate,
            valid_draft_tokens_by_position,
            valid_accepted_tokens_by_position,
            valid_accept_rate_by_position,
        ) = _get_valid_draft_acceptance_stats(meta_info)
        if mode == "decoupled_spec":
            (
                accept_length,
                accept_rate,
                accepted_tokens,
                draft_tokens,
                verify_ct,
            ) = _get_decoupled_verify_acceptance_stats(meta_info)
        else:
            (
                accept_length,
                accept_rate,
                accepted_tokens,
                draft_tokens,
                verify_ct,
            ) = _get_real_verify_acceptance_stats(meta_info)
        total_accepted_tokens += accepted_tokens
        total_draft_tokens += draft_tokens
        total_valid_draft_tokens += valid_draft_tokens
        total_valid_accepted_tokens += valid_accepted_tokens
        _add_position_counts(
            total_valid_draft_tokens_by_position,
            valid_draft_tokens_by_position,
        )
        _add_position_counts(
            total_valid_accepted_tokens_by_position,
            valid_accepted_tokens_by_position,
        )
        total_verify_ct += verify_ct
        output_text = output.get("text", "")
        finish_reason = meta_info.get("finish_reason")
        request_latency_s = _float_or_none(meta_info.get("e2e_latency"))
        if request_latency_s is None:
            raise RuntimeError(
                f"{mode} output for batch_index={index}, row_index={sample.row_index} "
                "is missing meta_info['e2e_latency']; cannot compute request "
                "duration without script-side timing."
            )
        has_spec_acceptance_stats = (
            accept_length is not None
            or accept_rate is not None
            or accepted_tokens > 0
            or draft_tokens > 0
            or verify_ct > 0
        )

        request_metrics = {
            "batch_index": index,
            "row_index": sample.row_index,
            "verifier_rank": (
                verifier_assignments[index]
                if verifier_assignments is not None
                else None
            ),
            "prompt_text": sample.prompt,
            "prompt_tokens": sample.prompt_tokens,
            "generated_tokens": generated_tokens,
            "request_latency_s": request_latency_s,
            "spec_accept_length": accept_length,
            "spec_accept_rate": accept_rate,
            "spec_accepted_drafts": (
                accepted_tokens if has_spec_acceptance_stats else None
            ),
            "spec_proposed_drafts": (
                draft_tokens if has_spec_acceptance_stats else None
            ),
            "spec_valid_accept_rate": valid_accept_rate,
            "spec_valid_accept_token_num": valid_accepted_tokens or None,
            "spec_valid_draft_token_num": valid_draft_tokens or None,
            "spec_valid_accept_rate_by_position": valid_accept_rate_by_position,
            "spec_valid_accept_token_num_by_position": (
                valid_accepted_tokens_by_position
            ),
            "spec_valid_draft_token_num_by_position": valid_draft_tokens_by_position,
            "spec_verify_ct": verify_ct if has_spec_acceptance_stats else None,
            "finish_reason": finish_reason,
            "output_text_preview": (
                output_text[:512] if isinstance(output_text, str) else None
            ),
            "output_ids_head": (
                output_ids[:32] if isinstance(output_ids, list) else None
            ),
            "output_ids_tail": (
                output_ids[-32:] if isinstance(output_ids, list) else None
            ),
        }
        if include_output_text:
            request_metrics["output_text"] = (
                output_text if isinstance(output_text, str) else ""
            )
        per_request.append(request_metrics)

    generation_time_s = (
        max(item["request_latency_s"] for item in per_request) if per_request else 0.0
    )
    throughput = (
        total_generated_tokens / generation_time_s if generation_time_s > 0 else 0.0
    )
    avg_accept_length = (
        total_accepted_tokens / total_verify_ct if total_verify_ct > 0 else None
    )
    avg_accept_rate = (
        total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else None
    )
    avg_valid_accept_rate = (
        total_valid_accepted_tokens / total_valid_draft_tokens
        if total_valid_draft_tokens > 0
        else None
    )
    avg_valid_accept_rate_by_position = _position_accept_rates(
        total_valid_draft_tokens_by_position,
        total_valid_accepted_tokens_by_position,
    )
    return ModeMetrics(
        mode=mode,
        generation_time_s=generation_time_s,
        total_generated_tokens=total_generated_tokens,
        output_throughput_tok_per_s=throughput,
        per_request=per_request,
        avg_spec_accept_length=avg_accept_length,
        avg_spec_accept_rate=avg_accept_rate,
        avg_spec_valid_accept_rate=avg_valid_accept_rate,
        avg_spec_valid_accept_rate_by_position=avg_valid_accept_rate_by_position,
        total_spec_valid_draft_token_num=total_valid_draft_tokens,
        total_spec_valid_accept_token_num=total_valid_accepted_tokens,
        total_spec_valid_draft_token_num_by_position=(
            total_valid_draft_tokens_by_position
        ),
        total_spec_valid_accept_token_num_by_position=(
            total_valid_accepted_tokens_by_position
        ),
    )


_SPEC_VALID_TOP_LEVEL_KEYS = (
    "avg_spec_valid_accept_rate",
    "avg_spec_valid_accept_rate_by_position",
    "total_spec_valid_draft_token_num",
    "total_spec_valid_accept_token_num",
    "total_spec_valid_draft_token_num_by_position",
    "total_spec_valid_accept_token_num_by_position",
)

_SPEC_VALID_REQUEST_KEYS = (
    "spec_valid_accept_rate",
    "spec_valid_accept_rate_by_position",
    "spec_valid_accept_token_num",
    "spec_valid_accept_token_num_by_position",
    "spec_valid_draft_token_num",
    "spec_valid_draft_token_num_by_position",
)


def _mode_metrics_result_dict(metrics: ModeMetrics) -> dict[str, Any]:
    record = asdict(metrics)
    if metrics.mode == "mtp":
        for key in _SPEC_VALID_TOP_LEVEL_KEYS:
            record.pop(key, None)
        for item in record["per_request"]:
            for key in _SPEC_VALID_REQUEST_KEYS:
                item.pop(key, None)
    return record


def build_result(
    *,
    args: argparse.Namespace,
    target_nnodes: int,
    target_gpus_per_node: int,
    prompt_column: str,
    total_rows: int,
    prompt_samples: list[PromptSample],
    spec_metrics: ModeMetrics,
    baseline_metrics: ModeMetrics | None = None,
) -> dict[str, Any]:
    speedup = (
        baseline_metrics.generation_time_s / spec_metrics.generation_time_s
        if baseline_metrics is not None and spec_metrics.generation_time_s > 0
        else None
    )
    baseline = baseline_metrics.mode if baseline_metrics is not None else "none"
    actor_env_vars = get_decoupled_spec_actor_env_vars()
    result = {
        "config": {
            "dataset_path": args.dataset_path,
            "dataset_format": args.dataset_format,
            "prompt_column": prompt_column,
            "engine_input": "input_ids",
            "code_language": args.code_language,
            "offset": args.offset,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "max_new_tokens": args.context_length,
            "max_prompt_length": args.max_prompt_length,
            "target_model_path": args.target_model_path,
            "draft_model_path": args.draft_model_path,
            "tokenizer_path": args.tokenizer_path or args.target_model_path,
            "target_tp_size": args.target_tp_size,
            "target_ep_size": args.target_ep_size,
            "target_moe_a2a_backend": args.target_moe_a2a_backend,
            "num_verifier_replicas": args.num_verifier_replicas,
            "verify_ngpus": args.verify_ngpus,
            "draft_tp_size": args.draft_tp_size,
            "draft_ngpus": args.draft_ngpus,
            "num_speculative_steps": args.num_speculative_steps,
            "temperature": args.temperature,
            "deterministic": args.deterministic,
            "ignore_eos": args.ignore_eos,
            "nnodes": args.nnodes,
            "n_gpu_per_node": args.n_gpu_per_node,
            "target_nnodes": target_nnodes,
            "target_gpus_per_node": target_gpus_per_node,
            "num_draft_replicas": args.num_draft_replicas,
            "baseline": args.baseline,
            "show_responses": args.show_responses,
            "spec_trace_dir": args.spec_trace_dir,
            "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL": actor_env_vars[
                "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL"
            ],
        },
        "dataset": {
            "total_rows": total_rows,
            "loaded_rows": [sample.row_index for sample in prompt_samples],
            "total_prompt_tokens": sum(
                sample.prompt_tokens for sample in prompt_samples
            ),
            "prompt_samples": [
                {
                    "row_index": sample.row_index,
                    "prompt_tokens": sample.prompt_tokens,
                    "prompt": sample.prompt,
                    "prompt_head": sample.prompt[:1024],
                    "prompt_tail": sample.prompt[-1024:],
                }
                for sample in prompt_samples
            ],
        },
        "decoupled_spec": _mode_metrics_result_dict(spec_metrics),
    }
    if baseline_metrics is not None:
        result[baseline] = _mode_metrics_result_dict(baseline_metrics)
        result["baseline"] = baseline
        result["e2e_speedup"] = speedup
    return result


def _iter_output_modes(result: dict[str, Any]):
    yield "decoupled_spec", "decoupled-spec"
    if "decode" in result:
        yield "decode", "decode"
    if "mtp" in result:
        yield "mtp", "mtp"


def _request_output_record(item: dict[str, Any]) -> dict[str, Any]:
    steps = item["spec_verify_ct"] or item["generated_tokens"]
    return {
        "index": item["batch_index"],
        "offset": item["row_index"],
        "verifier-rank": item["verifier_rank"],
        "prompt-length": item["prompt_tokens"],
        "response-length": item["generated_tokens"],
        "steps": steps,
        "duration": item["request_latency_s"],
    }


def _csv_fieldnames_for_mode(mode_key: str) -> list[str]:
    fieldnames = [
        "index",
        "offset",
        "verifier-rank",
        "prompt-length",
        "response-length",
        "steps",
        "duration",
    ]
    if mode_key == "decoupled_spec":
        fieldnames.extend(
            [
                "num_speculative_steps",
            ]
        )
    if mode_key in ("decoupled_spec", "mtp"):
        fieldnames.extend(
            [
                "spec_accept_length",
                "spec_accept_rate",
                "spec_verify_ct",
                "spec_accepted_drafts",
                "spec_proposed_drafts",
            ]
        )
    if mode_key == "decoupled_spec":
        fieldnames.extend(
            [
                "spec_valid_accept_rate",
                "spec_valid_accept_rate_by_position",
                "spec_valid_accept_token_num",
                "spec_valid_accept_token_num_by_position",
                "spec_valid_draft_token_num",
                "spec_valid_draft_token_num_by_position",
            ]
        )
    return fieldnames


def _csv_output_record(
    mode_key: str,
    item: dict[str, Any],
    num_speculative_steps: int | None = None,
) -> dict[str, Any]:
    record = _request_output_record(item)
    if mode_key == "decoupled_spec":
        record.update({"num_speculative_steps": num_speculative_steps})
    if mode_key in ("decoupled_spec", "mtp"):
        record.update(
            {
                "spec_accept_length": item["spec_accept_length"],
                "spec_accept_rate": item["spec_accept_rate"],
                "spec_verify_ct": item["spec_verify_ct"],
                "spec_accepted_drafts": item["spec_accepted_drafts"],
                "spec_proposed_drafts": item["spec_proposed_drafts"],
            }
        )
    if mode_key == "decoupled_spec":
        record.update(
            {
                "spec_valid_accept_rate": item["spec_valid_accept_rate"],
                "spec_valid_accept_rate_by_position": json.dumps(
                    item["spec_valid_accept_rate_by_position"],
                    ensure_ascii=False,
                ),
                "spec_valid_accept_token_num": item["spec_valid_accept_token_num"],
                "spec_valid_accept_token_num_by_position": json.dumps(
                    item["spec_valid_accept_token_num_by_position"],
                    ensure_ascii=False,
                ),
                "spec_valid_draft_token_num": item["spec_valid_draft_token_num"],
                "spec_valid_draft_token_num_by_position": json.dumps(
                    item["spec_valid_draft_token_num_by_position"],
                    ensure_ascii=False,
                ),
            }
        )
    return record


def write_output_files(result: dict[str, Any], output_dir: str) -> list[Path]:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    num_speculative_steps = result.get("config", {}).get("num_speculative_steps")

    written_paths = []
    for mode_key, file_prefix in _iter_output_modes(result):
        mode_items = result[mode_key]["per_request"]

        csv_path = output_path / f"{file_prefix}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = _csv_fieldnames_for_mode(mode_key)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in mode_items:
                writer.writerow(
                    _csv_output_record(
                        mode_key,
                        item,
                        num_speculative_steps=num_speculative_steps,
                    )
                )
        written_paths.append(csv_path)

        json_path = output_path / f"{file_prefix}.json"
        requests = []
        for item in mode_items:
            record = _request_output_record(item)
            if mode_key in ("decoupled_spec", "mtp"):
                record.update(
                    {
                        "spec_accept_length": item["spec_accept_length"],
                        "spec_accept_rate": item["spec_accept_rate"],
                        "spec_verify_ct": item["spec_verify_ct"],
                        "spec_accepted_drafts": item["spec_accepted_drafts"],
                        "spec_proposed_drafts": item["spec_proposed_drafts"],
                    }
                )
            if mode_key == "decoupled_spec":
                record.update(
                    {
                        "spec_valid_accept_rate": item["spec_valid_accept_rate"],
                        "spec_valid_accept_rate_by_position": item[
                            "spec_valid_accept_rate_by_position"
                        ],
                        "spec_valid_accept_token_num": item[
                            "spec_valid_accept_token_num"
                        ],
                        "spec_valid_accept_token_num_by_position": item[
                            "spec_valid_accept_token_num_by_position"
                        ],
                        "spec_valid_draft_token_num": item[
                            "spec_valid_draft_token_num"
                        ],
                        "spec_valid_draft_token_num_by_position": item[
                            "spec_valid_draft_token_num_by_position"
                        ],
                    }
                )
            record.update(
                {
                    "prompt": item.get("prompt_text", ""),
                    "response": item.get("output_text", ""),
                }
            )
            requests.append(record)
        json_path.write_text(
            json.dumps(
                {"mode": mode_key, "requests": requests},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written_paths.append(json_path)

    return written_paths


def _print_response_block(label: str, text: str, *, indent: str = "    ") -> None:
    print(f"{indent}{label}:")
    if not text:
        print(f"{indent}  <empty>")
        return
    for line in text.splitlines():
        print(f"{indent}  {line}")


def print_summary(result: dict[str, Any]) -> None:
    spec = result["decoupled_spec"]
    baseline_name = result.get("baseline")
    baseline = result.get(baseline_name) if baseline_name else None
    speedup = result.get("e2e_speedup")
    title = (
        f"decoupled_spec_vs_{baseline_name}_batch"
        if baseline is not None
        else "decoupled_spec_batch"
    )
    print(f"=== {title} ===")
    print(f"dataset_path: {result['config']['dataset_path']}")
    print(f"dataset_format: {result['config']['dataset_format']}")
    print(f"prompt_column: {result['config']['prompt_column']}")
    print(f"batch_size: {result['config']['batch_size']}")
    print(f"verify_ngpus: {result['config']['verify_ngpus']}")
    print(f"draft_ngpus: {result['config']['draft_ngpus']}")
    print(f"num_verifier_replicas: {result['config']['num_verifier_replicas']}")
    print(f"num_draft_replicas: {result['config']['num_draft_replicas']}")
    print(f"max_new_tokens: {result['config']['max_new_tokens']}")
    print(f"total_prompt_tokens: {result['dataset']['total_prompt_tokens']}")
    print(
        "decoupled_spec: "
        f"generation_time_s={spec['generation_time_s']:.3f}, "
        f"generated_tokens={spec['total_generated_tokens']}, "
        f"output_throughput={spec['output_throughput_tok_per_s']:.3f} tok/s, "
        f"avg_spec_accept_length={spec['avg_spec_accept_length']}, "
        f"avg_spec_accept_rate={spec['avg_spec_accept_rate']}, "
        f"avg_spec_valid_accept_rate={spec['avg_spec_valid_accept_rate']}, "
        f"avg_spec_valid_accept_rate_by_position="
        f"{spec['avg_spec_valid_accept_rate_by_position']}, "
        f"valid_accept_tokens={spec['total_spec_valid_accept_token_num']}, "
        f"valid_accept_tokens_by_position="
        f"{spec['total_spec_valid_accept_token_num_by_position']}, "
        f"valid_draft_tokens={spec['total_spec_valid_draft_token_num']}, "
        f"valid_draft_tokens_by_position="
        f"{spec['total_spec_valid_draft_token_num_by_position']}"
    )
    if baseline is not None:
        baseline_line = (
            f"{baseline_name}: "
            f"generation_time_s={baseline['generation_time_s']:.3f}, "
            f"generated_tokens={baseline['total_generated_tokens']}, "
            f"output_throughput={baseline['output_throughput_tok_per_s']:.3f} tok/s"
        )
        if baseline_name == "mtp":
            baseline_line += (
                f", avg_spec_accept_length={baseline['avg_spec_accept_length']}, "
                f"avg_spec_accept_rate={baseline['avg_spec_accept_rate']}"
            )
        print(baseline_line)
        print(
            f"e2e_speedup_vs_{baseline_name}: {speedup:.4f}"
            if speedup is not None
            else f"e2e_speedup_vs_{baseline_name}: None"
        )
    print("per_request:")
    for item in spec["per_request"]:
        print(
            "  "
            f"batch_index={item['batch_index']}, "
            f"row_index={item['row_index']}, "
            f"verifier_rank={item['verifier_rank']}, "
            f"prompt_tokens={item['prompt_tokens']}, "
            f"generated_tokens={item['generated_tokens']}, "
            f"request_latency_s={item['request_latency_s']}, "
            f"spec_accept_length={item['spec_accept_length']}, "
            f"spec_accept_rate={item['spec_accept_rate']}, "
            f"spec_valid_accept_rate={item['spec_valid_accept_rate']}, "
            f"spec_valid_accept_rate_by_position="
            f"{item['spec_valid_accept_rate_by_position']}, "
            f"spec_valid_accept_token_num={item['spec_valid_accept_token_num']}, "
            f"spec_valid_accept_token_num_by_position="
            f"{item['spec_valid_accept_token_num_by_position']}, "
            f"spec_valid_draft_token_num={item['spec_valid_draft_token_num']}, "
            f"spec_valid_draft_token_num_by_position="
            f"{item['spec_valid_draft_token_num_by_position']}, "
            f"spec_verify_ct={item['spec_verify_ct']}"
        )
    if baseline_name == "mtp" and baseline is not None:
        print("mtp_per_request:")
        for item in baseline["per_request"]:
            print(
                "  "
                f"batch_index={item['batch_index']}, "
                f"row_index={item['row_index']}, "
                f"verifier_rank={item['verifier_rank']}, "
                f"prompt_tokens={item['prompt_tokens']}, "
                f"generated_tokens={item['generated_tokens']}, "
                f"request_latency_s={item['request_latency_s']}, "
                f"spec_accept_length={item['spec_accept_length']}, "
                f"spec_accept_rate={item['spec_accept_rate']}, "
                f"spec_verify_ct={item['spec_verify_ct']}, "
                f"spec_accepted_drafts={item['spec_accepted_drafts']}, "
                f"spec_proposed_drafts={item['spec_proposed_drafts']}"
            )
    if result["config"].get("show_responses"):
        print("responses:")
        baseline_items = (
            baseline["per_request"]
            if baseline is not None
            else [None] * len(spec["per_request"])
        )
        for spec_item, baseline_item in zip(
            spec["per_request"], baseline_items, strict=True
        ):
            print(
                "  "
                f"batch_index={spec_item['batch_index']}, "
                f"row_index={spec_item['row_index']}, "
                f"verifier_rank={spec_item['verifier_rank']}"
            )
            _print_response_block(
                "decoupled_spec_response",
                spec_item.get("output_text", ""),
            )
            if baseline_item is not None:
                if (
                    spec_item["batch_index"] != baseline_item["batch_index"]
                    or spec_item["row_index"] != baseline_item["row_index"]
                ):
                    raise RuntimeError(
                        "Mismatched per-request ordering between decoupled_spec "
                        f"and {baseline_name}"
                    )
                _print_response_block(
                    f"{baseline_name}_response",
                    baseline_item.get("output_text", ""),
                )
