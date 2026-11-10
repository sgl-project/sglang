"""SGLang CI transport and metric reporting for sgl-eval benchmarks."""

import argparse
import json
import multiprocessing
import os
import threading
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

THINKING_MODE_CHOICES = ["deepseek-v3", "qwen-3", "glm-45", "kimi-k2"]

# Benchmarks scored only by sgl-eval; run_eval does not serve them.
SGL_EVAL_BENCHMARKS = frozenset(
    {"mmlu", "gpqa", "mmmu_pro", "mmmu_pro_vision", "aime25", "aime26"}
)


def get_thinking_kwargs(args):
    thinking_mode = getattr(args, "thinking_mode", None)
    if thinking_mode in THINKING_MODE_CHOICES:
        if thinking_mode in ["deepseek-v3", "kimi-k2"]:
            thinking_param = "thinking"
        else:
            # All models other than dpsk v3/kimi_k2
            thinking_param = "enable_thinking"
        return {thinking_param: True}
    return {}


def parse_json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError("must be a valid JSON object string") from e

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")

    return parsed


def api_base_url(args):
    base_url = getattr(args, "base_url", None)
    if not base_url:
        host = getattr(args, "host", "127.0.0.1").rstrip("/")
        if "://" not in host:
            host = f"http://{host}"
        base_url = f"{host}:{getattr(args, 'port', None) or 30000}"
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/v1") else base_url + "/v1"


def _print_truncated_samples(result):
    truncated = (
        (example, repeat, sample)
        for example in result.per_example
        for repeat, sample in enumerate(example.samples)
        if sample.finish_reason == "length"
    )
    for example, repeat, sample in islice(truncated, 3):
        preview = {
            "example_id": example.example.id,
            "repeat": repeat,
            "finish_reason": sample.finish_reason,
            "completion_tokens": sample.completion_tokens,
            "score": example.scores[repeat],
            "text_chars": len(sample.text),
            "head": sample.text[:500],
            "tail": sample.text[-500:],
        }
        print(f"sgl-eval truncated sample: {json.dumps(preview)}", flush=True)


def run_sgl_eval(args):
    # math_verify uses SIGALRM, which requires the process's main thread.
    if threading.current_thread() is not threading.main_thread():
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            return executor.submit(run_sgl_eval, args).result()

    from sgl_eval.predictions import PredictionsWriter
    from sgl_eval.registry import get
    from sgl_eval.sampler import ChatCompletionSampler

    from sglang.test.test_utils import dump_metric

    spec = get(args.eval_name)
    model = getattr(args, "model", None)
    default_gen = spec.default_gen
    model_preset_id = getattr(args, "load_preset_from_model_id", None)
    if model_preset_id:
        from sgl_eval.model_preset import load_model_preset
        from sgl_eval.preset import apply_to_gen

        model_preset = load_model_preset(model_preset_id)
        unset = dict.fromkeys(
            ("thinking", "temperature", "top_p", "max_tokens", "reasoning_effort")
        )
        default_gen = apply_to_gen(
            default_gen, None, SimpleNamespace(**unset), model_preset
        )
        model = model or model_preset.model

    overrides = {}
    # Omitted max_tokens keeps the 2048 CI cap; an explicit None defers to the server.
    if hasattr(args, "max_tokens"):
        if args.max_tokens is None:
            warnings.warn(
                f"sgl-eval {spec.name}: max_tokens=None leaves the output length "
                "uncapped; generation stops only at the server limit.",
                stacklevel=2,
            )
        overrides["max_tokens"] = args.max_tokens
    elif not model_preset_id:
        overrides["max_tokens"] = 2048
    for key in (
        "temperature",
        "top_p",
        "min_p",
        "seed",
        "reasoning_effort",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    chat_kwargs = getattr(args, "chat_template_kwargs", None) or {}
    if isinstance(chat_kwargs, str):
        chat_kwargs = parse_json_object(chat_kwargs)
    chat_kwargs = {**get_thinking_kwargs(args), **chat_kwargs}
    thinking = getattr(args, "sgl_eval_thinking", None)
    if thinking is not None:
        chat_kwargs.setdefault("thinking", thinking)
        chat_kwargs.setdefault("enable_thinking", thinking)
    if chat_kwargs:
        overrides["chat_template_kwargs"] = {
            **(default_gen.chat_template_kwargs or {}),
            **chat_kwargs,
        }
    if getattr(args, "top_k", None) is not None:
        overrides["extra_body"] = {"top_k": args.top_k}
    gen = replace(default_gen, **overrides)
    generation = {
        key: getattr(gen, key)
        for key in ("max_tokens", "temperature", "top_p", "min_p", "seed")
    } | {"chat_template_kwargs": gen.chat_template_kwargs, **(gen.extra_body or {})}
    print(f"sgl-eval {spec.name} generation: {generation}", flush=True)
    sampler = ChatCompletionSampler(
        base_url=api_base_url(args),
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    out_parent = Path(
        getattr(args, "sgl_eval_out_dir", None)
        or Path.home() / ".sgl_eval" / "sglang_run_eval"
    ).expanduser()
    out_dir = out_parent / f"sgl_eval_{spec.name}_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    repeats = getattr(args, "repeat", None) or spec.default_n_repeats
    writer = PredictionsWriter(out_dir, repeats, spec.pred_schema)
    try:
        result = spec.run(
            sampler=sampler,
            gen=gen,
            n_repeats=repeats,
            num_examples=getattr(args, "num_examples", None),
            num_threads=getattr(args, "num_threads", None) or spec.default_num_threads,
            predictions_writer=writer,
            load_examples=None,
        )
    finally:
        writer.close()
        sampler.abort()
    if result.partial:
        raise RuntimeError(f"Incomplete {spec.name} evaluation; results: {out_dir}")
    extracted = [
        answer for example in result.per_example for answer in example.extracted
    ]
    if not extracted:
        raise RuntimeError(f"Empty {spec.name} evaluation; results: {out_dir}")
    metrics = {
        **result.aggregate,
        "accuracy": result.aggregate["score"],
        "invalid": sum(answer is None for answer in extracted) / len(extracted),
        "latency": result.latency,
        "output_throughput": result.output_throughput,
        "sgl_eval_metrics_path": str(out_dir / "metrics.json"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "name": spec.name,
                "model": sampler.model,
                "generation": generation,
                "aggregate": result.aggregate,
                "num_examples": result.num_examples,
                "n_repeats": result.n_repeats,
                "latency_seconds": result.latency,
                "output_throughput_tps": result.output_throughput,
            },
            indent=2,
        )
    )
    _print_truncated_samples(result)
    if result.aggregate.get("error_rate", 0) > 0:
        raise RuntimeError(
            f"{spec.name} evaluation had request errors "
            f"(error_rate={result.aggregate['error_rate']}); results: {out_dir}"
        )
    for name, value in (("score", metrics["score"]), ("latency", metrics["latency"])):
        dump_metric(
            f"{spec.name}_{name}",
            value,
            labels={"model": sampler.model, "eval": spec.name},
        )
    print(f"sgl-eval {spec.name}: {metrics}")
    return (
        (metrics, result.latency) if getattr(args, "return_latency", False) else metrics
    )
