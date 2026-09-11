"""SGLang CI transport and metric reporting for sgl-eval benchmarks."""

import json
import multiprocessing
import os
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path


def api_base_url(args):
    base_url = getattr(args, "base_url", None)
    if not base_url:
        host = getattr(args, "host", "127.0.0.1").rstrip("/")
        if "://" not in host:
            host = f"http://{host}"
        base_url = f"{host}:{getattr(args, 'port', None) or 30000}"
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/v1") else base_url + "/v1"


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

    from sglang.test.run_eval import get_thinking_kwargs, parse_json_object
    from sglang.test.test_utils import dump_metric

    spec = get(args.eval_name)
    overrides = {"max_tokens": 2048}
    for key in (
        "max_tokens",
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
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
            **(spec.default_gen.chat_template_kwargs or {}),
            **chat_kwargs,
        }
    extra_body = dict(getattr(args, "extra_body", None) or {})
    for key in ("top_k", "presence_penalty", "frequency_penalty"):
        value = getattr(args, key, None)
        if value is not None:
            extra_body[key] = value
    if extra_body:
        overrides["extra_body"] = extra_body
    gen = replace(spec.default_gen, **overrides)
    sampler = ChatCompletionSampler(
        base_url=api_base_url(args),
        model=getattr(args, "model", None),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    load_examples = None
    if getattr(args, "from_dataset", None):
        from sgl_eval.evals._loader import load_from_path

        load_examples = load_from_path(args.from_dataset)
    out_parent = Path(
        getattr(args, "sgl_eval_out_dir", None)
        or Path.home() / ".sgl_eval" / "sglang_run_eval"
    ).expanduser()
    out_dir = out_parent / f"sgl_eval_{spec.name}_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    repeats = getattr(args, "repeat", 1)
    writer = PredictionsWriter(out_dir, repeats, spec.pred_schema)
    try:
        result = spec.run(
            sampler=sampler,
            gen=gen,
            n_repeats=repeats,
            num_examples=getattr(args, "num_examples", None),
            num_threads=getattr(args, "num_threads", None) or spec.default_num_threads,
            predictions_writer=writer,
            load_examples=load_examples,
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
                "aggregate": result.aggregate,
                "num_examples": result.num_examples,
                "n_repeats": result.n_repeats,
                "latency_seconds": result.latency,
                "output_throughput_tps": result.output_throughput,
            },
            indent=2,
        )
    )
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
