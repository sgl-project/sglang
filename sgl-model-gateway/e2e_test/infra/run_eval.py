"""Run gateway accuracy checks with sgl-eval's dataset, prompt and grader."""

import os
from dataclasses import replace


def run_eval(args):
    from sgl_eval.registry import get
    from sgl_eval.sampler import ChatCompletionSampler

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    spec = get(args.eval_name)
    sampler = ChatCompletionSampler(
        base_url=base_url,
        model=getattr(args, "model", None),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    gen = replace(
        spec.default_gen,
        temperature=getattr(args, "temperature", 0.0),
        max_tokens=getattr(args, "max_tokens", 2048),
    )
    try:
        result = spec.run(
            sampler=sampler,
            gen=gen,
            n_repeats=1,
            num_examples=getattr(args, "num_examples", 64),
            num_threads=getattr(args, "num_threads", 32),
            predictions_writer=None,
            load_examples=None,
        )
    finally:
        sampler.abort()
    if result.partial:
        raise RuntimeError("Incomplete sgl-eval evaluation")
    return {**result.aggregate, "latency": result.latency}
