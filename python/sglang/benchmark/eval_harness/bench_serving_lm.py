"""BenchServingLM — an lm_eval.api.model.LM that routes generation through
sglang.bench_serving.

Usage:
    lm = BenchServingLM(base_url="http://host:port", backend="sglang-oai",
                        model_id="Qwen/...", tokenizer=tok, ...)
    results = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"], ...)
    perf    = lm.last_perf   # populated after the run

simple_evaluate handles prompt construction (fewshot + chat template +
enable_thinking via our apply_chat_template override), filter application,
and aggregation. We only implement generate_until; loglikelihood raises.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from typing import Any, Dict, List, Optional

from lm_eval.api.model import LM

from sglang.benchmark.datasets.common import DatasetRow


class BenchServingLM(LM):
    def __init__(
        self,
        *,
        base_url: str,
        backend: str,
        model_id: str,
        tokenizer,
        request_rate: float = float("inf"),
        max_concurrency: Optional[int] = None,
        enable_thinking: bool = False,
        warmup_requests: int = 0,
        flush_cache: bool = False,
        extra_request_body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.request_rate = request_rate
        self.max_concurrency = max_concurrency
        self.enable_thinking = enable_thinking
        self.warmup_requests = warmup_requests
        self.flush_cache = flush_cache
        self._global_extra = dict(extra_request_body or {})
        self.last_perf: Optional[Dict[str, Any]] = None

        # lm_eval inspects these attributes.
        self._rank = 0
        self._world_size = 1
        self.batch_size_per_gpu = max_concurrency or 1

    # ---- chat template ---------------------------------------------------

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        if self.enable_thinking:
            kwargs["enable_thinking"] = True
        prompt = self.tokenizer.apply_chat_template(chat_history, **kwargs)
        bos = getattr(self.tokenizer, "bos_token", None)
        if bos and prompt.startswith(bos):
            prompt = prompt[len(bos):]  # bench_serving backend re-adds it.
        return prompt

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", "bench_serving_lm")

    # ---- loglikelihood: unsupported -------------------------------------

    def loglikelihood(self, requests):
        raise NotImplementedError(
            "BenchServingLM supports only generative tasks. Use a CoT / "
            "generative variant (e.g. mmlu_flan_cot_zeroshot instead of mmlu)."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "BenchServingLM supports only generative tasks."
        )

    # ---- generate_until: the whole point --------------------------------

    def generate_until(self, requests) -> List[str]:
        from sglang import bench_serving

        rows: List[DatasetRow] = []
        for req in requests:
            prompt, gen_kwargs = req.args[0], req.args[1]
            stop = gen_kwargs.get("until") or []
            max_gen_toks = gen_kwargs.get("max_gen_toks") or 2048
            temperature = gen_kwargs.get("temperature")

            per_req_extra: Dict[str, Any] = {}
            if stop:
                per_req_extra["stop"] = list(stop)
            if temperature is not None:
                per_req_extra["temperature"] = temperature

            prompt_ids = self.tokenizer.encode(prompt)
            rows.append(DatasetRow(
                prompt=prompt,
                prompt_len=len(prompt_ids),
                output_len=max_gen_toks,
                extra_request_body=per_req_extra,
            ))

        api_url = self.base_url + (
            "/v1/completions" if self.backend == "sglang-oai" else "/generate"
        )

        # bench_serving reads module-level args in a few places (warmup
        # branching, flush_cache, dataset_name guards). Populate it.
        args = Namespace(
            dataset_name="bench_eval",
            backend=self.backend,
            tag=None,
            sharegpt_output_len=None,
            random_input_len=0,
            random_output_len=0,
            random_range_ratio=1.0,
            output_file=None,
            output_details=False,
            num_prompts=len(rows),
        )
        bench_serving.set_global_args(args)
        bench_serving._apply_arg_defaults(args)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no loop running — good
        else:
            raise RuntimeError(
                "BenchServingLM.generate_until cannot be called from inside a running "
                "asyncio event loop. lm_eval.simple_evaluate is synchronous; call it "
                "from synchronous code, or run it in a thread if you must drive it "
                "from an async context."
            )

        perf = asyncio.run(bench_serving.benchmark(
            backend=self.backend,
            api_url=api_url,
            base_url=self.base_url,
            model_id=self.model_id,
            tokenizer=self.tokenizer,
            input_requests=rows,
            request_rate=self.request_rate,
            max_concurrency=self.max_concurrency,
            disable_tqdm=False,
            lora_names=[],
            lora_request_distribution=None,
            lora_zipf_alpha=None,
            extra_request_body=dict(self._global_extra),
            profile=False,
            flush_cache=self.flush_cache,
            warmup_requests=self.warmup_requests,
        ))
        self.last_perf = perf
        return list(perf["generated_texts"])
