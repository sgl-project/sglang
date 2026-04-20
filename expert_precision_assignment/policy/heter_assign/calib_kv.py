"""Measure actual per-request token distribution for KV sizing.

We size the KV reserve from ``vest.SLO``. Two sources feed it:

1. **Prompt calibration** (no server) — reconstruct the prompts
   ``bench_eval`` would send for a task via lm_eval's own
   ``simple_evaluate`` path (same chat template + fewshot + thinking),
   tokenize them, record prompt length stats. Drives the worst-case
   envelope (``max_prompt_len``).

2. **Bench-details ingestion** (post-hoc) — read an existing
   ``bench_serving --output-details`` JSONL (has ``input_lens`` +
   ``output_lens`` per request). Compute ``total_len = input + output``
   mean/std across requests, giving an **amortized** envelope
   (``mean_total_len`` + ``std_total_len``) that ``vest.kv_bytes``
   consumes as ``mc × (μ + k·σ)`` — much tighter than worst case when
   real requests rarely hit the envelope simultaneously.

Either mode is usable alone; providing both gives the fullest picture.

Example — bench_eval prompt calibration:
    python calib_kv.py \\
        --task gsm8k --model <snapshot> \\
        --num_samples 64 --num_fewshot 5 --max_gen_toks 512 \\
        --out_file kv_calib/gsm8k.json

Example — amortized from a past bench_serving run:
    python calib_kv.py \\
        --bench_details_jsonl .../sharegpt/mc128_thr128_n1024.jsonl \\
        --out_file kv_calib/sharegpt.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _resolve_hf_path(path: str) -> str:
    m = re.match(r"^(.+)/hub/models--(.+?)--(.+?)/snapshots/[a-f0-9]+$", path)
    if m:
        os.environ.setdefault("HF_HOME", m.group(1))
        return f"{m.group(2)}/{m.group(3)}"
    return path


def _make_tokenize_only_lm(tokenizer, enable_thinking: bool):
    """Real ``lm_eval.api.model.LM`` subclass that records prompt lengths.

    Defined inside a factory so the ``lm_eval`` import stays lazy (keeps
    --help fast and avoids a hard dep at module import time).
    """
    from lm_eval.api.model import LM

    class _TokenizeOnlyLM(LM):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.enable_thinking = enable_thinking
            self.prompt_lens: List[int] = []
            self.max_gen_toks_seen: List[int] = []
            # lm_eval inspects these.
            self._rank = 0
            self._world_size = 1
            self.batch_size_per_gpu = 1

        def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
            kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
            if self.enable_thinking:
                kwargs["enable_thinking"] = True
            prompt = self.tokenizer.apply_chat_template(chat_history, **kwargs)
            bos = getattr(self.tokenizer, "bos_token", None)
            if bos and prompt.startswith(bos):
                prompt = prompt[len(bos):]
            return prompt

        @property
        def tokenizer_name(self) -> str:
            return getattr(self.tokenizer, "name_or_path", "tokenize_only_lm")

        def loglikelihood(self, requests):
            raise NotImplementedError(
                "calib_kv supports only generative tasks (match bench_eval). "
                "Use a CoT / generative variant (e.g. mmlu_flan_cot_zeroshot)."
            )

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("calib_kv supports only generative tasks.")

        def generate_until(self, requests) -> List[str]:
            outs: List[str] = []
            for req in requests:
                prompt, gen_kwargs = req.args[0], req.args[1]
                self.prompt_lens.append(len(self.tokenizer.encode(prompt)))
                self.max_gen_toks_seen.append(gen_kwargs.get("max_gen_toks") or 2048)
                outs.append("")
            return outs

    return _TokenizeOnlyLM()


def _quantiles(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> int:
        idx = min(n - 1, max(0, int(math.ceil(p * n)) - 1))
        return s[idx]

    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    return {
        "n": n,
        "min": s[0],
        "mean": round(mean, 2),
        "std": round(math.sqrt(var), 2),
        "p50": q(0.50),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": s[-1],
    }


def _load_bench_details(path: Path) -> Dict[str, List[int]]:
    """Read a bench_serving --output-details JSONL.

    Each line is one run with ``input_lens`` and ``output_lens`` arrays.
    We concatenate across all lines (useful if the file accumulated
    multiple (mc, variant) runs) and filter non-positive lens (errored
    requests get 0-output).
    """
    input_lens: List[int] = []
    output_lens: List[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ins = rec.get("input_lens") or []
            outs = rec.get("output_lens") or []
            if not ins or not outs:
                continue
            for i, o in zip(ins, outs):
                if i <= 0 or o <= 0:
                    continue
                input_lens.append(int(i))
                output_lens.append(int(o))
    return {"input_lens": input_lens, "output_lens": output_lens}


def _run_prompt_calibration(args) -> Dict[str, Any]:
    """lm_eval tokenize-only run → prompt-length stats (no server)."""
    from transformers import AutoTokenizer
    import lm_eval
    from lm_eval.tasks import TaskManager

    hf_name = _resolve_hf_path(args.model)
    logger.info("Loading tokenizer: %s", hf_name)
    tok = AutoTokenizer.from_pretrained(
        hf_name, trust_remote_code=True, local_files_only=True
    )

    lm = _make_tokenize_only_lm(tok, args.enable_thinking)

    task_manager = TaskManager()
    logger.info(
        "Running lm_eval (task=%s, num_fewshot=%d, limit=%d, chat=%s, thinking=%s)",
        args.task, args.num_fewshot, args.num_samples,
        args.apply_chat_template, args.enable_thinking,
    )
    # simple_evaluate may raise at the scoring step because our generations
    # are empty; we only care about prompts captured in generate_until,
    # which happens before scoring. Swallow and continue.
    try:
        lm_eval.simple_evaluate(
            model=lm,
            tasks=[args.task],
            num_fewshot=args.num_fewshot,
            limit=args.num_samples,
            apply_chat_template=args.apply_chat_template,
            task_manager=task_manager,
            gen_kwargs=f"max_gen_toks={args.max_gen_toks}",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "lm_eval scoring failed (expected — empty generations): %s", e
        )

    if not lm.prompt_lens:
        raise RuntimeError(
            "No prompts captured — task registration or limit wrong."
        )

    return {
        "model": hf_name,
        "num_fewshot": args.num_fewshot,
        "max_gen_toks": args.max_gen_toks,
        "apply_chat_template": args.apply_chat_template,
        "enable_thinking": args.enable_thinking,
        "prompt_len": _quantiles(lm.prompt_lens),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Mode A: prompt calibration via lm_eval (no server)
    ap.add_argument("--task",
                    help="lm_eval task name (e.g. gsm8k, mmlu_flan_cot_zeroshot). "
                         "Required for prompt calibration; omit if you only have "
                         "--bench_details_jsonl.")
    ap.add_argument("--model", help="HF model path or repo id.")
    ap.add_argument("--num_samples", type=int, default=64,
                    help="Number of docs to draw (limit passed to lm_eval).")
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--max_gen_toks", type=int, default=512,
                    help="Same value you plan to pass to bench_eval.")
    ap.add_argument("--apply_chat_template", action="store_true", default=True)
    ap.add_argument("--no_apply_chat_template",
                    dest="apply_chat_template", action="store_false")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--margin_frac", type=float, default=0.10,
                    help="Safety margin on top of observed prompt max when "
                         "recommending max_prompt_len (default 10%%).")
    # Mode B: amortized KV from an existing bench_serving --output-details JSONL
    ap.add_argument("--bench_details_jsonl",
                    help="Path to a bench_serving --output-details JSONL. "
                         "Provides per-request (input_lens, output_lens) used "
                         "to compute amortized μ+k·σ for total_len.")
    ap.add_argument("--headroom_sigmas", type=float, default=2.0,
                    help="k for amortized SLO: mean_total_len + k·std "
                         "(default 2.0; matches vest.BudgetKnobs default).")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.task and not args.bench_details_jsonl:
        ap.error("need at least one of --task or --bench_details_jsonl")
    if args.task and not args.model:
        ap.error("--task requires --model")

    report: Dict[str, Any] = {}
    recommended: Dict[str, Any] = {}

    if args.task:
        prompt_info = _run_prompt_calibration(args)
        q_prompt = prompt_info["prompt_len"]
        recommended_max_prompt = int(
            math.ceil(q_prompt["max"] * (1.0 + args.margin_frac))
        )
        report.update({
            "task": args.task,
            **prompt_info,
            "num_samples": q_prompt["n"],
        })
        recommended.update({
            "max_prompt_len": recommended_max_prompt,
            "max_output_len": args.max_gen_toks,
            "margin_frac": args.margin_frac,
        })
        logger.info(
            "prompt_len  n=%(n)d  min=%(min)d  p50=%(p50)d  p95=%(p95)d  "
            "p99=%(p99)d  max=%(max)d  mean=%(mean).1f  std=%(std).1f",
            q_prompt,
        )
        logger.info(
            "worst-case SLO: max_prompt_len=%d (max+%.0f%%), max_output_len=%d",
            recommended_max_prompt, args.margin_frac * 100, args.max_gen_toks,
        )

    if args.bench_details_jsonl:
        details = _load_bench_details(Path(args.bench_details_jsonl))
        n = len(details["input_lens"])
        if n == 0:
            logger.error(
                "No per-request lens in %s — was it produced with "
                "--output-details?", args.bench_details_jsonl,
            )
            return 2
        totals = [i + o for i, o in zip(details["input_lens"],
                                        details["output_lens"])]
        q_input = _quantiles(details["input_lens"])
        q_output = _quantiles(details["output_lens"])
        q_total = _quantiles(totals)
        report["bench_details"] = {
            "source": str(Path(args.bench_details_jsonl).resolve()),
            "input_len": q_input,
            "output_len": q_output,
            "total_len": q_total,
        }
        recommended["mean_total_len"] = q_total["mean"]
        recommended["std_total_len"] = q_total["std"]
        recommended["kv_headroom_sigmas"] = args.headroom_sigmas
        amortized_peak = q_total["mean"] + args.headroom_sigmas * q_total["std"]
        worst_peak = q_total["max"]
        logger.info(
            "total_len  n=%(n)d  mean=%(mean).1f  std=%(std).1f  "
            "p50=%(p50)d  p95=%(p95)d  p99=%(p99)d  max=%(max)d",
            q_total,
        )
        logger.info(
            "amortized SLO: mean_total_len=%.1f std_total_len=%.1f  "
            "→ per-req μ+%.1fσ=%.1f tokens  (worst-case max=%d, ratio=%.2f×)",
            q_total["mean"], q_total["std"], args.headroom_sigmas,
            amortized_peak, worst_peak,
            worst_peak / max(amortized_peak, 1.0),
        )

    report["recommended_slo"] = recommended

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
