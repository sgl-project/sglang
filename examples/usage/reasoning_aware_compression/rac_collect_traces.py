#!/usr/bin/env python3
"""
Phase I of Reasoning-Aware Compression (RAC): collect on-policy chain-of-thought
traces with SGLang and write them out as a pruning calibration set.

RAC ("Reasoning Models Can be Accurately Pruned Via Chain-of-Thought
Reconstruction", ICLR 2026, https://arxiv.org/abs/2509.12464) starts from the
observation that one-shot pruning methods minimize a layer-wise reconstruction
error

    min_{W'} || W X - W' X ||_F^2   s.t.  ||W'||_0 <= S

against a calibration activation matrix X built from *prompt* tokens only. A
reasoning model, however, spends most of its forward passes on tokens it
generated itself (|decode| >> |prompt|), so prompt-only calibration is
distribution-shifted away from what the pruned model will actually run.

RAC's fix is to build the calibration matrix from the dense model's own rollout:

    X_l^RAC = [ X_l^prompt , X_l^decode ]        (paper Eq. 7)

This script is Phase I of the paper's Algorithm 1 -- sampling that rollout --
which is the expensive half (the paper uses a 1M token budget). Batched
generation is exactly what SGLang is good at, so it is a much cheaper way to get
there than the Hugging Face `generate` loop used by the reference
implementation. Phase II (the pruning solver) lives in `rac_prune.py`.

Each output row is one calibration sequence: the chat-templated prompt followed
by the model's own continuation, as token ids. Emitting token ids rather than
text means the sequence fed to the pruner is exactly the sequence the model
produced, with no detokenize/retokenize drift.

Example (paper's math setup):

    python rac_collect_traces.py \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --dataset open-r1/OpenR1-Math-220k \
        --prompt-column problem \
        --output-dir ./rac_traces_math

To produce the paper's "prompt only" ablation baseline from the same prompts,
re-run with `--calibration-mode prompt_only`.
"""

import argparse
import json
import os
import time
from typing import Iterator, List, Optional

import msgspec

import sglang as sgl
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

# The system prompt used by open-r1's GRPO recipes, which is what the RAC
# reference implementation generated its published traces with. Keeping it
# identical matters: the calibration distribution is the method.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. You first think about the reasoning process as an internal "
    "monologue and then provide the user with the answer. Respond in the "
    "following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
)


class TraceStats(msgspec.Struct, frozen=True):
    """What one collection run actually produced."""

    num_rows: int
    num_prompt_tokens: int
    num_decode_tokens: int
    elapsed_seconds: float

    @property
    def num_total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_decode_tokens


class TraceManifest(msgspec.Struct, frozen=True):
    """Provenance for one calibration set, written next to the traces."""

    model_path: str
    calibration_mode: str
    dataset: str
    prompt_column: str
    system_prompt: Optional[str]
    num_rows: int
    num_prompt_tokens: int
    num_decode_tokens: int
    num_total_tokens: int
    target_tokens: int
    num_generations: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect on-policy CoT calibration traces for RAC pruning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Dense reasoning model to collect traces from.",
    )
    parser.add_argument(
        "--dataset",
        default="open-r1/OpenR1-Math-220k",
        help="Hugging Face dataset id, or a path to a local .json/.jsonl file.",
    )
    parser.add_argument("--dataset-config-name", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--prompt-column",
        default="problem",
        help="Column holding the question. 'problem' for math, 'prompt' for code.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Cap on prompts read from the dataset. Default: read until the "
        "token budget is met.",
    )
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--calibration-mode",
        choices=["rac", "prompt_only"],
        default="rac",
        help="'rac' appends on-policy CoT activations (paper Eq. 7). "
        "'prompt_only' emits prompts alone, i.e. the paper's ablation baseline.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=1_000_000,
        help="Calibration token budget. The paper uses 1M.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help="Rollouts sampled per prompt. The paper uses 2.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="T_max, the per-rollout CoT length cap. The paper uses 8192.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Pass an empty string to omit the system message.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Prompts per engine call. Bounds how far past the token budget a "
        "run can overshoot.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Omit the decoded 'text' field from each row to shrink the file. "
        "Token ids are what the pruner actually reads; the text is for humans.",
    )

    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_rows(
    *,
    dataset: str,
    config_name: Optional[str],
    split: str,
    prompt_column: str,
    max_prompts: Optional[int],
):
    """Open the prompt corpus that seeds the rollouts."""
    from datasets import load_dataset

    if os.path.exists(dataset):
        rows = load_dataset("json", data_files=dataset, split="train")
    else:
        rows = load_dataset(dataset, config_name, split=split)

    if prompt_column not in rows.column_names:
        raise ValueError(
            f"Column '{prompt_column}' not in {dataset}. "
            f"Available columns: {rows.column_names}"
        )
    if max_prompts is not None:
        rows = rows.select(range(min(max_prompts, len(rows))))

    return rows


def build_prompt_token_ids(
    *, tokenizer, question: str, system_prompt: str
) -> List[int]:
    """Chat-template one question into the token ids the model would see."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )


def iter_question_chunks(
    *, rows, prompt_column: str, chunk_size: int
) -> Iterator[List[str]]:
    """Yield questions a chunk at a time.

    Chunking matters beyond batching: the corpus (220k rows for the paper's math
    set) is far larger than any token budget needs, so templating and rolling out
    lazily means a 1M-token run only touches the prompts it actually uses.
    """
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size][prompt_column]


def rollout(
    *, llm, prompt_ids_batch: List[List[int]], sampling_params: dict
) -> List[List[int]]:
    """Sample one on-policy continuation per entry (Algorithm 1, decode phase)."""
    outputs = llm.generate(input_ids=prompt_ids_batch, sampling_params=sampling_params)
    return [output["output_ids"] for output in outputs]


def collect_traces(
    *,
    llm,
    tokenizer,
    rows,
    prompt_column: str,
    system_prompt: str,
    sampling_params: dict,
    calibration_mode: str,
    target_tokens: int,
    num_generations: int,
    chunk_size: int,
    emit_text: bool,
    trace_path: str,
) -> TraceStats:
    """Stream calibration rows to disk until the token budget is met."""
    num_rows = 0
    num_prompt_tokens = 0
    num_decode_tokens = 0
    started_at = time.perf_counter()

    with open(trace_path, "w", encoding="utf-8") as trace_file:
        for questions in iter_question_chunks(
            rows=rows, prompt_column=prompt_column, chunk_size=chunk_size
        ):
            chunk = [
                build_prompt_token_ids(
                    tokenizer=tokenizer,
                    question=question,
                    system_prompt=system_prompt,
                )
                for question in questions
            ]
            batch = [ids for ids in chunk for _ in range(num_generations)]

            if calibration_mode == "rac":
                decode_ids_batch = rollout(
                    llm=llm,
                    prompt_ids_batch=batch,
                    sampling_params=sampling_params,
                )
            else:
                decode_ids_batch = [[] for _ in batch]

            for prompt_ids, decode_ids in zip(batch, decode_ids_batch):
                input_ids = list(prompt_ids) + list(decode_ids)
                row = {
                    "input_ids": input_ids,
                    "num_prompt_tokens": len(prompt_ids),
                    "num_decode_tokens": len(decode_ids),
                }
                if emit_text:
                    row["text"] = tokenizer.decode(input_ids)
                trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")

                num_rows += 1
                num_prompt_tokens += len(prompt_ids)
                num_decode_tokens += len(decode_ids)

            total = num_prompt_tokens + num_decode_tokens
            print(
                f"[rac] rows={num_rows} "
                f"tokens={total}/{target_tokens} "
                f"(prompt={num_prompt_tokens} decode={num_decode_tokens})",
                flush=True,
            )
            if total >= target_tokens:
                break

    return TraceStats(
        num_rows=num_rows,
        num_prompt_tokens=num_prompt_tokens,
        num_decode_tokens=num_decode_tokens,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def report(manifest: TraceManifest) -> None:
    """Print the prompt/decode split, which is the paper's core diagnostic."""
    total = manifest.num_total_tokens
    decode_share = manifest.num_decode_tokens / total if total else 0.0

    print("\n=== RAC calibration set ===")
    print(f"  rows            : {manifest.num_rows}")
    print(f"  prompt tokens   : {manifest.num_prompt_tokens}")
    print(f"  decode tokens   : {manifest.num_decode_tokens}")
    print(f"  total tokens    : {total}")
    print(f"  decode share    : {decode_share:.1%}")
    print(f"  wall clock      : {manifest.elapsed_seconds/60:.1f} min")
    if manifest.calibration_mode == "rac":
        print(
            "\nThe decode share is the activation mass that prompt-only "
            "calibration throws away."
        )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.model_path)
    rows = load_rows(
        dataset=args.dataset,
        config_name=args.dataset_config_name,
        split=args.dataset_split,
        prompt_column=args.prompt_column,
        max_prompts=args.max_prompts,
    )
    print(f"[rac] {len(rows)} prompts available in {args.dataset}")

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    engine_kwargs = {
        "model_path": args.model_path,
        "skip_tokenizer_init": True,
        "tp_size": args.tp_size,
        "random_seed": args.seed,
    }
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static

    trace_path = os.path.join(args.output_dir, "traces.jsonl")

    # prompt_only needs no rollout, so it needs no engine either.
    llm = sgl.Engine(**engine_kwargs) if args.calibration_mode == "rac" else None
    try:
        stats = collect_traces(
            llm=llm,
            tokenizer=tokenizer,
            rows=rows,
            prompt_column=args.prompt_column,
            system_prompt=args.system_prompt,
            sampling_params=sampling_params,
            calibration_mode=args.calibration_mode,
            target_tokens=args.target_tokens,
            num_generations=args.num_generations,
            chunk_size=args.chunk_size,
            emit_text=not args.no_text,
            trace_path=trace_path,
        )
    finally:
        if llm is not None:
            llm.shutdown()

    manifest = TraceManifest(
        model_path=args.model_path,
        calibration_mode=args.calibration_mode,
        dataset=args.dataset,
        prompt_column=args.prompt_column,
        system_prompt=args.system_prompt or None,
        num_rows=stats.num_rows,
        num_prompt_tokens=stats.num_prompt_tokens,
        num_decode_tokens=stats.num_decode_tokens,
        num_total_tokens=stats.num_total_tokens,
        target_tokens=args.target_tokens,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        elapsed_seconds=stats.elapsed_seconds,
    )

    manifest_path = os.path.join(args.output_dir, "rac_manifest.json")
    with open(manifest_path, "wb") as manifest_file:
        manifest_file.write(msgspec.json.format(msgspec.json.encode(manifest)))

    report(manifest)
    print(f"\nTraces    : {trace_path}")
    print(f"Manifest  : {manifest_path}")
    print("\nNext, prune with these activations:")
    print(
        f"  python rac_prune.py --model-path {args.model_path} "
        f"--calibration {trace_path} --sparsity 0.5 --output-dir ./rac_pruned"
    )


# sgl.Engine spawns subprocesses, so the entry point must be guarded.
if __name__ == "__main__":
    main()
