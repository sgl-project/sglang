"""MoE LoRA prefill graphs must replay adapters and match eager prefill, and
the same engines' decode graphs must match eager decode.

Shared by the per-PR and the extra-tier e2e files. Each file passes its own
configuration list and starts with the eager reference, because results never
cross CI jobs.
"""

import os

import torch

import sglang as sgl
from sglang.test.lora_utils import (
    MOE_BASE_MODEL_PATH,
    MOE_LORA_PATH,
    MOE_LORA_TEST_PROMPTS,
)

# Missing adapters shift logprobs by 7-17.
LOGPROB_THRESHOLD = 1.0
MAX_NEW_TOKENS = 8
# Keep padded buckets within the runner's 2x token-count limit.
PREFILL_GRAPH_BATCH_SIZES = [512, 1024, 2048, 4096]
# (lora_backend, prefill graph backend, moe_runner_backend)
EAGER_REFERENCE = ("triton", "disabled", None)
# No more requests than cuda_graph_max_bs_decode, so every decode step can
# replay a graph; adapter rows next to a base row.
DECODE_PROMPTS = MOE_LORA_TEST_PROMPTS[:4]
DECODE_LORA_PATHS = ["moe_lora", None, "moe_lora", "moe_lora"]
# Decode positions after token 0 compared on a shared prefix: every row of
# DECODE_PROMPTS must reach one, and all rows together at least this many. The
# eager reference decodes 20 (two prompts stop early).
MIN_DECODE_POSITIONS = 14


def _graph_passes(engine, *modes):
    """The engine's CUDA graph pass counters for `modes`."""
    from prometheus_client import CollectorRegistry, multiprocess

    # Wait for scheduler metric reporting after the response.
    engine.get_server_info()
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    samples = [
        sample
        for metric in registry.collect()
        for sample in metric.samples
        if sample.name == "sglang:cuda_graph_passes_total"
    ]
    if not samples:
        raise AssertionError(
            "no sglang:cuda_graph_passes_total samples: metrics are off or the "
            "scheduler has not reported them"
        )
    return {
        mode: sum(
            sample.value for sample in samples if sample.labels.get("mode") == mode
        )
        for mode in modes
    }


def _scores_per_position(case, output, where):
    """[(chosen token, {top-5 token: logprob})] for every generated position."""
    meta = output["meta_info"]
    chosen, tops = meta["output_token_logprobs"], meta["output_top_logprobs"]
    case.assertTrue(chosen, f"{where}: no generated token")
    case.assertEqual(len(tops), len(chosen), f"{where}: one top-5 list per token")
    scores = []
    for position, ((_, token, _), top) in enumerate(zip(chosen, tops)):
        top = {top_token: lp for lp, top_token, _ in top}
        # Greedy decoding picks its own top-1 token.
        case.assertIn(token, top, f"{where}, position {position}: own top-5")
        scores.append((token, top))
    return scores


def run_prefill_graph_matrix(case, configs):
    """Run every (lora_backend, prefill graph backend, moe_runner_backend)
    config and compare each graph arm with the first, which must be the eager
    reference (decode included). Every graph arm must also report two graph
    prefills, and decode graphs for every DECODE_PROMPTS step, with no eager
    fallback, so a silent fallback cannot make the comparison pass."""
    from prometheus_client import REGISTRY

    assert configs[0] == EAGER_REFERENCE, "the first config is the eager reference"
    prompts = (MOE_LORA_TEST_PROMPTS * 4)[:64]
    lora_paths = [None if i % 3 == 1 else "moe_lora" for i in range(len(prompts))]
    results = {}
    for lora_backend, backend, moe_runner_backend in configs:
        label = f"{lora_backend}/{backend}"
        if moe_runner_backend is not None:
            label += f"/{moe_runner_backend}"
        prefill_graph = backend != "disabled"
        if prefill_graph:
            # Isolate replay counts between engines.
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        kwargs = dict(
            model_path=MOE_BASE_MODEL_PATH,
            enable_lora=True,
            lora_paths={"moe_lora": MOE_LORA_PATH},
            max_loras_per_batch=2,
            lora_backend=lora_backend,
            attention_backend="flashinfer",
            trust_remote_code=True,
            enable_tokenizer_batch_encode=True,
            enable_metrics=prefill_graph,
            disable_radix_cache=True,
            mem_fraction_static=0.8,
            max_running_requests=len(prompts),
            chunked_prefill_size=4096,
            cuda_graph_max_bs_decode=4,
            cuda_graph_backend_prefill=backend,
        )
        if prefill_graph:
            kwargs["cuda_graph_bs_prefill"] = PREFILL_GRAPH_BATCH_SIZES
        else:
            kwargs["cuda_graph_backend_decode"] = "disabled"
        if backend == "full":
            kwargs["cuda_graph_config"] = {
                "prefill": {"full_prefill_max_req": len(prompts)}
            }
        if moe_runner_backend is not None:
            kwargs["moe_runner_backend"] = moe_runner_backend

        collectors_before = set(REGISTRY._collector_to_names)
        engine = None
        try:
            engine = sgl.Engine(**kwargs)
            # Compare prefill outputs before decoding.
            prompt_out = engine.generate(
                prompts,
                sampling_params={"max_new_tokens": 0, "temperature": 0.0},
                return_logprob=True,
                logprob_start_len=0,
                lora_path=lora_paths,
            )
            prompt_logprobs = [
                torch.tensor(
                    [lp for lp, _, _ in o["meta_info"]["input_token_logprobs"][1:]]
                )
                for o in prompt_out
            ]
            gen_out = engine.generate(
                prompts,
                sampling_params={
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.0,
                },
                lora_path=lora_paths,
                return_logprob=True,
                logprob_start_len=-1,
                top_logprobs_num=5,
            )
            decode_modes = ("decode_cuda_graph", "decode_none")
            if prefill_graph:
                case.assertEqual(
                    _graph_passes(engine, "prefill_cuda_graph", "prefill_none"),
                    {"prefill_cuda_graph": 2, "prefill_none": 0},
                    f"{label}: expected two 64-request graph prefills",
                )
                decode_before = _graph_passes(engine, *decode_modes)
            decode_out = engine.generate(
                DECODE_PROMPTS,
                sampling_params={
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.0,
                },
                lora_path=DECODE_LORA_PATHS,
                return_logprob=True,
                logprob_start_len=-1,
                top_logprobs_num=5,
            )
            case.assertEqual(
                len(decode_out), len(DECODE_PROMPTS), f"{label}: decode responses"
            )
            if prefill_graph:
                decode_after = _graph_passes(engine, *decode_modes)
                graph_steps, eager_steps = (
                    decode_after[mode] - decode_before[mode] for mode in decode_modes
                )
                case.assertGreater(graph_steps, 0, f"{label}: no decode graph replay")
                case.assertEqual(eager_steps, 0, f"{label}: eager decode steps")
            results[label] = {
                "prompt_logprobs": prompt_logprobs,
                "next_token_scores": [
                    (
                        o["meta_info"]["output_token_logprobs"][0][1],
                        {
                            token: lp
                            for lp, token, _ in o["meta_info"]["output_top_logprobs"][0]
                        },
                    )
                    for o in gen_out
                ],
                "decode_scores": [
                    _scores_per_position(case, o, f"{label}, decode prompt {i}")
                    for i, o in enumerate(decode_out)
                ],
            }
        finally:
            if engine is not None:
                engine.shutdown()
            for collector in set(REGISTRY._collector_to_names) - collectors_before:
                REGISTRY.unregister(collector)
            torch.cuda.empty_cache()

    eager = results.pop("triton/disabled")
    for backend, graph in results.items():
        for i, prompt in enumerate(prompts):
            e_lp, g_lp = eager["prompt_logprobs"][i], graph["prompt_logprobs"][i]
            case.assertEqual(e_lp.numel(), g_lp.numel(), f"prompt {i}: token count")
            max_diff = (e_lp - g_lp).abs().max().item()
            case.assertLess(
                max_diff,
                LOGPROB_THRESHOLD,
                f"{backend}, prompt {i} ({prompt[:40]!r}): logprobs drift "
                f"{max_diff:.2e} from eager",
            )
            # Compare first-token scores; argmax can flip near ties.
            e_token, e_scores = eager["next_token_scores"][i]
            g_token, g_scores = graph["next_token_scores"][i]
            for token in {e_token, g_token}:
                case.assertIn(token, e_scores, f"{backend}, prompt {i}: eager top-5")
                case.assertIn(token, g_scores, f"{backend}, prompt {i}: graph top-5")
                case.assertLess(
                    abs(e_scores[token] - g_scores[token]),
                    LOGPROB_THRESHOLD,
                    f"{backend}, prompt {i}: first-token logprob drift",
                )

        compared = 0
        for i, (e_steps, g_steps) in enumerate(
            zip(eager["decode_scores"], graph["decode_scores"])
        ):
            row_compared = 0
            for position, ((e_token, e_top), (g_token, g_top)) in enumerate(
                zip(e_steps, g_steps)
            ):
                where = f"{backend}, decode prompt {i}, position {position}"
                for token in {e_token, g_token}:
                    case.assertIn(token, e_top, f"{where}: eager top-5")
                    case.assertIn(token, g_top, f"{where}: graph top-5")
                    case.assertLess(
                        abs(e_top[token] - g_top[token]),
                        LOGPROB_THRESHOLD,
                        f"{where}: logprob drift",
                    )
                row_compared += position > 0
                if e_token != g_token:
                    break  # Later positions condition on different prefixes.
            else:
                # Same tokens throughout, so neither response may stop early.
                case.assertEqual(
                    len(g_steps),
                    len(e_steps),
                    f"{backend}, decode prompt {i}: positions returned",
                )
            case.assertGreater(
                row_compared,
                0,
                f"{backend}, decode prompt {i}: no decode position compared",
            )
            compared += row_compared
        case.assertGreaterEqual(
            compared,
            MIN_DECODE_POSITIONS,
            f"{backend}: decode positions compared on a shared prefix",
        )
