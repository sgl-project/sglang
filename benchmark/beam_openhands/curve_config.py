"""Beam OpenHands configuration shared by planning, launch and reporting."""

from pathlib import Path

ENGINE_COMMIT = "f7473673ddf6175f301782af57e684d33d3e3f96"
MODEL_REVISION = "b23f9e28019fc26dc0c5b5d4dc7382cee1ceb24f"
DRAFT_REVISION = "8d26c952ec6a82a9a18825d6126c9f6948149ad9"
CONCURRENCIES = (1, 4, 8, 16, 32, 48, 64, 128)
DEFAULT_CONCURRENCIES = (1, 4, 8, 16, 32)
SIMULATION = {
    "SGLANG_SIMULATE_ACC_LEN": "4",
    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
    # Forced proposals need not satisfy Beam's inferred turn grammar.
    "SGLANG_RESPONSE_TEMPLATE_TURN_GRAMMAR": "0",
}


def profile(layout, cc, hicache_size=88):
    if layout not in ("tp8", "tep8", "tp4", "tep4") or cc not in CONCURRENCIES:
        raise ValueError("unsupported layout or concurrency")
    if hicache_size not in (64, 88):
        raise ValueError("HiCache size requires a recorded memory budget")
    size = int(layout[-1])
    args = dict(
        tp_size=size,
        ep_size=size if layout.startswith("tep") else 1,
        attn_dp_size=1,
        nnodes=size // 4,
        quantization="modelopt_fp4",
        attention_backend="trtllm_mha",
        page_size=64,
        moe_a2a_backend="none",
        moe_runner_backend="flashinfer_trtllm",
        context_length=1048576,
        mem_fraction_static=0.80,
        enable_unified_memory=True,
        chunked_prefill_size=16384,
        prefill_decode_interval=0,
        max_running_requests=cc,
        cuda_graph_max_bs_decode=cc,
        speculative_algorithm="DFLASH",
        speculative_num_draft_tokens=8,
        speculative_draft_attention_backend="trtllm_mha",
        enable_hierarchical_cache=cc >= 16,
        enable_cache_report=True,
        reasoning_parser="beam",
        tool_call_parser="beam",
        tokenizer_worker_num=1,
        disable_radix_cache=False,
    )
    if cc >= 16:
        args["hicache_size"] = hicache_size
    return {
        "name": f"beam-nvfp4-{layout}-c{cc}-d8-al4",
        "resolved_args": args,
        "simulation": SIMULATION,
        "n_gpus": size,
    }


def server_args(
    layout, cc, rank, model, draft, head, port=30080, dist_port=39746, hicache_size=88
):
    p = profile(layout, cc, hicache_size)["resolved_args"]
    if rank not in range(p["nnodes"]):
        raise ValueError("rank outside selected layout")
    # Default-valued checks still belong in the profile, not necessarily the CLI.
    omitted = {
        "disable_radix_cache",
        "tokenizer_worker_num",
        "moe_a2a_backend",
        "moe_runner_backend",
    }
    args = [
        "--model-path",
        model,
        "--served-model-name",
        "beam",
        "--speculative-draft-model-path",
        draft,
        "--node-rank",
        str(rank),
        "--dist-init-addr",
        f"{head}:{dist_port}",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--enable-metrics",
    ]
    for k, v in p.items():
        if k in omitted or v is False:
            continue
        args.append("--" + k.replace("_", "-"))
        if v is not True:
            args.append(str(v))
    return args


def inside_repo(path):
    repo = Path(__file__).resolve().parents[2]
    path = Path(path).resolve()
    if not path.is_relative_to(repo):
        raise ValueError(f"new artifacts must stay inside Beam checkout: {path}")
    return path
