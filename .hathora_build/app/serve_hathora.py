import json
import logging
import os
import sys
from typing import Optional


def _env_truthy(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _get_env_or_cfg(key: str, cfg: Optional[dict], cfg_keys: list[str], default=None):
    if key in os.environ and os.environ[key]:
        return os.environ[key]
    if cfg:
        for ck in cfg_keys:
            v = cfg.get(ck)
            if v not in (None, ""):
                return v
    return default


def _add_arg_if_set(argv: list[str], flag: str, value) -> None:
    if value:
        argv.extend([flag, str(value)])


def _build_sglang_argv_from_env() -> list[str]:
    cfg_json = os.environ.get("DEPLOYMENT_CONFIG_JSON")
    cfg = json.loads(cfg_json) if cfg_json else None

    model_path = _get_env_or_cfg("MODEL_PATH", cfg, ["model_id", "model_path"])
    if not model_path:
        raise RuntimeError("MODEL_PATH is required")

    argv = [
        "--model-path", str(model_path),
        "--tp-size", _get_env_or_cfg("TP_SIZE", cfg, ["tp_size"], "1"),
        "--dp-size", _get_env_or_cfg("DP_SIZE", cfg, ["dp_size"], "1"),
        "--dtype", _get_env_or_cfg("DTYPE", cfg, ["dtype"], "auto"),
        "--kv-cache-dtype", _get_env_or_cfg("KV_CACHE_DTYPE", cfg, ["kv_cache_dtype"], "auto"),
        "--host", os.environ.get("HOST", "0.0.0.0"),
        "--port", os.environ.get("PORT", os.environ.get("HATHORA_DEFAULT_PORT", "8000")),
        "--log-level", os.environ.get("LOG_LEVEL", "info").lower(),
        "--schedule-conservativeness", os.environ.get("SCHEDULE_CONSERVATIVENESS", "1.0"),
        "--skip-server-warmup",
        "--scheduler-recv-interval", "0",
        "--disable-custom-all-reduce",
    ]

    # Optional server arguments
    _add_arg_if_set(argv, "--quantization", _get_env_or_cfg("QUANTIZATION", cfg, ["quantization"]))
    _add_arg_if_set(argv, "--context-length", _get_env_or_cfg("CONTEXT_LENGTH", cfg, ["context_length", "max_context_length"]))
    _add_arg_if_set(argv, "--max-total-tokens", _get_env_or_cfg("MAX_TOTAL_TOKENS", cfg, ["max_total_tokens"]))
    _add_arg_if_set(argv, "--mem-fraction-static", _get_env_or_cfg("MEM_FRACTION_STATIC", cfg, ["mem_fraction_static"]))
    _add_arg_if_set(argv, "--max-running-requests", os.environ.get("MAX_RUNNING_REQUESTS"))
    _add_arg_if_set(argv, "--cuda-graph-max-bs", os.environ.get("CUDA_GRAPH_MAX_BS"))
    _add_arg_if_set(argv, "--chunked-prefill-size", os.environ.get("CHUNKED_PREFILL_SIZE"))
    _add_arg_if_set(argv, "--max-prefill-tokens", os.environ.get("MAX_PREFILL_TOKENS"))
    _add_arg_if_set(argv, "--chat-template", os.environ.get("CHAT_TEMPLATE"))
    _add_arg_if_set(argv, "--tool-call-parser", os.environ.get("TOOL_CALL_PARSER"))
    _add_arg_if_set(argv, "--api-key", os.environ.get("API_KEY") or os.environ.get("HATHORA_APP_SECRET"))

    # Embedding mode
    is_embedding = _env_truthy(os.environ.get("IS_EMBEDDING")) or "embedding" in model_path.lower()
    if is_embedding:
        argv.extend(["--is-embedding", "--disable-radix-cache"])

    # Boolean flags
    if _env_truthy(os.environ.get("ENABLE_METRICS"), True):
        argv.append("--enable-metrics")
    if _env_truthy(os.environ.get("TRUST_REMOTE_CODE")):
        argv.append("--trust-remote-code")
    if _env_truthy(os.environ.get("ENABLE_DP_ATTENTION")):
        argv.append("--enable-dp-attention")
    if _env_truthy(os.environ.get("ENABLE_DP_LM_HEAD")):
        argv.append("--enable-dp-lm-head")
    if _env_truthy(os.environ.get("ENABLE_MEMORY_SAVER")):
        argv.append("--enable-memory-saver")

    # MoE optimizations
    _add_arg_if_set(argv, "--moe-runner-backend", os.environ.get("MOE_RUNNER_BACKEND"))
    if _env_truthy(os.environ.get("ENABLE_EPLB")):
        argv.append("--enable-eplb")
        _add_arg_if_set(argv, "--eplb-algorithm", os.environ.get("EPLB_ALGORITHM"))
        _add_arg_if_set(argv, "--eplb-rebalance-num-iterations", os.environ.get("EPLB_REBALANCE_NUM_ITERATIONS"))

    # Speculative decoding
    if _env_truthy(os.environ.get("SPEC_DECODE")):
        _add_arg_if_set(argv, "--speculative-algorithm", os.environ.get("SPECULATIVE_ALGORITHM"))
        _add_arg_if_set(argv, "--speculative-draft-model-path", os.environ.get("SPECULATIVE_DRAFT_MODEL_PATH"))
        _add_arg_if_set(argv, "--speculative-num-steps", os.environ.get("SPECULATIVE_NUM_STEPS"))

    # Disaggregation
    _add_arg_if_set(argv, "--disaggregation-decode-tp", os.environ.get("DISAGGREGATION_DECODE_TP") or os.environ.get("DISAGG_DECODE_TP"))
    _add_arg_if_set(argv, "--disaggregation-prefill-pp", os.environ.get("DISAGGREGATION_PREFILL_PP"))

    # Multi-node distributed serving
    _add_arg_if_set(argv, "--dist-init-addr", os.environ.get("DIST_INIT_ADDR"))
    _add_arg_if_set(argv, "--nnodes", os.environ.get("NNODES"))
    _add_arg_if_set(argv, "--node-rank", os.environ.get("NODE_RANK"))

    # Router configuration
    _add_arg_if_set(argv, "--router-policy", os.environ.get("ROUTER_POLICY", "cache_aware"))
    _add_arg_if_set(argv, "--router-api-key", os.environ.get("API_KEY") or os.environ.get("HATHORA_APP_SECRET"))
    _add_arg_if_set(argv, "--router-request-timeout-secs", os.environ.get("ROUTER_REQUEST_TIMEOUT_SECS"))
    _add_arg_if_set(argv, "--router-max-concurrent-requests", os.environ.get("ROUTER_MAX_CONCURRENT_REQUESTS"))
    _add_arg_if_set(argv, "--router-queue-size", os.environ.get("ROUTER_QUEUE_SIZE"))
    _add_arg_if_set(argv, "--router-queue-timeout-secs", os.environ.get("ROUTER_QUEUE_TIMEOUT_SECS"))
    _add_arg_if_set(argv, "--max-queued-requests", os.environ.get("MAX_QUEUED_REQUESTS"))
    
    if _env_truthy(os.environ.get("ROUTER_DISABLE_RETRIES")):
        argv.append("--router-disable-retries")
    
    if "kimi" in model_path.lower():
        argv.extend(["--router-reasoning-parser", "kimi"])

    return argv


def main():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level), stream=sys.stdout)

    # Pass HF token from deployment config if available
    cfg_json = os.environ.get("DEPLOYMENT_CONFIG_JSON")
    if cfg_json and not os.environ.get("HF_TOKEN"):
        try:
            cfg = json.loads(cfg_json)
            if cfg.get("hf_token"):
                os.environ["HF_TOKEN"] = str(cfg["hf_token"])
        except Exception:
            pass

    # Launch with router
    argv = _build_sglang_argv_from_env()
    sys.argv = ["sglang_router.launch_server"] + argv

    from sglang_router.launch_server import main as router_main
    router_main()


if __name__ == "__main__":
    main()
