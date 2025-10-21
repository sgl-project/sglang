import json
import logging
import os
import sys
from typing import Optional

from sglang.srt.server_args import prepare_server_args
from sglang.srt.entrypoints.http_server import launch_server


def _env_truthy(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _get_env_or_cfg(key: str, cfg: Optional[dict], cfg_keys: list[str], default=None):
    if key in os.environ and os.environ[key] != "":
        return os.environ[key]
    if cfg is not None:
        for ck in cfg_keys:
            v = cfg.get(ck)
            if v not in (None, ""):
                return v
    return default


def _build_sglang_argv_from_env() -> list[str]:
    cfg_json = os.environ.get("DEPLOYMENT_CONFIG_JSON")
    cfg: Optional[dict] = None
    if cfg_json:
        try:
            cfg = json.loads(cfg_json)
        except Exception:
            cfg = None

    argv: list[str] = []

    model_path = _get_env_or_cfg("MODEL_PATH", cfg, ["model_id", "model_path"]) or ""
    if not model_path:
        raise RuntimeError("MODEL_PATH is required (or provide model_id in DEPLOYMENT_CONFIG_JSON)")

    tp_size = _get_env_or_cfg("TP_SIZE", cfg, ["tp_size"], "1")
    dtype = _get_env_or_cfg("DTYPE", cfg, ["dtype"], "auto")
    kv_cache_dtype = _get_env_or_cfg("KV_CACHE_DTYPE", cfg, ["kv_cache_dtype"], "auto")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", os.environ.get("HATHORA_DEFAULT_PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    schedule_conservativeness = os.environ.get("SCHEDULE_CONSERVATIVENESS", "1.0")

    argv += [
        "--model-path", str(model_path),
        "--tp-size", str(tp_size),
        "--dtype", str(dtype),
        "--kv-cache-dtype", str(kv_cache_dtype),
        "--host", str(host),
        "--port", str(port),
        "--log-level", str(log_level),
        "--schedule-conservativeness", str(schedule_conservativeness),
        "--skip-server-warmup",
        "--scheduler-recv-interval", "0",
    ]

    # Optional arguments (kept minimal, parity with native entrypoint)
    quantization = _get_env_or_cfg("QUANTIZATION", cfg, ["quantization"], None)
    if quantization:
        argv += ["--quantization", str(quantization)]

    max_total_tokens = _get_env_or_cfg("MAX_TOTAL_TOKENS", cfg, ["max_total_tokens"], None)
    if max_total_tokens:
        argv += ["--max-total-tokens", str(max_total_tokens)]

    mem_fraction_static = _get_env_or_cfg("MEM_FRACTION_STATIC", cfg, ["mem_fraction_static"], None)
    if mem_fraction_static:
        argv += ["--mem-fraction-static", str(mem_fraction_static)]

    max_running_requests = os.environ.get("MAX_RUNNING_REQUESTS")
    if max_running_requests:
        argv += ["--max-running-requests", str(max_running_requests)]

    cuda_graph_max_bs = os.environ.get("CUDA_GRAPH_MAX_BS")
    if cuda_graph_max_bs:
        argv += ["--cuda-graph-max-bs", str(cuda_graph_max_bs)]

    chunked_prefill_size = os.environ.get("CHUNKED_PREFILL_SIZE")
    if chunked_prefill_size:
        argv += ["--chunked-prefill-size", str(chunked_prefill_size)]

    chat_template = os.environ.get("CHAT_TEMPLATE")
    if chat_template:
        argv += ["--chat-template", str(chat_template)]

    tool_call_parser = os.environ.get("TOOL_CALL_PARSER")
    if tool_call_parser:
        argv += ["--tool-call-parser", str(tool_call_parser)]

    api_key = os.environ.get("API_KEY") or os.environ.get("HATHORA_APP_SECRET")
    if api_key:
        argv += ["--api-key", str(api_key)]

    # Embedding mode
    is_embedding = _env_truthy(os.environ.get("IS_EMBEDDING"), False)
    try:
        mp_lower = (model_path or "").lower()
        if "embedding" in mp_lower:
            is_embedding = True
    except Exception:
        pass
    if is_embedding:
        argv += ["--is-embedding", "--disable-radix-cache"]

    # Metrics and trust-remote-code
    if _env_truthy(os.environ.get("ENABLE_METRICS"), True):
        argv += ["--enable-metrics"]
    if _env_truthy(os.environ.get("TRUST_REMOTE_CODE"), False):
        argv += ["--trust-remote-code"]

    # Memory saver (only if explicitly requested)
    if _env_truthy(os.environ.get("ENABLE_MEMORY_SAVER"), False):
        argv += ["--enable-memory-saver"]

    # Speculative decoding (minimal passthrough)
    if _env_truthy(os.environ.get("SPEC_DECODE"), False):
        spec_algo = os.environ.get("SPECULATIVE_ALGORITHM")
        if spec_algo:
            argv += ["--speculative-algorithm", spec_algo]
            draft_path = os.environ.get("SPECULATIVE_DRAFT_MODEL_PATH")
            if draft_path:
                argv += ["--speculative-draft-model-path", draft_path]
            num_steps = os.environ.get("SPECULATIVE_NUM_STEPS")
            if num_steps:
                argv += ["--speculative-num-steps", str(num_steps)]

    # Disaggregation passthrough
    disagg_decode_tp = os.environ.get("DISAGGREGATION_DECODE_TP") or os.environ.get("DISAGG_DECODE_TP")
    if disagg_decode_tp:
        argv += ["--disaggregation-decode-tp", str(disagg_decode_tp)]
    disagg_prefill_pp = os.environ.get("DISAGGREGATION_PREFILL_PP")
    if disagg_prefill_pp:
        argv += ["--disaggregation-prefill-pp", str(disagg_prefill_pp)]

    # Custom all-reduce disabled (parity with native entrypoint)
    argv += ["--disable-custom-all-reduce"]

    return argv


def main():
    # Setup minimal logging
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level), stream=sys.stdout)

    # HF token passthrough if provided via config json
    try:
        cfg_json = os.environ.get("DEPLOYMENT_CONFIG_JSON")
        if cfg_json and not os.environ.get("HF_TOKEN"):
            _cfg = json.loads(cfg_json)
            if isinstance(_cfg, dict) and _cfg.get("hf_token"):
                os.environ["HF_TOKEN"] = str(_cfg["hf_token"])  # export for SGLang
    except Exception:
        pass

    argv = _build_sglang_argv_from_env()
    server_args = prepare_server_args(argv)
    launch_server(server_args)


if __name__ == "__main__":
    main()
