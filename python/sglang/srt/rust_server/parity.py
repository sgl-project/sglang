"""Which configuration the Python frontend reads that Rust-server mode
(``SGLANG_RUST_SERVER=1``) honors, refuses, or ignores.

The Rust server replaces the Python HTTP server, ``TokenizerManager``,
``DetokenizerManager`` and multimodal processors, and reads only what its
launch code passes it (or what it reads from the environment itself). Every
server argument or ``SGLANG_*`` environment variable the Python frontend reads
must therefore be honored by that launch code, refused at startup
(``RUST_SERVER_UNSUPPORTED_ARGS``), or listed below with why it is ignored.
A parity test (run downstream, with the model packages' own tables) scans the
frontend for those reads, so a new one fails CI until it is classified.

The guard is one-way: it catches a new read, not an entry that outlived its
read. Deleting a read, honoring it or refusing it never fails; drop its entry
here when you do. Classification is per name: a name the launch code passes
counts as honored at every read, even one whose effect Rust lacks. Such a
read's gap is listed under another name it shares the code path with (the
per-request speculative stats behind ``speculative_algorithm`` are listed
under ``speculative_num_draft_tokens``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Collection, Dict, Iterable, Optional, Set, Tuple

_SRT = Path(__file__).resolve().parents[1]

# The Python frontend the Rust server replaces.
PYTHON_FRONTEND_SOURCES: Tuple[Path, ...] = (
    _SRT / "managers/tokenizer_manager.py",
    _SRT / "managers/tokenizer_control_mixin.py",
    _SRT / "managers/tokenizer_manager_score_mixin.py",
    _SRT / "managers/multi_tokenizer_mixin.py",
    _SRT / "managers/detokenizer_manager.py",
    _SRT / "entrypoints/http_server.py",
    *sorted((_SRT / "entrypoints/openai").glob("serving_*.py")),
    _SRT / "observability/request_logger.py",
    _SRT / "multimodal/processors/base_processor.py",
    _SRT / "function_call/function_call_parser.py",
)

# The Python code that configures the Rust server, with the top-level
# functions to scan (``None``: the whole module). A read here is honored.
RUST_SERVER_LAUNCH_SOURCES: Tuple[Tuple[Path, Optional[Tuple[str, ...]]], ...] = (
    (_SRT / "rust_server/config.py", None),
    (_SRT / "rust_server/server.py", None),
    (_SRT / "rust_server/multimodal.py", None),
    (_SRT / "observability/metrics_collector.py", ("tokenizer_metric_labels",)),
    (_SRT / "managers/utils.py", ("compute_num_reserved_tokens",)),
)

# Python-frontend reads Rust-server mode ignores on purpose: no effect on
# what a request gets, or a feature that fails loudly without them.
RUST_SERVER_IGNORED_BY_DESIGN: Dict[str, str] = {
    "SGLANG_RUST_SERVER": "The frontend selector itself.",
    # Python's own process, thread and event-loop plumbing.
    "SGLANG_CPU_WORKERS": "Sizes Python's multimodal process pool; Rust MM workers follow --mm-processor-worker-num.",
    "SGLANG_IO_WORKERS": "Sizes Python's media-loading pool; the Rust MM workers load media themselves.",
    "mm_io_worker_num": "Sizes Python's media-loading pool; the Rust MM workers load media themselves.",
    "rl_on_policy_target": "Puts Python's fast image processor on CPU and caps its workers; the Rust resize runs on CPU.",
    "SGLANG_USE_IPC_POOL_HANDLE_CACHE": "Python's CUDA IPC feature transport; Rust pipelines pick their own.",
    "mm_feature_transport": "Python's feature transport; Rust pipelines pick their own.",
    "SGLANG_DETOKENIZER_MAX_STATES": "Bounds Python's detokenizer state LRU; Rust keeps a request's state for its life.",
    "disable_tokenizer_batch_decode": "Python detokenizer batching.",
    "enable_tokenizer_batch_encode": "Python tokenizer batching.",
    "attn_dp_enabled": "Only picks Python's batch tokenization path.",
    "enable_dynamic_batch_tokenizer": "Python tokenizer batching.",
    "dynamic_batch_tokenizer_batch_size": "Python tokenizer batching.",
    "dynamic_batch_tokenizer_batch_timeout": "Python tokenizer batching.",
    "batch_notify_size": "Python event-loop batching of output notifications.",
    "SGLANG_REQUEST_STATE_WAIT_TIMEOUT": "Poll interval of Python's per-request wait loop.",
    "SGL_FORCE_SHUTDOWN": "Python frontend's SIGTERM handling.",
    "trust_remote_code": "Rust tokenizes with tokenizer.json; no tokenizer code is loaded.",
    "SGLANG_MM_AVOID_RETOKENIZE": "Rust always keeps the prompt's own token ids, which this opts into.",
    "disable_radix_cache": "Python's precondition for input_embeds requests; Rust has no input_embeds field and drops it like any unknown field.",
    "SGLANG_BATCH_TOKENIZER_OUTPUTS": "Routes Python's multi-tokenizer outputs; Rust runs no tokenizer workers.",
    "enable_dp_attention_local_control_broadcast": "Fans Python's control requests out to DP ranks; each Rust listener serves its own rank.",
    # The Python HTTP server.
    "SGLANG_HTTP_SERVER_APP_PATH": "Python HTTP server (Granian) setting.",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "Python HTTP server idle timeout; axum::serve sets none, which keeps the router's >50 s idle connections.",
    "SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT": "Python HTTP server (uvicorn) setting.",
    "fastapi_root_path": "Python HTTP server (FastAPI) setting.",
    "enable_http_worker_reuseport": "Python HTTP server (multi-worker) setting.",
    "grpc_worker_threads": "Python gRPC server thread pool.",
    "http2_initial_connection_window_size": "Only with --enable-http2, which is refused.",
    "http2_max_concurrent_streams": "Only with --enable-http2, which is refused.",
    # Routes the Rust server does not serve: their clients get a 404.
    "SGLANG_OLLAMA_ROOT_ROUTE": "Ollama routes are not served.",
    "SGLANG_OLLAMA_CHAT_ROUTE": "Ollama routes are not served.",
    "SGLANG_OLLAMA_GENERATE_ROUTE": "Ollama routes are not served.",
    "SGLANG_OLLAMA_TAGS_ROUTE": "Ollama routes are not served.",
    "SGLANG_OLLAMA_SHOW_ROUTE": "Ollama routes are not served.",
    "asr_max_concurrent_sessions": "/v1/realtime is not served.",
    "enable_response_store": "/v1/responses is not served.",
    "tool_server": "/v1/responses is not served.",
    "enable_mis": "/v1/decisions is not served.",
    "dllm_algorithm": "/v1/decisions is not served.",
    "engine_info_bootstrap_url": "/get_remote_instance_transfer_engine_info is not served.",
    "chunked_prefill_size": "/v1/score is not served.",
    "dp_size": "Only asserted by the weight-update and LoRA routes, which are not served.",
    "enable_pd_role_switch": "/pd_role_switch is not served.",
    "enable_streaming_session": "/open_session is not served.",
    "speculative_ngram_external_corpus_max_tokens": "/add_external_corpus is not served.",
    # Settings of features Rust-server mode refuses.
    "SGLANG_WAIT_WEIGHTS_READY_TIMEOUT": "Only with --checkpoint-engine-wait-weights-before-ready, which is refused.",
    "max_loaded_loras": "LoRA (--enable-lora, --lora-paths) is refused.",
    "tokenizer_metrics_custom_labels_header": "Only with --tokenizer-metrics-allowed-custom-labels, which is refused.",
    "otlp_service_name": "Only with --enable-trace, which is refused.",
    "otlp_traces_endpoint": "Only with --enable-trace, which is refused.",
    "trace_modules": "Only with --enable-trace, which is refused.",
    "trust_mm_content_hashes": "Only lets Python's preprocessing cache skip reading media it holds; Rust reads every item, with the same result.",
    "watermark_config": "Only with --enable-watermark, which is refused.",
    "watermark_context_window": "Only with --enable-watermark, which is refused.",
    "watermark_default_enabled": "Only with --enable-watermark, which is refused.",
    "watermark_enforce_all": "Only with --enable-watermark, which is refused.",
    "watermark_key": "Only with --enable-watermark, which is refused.",
    "SGLANG_ENCODER_DISPATCH_MIN_ITEMS": "Encoder dispatch of --language-only, which is refused.",
    "enable_adaptive_dispatch_to_encoder": "Encoder dispatch of --language-only, which is refused.",
    "encoder_bootstrap_port": "Encoder dispatch of --language-only, which is refused.",
    "encoder_transfer_backend": "Encoder dispatch of --language-only, which is refused.",
    "encoder_urls": "Encoder dispatch of --language-only, which is refused.",
    # Logs, tests and debugging.
    "ep_join_mode": "Only named in a log line.",
    "soft_watchdog_timeout": "Log-only watchdog over the Python frontend's loops.",
    "SGLANG_TEST_STUCK_TOKENIZER": "Test hook of the Python frontend's watchdog.",
    "SGLANG_TEST_STUCK_DETOKENIZER": "Test hook of the Python frontend's watchdog.",
    "SGLANG_DIAG_BYPASS_HEALTH_GENERATE": "Diagnostic hook for an external benchmark driver.",
    "disaggregation_transfer_backend": "Python numbers bootstrap rooms for the test-only `fake` backend.",
    "debug_tensor_dump_input_file": "Offline tensor-dump debugging: Python exits after warmup.",
}

# Python-frontend reads the Rust launch code honors in another form, or that a
# dedicated guard refuses in Rust-server mode.
RUST_SERVER_HANDLED_ELSEWHERE: Dict[str, str] = {
    "SGLANG_TOOL_STRICT_LEVEL": "Refused above OFF (rust_server_unsupported_args): the Rust chat route applies only each tool's own strict.",
    "context_length": "Passed as model_config.context_len.",
    "language_model_only": "Passed as model_config.is_multimodal.",
    "image_processor_backend": "Selects the HF image processor the Rust resize mirrors (RustMmFamily).",
    "disable_fast_image_processor": "Selects the HF image processor the Rust resize mirrors (RustMmFamily).",
    "enable_watermark": "Refused by check_watermark_server_args.",
    "sidecar": "Refused by the --sidecar validation (serving_hook).",
    "log_requests_format": "With --log-requests, refused unless text (check_rust_server_request_logging).",
    "log_requests_target": "With --log-requests, refused unless stdout (check_rust_server_request_logging).",
    "SGLANG_LOG_REQUEST_HEADERS": "Passed as RequestLogConfig.headers, request_logger's WHITELISTED_HEADERS (request_log_settings).",
    "SGLANG_LOG_REQUEST_EXCEEDED_MS": "Passed as RequestLogConfig.exceeded_ms, RequestLogger.log_exceeded_ms (request_log_settings).",
    "SGLANG_PROFILE_RECORD_SHAPES": "Read by the Rust /start_profile through an injected env reader (resolve_start_profile).",
    "SGLANG_PROFILE_WITH_STACK": "Read by the Rust /start_profile through an injected env reader (resolve_start_profile).",
}

# Python-frontend reads Rust-server mode ignores but should not: each changes
# what a client or operator gets. Refuse or implement each, then drop it here.
RUST_SERVER_KNOWN_GAPS: Dict[str, str] = {
    "SGLANG_ENABLE_REQUEST_DECOMPRESSION": "Compressed request bodies get a 400. TODO: drop when the Rust server decompresses them.",
    "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "No crash diagnostics when a scheduler crashes.",
    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "No crash diagnostics when a scheduler crashes.",
    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS": "No crash diagnostics when a scheduler crashes.",
    "speculative_num_draft_tokens": "No per-request speculative-decoding stats in meta_info (also behind the speculative_algorithm read, honored by name).",
    "return_hidden_states_mode": "Rust gets the resolved enable_return_hidden_states only: a request above the server's mode is not refused.",
    "grpc_response_timeout_secs": "Unverified: whether the Rust gRPC listener times responses out.",
    "SGLANG_ENABLE_COLOCATED_BATCH_GEN": "Unverified: whether a batch's items are scheduled together.",
    "enable_encoder_swa_bounded_replay": "Unverified: Python refuses image requests off CUDA and warms up with text.",
    "SGLANG_DEFAULT_THINKING": "Chat: the Rust chat route has no DeepSeek-V4 recipe renderer.",
    "SGLANG_DSV4_REASONING_EFFORT": "Chat: the Rust chat route has no DeepSeek-V4 recipe renderer.",
    "SGLANG_DSV41_REASONING_EFFORT": "Chat: the Rust chat route has no DeepSeek-V4.1 recipe renderer.",
    "SGLANG_INKLING_DEFAULT_REASONING_EFFORT": "Chat: the Rust chat route has no Inkling renderer.",
    "trust_request_chat_template": "Chat: the Rust chat route never renders a request's own template.",
    "SGLANG_ENABLE_RESPONSE_TEMPLATE_TURN_GRAMMAR": "Chat: the Rust chat route builds no per-turn response-template tool grammar.",
    "grammar_backend": "Chat: the Rust chat route builds no per-turn response-template tool grammar.",
}

# Getters of the resolved config namespaces (``runtime_context``), whose
# attributes are server arguments or values derived from them.
_CONFIG_GETTERS = frozenset(
    {
        "get_device",
        "get_model",
        "get_exec",
        "get_schedule",
        "get_memory",
        "get_spec",
        "get_lora",
        "get_mm",
        "get_disagg",
        "get_serving",
        "get_observability",
        "get_parallel",
        "get_server_args",
        "resolving_view",
    }
)
# Readers of the environment variable their first argument names.
_ENV_READERS = frozenset(
    {
        "get_bool_env_var",
        "get_int_env_var",
        "getenv",
    }
)
_ENV_NAME = re.compile(r"^SGL(ANG)?_[A-Z0-9_]+$")
# The env readers the server crates call (`utils/environ.rs` and std): their
# first argument, a literal or a `const` naming one, is the variable.
_RUST_ENV_READ = re.compile(
    r"\b(?:env_bool|env_i64|env_f64|get_bool_env_var|env::var|env::var_os)"
    r'\(\s*(?:"((?:SGLANG|SGL)_[A-Z0-9_]+)"|([A-Z][A-Z0-9_]*)\s*[,)])'
)
_RUST_ENV_CONST = re.compile(
    r'\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*"((?:SGLANG|SGL)_[A-Z0-9_]+)"'
)
_RUST_TEST_MODULE = re.compile(
    r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*([{;])"
)


def config_subgroups() -> Set[str]:
    """Names of the nested config namespaces (``get_exec().moe``), which are
    not settings themselves."""
    from sglang.srt.arg_groups.arg_utils import namespace_of
    from sglang.srt.server_args import ServerArgs

    return {
        part
        for path in namespace_of(ServerArgs).values()
        for part in path.split(".")[1:]
    }


def _is_config_root(node: ast.AST, aliases: Collection[str]) -> bool:
    """Whether ``node`` evaluates to the server arguments or a config bag."""
    if isinstance(node, ast.Name):
        return node.id == "server_args" or node.id in aliases
    if isinstance(node, ast.Attribute):
        return node.attr == "server_args"
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _CONFIG_GETTERS
    )


def _config_chain(
    node: ast.AST, aliases: Collection[str], subgroups: Collection[str]
) -> Optional[list]:
    """The attribute names from a config root to ``node``, less the subgroups
    they pass through; ``None`` when ``node`` is not rooted at one."""
    chain = []
    while isinstance(node, ast.Attribute) and not _is_config_root(node, aliases):
        chain.append(node.attr)
        node = node.value
    if not _is_config_root(node, aliases):
        return None
    return [name for name in reversed(chain) if name not in subgroups]


def _string_arg(call: ast.Call, index: int = 0) -> Optional[str]:
    if len(call.args) > index and isinstance(call.args[index], ast.Constant):
        value = call.args[index].value
        return value if isinstance(value, str) else None
    return None


def _is_os_environ(node: ast.AST) -> bool:
    """``os.environ``, or ``environ`` imported from ``os``."""
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _alias_binding(
    node: ast.AST, aliases: Collection[str], subgroups: Collection[str]
) -> Optional[str]:
    """The name ``node`` binds to the server arguments or a config bag
    (``cfg = …``, ``cfg: T = …``, ``(cfg := …)``), if it binds one."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target, value = node.targets[0], node.value
    elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
        target, value = node.target, node.value
    else:
        return None
    if (
        isinstance(target, ast.Name)
        and value is not None
        and _config_chain(value, aliases, subgroups) == []
    ):
        return target.id
    return None


def _scope_aliases(
    scope: ast.AST, enclosing: Collection[str], subgroups: Collection[str]
) -> Set[str]:
    """The aliases visible in ``scope``: the enclosing ones and those bound in
    its own body (``cfg = resolving_view(server_args)``), nested scopes
    excluded."""
    body = []
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        body.append(node)
        if not isinstance(node, _SCOPES):
            pending.extend(ast.iter_child_nodes(node))
    aliases = set(enclosing)
    changed = True
    while changed:
        changed = False
        for node in body:
            name = _alias_binding(node, aliases, subgroups)
            if name is not None and name not in aliases:
                aliases.add(name)
                changed = True
    return aliases


class _ReadCollector(ast.NodeVisitor):
    def __init__(self, aliases: Collection[str], subgroups: Collection[str]):
        self.aliases = aliases
        self.subgroups = subgroups
        self.reads: Dict[str, Set[int]] = {}

    def _add(self, name: str, node: ast.AST) -> None:
        if not name.startswith("__"):
            self.reads.setdefault(name, set()).add(node.lineno)

    def _visit_scope(self, node: ast.AST) -> None:
        enclosing = self.aliases
        self.aliases = _scope_aliases(node, enclosing, self.subgroups)
        self.generic_visit(node)
        self.aliases = enclosing

    # A class body's own names are not visible in its methods: it adds none.
    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_scope

    def _visit_binding(self, node: ast.AST) -> None:
        # `features = get_exec().features` binds an alias; its reads count.
        if _alias_binding(node, self.aliases, self.subgroups) is None:
            self.generic_visit(node)

    visit_Assign = visit_AnnAssign = visit_NamedExpr = _visit_binding

    def visit_Compare(self, node: ast.Compare) -> None:
        # `"SGLANG_X" in os.environ`
        left = node.left
        if (
            isinstance(left, ast.Constant)
            and isinstance(left.value, str)
            and _ENV_NAME.match(left.value)
            and isinstance(node.ops[0], (ast.In, ast.NotIn))  # codespell:ignore notin
            and _is_os_environ(node.comparators[0])
        ):
            self._add(left.value, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        env = _string_arg(node)
        if name == "config_value" and env is not None:
            self._add(env, node)
        elif name == "getattr" and _string_arg(node, 1) is not None:
            if _config_chain(node.args[0], self.aliases, self.subgroups) == []:
                self._add(_string_arg(node, 1), node)
        elif env is not None and _ENV_NAME.match(env):
            if name in _ENV_READERS or (
                name in ("get", "setdefault")
                and isinstance(func, ast.Attribute)
                and _is_os_environ(func.value)
            ):
                self._add(env, node)
        chain = (
            _config_chain(func, self.aliases, self.subgroups)
            if isinstance(func, ast.Attribute)
            else None
        )
        if chain is None:
            self.generic_visit(node)
            return
        # A method of a setting reads the setting; a method of the config
        # itself (`server_args.get_model_config()`) reads nothing.
        if len(chain) > 1:
            self._add(chain[0], node)
        for child in [*node.args, *node.keywords]:
            self.visit(child)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = node.slice
        if (
            isinstance(node.ctx, ast.Load)
            and _is_os_environ(node.value)
            and isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and _ENV_NAME.match(key.value)
        ):
            self._add(key.value, node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id.endswith("envs"):
            if _ENV_NAME.match(node.attr) and isinstance(node.ctx, ast.Load):
                self._add(node.attr, node)
            return
        chain = _config_chain(node, self.aliases, self.subgroups)
        if chain is None:
            self.generic_visit(node)
        elif chain and isinstance(node.ctx, ast.Load):
            # A store or `del` of a setting is not a read of it.
            self._add(chain[0], node)


def _canonical_imports(tree: ast.Module, path: Path) -> ast.Module:
    """``tree`` with each name imported as an alias of a config getter or of
    ``envs`` (``from … import get_serving as serving``) renamed back, so the
    scan sees ``get_serving().X`` and ``envs.X``. An alias names one target
    per file: the renaming ignores scopes."""
    renames: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if not alias.asname or alias.asname == alias.name:
                continue
            if alias.name not in _CONFIG_GETTERS and alias.name != "envs":
                continue
            if renames.setdefault(alias.asname, alias.name) != alias.name:
                raise ValueError(
                    f"{path}: `{alias.asname}` aliases both "
                    f"`{renames[alias.asname]}` and `{alias.name}`; the "
                    "Rust-server parity scan needs one name for each."
                )
    if renames:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in renames:
                node.id = renames[node.id]
    return tree


def scan_config_reads(
    path: Path,
    subgroups: Collection[str],
    functions: Optional[Collection[str]] = None,
) -> Dict[str, Set[int]]:
    """The server arguments (lowercase) and ``SGLANG_*`` environment variables
    (uppercase) a Python module reads, with the lines that read them.

    Server arguments are read as ``server_args.X`` (or ``….server_args.X``),
    ``getattr(server_args, "X", …)``, ``get_<namespace>().X``,
    ``config_value("X")``, or through a name bound to one of those;
    environment variables as ``envs.X`` (of any ``…envs`` module) or by name
    through ``os.environ`` (``[…]``, ``.get``, ``.setdefault``, ``in``),
    ``os.getenv`` or the ``get_*_env_var`` helpers. Aliases are scoped like
    Python names; stores and ``del`` are not reads. With
    ``functions``, only the top-level functions and classes so named are
    scanned. Reads behind a call into another module are not followed.
    """
    tree = _canonical_imports(ast.parse(path.read_text(), filename=str(path)), path)
    collector = _ReadCollector(_scope_aliases(tree, (), subgroups), subgroups)
    if functions is None:
        collector.generic_visit(tree)
    else:
        for node in tree.body:
            scope = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            if isinstance(node, scope) and node.name in functions:
                collector.visit(node)
    return collector.reads


def _rust_production_code(source: str) -> str:
    """``source`` without its comments and ``#[cfg(test)]`` modules, and with
    every string literal but a bare environment variable name emptied."""
    out = []
    i, n = 0, len(source)
    while i < n:
        test_module = _RUST_TEST_MODULE.match(source, i)
        if test_module and test_module.group(2) == "{":
            # Skip to the module's closing brace, counting code braces only.
            depth, i = 1, test_module.end()
            while i < n and depth:
                j = _skip_rust_literal_or_comment(source, i)
                if j > i:
                    i = j
                    continue
                if source[i] in "{}":
                    depth += 1 if source[i] == "{" else -1
                i += 1
            continue
        j = _skip_rust_literal_or_comment(source, i)
        if j > i:
            literal = source[i:j]
            if re.fullmatch(r'"(?:SGLANG|SGL)_[A-Z0-9_]+"', literal):
                out.append(literal)
            elif not source.startswith(("//", "/*"), i):
                out.append('""')
            i = j
            continue
        out.append(source[i])
        i += 1
    return "".join(out)


def _skip_rust_literal_or_comment(source: str, i: int) -> int:
    """The index past the string literal or comment starting at ``i``, or
    ``i`` when none starts there."""
    if source.startswith("//", i):
        end = source.find("\n", i)
        return len(source) if end < 0 else end
    if source.startswith("/*", i):
        depth, i = 1, i + 2
        while i < len(source) and depth:
            if source.startswith("/*", i):
                depth, i = depth + 1, i + 2
            elif source.startswith("*/", i):
                depth, i = depth - 1, i + 2
            else:
                i += 1
        return i
    prefix = source[max(i - 2, 0) : i]
    starts_token = not re.search(r"\w$", prefix) or re.fullmatch(r"\Wb|b", prefix)
    raw = re.match(r'r(#*)"', source[i : i + 260])
    if raw and starts_token:
        end = source.find('"' + raw.group(1), i + raw.end())
        return len(source) if end < 0 else end + 1 + len(raw.group(1))
    char = re.match(r"'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'])'", source[i : i + 12])
    if char:
        return i + char.end()
    if source[i] == '"':
        i += 1
        while i < len(source) and source[i] != '"':
            i += 2 if source[i] == "\\" else 1
        return i + 1
    return i


def scan_rust_env_reads(root: Path) -> Set[str]:
    """The ``SGLANG_*`` environment variables a Rust crate's production code
    reads through an env reader: the embedded server reads them from the
    scheduler process's environment. Comments, ``#[cfg(test)]`` modules and
    files of out-of-line ones (``#[cfg(test)] mod tests;``) do not count."""
    if not root.is_dir():
        raise FileNotFoundError(f"no Rust sources at {root}")
    sources = {path: path.read_text() for path in root.rglob("*.rs")}
    test_modules = {
        (path.parent / (path.stem if path.stem not in ("lib", "main", "mod") else ""))
        / name
        for path, source in sources.items()
        for name, body in _RUST_TEST_MODULE.findall(source)
        if body == ";"
    }
    names: Set[str] = set()
    for path, source in sources.items():
        module = path.parent if path.stem == "mod" else path.with_suffix("")
        if any(module == test or test in module.parents for test in test_modules):
            continue
        code = _rust_production_code(source)
        consts = dict(_RUST_ENV_CONST.findall(code))
        for literal, const in _RUST_ENV_READ.findall(code):
            if literal:
                names.add(literal)
            elif const in consts:
                names.add(consts[const])
    return names


def frontend_reads(
    sources: Iterable[Path], subgroups: Collection[str]
) -> Dict[str, Set[str]]:
    """Every read in ``sources``, as ``{name: {"file.py:line", …}}``."""
    reads: Dict[str, Set[str]] = {}
    for path in sources:
        for name, lines in scan_config_reads(path, subgroups).items():
            reads.setdefault(name, set()).update(f"{path.name}:{n}" for n in lines)
    return reads


def honored_reads(
    launch_sources: Iterable[Tuple[Path, Optional[Collection[str]]]],
    rust_crates: Iterable[Path],
    subgroups: Collection[str],
) -> Set[str]:
    """What the Rust launch code reads, and the environment variables the Rust
    crates read themselves."""
    honored: Set[str] = set()
    for path, functions in launch_sources:
        honored.update(scan_config_reads(path, subgroups, functions))
    for crate in rust_crates:
        honored.update(scan_rust_env_reads(crate))
    return honored
