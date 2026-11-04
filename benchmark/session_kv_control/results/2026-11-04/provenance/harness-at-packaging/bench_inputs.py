"""Local JSONL input provenance; user supplied records are never relabelled real."""
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tokenizer_identity(model):
    root = Path(model).resolve()
    files = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
        path = root/name
        if path.is_file():
            files[name] = sha256(path.read_bytes())
    return {"local_model_path": str(root), "local_tokenizer_path": str(root), "files_sha256": files,
            "tokenization": "no added special tokens; token arrays used verbatim"}


def _tokens(value, label):
    if not isinstance(value, list) or not value or any(type(x) is not int or not 0 <= x < 2**32 for x in value):
        raise ValueError(label+" must be a nonempty array of u32 token IDs")
    return value


def load_trace(path, model):
    """Schema: id, token_prefix|prefix_text, tool_suffix(list|text), tool_wait_s.

    Optional model/tokenizer paths are assertions, and must match local model.
    declared_origin is provenance from the submitter, never independently verified.
    """
    source = Path(path).resolve()
    raw = source.read_bytes()
    identity = tokenizer_identity(model)
    tokenizer, cases, ids = None, [], set()
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"trace line {line_no} must be an object")
        case_id = str(record.get("id", f"line-{line_no}"))
        if not case_id or len(case_id) > 128 or case_id in ids:
            raise ValueError(f"trace line {line_no} has invalid/duplicate id")
        ids.add(case_id)
        for field in ("model", "tokenizer"):
            if record.get(field) is not None and Path(record[field]).resolve() != Path(model).resolve():
                raise ValueError(f"trace line {line_no} {field} differs from local model/tokenizer")
        if ("token_prefix" in record) == ("prefix_text" in record):
            raise ValueError(f"trace line {line_no} requires exactly one of token_prefix/prefix_text")
        suffix = record.get("tool_suffix")
        if not isinstance(suffix, (list, str)):
            raise ValueError(f"trace line {line_no} requires tool_suffix token array or text")
        if "prefix_text" in record or isinstance(suffix, str):
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        prefix = (_tokens(record["token_prefix"], "token_prefix") if "token_prefix" in record
                  else _tokens(tokenizer.encode(record["prefix_text"], add_special_tokens=False), "encoded prefix"))
        suffix = (_tokens(suffix, "tool_suffix") if isinstance(suffix, list)
                  else _tokens(tokenizer.encode(suffix, add_special_tokens=False), "encoded suffix"))
        wait = record.get("tool_wait_s")
        if type(wait) not in (float, int) or not 0 <= wait < float("inf"):
            raise ValueError(f"trace line {line_no} requires finite nonnegative tool_wait_s")
        lead = record.get("lead_s")
        if lead is not None and (type(lead) not in (float, int) or not 0 <= lead < float("inf")):
            raise ValueError(f"trace line {line_no} invalid lead_s")
        provenance = {"input_mode": "local_jsonl", "source_file": str(source), "source_file_sha256": sha256(raw),
                      "source_line": line_no, "source_line_sha256": sha256(line),
                      "declared_origin": record.get("declared_origin", "user_supplied"),
                      "origin_independently_verified": False, "model_tokenizer": identity,
                      "prefix_tokens_sha256": sha256(json.dumps(prefix, separators=(",", ":")).encode()),
                      "suffix_tokens_sha256": sha256(json.dumps(suffix, separators=(",", ":")).encode())}
        cases.append({"id": case_id, "prefix": prefix, "suffix": suffix,
                      "tool_wait_s": float(wait), "lead_s": lead, "provenance": provenance})
    if not cases:
        raise ValueError("trace has no input records")
    return cases


def estimate_kv_bytes_per_token(model, override=None):
    if override is not None:
        if override <= 0:
            raise ValueError("KV byte estimate must be positive")
        return {"bytes_per_token": override, "basis": "explicit_user_estimate", "physical_io_bytes": None}
    path = Path(model)/"config.json"
    if not path.is_file():
        return {"bytes_per_token": None, "basis": "unavailable", "physical_io_bytes": None}
    c = json.loads(path.read_text())
    dtype = c.get("torch_dtype", c.get("dtype"))
    size = {"bfloat16": 2, "float16": 2, "float32": 4}.get(dtype)
    heads, layers = c.get("num_key_value_heads"), c.get("num_hidden_layers")
    dim = c.get("head_dim") or (c.get("hidden_size", 0)//c.get("num_attention_heads", 1))
    supported = size and heads and layers and dim and not c.get("use_sliding_window")
    return {"bytes_per_token": 2*layers*heads*dim*size if supported else None,
            "basis": "FULL unquantized K+V model-config estimate" if supported else "unavailable_for_model_layout",
            "model_config_sha256": sha256(path.read_bytes()), "dtype": dtype, "physical_io_bytes": None}
