"""Dynamo 1.6.0 SGLang edits for Kimi-K3.1 (DSA decode context parallel, kimi_k3 model card).

Run by infra-d001v2-d002-dynamo-k31.sh under its flock; do not run directly.
  python3 -I -B dynamo_sglang_k31_patch.py <dynamo/sglang dir> <model dir> <staged dir>

Edits (each an exact anchor that must match once; already-applied text is detected and kept):
  _compat.py  the earlier edit, unchanged: sglang_default_mla_attention_backend also defers
              KimiK31ForConditionalGeneration to engine resolution.
  args.py     K3.1 DSA DCP decode: _validate_mla_dcp_decode_backend returns early for
              KimiK31ForConditionalGeneration (read via _compat.get_sglang_model_config).
  register.py K3.1 model card: for KimiK31ForConditionalGeneration, _register_model_source_path
              returns the per-job staged model-card directory.
Staging (model card): <staged dir> holds a symlink to every non-weight file of <model dir> and a
copy of config.json whose top-level model_type is "kimi_k3" (the only changed byte range).
Prints one "key=value" line per item for the wrapper script; exits 1 with a message on failure.
"""

import json
import os
import shutil
import sys
import tempfile

K31 = "KimiK31ForConditionalGeneration"
STAGED_MODEL_TYPE = "kimi_k3"

COMPAT_OLD = 'if "KimiK3ForConditionalGeneration" in ('
COMPAT_NEW = (
    'if {"KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration"} & set('
)

ARGS_OLD = '''    unpack``, deep inside engine startup.
    """
    resolved = resolved_server_args(server_args)
    # Prefill workers never run the DCP decode branch.
'''
ARGS_BLOCK = """    # K3.1 DSA DCP decode edit: Kimi-K3.1 decodes DCP through
    # SGLang's DSA path, so its 'dsa' decode backend is not rejected, neither
    # the CLI-time prediction nor the engine's resolved value.
    try:
        from dynamo.sglang._compat import get_sglang_model_config

        _hf_config = getattr(get_sglang_model_config(server_args), "hf_config", None)
        if "KimiK31ForConditionalGeneration" in (
            getattr(_hf_config, "architectures", None) or []
        ):
            return
    except Exception as exc:
        logging.warning("K3.1 DSA DCP decode: could not read the model architecture: %s", exc)
"""
ARGS_NEW = ARGS_OLD.replace(
    '    """\n    resolved', '    """\n' + ARGS_BLOCK + "    resolved", 1
)

REGISTER_OLD = """    weights = getattr(mc, "model_weights", None)
    if weights:
        return mc.model_path
    return server_args.model_path


def _build_media_decoder_and_fetcher():
"""
REGISTER_BLOCK = """# K3.1 model card edit: Dynamo's Rust tokenizer picks the
# tiktoken BPE regex from config.json model_type and has none for Kimi-K3.1's
# 'kimi_k31'. For KimiK31ForConditionalGeneration only, build the model card
# from the per-job directory the worker setup script staged: a symlink to every
# non-weight file of the model plus a config.json copy whose only change is
# model_type 'kimi_k3' (identical tokenizer regex). SGLang keeps loading the
# original model path.
_K31_MODEL_CARD_STAGE_ROOT = "/tmp/k31-dynamo-model-card"
_k31_model_card_original_source_path = _register_model_source_path


def _k31_model_card_staged_dir() -> str:
    job = os.environ.get("SLURM_JOB_ID") or "nojob"
    return os.path.join(_K31_MODEL_CARD_STAGE_ROOT, f"mdc-src-{job}")


def _k31_model_card_is_weight_file(name: str) -> bool:
    return name.endswith(".safetensors") or name.endswith(".safetensors.index.json")


def _register_model_source_path(
    engine: Optional[sgl.Engine], server_args: ServerArgs
) -> str:
    source = _k31_model_card_original_source_path(engine, server_args)
    try:
        with open(os.path.join(source, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return source
    if not isinstance(config, dict) or "KimiK31ForConditionalGeneration" not in (
        config.get("architectures") or []
    ):
        return source
    staged = _k31_model_card_staged_dir()
    try:
        with open(os.path.join(staged, "config.json"), encoding="utf-8") as f:
            staged_config = json.load(f)
        expected = sorted(
            name
            for name in os.listdir(source)
            if name != "config.json"
            and not _k31_model_card_is_weight_file(name)
            and os.path.isfile(os.path.join(source, name))
        )
        links = sorted(name for name in os.listdir(staged) if name != "config.json")
        mirrors = links == expected and all(
            os.path.islink(os.path.join(staged, name))
            and os.path.realpath(os.path.join(staged, name))
            == os.path.realpath(os.path.join(source, name))
            for name in links
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"K3.1 model card: staged model card directory {staged} is unusable ({exc}); "
            "was the model card staged in this job?"
        ) from exc
    if not mirrors or staged_config != {**config, "model_type": "kimi_k3"}:
        raise RuntimeError(
            f"K3.1 model card: {staged} does not mirror {source} with model_type 'kimi_k3'"
        )
    logging.info("K3.1 model card: model card for %s built from %s", source, staged)
    return staged
"""
REGISTER_NEW = REGISTER_OLD.replace(
    "\n\ndef _build_media_decoder_and_fetcher():\n",
    "\n\n" + REGISTER_BLOCK + "\n\ndef _build_media_decoder_and_fetcher():\n",
    1,
)


def die(msg):
    print(f"K3.1 Dynamo SGLang edits FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def atomic_write(path, data, mode_from=None):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix="." + os.path.basename(path) + ".infra.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, os.stat(mode_from).st_mode & 0o7777 if mode_from else 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def edit(path, old, new):
    """Replace the single occurrence of old with new; detect an already-applied file."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        die(f"cannot read {path}: {exc}")
    n_old, n_new = text.count(old), text.count(new)
    if n_old == 0 and n_new == 1:
        return "already applied"
    if n_old != 1 or n_new != 0:
        die(
            f"{path}: expected exactly 1 anchor and 0 patched blocks, found "
            f"{n_old} anchor(s) and {n_new} patched block(s); first anchor line: "
            f"{old.splitlines()[0]!r}"
        )
    patched = text.replace(old, new, 1)
    if patched.count(new) != 1 or patched.count(old) != 0:
        die(f"{path}: replacement did not produce exactly one patched block")
    atomic_write(path, patched.encode("utf-8"), mode_from=path)
    return "applied"


def top_level_value_span(text, key):
    """Return (start, end) of the string value of a depth-1 key in a JSON object text."""
    depth, i, n, spans = 0, 0, len(text), []
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            token = text[i + 1 : j]
            k = j + 1
            while k < n and text[k] in " \t\r\n":
                k += 1
            if depth == 1 and token == key and k < n and text[k] == ":":
                v = k + 1
                while text[v] in " \t\r\n":
                    v += 1
                if text[v] != '"':
                    die(f"top-level {key!r} is not a string")
                e = v + 1
                while text[e] != '"':
                    e += 2 if text[e] == "\\" else 1
                spans.append((v + 1, e))
            i = j + 1
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        i += 1
    if len(spans) != 1:
        die(
            f"expected exactly 1 top-level {key!r} key in config.json, found {len(spans)}"
        )
    return spans[0]


def is_weight_file(name):
    return name.endswith(".safetensors") or name.endswith(".safetensors.index.json")


def staged_config_bytes(src_text):
    config = json.loads(src_text)
    start, end = top_level_value_span(src_text, "model_type")
    if src_text[start:end] != "kimi_k31":
        die(f"top-level model_type is {src_text[start:end]!r}, expected 'kimi_k31'")
    new_text = src_text[:start] + STAGED_MODEL_TYPE + src_text[end:]
    if json.loads(new_text) != {**config, "model_type": STAGED_MODEL_TYPE}:
        die("staged config.json differs from the original beyond model_type")
    old_lines, new_lines = src_text.splitlines(), new_text.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    if len(old_lines) != len(new_lines) or len(changed) != 1:
        die(f"staged config.json must differ in exactly one line, differs in {changed}")
    return new_text.encode("utf-8")


def stage_valid(model_dir, staged, names, config_bytes):
    try:
        entries = sorted(os.listdir(staged))
        with open(os.path.join(staged, "config.json"), "rb") as f:
            current = f.read()
    except OSError:
        return False
    if entries != sorted(names + ["config.json"]) or current != config_bytes:
        return False
    if os.path.islink(os.path.join(staged, "config.json")):
        return False
    return all(
        os.path.islink(os.path.join(staged, n))
        and os.readlink(os.path.join(staged, n)) == os.path.join(model_dir, n)
        for n in names
    )


def stage(model_dir, staged):
    cfg = os.path.join(model_dir, "config.json")
    try:
        with open(cfg, encoding="utf-8") as f:
            src_text = f.read()
    except FileNotFoundError:
        return "skipped (no config.json in model dir)"
    config = json.loads(src_text)
    if K31 not in (config.get("architectures") or []):
        return "skipped (not KimiK31ForConditionalGeneration)"
    config_bytes = staged_config_bytes(src_text)
    names = sorted(
        n
        for n in os.listdir(model_dir)
        if n != "config.json"
        and not is_weight_file(n)
        and os.path.isfile(os.path.join(model_dir, n))
    )
    if stage_valid(model_dir, staged, names, config_bytes):
        return "already staged"
    root = os.path.dirname(staged)
    os.makedirs(root, mode=0o755, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=root, prefix=os.path.basename(staged) + ".tmp.")
    try:
        os.chmod(tmp, 0o755)
        for n in names:
            os.symlink(os.path.join(model_dir, n), os.path.join(tmp, n))
        with open(os.path.join(tmp, "config.json"), "wb") as f:
            f.write(config_bytes)
        os.chmod(os.path.join(tmp, "config.json"), 0o644)
        old = None
        if os.path.lexists(staged):
            old = tempfile.mkdtemp(dir=root, prefix=os.path.basename(staged) + ".old.")
            os.rename(staged, os.path.join(old, "d"))
        os.rename(tmp, staged)
        if old:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if not stage_valid(model_dir, staged, names, config_bytes):
        die(f"staged directory {staged} failed verification")
    return f"created ({len(names)} symlinks + config.json)"


def main():
    if len(sys.argv) != 4:
        die(
            "usage: dynamo_sglang_k31_patch.py <dynamo/sglang dir> <model dir> <staged dir>"
        )
    pkg, model_dir, staged = sys.argv[1:]
    print(
        "compat="
        + edit(os.path.join(pkg, "_compat.py"), COMPAT_OLD, COMPAT_NEW).replace(
            " ", "_"
        )
    )
    print(
        "args="
        + edit(os.path.join(pkg, "args.py"), ARGS_OLD, ARGS_NEW).replace(" ", "_")
    )
    print(
        "register="
        + edit(os.path.join(pkg, "register.py"), REGISTER_OLD, REGISTER_NEW).replace(
            " ", "_"
        )
    )
    print("stage=" + stage(model_dir, staged).replace(" ", "_"))


if __name__ == "__main__":
    main()
