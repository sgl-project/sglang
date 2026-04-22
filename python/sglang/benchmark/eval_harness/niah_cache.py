"""Opt-in disk cache for lm_eval RULER niah_* dataset builders.

RULER's ``lm_eval/tasks/ruler/prepare_niah.py:generate_samples`` is a
serial Python loop that tokenizes 500 haystack samples per (task,
max_seq_length). At long contexts this takes minutes and the output
is deterministic (random_seed=42, num_samples hardcoded), so building
it once per invocation is pure waste across a sweep or across the
calib → sweep handoff.

Usage::

    # Enable via env var; opt-in means we leave default lm_eval
    # behaviour alone when unset.
    export NIAH_CACHE_DIR=/some/path

    # Call once before lm_eval.simple_evaluate — this patches
    # lm_eval.utils.import_function, which is what the YAML ``!function``
    # tag resolves through, so niah_* functions get wrapped on the way
    # out with Dataset.save_to_disk / load_from_disk.
    from sglang.benchmark.eval_harness.niah_cache import install_niah_disk_cache
    install_niah_disk_cache()

Cache key is ``md5(fn_name + max_seq_lengths + tokenizer)[:12]``.
Races between concurrent writers are handled by tmp-dir + ``os.rename``
— the loser cleans up their tmp and the next run sees the winner's
entry.

Why patching ``lm_eval.utils.import_function`` (not the niah_utils
module): lm_eval's YAML loader resolves ``!function niah_utils.X`` via
``importlib.util.spec_from_file_location`` → a *fresh* module object
each parse, never placed in ``sys.modules``. Patches to the normally
imported ``lm_eval.tasks.ruler.niah_utils`` therefore don't reach the
function refs lm_eval actually calls. Wrapping the returned function
at the ``import_function`` layer catches all loaders, including these
ad-hoc ones.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from typing import Any, Callable

logger = logging.getLogger(__name__)

# RULER builder fn names registered via YAML `!function`. All produce a
# deterministic {"test": Dataset} keyed by (max_seq_lengths, tokenizer), so
# they share the same cache wrapper. Covers the 13 canonical RULER tasks:
#   8 niah_* (retrieval), 2 qa_* (QA), vt (tracing), cwe + fwe (aggregation).
_RULER_DATASET_FN_NAMES = frozenset({
    # NIAH family
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    # QA family
    "get_squad",      # ruler_qa_squad  (paper: qa_1)
    "get_hotpotqa",   # ruler_qa_hotpot (paper: qa_2)
    # Variable tracking
    "get_vt_dataset",
    # Aggregation
    "get_cw_dataset",
    "fwe_download",
})


def _cache_key(fname: str, kwargs: dict) -> str:
    k = {
        "fn": fname,
        "seq": kwargs.get("max_seq_lengths"),
        "tok": kwargs.get("tokenizer") or kwargs.get("pretrained"),
    }
    return hashlib.md5(
        json.dumps(k, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _make_cached_wrapper(fname: str, orig: Callable, cache_root: str) -> Callable:
    import datasets

    def wrapper(**kwargs: Any) -> dict:
        key = _cache_key(fname, kwargs)
        path = os.path.join(cache_root, fname, key)
        if os.path.isdir(path):
            logger.info(
                "niah_cache HIT %s seq=%s key=%s",
                fname, kwargs.get("max_seq_lengths"), key,
            )
            return {"test": datasets.Dataset.load_from_disk(path)}
        logger.info(
            "niah_cache MISS %s seq=%s key=%s — building",
            fname, kwargs.get("max_seq_lengths"), key,
        )
        result = orig(**kwargs)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp.{os.getpid()}"
        try:
            result["test"].save_to_disk(tmp)
            os.rename(tmp, path)
            logger.info("niah_cache SAVED %s → %s", fname, path)
        except OSError:
            # Lost a race (another worker finished first) or target
            # already populated. Drop our tmp and trust the winner.
            shutil.rmtree(tmp, ignore_errors=True)
        return result

    wrapper._niah_cached = True
    wrapper.__name__ = fname
    return wrapper


def install_niah_disk_cache() -> bool:
    """Patch ``lm_eval.utils.import_function`` so that any ``!function
    niah_*`` YAML tag resolved during task-loading returns a wrapped
    function backed by a disk cache. Opt-in via ``NIAH_CACHE_DIR``.

    Returns True if the patch was installed, False if the env var is
    unset (no-op) or lm_eval isn't importable.
    """
    cache_root = os.environ.get("NIAH_CACHE_DIR")
    if not cache_root:
        return False

    try:
        from lm_eval import utils as lm_eval_utils
    except ImportError:
        logger.warning(
            "NIAH_CACHE_DIR=%s set but lm_eval not importable; "
            "niah cache disabled.", cache_root,
        )
        return False

    if getattr(lm_eval_utils.import_function, "_niah_cache_patched", False):
        return True

    os.makedirs(cache_root, exist_ok=True)

    orig_import_function = lm_eval_utils.import_function

    def patched_import_function(loader, node, yaml_path):
        fn = orig_import_function(loader, node, yaml_path)
        name = getattr(fn, "__name__", "")
        if name in _RULER_DATASET_FN_NAMES:
            return _make_cached_wrapper(name, fn, cache_root)
        return fn

    patched_import_function._niah_cache_patched = True
    lm_eval_utils.import_function = patched_import_function
    logger.info("niah_cache installed; root=%s", cache_root)
    return True
