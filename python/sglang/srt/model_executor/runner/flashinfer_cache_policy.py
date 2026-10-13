"""Lab compatibility shim for policy-less FlashInfer 0.7 tactic JSON files.

Persist measurement provenance separately, bound to the exact tactic file.
Only replay a saved winner when its policy matches the requested measurement.
The lookup shim is installed during startup tuning only, not serving.
"""

import contextlib
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def persisted_profiling_policies(tuner, cache_path):
    cache_path = Path(cache_path)
    sidecar = cache_path.with_suffix(".policies.json")
    policies = {}
    try:
        saved = json.loads(sidecar.read_text())
        if (
            isinstance(saved, dict)
            and saved["cache_sha256"] == _digest(cache_path)
            and isinstance(saved["policies"], dict)
        ):
            policies = saved["policies"]
    except (OSError, ValueError, KeyError):
        pass

    original = tuner.search_cache
    hits = 0

    def search(custom_op, runners, input_shapes, tuning_config, inputs=None):
        nonlocal hits
        found = original(custom_op, runners, input_shapes, tuning_config, inputs)
        if (
            found[0]
            or not tuner.is_tuning_mode
            or tuner._active_managed_store is not None
            or tuner._effective_measure_policy is not None
        ):
            return found
        requested = tuner._profiling_policy(tuning_config)
        with tuner._lock:
            for index, runner in enumerate(runners):
                key = tuner._get_cache_key(
                    custom_op,
                    runner,
                    input_shapes,
                    tuning_config,
                    runner.get_cache_key_extras(inputs) if inputs is not None else (),
                )
                if policies.get(key.file_key) != list(requested):
                    continue
                entry = tuner._file_configs.get(key.file_key)
                if entry is None or entry[0] != runner.__class__.__name__:
                    continue
                tactic = entry[1]
                if not tuner._tactic_still_valid(
                    runner, inputs, tactic, custom_op, "policy sidecar"
                ):
                    continue
                # Match the in-memory state of a freshly tuned winner, so
                # subsequent serving retains its original lookup fast path.
                tuner.profiling_cache[key] = (tactic, None)
                tuner._profiling_cache_policies[key] = requested
                hits += 1
                return True, index, tactic, None
        return found

    tuner.search_cache = search
    try:
        yield
        # The caller saves the tactic file before leaving this context.
        for key, policy in tuner._profiling_cache_policies.items():
            if key in tuner.profiling_cache:
                policies[key.file_key] = list(policy)
        if cache_path.is_file():
            payload = {"cache_sha256": _digest(cache_path), "policies": policies}
            temporary = sidecar.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            temporary.replace(sidecar)
        logger.info("FlashInfer policy cache: reused %d persisted profiles.", hits)
    finally:
        tuner.search_cache = original
