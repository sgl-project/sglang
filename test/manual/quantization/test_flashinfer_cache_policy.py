"""Persisted tactics must match both the file and measurement policy."""

import hashlib
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

_path = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/model_executor/runner/flashinfer_cache_policy.py"
)
_spec = importlib.util.spec_from_file_location("policy_cache", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class Key:
    file_key = "gemm-shape-key"


class Runner:
    def get_cache_key_extras(self, inputs):
        return ()


class PolicyCacheTest(unittest.TestCase):
    def test_reuse_requires_matching_file_and_policy(self):
        for stale_file, stale_policy in ((False, False), (True, False), (False, True)):
            with self.subTest(stale_file=stale_file, stale_policy=stale_policy):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "cache.json"
                    path.write_text("{}\n")
                    cold = ("cuda_graph_profile_replays", 1, "l2_cache_policy", "cold")
                    policy = cold[:-1] + ("hot",) if stale_policy else cold
                    sidecar = {
                        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "policies": {Key.file_key: list(policy)},
                    }
                    path.with_suffix(".policies.json").write_text(json.dumps(sidecar))
                    if stale_file:
                        path.write_text('{"changed": true}\n')
                    key = Key()
                    original = lambda *args: (False, 0, -1, None)
                    tuner = SimpleNamespace(
                        search_cache=original,
                        is_tuning_mode=True,
                        _active_managed_store=None,
                        _effective_measure_policy=None,
                        _profiling_policy=lambda config: cold,
                        _get_cache_key=lambda *args: key,
                        _lock=threading.RLock(),
                        _file_configs={key.file_key: ("Runner", 7)},
                        _tactic_still_valid=lambda *args: True,
                        profiling_cache={},
                        _profiling_cache_policies={},
                    )
                    with _module.persisted_profiling_policies(tuner, path):
                        hit = tuner.search_cache("gemm", [Runner()], (), None)
                        self.assertEqual(hit[0], not (stale_file or stale_policy))
                        if hit[0]:
                            self.assertEqual(hit[2], 7)
                            self.assertEqual(tuner._profiling_cache_policies[key], cold)
                    self.assertIs(tuner.search_cache, original)


if __name__ == "__main__":
    unittest.main()
