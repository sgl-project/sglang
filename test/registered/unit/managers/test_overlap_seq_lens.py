"""Regression coverage for spec-v2 overlap seq-lens relay.

The fully-overlapped spec-v2 path must not pull ``seq_lens`` back to CPU when
all active attention backends declare that their metadata replay is GPU-only.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import FutureMap, decide_needs_cpu_seq_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


class _FailingD2HStream:
    def synchronize(self):
        raise AssertionError("GPU-only seq-lens path must not synchronize D2H")


def _server_args(
    speculative_algorithm: str = "EAGLE",
    enable_two_batch_overlap: bool = False,
):
    return SimpleNamespace(
        speculative_algorithm=speculative_algorithm,
        enable_two_batch_overlap=enable_two_batch_overlap,
    )


def _backend(needs_cpu_seq_lens: bool):
    return SimpleNamespace(needs_cpu_seq_lens=needs_cpu_seq_lens)


class TestOverlapSeqLensRelay(CustomTestCase):
    def test_decide_needs_cpu_seq_lens_allows_all_backend_opt_out(self):
        args = _server_args()

        self.assertFalse(
            decide_needs_cpu_seq_lens(
                args,
                [
                    _backend(False),
                    _backend(False),
                    None,
                    _backend(False),
                ],
            )
        )
        self.assertTrue(
            decide_needs_cpu_seq_lens(args, [_backend(False), _backend(True)])
        )

    def test_decide_needs_cpu_seq_lens_keeps_forced_legacy_paths(self):
        self.assertTrue(
            decide_needs_cpu_seq_lens(
                _server_args(enable_two_batch_overlap=True), [_backend(False)]
            )
        )
        self.assertTrue(
            decide_needs_cpu_seq_lens(
                _server_args(speculative_algorithm="NGRAM"), [_backend(False)]
            )
        )

    def test_resolve_seq_lens_gpu_only_skips_cpu_mirror_and_d2h_stream(self):
        future_indices = torch.tensor([1, 3], dtype=torch.long)
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(future_indices=future_indices),
            seq_lens_cpu=torch.tensor([99, 99]),
            seq_lens_sum=198,
            req_pool_indices_cpu=torch.tensor([1, 3], dtype=torch.long),
        )
        publish_ready = _FakeEvent()

        future_map = object.__new__(FutureMap)
        future_map.needs_cpu_seq_lens = False
        future_map.publish_ready = publish_ready
        future_map.new_seq_lens_buf = torch.tensor(
            [10, 20, 30, 40], dtype=torch.long
        )
        future_map.fwd_prepare_d2h_stream = _FailingD2HStream()

        future_map.resolve_seq_lens_cpu(batch)

        self.assertEqual(publish_ready.wait_calls, 1)
        self.assertEqual(batch.seq_lens.tolist(), [20, 40])
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)

    def test_dsa_backends_keep_cpu_seq_lens_opt_out(self):
        repo_root = Path(__file__).resolve().parents[4]
        path = (
            repo_root
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "attention"
            / "dsa_backend.py"
        )
        tree = ast.parse(path.read_text())

        expected_classes = {
            "DeepseekSparseAttnBackend",
            "DeepseekSparseAttnMultiStepBackend",
        }
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in expected_classes:
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)
                        and stmt.target.id == "needs_cpu_seq_lens"
                    ):
                        found[node.name] = (
                            isinstance(stmt.value, ast.Constant)
                            and stmt.value.value is False
                        )

        self.assertEqual(set(found), expected_classes)
        self.assertTrue(all(found.values()))


if __name__ == "__main__":
    unittest.main()
