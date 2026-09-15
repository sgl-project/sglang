"""CPU distributed regression tests; no model download or NPU is required."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from check_equivalence import compare_artifacts
from workload import gather_outputs, make_input, token_counts


def _distributed_check(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        for n in (1, 5, 8):
            tp, tp_counts, tp_hash = make_input(n, 4, "none", 17, "cpu")
            ep, ep_counts, ep_hash = make_input(n, 4, "deepep", 17, "cpu")
            assert tp_hash == ep_hash
            assert sum(ep_counts) == n
            start = sum(ep_counts[:rank])
            torch.testing.assert_close(ep, tp[start : start + ep_counts[rank]])
            # Linear TP shards must reduce contributions from the same token.
            weight = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
            partial = (
                tp.float()[:, rank * 2 : rank * 2 + 2] @ weight[rank * 2 : rank * 2 + 2]
            )
            dist.all_reduce(partial)
            expected = tp.float() @ weight
            torch.testing.assert_close(partial, expected)
            tp_full, replicas = gather_outputs(partial, tp_counts, "none")
            ep_local = ep.float() @ weight
            ep_full, _ = gather_outputs(ep_local, ep_counts, "deepep")
            common = dict(
                global_tokens=n,
                input_sha256=tp_hash,
                weights_sha256="fixture",
                phase="prefill",
                layer_id=0,
                skew_experts=0,
                world_size=2,
            )
            a = dict(common, backend="none", output=tp_full, replicas=replicas)
            b = dict(common, backend="deepep", output=ep_full, replicas=[])
            assert compare_artifacts(a, b)["passed"]
            if n > 1:
                # A broken rank 1 must fail after reconstructing the global output.
                if rank == 1:
                    ep_local.add_(100)
                broken, _ = gather_outputs(ep_local, ep_counts, "deepep")
                assert not compare_artifacts(a, dict(b, output=broken))["passed"]
            corrupted = [x.clone() for x in replicas]
            corrupted[1].add_(100)
            assert not compare_artifacts(dict(a, replicas=corrupted), b)["passed"]
            try:
                compare_artifacts(a, dict(b, input_sha256="different-input"))
            except ValueError:
                pass
            else:
                raise AssertionError("mismatched inputs accepted")
    finally:
        dist.destroy_process_group()


class WorkloadTest(unittest.TestCase):
    def test_layout_and_all_rank_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(
                _distributed_check,
                args=(str(Path(temp) / "rdzv"),),
                nprocs=2,
                join=True,
            )

    def test_timing_uses_inference_mode(self):
        from bench_moe_tp_ep import time_forward

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pass

            def elapsed_time(self, other):
                return 1.0

        observed = []

        def block(hidden, batch):
            observed.append(torch.is_inference_mode_enabled())
            return hidden + 1

        def gather(outputs, tensor):
            outputs[0].copy_(tensor)

        with (
            patch.object(
                torch,
                "npu",
                SimpleNamespace(Event=Event, synchronize=lambda: None),
                create=True,
            ),
            patch.object(dist, "barrier"),
            patch.object(dist, "get_world_size", return_value=1),
            patch.object(dist, "all_gather", side_effect=gather),
        ):
            samples, _ = time_forward(block, torch.ones(1, 2), None, 2, 3)
        self.assertEqual(observed, [True] * 5)
        self.assertEqual([row[0] for row in samples], [1.0] * 3)

    def test_empty_rank_counts(self):
        self.assertEqual(token_counts(1, 4), [1, 0, 0, 0])
        self.assertEqual(token_counts(7, 4), [2, 2, 2, 1])
        with self.assertRaises(ValueError):
            token_counts(0, 2)


if __name__ == "__main__":
    unittest.main()
