import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.tune.device import DeviceInfo
from sglang.tune.orchestrate import run_tune, summarize
from sglang.tune.shapes import AttnProfile

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MHA = AttnProfile(40, 8, 128, "bfloat16")


class TestEndToEndMock(unittest.TestCase):
    def test_full_run_writes_both_artifacts_and_bodies(self):
        with tempfile.TemporaryDirectory() as cfg, tempfile.TemporaryDirectory() as cache:
            dev = DeviceInfo("NVIDIA H20", 90, cuda_version="12.4")
            # in-process (isolate=False) keeps the test fast and deterministic
            config = run_tune(
                dev,
                MHA,
                packaged_dir=cfg,
                local_cache_dir=cache,
                mock=True,
                isolate=False,
            )
            self.assertTrue(config["decode"])
            self.assertTrue(config["prefill"])
            # every decode cell carries a derived page_size
            self.assertTrue(all("page_size" in v for v in config["decode"].values()))
            # static prune is recorded (trtllm_mha not valid for prefill on sm90)
            self.assertTrue(any("trtllm_mha" in k for k in config["skipped"]))
            self.assertIn("decode:", summarize(config))

    def test_decode_has_a_shape_crossover(self):
        # The core value case: within ONE device the best decode backend CHANGES with batch
        # size — a compute-tile kernel (fa3/fa4) that wins big-batch decode loses small-batch
        # decode on a bandwidth-divergent SKU. A shape-blind SM heuristic picks one and is
        # wrong on the other end; Attune keys to the shape.
        with tempfile.TemporaryDirectory() as cfg:
            h20 = run_tune(
                DeviceInfo("NVIDIA H20", 90),
                MHA,
                packaged_dir=cfg,
                mock=True,
                isolate=False,
                phases=("decode",),
            )
            winners = {v["backend"] for v in h20["decode"].values()}
            self.assertGreater(
                len(winners), 1, "expected a decode crossover across shapes"
            )
            # the small-batch decode winner must NOT be a 128-row-tile kernel on H20
            small = h20["decode"]["1:1024"]["backend"]
            self.assertNotIn(small, ("fa3", "fa4"))

    def test_divergent_sku_penalizes_compute_tile_kernel(self):
        # Two parts sharing the same SM predicate (both sm90) but different bandwidth: the
        # 128-row-tile decode kernel is slower at small batch on the divergent one. This is
        # the fact the SM heuristic cannot see (it keys only on sm90) and Attune measures.
        from sglang.tune.harness import mock_decode_latency
        from sglang.tune.shapes import DecodeShape

        sh = DecodeShape(1, 1024)
        fa4_h20 = mock_decode_latency("fa4", sh, MHA, bandwidth_divergent=True)
        fa4_h100 = mock_decode_latency("fa4", sh, MHA, bandwidth_divergent=False)
        self.assertGreater(fa4_h20, fa4_h100)


if __name__ == "__main__":
    unittest.main()
