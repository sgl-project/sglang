import json
import os
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.tune.device import DeviceInfo
from sglang.tune.loader import (
    ENV_FOLDER,
    attune_select,
    get_attune_config,
    pick_backends,
)
from sglang.tune.shapes import AttnProfile
from sglang.tune.writer import (
    SCHEMA_VERSION,
    build_config,
    config_filename,
    fingerprint,
    save_committed,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DEV = DeviceInfo("NVIDIA H20", 90, cuda_version="12.4", sm_clock_max_mhz=1980)
PROF = AttnProfile(40, 8, 128, "bfloat16", tp_size=1)


def _mini_config():
    return build_config(
        DEV,
        PROF,
        decode_body={
            "1:1024": {"backend": "flashinfer", "page_size": 1, "latency_us": 41.0},
            "256:1024": {"backend": "fa4", "page_size": 128, "latency_us": 90.0},
        },
        prefill_body={"1:512": {"backend": "fa3", "latency_us": 88.0}},
        skipped={"prefill/trtllm_mha": "trtllm_mha prefill unsupported on sm90"},
        provenance={"attune_version": "test"},
    )


class TestWriter(unittest.TestCase):
    def test_filename_is_coarse_and_explicit(self):
        fn = config_filename(DEV, PROF)
        self.assertTrue(fn.startswith("attn,device_name=NVIDIA_H20,sm=90,family=gqa"))
        self.assertIn("tp=1,ep=1,dp=1", fn)  # topology explicit

    def test_fingerprint_captures_extended_hw(self):
        # Guardrail #1: CUDA version and clock state change the LOCAL fingerprint,
        # so a throttled or CUDA-bumped box gets its own re-tune.
        base = fingerprint(DEV, PROF)
        bumped_cuda = fingerprint(
            DeviceInfo("NVIDIA H20", 90, cuda_version="13.0", sm_clock_max_mhz=1980),
            PROF,
        )
        throttled = fingerprint(
            DeviceInfo("NVIDIA H20", 90, cuda_version="12.4", sm_clock_max_mhz=1200),
            PROF,
        )
        self.assertNotEqual(base, bumped_cuda)
        self.assertNotEqual(base, throttled)

    def test_schema_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = save_committed(d, DEV, PROF, _mini_config())
            c = json.load(open(p))
            self.assertEqual(c["schema_version"], SCHEMA_VERSION)
            self.assertIn("skipped", c)


class TestLoader(unittest.TestCase):
    def test_nearest_bucket_and_pick(self):
        cfg = _mini_config()
        # decode has flashinfer at low batch, fa4 at high -> vote picks the majority winner
        prefill, decode, page = pick_backends(cfg, ["fa3"], ["flashinfer", "fa4"])
        self.assertEqual(prefill, "fa3")
        self.assertIn(decode, ("flashinfer", "fa4"))

    def test_workload_hint_uses_nearest_bucket(self):
        cfg = _mini_config()
        # A hint near the big-batch bucket must pick that bucket's winner (fa4), which
        # a plain equal-weight vote over {flashinfer, fa4} would not guarantee.
        _, decode_big, _ = pick_backends(
            cfg,
            ["fa3"],
            ["flashinfer", "fa4"],
            workload_hint={"decode": {"batch": 128, "ctx_len": 1024}},
        )
        self.assertEqual(decode_big, "fa4")
        _, decode_small, _ = pick_backends(
            cfg,
            ["fa3"],
            ["flashinfer", "fa4"],
            workload_hint={"decode": {"batch": 1, "ctx_len": 1024}},
        )
        self.assertEqual(decode_small, "flashinfer")

    def test_workload_hint_ineligible_winner_falls_back_to_vote(self):
        cfg = _mini_config()
        # Nearest bucket says fa4, but fa4 is not currently eligible -> the hint must
        # NOT bypass the gate; fall back to the vote among eligible candidates.
        _, decode, _ = pick_backends(
            cfg,
            ["fa3"],
            ["flashinfer"],
            workload_hint={"decode": {"batch": 256, "ctx_len": 1024}},
        )
        self.assertEqual(decode, "flashinfer")

    def test_failsafe_never_picks_ineligible(self):
        cfg = _mini_config()
        # fa3 not eligible for prefill -> must not be selected
        prefill, decode, page = pick_backends(cfg, ["triton"], ["triton"])
        self.assertNotEqual(prefill, "fa3")

    def test_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            other = AttnProfile(128, 128, 576, "bfloat16", is_mla=True)
            self.assertIsNone(get_attune_config(DEV, other, d))
            self.assertIsNone(attune_select(DEV, other, d, ["triton"], ["triton"]))

    def test_env_folder_override(self):
        with tempfile.TemporaryDirectory() as d:
            save_committed(d, DEV, PROF, _mini_config())
            os.environ[ENV_FOLDER] = d
            try:
                (
                    get_attune_config.__wrapped__
                    if hasattr(get_attune_config, "__wrapped__")
                    else None
                )
                cfg = get_attune_config(DEV, PROF, "/nonexistent-packaged-dir")
                self.assertIsNotNone(cfg)
            finally:
                del os.environ[ENV_FOLDER]

    def test_select_returns_overrides_dict(self):
        with tempfile.TemporaryDirectory() as d:
            save_committed(d, DEV, PROF, _mini_config())
            out = attune_select(DEV, PROF, d, ["fa3"], ["flashinfer", "fa4"])
            self.assertIsInstance(out, dict)
            self.assertIn("decode_attention_backend", out)


if __name__ == "__main__":
    unittest.main()
