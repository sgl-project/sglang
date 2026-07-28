"""Guards for the bench evidence machinery (5th gate-4 review).

Every case here corresponds to a defect that reached a review:

* the chunked gate must return the SAME verdict as the direct gate — a
  cheaper gate that disagrees is a silent correctness hole (review 5
  rejected row sampling for exactly this reason);
* a mutated config table must be refused — ``table_content_digest`` was
  computed but never verified, so selected configs could be edited
  without failing provenance;
* a table tuned for another workload must be refused — provenance
  checked device/toolchain but not model geometry;
* the canonical pair list must contain ``one_launch vs rank_split`` —
  rank_split was only ever compared against the LEGACY kernel, which
  produced a wrong "reopened" ruling.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

from benchmark.kernels.lora_moe.bench_common import (
    TABLE_SCHEMA_VERSION,
    config_key,
    regime_of,
    require_delta_close_chunked,
    require_table_provenance,
    table_content_digest,
)
from benchmark.kernels.lora_moe.signal_gates import (
    DegenerateSignalError,
    require_delta_close,
)


def _verdict(fn, observed, reference, **kwargs):
    try:
        fn(observed, reference, gate_dtype=torch.bfloat16, label="t", **kwargs)
        return "pass"
    except AssertionError:
        return "fail"


class TestChunkedGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def test_chunked_gate_matches_direct_gate_including_across_chunk_boundaries(self):
        # Row counts straddle the 4,096-row chunk size on purpose: an
        # accumulator that resets per chunk, or a max that only keeps the
        # last chunk's value, disagrees here and nowhere else.
        generator = torch.Generator(device="cpu").manual_seed(5)
        for rows in (100, 4096, 4097, 9000):
            reference = (
                (torch.randn(rows, 128, generator=generator) * 0.05)
                .to(torch.bfloat16)
                .to(self.device)
            )
            scale = float(reference.abs().max())
            for noise, expected in ((0.0, "pass"), (1e-4, "pass"), (0.5, "fail")):
                perturbation = (
                    torch.randn(rows, 128, generator=generator).to(self.device)
                    * noise
                    * scale
                )
                observed = (reference.float() + perturbation).to(torch.bfloat16)
                direct = _verdict(
                    require_delta_close, observed.float(), reference.float()
                )
                chunked = _verdict(require_delta_close_chunked, observed, reference)
                with self.subTest(rows=rows, noise=noise):
                    self.assertEqual(direct, chunked)
                    self.assertEqual(chunked, expected)

    def test_accumulate_form_subtracts_the_matched_base_per_chunk(self):
        # The accumulate arms observe base + delta. The gate must subtract
        # the matched base so all thresholds come from the DELTA domain;
        # omitting it compares base+delta against delta and must fail.
        # Delta is 30% of the base here, which clears the 12.5% BF16
        # noise-floor validity rule (32 * 2**-8).
        generator = torch.Generator(device="cpu").manual_seed(7)
        delta = (
            (torch.randn(5000, 64, generator=generator) * 0.3)
            .to(torch.bfloat16)
            .to(self.device)
        )
        base = (
            torch.randn(5000, 64, generator=generator)
            .to(torch.bfloat16)
            .to(self.device)
        )
        observed = (base.float() + delta.float()).to(torch.bfloat16)
        require_delta_close_chunked(
            observed,
            delta,
            observed_base=base,
            gate_dtype=torch.bfloat16,
            label="base subtracted",
        )
        with self.assertRaises(AssertionError):
            require_delta_close_chunked(
                observed, delta, gate_dtype=torch.bfloat16, label="base omitted"
            )

    def test_chunked_gate_peak_memory_does_not_scale_with_rows(self):
        generator = torch.Generator(device="cpu").manual_seed(11)
        reference = (
            (torch.randn(40000, 512, generator=generator) * 0.05)
            .to(torch.bfloat16)
            .to(self.device)
        )
        observed = reference.clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        require_delta_close_chunked(
            observed, reference, gate_dtype=torch.bfloat16, label="mem"
        )
        chunked_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        require_delta_close(
            observed.float(),
            reference.float(),
            gate_dtype=torch.bfloat16,
            label="mem",
        )
        direct_peak = torch.cuda.max_memory_allocated()
        self.assertLess(chunked_peak * 2, direct_peak)


class TestChunkedGateHostileInputs(CustomTestCase):
    """Cases the 6th review found missing — each one caught a real hole."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def test_dropped_delta_is_rejected_at_every_base_scale(self):
        """A dropped LoRA delta must never pass, however large the base.

        Two distinct mechanisms cover the range, and the test asserts that
        SOME rejection always happens rather than assuming which:
        * moderate base -> the max-abs gate fails (error == the delta);
        * base >= ~8x the delta -> the case itself is invalid under the
          BF16 noise-floor rule (delta must exceed 12.5% of the base), so
          DegenerateSignalError fires first.
        A gate deriving S from base+delta instead of the delta would admit
        the large-base cases silently.
        """
        generator = torch.Generator(device="cpu").manual_seed(13)
        delta = (
            (torch.randn(3000, 128, generator=generator) * 0.3)
            .to(torch.bfloat16)
            .to(self.device)
        )
        for base_scale in (1.0, 3.0, 10.0, 100.0):
            base = (
                (torch.randn(3000, 128, generator=generator) * base_scale)
                .to(torch.bfloat16)
                .to(self.device)
            )
            dropped = base.clone()  # delta never applied
            with self.subTest(base_scale=base_scale):
                with self.assertRaises((AssertionError, DegenerateSignalError)):
                    require_delta_close_chunked(
                        dropped,
                        delta,
                        observed_base=base,
                        gate_dtype=torch.bfloat16,
                        label=f"dropped delta base x{base_scale}",
                    )

    def test_valid_ratio_passes_and_invalid_ratio_is_refused_as_degenerate(self):
        generator = torch.Generator(device="cpu").manual_seed(23)
        delta = (
            (torch.randn(2000, 64, generator=generator) * 0.3)
            .to(torch.bfloat16)
            .to(self.device)
        )
        for base_scale, expected in ((1.0, "pass"), (100.0, "degenerate")):
            base = (
                (torch.randn(2000, 64, generator=generator) * base_scale)
                .to(torch.bfloat16)
                .to(self.device)
            )
            observed = (base.float() + delta.float()).to(torch.bfloat16)
            with self.subTest(base_scale=base_scale):
                if expected == "pass":
                    require_delta_close_chunked(
                        observed,
                        delta,
                        observed_base=base,
                        gate_dtype=torch.bfloat16,
                        label="valid ratio",
                    )
                else:
                    with self.assertRaises(DegenerateSignalError):
                        require_delta_close_chunked(
                            observed,
                            delta,
                            observed_base=base,
                            gate_dtype=torch.bfloat16,
                            label="invalid ratio",
                        )

    def test_non_finite_values_are_rejected_not_silently_passed(self):
        # max(finite, nan) keeps the finite value and nan > gate is False,
        # so a NaN chunk could otherwise satisfy the final comparison.
        generator = torch.Generator(device="cpu").manual_seed(17)
        reference = (
            (torch.randn(9000, 64, generator=generator) * 0.05)
            .to(torch.bfloat16)
            .to(self.device)
        )
        for bad, name in ((float("nan"), "nan"), (float("inf"), "inf")):
            for row in (0, 5000, 8999):  # first chunk, later chunk, last row
                observed = reference.clone()
                observed[row, 0] = bad
                with self.subTest(value=name, row=row):
                    with self.assertRaises(AssertionError):
                        require_delta_close_chunked(
                            observed,
                            reference,
                            gate_dtype=torch.bfloat16,
                            label=f"{name}@{row}",
                        )

    def test_near_threshold_cases_actually_straddle_the_boundary(self):
        # The 6th review noted the earlier "near threshold" cases were not
        # near anything. Build one just inside and one just outside the
        # max-abs gate (S/10) and require opposite verdicts.
        generator = torch.Generator(device="cpu").manual_seed(19)
        reference = (
            (torch.randn(5000, 32, generator=generator) * 0.05)
            .to(torch.bfloat16)
            .to(self.device)
        )
        signal = float(reference.abs().max())
        gate = signal / 10.0
        for factor, should_pass in ((0.8, True), (1.5, False)):
            observed = reference.float().clone()
            observed[0, 0] += gate * factor
            observed = observed.to(torch.bfloat16)
            with self.subTest(factor=factor):
                if should_pass:
                    require_delta_close_chunked(
                        observed, reference, gate_dtype=torch.bfloat16, label="in"
                    )
                else:
                    with self.assertRaises(AssertionError):
                        require_delta_close_chunked(
                            observed, reference, gate_dtype=torch.bfloat16, label="out"
                        )


class TestTableProvenance(CustomTestCase):
    def _table(self):
        table = {
            "down": {"16": {"stock": {"decode": "bn64-bk16-m16-g8-w4-s2"}}},
            "_meta": {
                "schema_version": TABLE_SCHEMA_VERSION,
                "device_name": "x",
                "sweep_checkpoint_digest": f"sha256:{'1' * 64}",
                "sweep_skips_digest": f"sha256:{'2' * 64}",
                "workload": {"model_preset": "qwen35_35b"},
            },
        }
        table["_meta"]["table_content_digest"] = table_content_digest(table)
        return table

    def test_content_digest_excludes_meta_and_detects_config_mutation(self):
        table = self._table()
        recorded = table["_meta"]["table_content_digest"]
        # Unbound audit metadata may change without retagging the table.
        table["_meta"]["source_revision"] = "later"
        self.assertEqual(table_content_digest(table), recorded)
        # Both promoted configs and their source evidence are bound.
        table["down"]["16"]["stock"]["decode"] = "bn128-bk16-m16-g8-w4-s2"
        self.assertNotEqual(table_content_digest(table), recorded)
        table = self._table()
        recorded = table["_meta"]["table_content_digest"]
        table["_meta"]["sweep_checkpoint_digest"] = f"sha256:{'9' * 64}"
        self.assertNotEqual(table_content_digest(table), recorded)

    def test_content_digest_differs_across_devices_with_different_winners(self):
        first = self._table()
        second = self._table()
        second["down"]["16"]["stock"]["decode"] = "bn256-bk32-m16-g1-w8-s3"
        self.assertNotEqual(table_content_digest(first), table_content_digest(second))

    def test_execution_fingerprint_binds_the_producer_path(self):
        from benchmark.kernels.lora_moe.timing import execution_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "producer.py"
            second = root / "second" / "producer.py"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("VALUE = 1\n")
            second.write_text("VALUE = 1\n")
            self.assertNotEqual(
                execution_fingerprint(str(first)),
                execution_fingerprint(str(second)),
            )


class TestArtifactPublication(CustomTestCase):
    def _suite(self, *, source_revision="test"):
        from benchmark.kernels.lora_moe.timing import TimingSuite

        return TimingSuite(
            suite="lora_b_schedules",
            device_name="TEST-DEVICE",
            source_revision=source_revision,
            torch_version="test-torch",
            host="test-host",
            execution_digest="exec:test",
            producer_files=("producer.py",),
        )

    def test_sweep_checkpoint_is_marked_immutable_and_not_the_final_output(self):
        import hashlib
        import json

        from benchmark.kernels.lora_moe import bench_lora_b as main_b
        from benchmark.kernels.lora_moe import timing

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            timing,
            "execution_fingerprint",
            return_value="exec:test",
        ):
            final = Path(directory) / "result.json"
            final.write_bytes(b"last-known-good")
            first_path, first_digest = main_b.write_sweep_checkpoint(
                self._suite(), str(final)
            )
            self.assertEqual(final.read_bytes(), b"last-known-good")
            self.assertNotEqual(Path(first_path), final)
            first_payload = Path(first_path).read_bytes()
            self.assertEqual(hashlib.sha256(first_payload).hexdigest(), first_digest)
            self.assertEqual(
                json.loads(first_payload)["suite"],
                "lora_b_schedules_sweep_checkpoint",
            )
            self.assertIn(first_digest, Path(first_path).name)

            repeated_path, repeated_digest = main_b.write_sweep_checkpoint(
                self._suite(), str(final)
            )
            self.assertEqual(
                (repeated_path, repeated_digest), (first_path, first_digest)
            )

            second_path, second_digest = main_b.write_sweep_checkpoint(
                self._suite(source_revision="other"), str(final)
            )
            self.assertNotEqual(second_path, first_path)
            self.assertNotEqual(second_digest, first_digest)
            self.assertTrue(Path(first_path).exists())
            self.assertTrue(Path(second_path).exists())

    def test_atomic_table_write_preserves_old_target_when_replace_fails(self):
        from benchmark.kernels.lora_moe import bench_lora_b as main_b
        from benchmark.kernels.lora_moe import timing

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "table.json"
            target.write_bytes(b"last-known-good")
            with mock.patch.object(
                timing.os,
                "replace",
                side_effect=OSError("injected replace failure"),
            ), self.assertRaisesRegex(OSError, "injected"):
                main_b.write_config_table({"new": True}, str(target))
            self.assertEqual(target.read_bytes(), b"last-known-good")
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

            main_b.write_config_table({"new": True}, str(target))
            self.assertEqual(target.read_text(), '{\n "new": true\n}')

    def test_atomic_suite_write_preserves_target_and_honors_umask(self):
        from benchmark.kernels.lora_moe import timing

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            timing,
            "execution_fingerprint",
            return_value="exec:test",
        ):
            target = Path(directory) / "suite.json"
            target.write_bytes(b"last-known-good")
            target.chmod(0o640)
            with mock.patch.object(
                timing.os,
                "replace",
                side_effect=OSError("injected replace failure"),
            ), self.assertRaisesRegex(OSError, "injected"):
                timing.write_suite(self._suite(), str(target))
            self.assertEqual(target.read_bytes(), b"last-known-good")
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])
            # 12th review: run the existing-target replacement under a
            # umask that INTERSECTS the target's mode bits. Under common
            # umasks (022/002/027) 0o640 survives os.open's masking
            # unchanged, so deleting production's os.fchmod restoration
            # kept this assertion green. 0o640 & ~0o077 == 0o600 != 0o640,
            # so only the explicit fchmod can produce the expected mode.
            old_umask = timing.os.umask(0o077)
            try:
                timing.write_suite(self._suite(), str(target))
            finally:
                timing.os.umask(old_umask)
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)

            fresh = Path(directory) / "fresh.json"
            old_umask = timing.os.umask(0o077)
            try:
                timing.write_suite(self._suite(), str(fresh))
            finally:
                timing.os.umask(old_umask)
            self.assertEqual(fresh.stat().st_mode & 0o777, 0o600)

    def test_every_publication_surface_refuses_a_symlink_destination(self):
        """All three symlink guards, none previously covered (12th review).

        os.replace would swap out the LINK itself, silently redirecting
        canonical evidence to wherever the link pointed; deleting any of
        the three production guards must turn a test red.
        """
        from benchmark.kernels.lora_moe import timing
        from benchmark.kernels.lora_moe.bench_common import write_skip_sidecar

        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real.json"
            real.write_bytes(b"{}")
            link = Path(directory) / "link.json"
            link.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "symlink"):
                timing.atomic_write_bytes(link, b"payload")
            self.assertEqual(real.read_bytes(), b"{}")
            with mock.patch.object(
                timing, "execution_fingerprint", return_value="exec:test"
            ):
                suite = self._suite()
                _, digest = timing.write_content_addressed_suite(
                    suite, str(Path(directory) / "anchor.json"), label="probe"
                )
                addressed = Path(directory) / f"anchor.probe.sha256-{digest}.json"
                content = addressed.read_bytes()
                addressed.unlink()
                addressed.symlink_to(real)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    timing.write_content_addressed_suite(
                        suite, str(Path(directory) / "anchor.json"), label="probe"
                    )
            ledger_path, ledger_digest = write_skip_sidecar(
                str(Path(directory) / "sweep.json"), [], content_addressed=True
            )
            ledger = Path(ledger_path)
            ledger_bytes = ledger.read_bytes()
            ledger.unlink()
            ledger.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "symlink"):
                write_skip_sidecar(
                    str(Path(directory) / "sweep.json"), [], content_addressed=True
                )
            # refusal must leave the link in place and the target untouched
            self.assertTrue(addressed.is_symlink())
            self.assertTrue(ledger.is_symlink())
            self.assertEqual(real.read_bytes(), b"{}")
            self.assertNotEqual(content, b"{}")
            self.assertNotEqual(ledger_bytes, b"{}")

    def test_content_addressed_file_with_wrong_bytes_is_rejected(self):
        """The collision contract (12th review): a sha256-named file whose
        bytes do not match its own name is corruption and must refuse."""
        from benchmark.kernels.lora_moe import timing
        from benchmark.kernels.lora_moe.bench_common import write_skip_sidecar

        with tempfile.TemporaryDirectory() as directory:
            anchor = str(Path(directory) / "sweep.json")
            ledger_path, _ = write_skip_sidecar(anchor, [], content_addressed=True)
            Path(ledger_path).write_bytes(b"corrupted")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                write_skip_sidecar(anchor, [], content_addressed=True)
            with mock.patch.object(
                timing, "execution_fingerprint", return_value="exec:test"
            ):
                suite = self._suite()
                addressed, _ = timing.write_content_addressed_suite(
                    suite, anchor, label="probe"
                )
                Path(addressed).write_bytes(b"corrupted")
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    timing.write_content_addressed_suite(suite, anchor, label="probe")

    def test_publication_drift_guard_without_any_mock(self):
        """12th review: every other publication test mocks
        execution_fingerprint, so the drift guard itself had no coverage.
        Here the digest is REAL: recorded at new_suite() from this test
        file as the producer, recomputed at write_suite. A hand-stamped
        stale digest must refuse; the sentinel must refuse."""
        import msgspec

        from benchmark.kernels.lora_moe import timing

        with tempfile.TemporaryDirectory() as directory:
            suite = timing.new_suite(
                "drift_probe", source_revision="test", producer_files=(__file__,)
            )
            self.assertTrue(suite.execution_digest.startswith("exec:"))
            target = Path(directory) / "out.json"
            timing.write_suite(suite, str(target))
            self.assertTrue(target.exists())
            stale = msgspec.structs.replace(suite, execution_digest="exec:" + "0" * 64)
            with self.assertRaisesRegex(RuntimeError, "fingerprint drifted"):
                timing.write_suite(stale, str(target))
            unknown = msgspec.structs.replace(suite, execution_digest="unknown")
            with self.assertRaisesRegex(RuntimeError, "cannot be established"):
                timing.write_suite(unknown, str(target))

    def test_content_addressed_skip_ledgers_do_not_overwrite_each_other(self):
        from benchmark.kernels.lora_moe.bench_common import write_skip_sidecar

        with tempfile.TemporaryDirectory() as directory:
            anchor = str(Path(directory) / "result.json")
            first_path, first_digest = write_skip_sidecar(
                anchor,
                [{"config": "a"}],
                content_addressed=True,
            )
            second_path, second_digest = write_skip_sidecar(
                anchor,
                [{"config": "b"}],
                content_addressed=True,
            )
            self.assertNotEqual(
                (first_path, first_digest), (second_path, second_digest)
            )
            self.assertTrue(Path(first_path).exists())
            self.assertTrue(Path(second_path).exists())


class TestProvenanceRejection(CustomTestCase):
    """require_table_provenance must actually REJECT, not just hash."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _valid_table(self):
        import triton

        from benchmark.kernels.lora_moe.timing import (
            content_fingerprint,
            kernel_fingerprint,
        )

        table = {
            "down": {"16": {"stock": {"decode": "bn64-bk16-m16-g8-w4-s2"}}},
            "_meta": {
                "schema_version": TABLE_SCHEMA_VERSION,
                "device_name": torch.cuda.get_device_name(self.device),
                "torch_version": str(torch.__version__),
                "triton_version": triton.__version__,
                "kernel_digest": kernel_fingerprint(),
                "source_digest": content_fingerprint(),
                "sweep_checkpoint_digest": f"sha256:{'1' * 64}",
                "sweep_skips_digest": f"sha256:{'2' * 64}",
                "workload": {
                    "model_preset": "qwen35_35b",
                    "adapter_cell": "active4_cap8_base1",
                    "route_generator": "iid",
                    "weight_ownership": "shared_outer",
                    "topology": "tp8_ep8",
                    "ranks": "16",
                    "sweep_regimes": "4,64,2048,8192",
                    "expert_id_domain": "ep_local",
                },
            },
        }
        table["_meta"]["table_content_digest"] = table_content_digest(table)
        return table

    def _full_workload(self):
        # Every contract field, because retunable fields now fail CLOSED:
        # a caller must supply each one and either match or declare it.
        return {
            "model_preset": "qwen35_35b",
            "adapter_cell": "active4_cap8_base1",
            "route_generator": "iid",
            "weight_ownership": "shared_outer",
            "topology": "tp8_ep8",
            "ranks": "16",
            "sweep_regimes": "4,64,2048,8192",
            "expert_id_domain": "ep_local",
        }

    def test_valid_table_is_accepted(self):
        require_table_provenance(
            self._valid_table(), self.device, workload=self._full_workload()
        )

    def test_mutated_config_is_rejected(self):
        table = self._valid_table()
        table["down"]["16"]["stock"]["decode"] = "bn256-bk16-m16-g8-w4-s2"
        with self.assertRaisesRegex(ValueError, "modified after emission"):
            require_table_provenance(table, self.device, workload=self._full_workload())

    def test_deleted_digest_is_rejected_not_skipped(self):
        table = self._valid_table()
        del table["_meta"]["table_content_digest"]
        with self.assertRaisesRegex(ValueError, "no table_content_digest"):
            require_table_provenance(table, self.device, workload=self._full_workload())

    def test_missing_source_digest_is_rejected_centrally(self):
        table = self._valid_table()
        del table["_meta"]["source_digest"]
        with self.assertRaisesRegex(ValueError, "source_digest"):
            require_table_provenance(table, self.device, workload=self._full_workload())

    def test_missing_or_malformed_sweep_evidence_digest_is_rejected(self):
        for key in ("sweep_checkpoint_digest", "sweep_skips_digest"):
            for value in (None, "unknown", "sha256:not-a-digest", f"sha256:{'A' * 64}"):
                table = self._valid_table()
                if value is None:
                    del table["_meta"][key]
                else:
                    table["_meta"][key] = value
                table["_meta"]["table_content_digest"] = table_content_digest(table)
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, key
                ):
                    require_table_provenance(
                        table,
                        self.device,
                        workload=self._full_workload(),
                    )

    def test_missing_schema_version_is_rejected(self):
        table = self._valid_table()
        del table["_meta"]["schema_version"]
        with self.assertRaisesRegex(ValueError, "schema_version"):
            require_table_provenance(table, self.device, workload=self._full_workload())

    def test_locally_retuned_declaration_is_honoured_and_bounded(self):
        # Exercises the locally_retuned kwarg itself: a signature edit that
        # dropped it left the body referencing an unbound name, and only a
        # GPU smoke caught it. Also pins that a REQUIRED invariant cannot be
        # waived by declaring it locally-exempt.
        from benchmark.kernels.lora_moe.bench_common import (
            TABLE_LOCALLY_RETUNED_WORKLOAD,
        )

        table = self._valid_table()
        table["_meta"]["workload"]["topology"] = "tp8_ep8"
        table["_meta"]["table_content_digest"] = table_content_digest(table)
        differing = {
            **self._full_workload(),
            "topology": "tp8_ep2",  # this bench sweeps another EP geometry
        }
        with self.assertRaisesRegex(ValueError, "locally_retuned"):
            require_table_provenance(table, self.device, workload=differing)
        require_table_provenance(
            table,
            self.device,
            workload=differing,
            locally_retuned=("topology",),
        )
        self.assertIn("topology", TABLE_LOCALLY_RETUNED_WORKLOAD)
        with self.assertRaisesRegex(ValueError, "REQUIRED transfer invariants"):
            require_table_provenance(
                table,
                self.device,
                workload=differing,
                locally_retuned=("model_preset",),
            )

    def test_null_values_are_rejected_before_local_retuning_exemptions(self):
        from benchmark.kernels.lora_moe.bench_common import (
            TABLE_LOCALLY_RETUNED_WORKLOAD,
            TABLE_REQUIRED_WORKLOAD,
        )

        for key in (*TABLE_REQUIRED_WORKLOAD, *TABLE_LOCALLY_RETUNED_WORKLOAD):
            locally_retuned = (key,) if key in TABLE_LOCALLY_RETUNED_WORKLOAD else ()
            table = self._valid_table()
            table["_meta"]["workload"][key] = None
            table["_meta"]["table_content_digest"] = table_content_digest(table)
            with self.subTest(key=key, side="table"):
                with self.assertRaisesRegex(ValueError, "null"):
                    require_table_provenance(
                        table,
                        self.device,
                        workload=self._full_workload(),
                        locally_retuned=locally_retuned,
                    )
            with self.subTest(key=key, side="caller"):
                with self.assertRaisesRegex(ValueError, "null"):
                    require_table_provenance(
                        self._valid_table(),
                        self.device,
                        workload={**self._full_workload(), key: None},
                        locally_retuned=locally_retuned,
                    )

    def test_missing_required_workload_key_is_rejected(self):
        table = self._valid_table()
        incomplete = {
            k: v for k, v in self._full_workload().items() if k != "weight_ownership"
        }
        with self.assertRaisesRegex(ValueError, "must declare workload"):
            require_table_provenance(table, self.device, workload=incomplete)
        # and a missing RETUNABLE field is equally fatal (fail-closed)
        missing_retunable = {
            k: v for k, v in self._full_workload().items() if k != "topology"
        }
        with self.assertRaisesRegex(ValueError, "must declare workload"):
            require_table_provenance(table, self.device, workload=missing_retunable)

    def test_foreign_workload_is_rejected(self):
        table = self._valid_table()
        with self.assertRaisesRegex(ValueError, "workload"):
            require_table_provenance(
                table,
                self.device,
                workload={**self._full_workload(), "model_preset": "other_model"},
            )


class TestPromotedArmAdmission(CustomTestCase):
    """Durable guard for the P0 that only an operational smoke caught."""

    def test_every_promoted_arm_including_stock_is_admitted(self):
        # The shared-down decided phase previously looped ARMS[1:], leaving
        # the promoted stock config ungated while claiming otherwise, and
        # it called reference_delta with a stale 2-arg signature. Both
        # survived because nothing but a GPU run exercised this loop.
        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb

        seen = []
        reference = object()

        class _StubFixture:
            def reference_delta(self, label):
                seen.append(("oracle", label))
                return reference

        def _fake_admit(fixture, arm, tuned, observed_reference, label):
            self.assertIs(observed_reference, reference)
            seen.append(("arm", arm, label))

        with mock.patch.object(sdb, "_admit", new=_fake_admit):
            admitted = sdb.gate_promoted_arms(
                _StubFixture(), {arm: {} for arm in sdb.ARMS}, "stub"
            )
        self.assertEqual(admitted, sdb.ARMS)
        self.assertIn("stock_charged", admitted)
        self.assertEqual(seen[0], ("oracle", "stub"))
        self.assertEqual([event[1] for event in seen[1:]], list(sdb.ARMS))

    def test_reference_oracle_takes_only_a_label(self):
        # Pins the signature whose drift raised TypeError after an
        # expensive sweep: the oracle must not accept a caller-selected
        # config, because a selected config must never be its own oracle.
        import inspect

        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb

        parameters = list(
            inspect.signature(sdb._DownBFixture.reference_delta).parameters
        )
        self.assertEqual(parameters, ["self", "label"])


class _Args:
    """Stand-in for parsed argparse namespace (production's own input)."""

    def __init__(self, **kwargs):
        self.model_preset = kwargs.get("model_preset", "qwen35_35b")
        self.ranks = kwargs.get("ranks", "16,32,64,128")
        self.sweep_regimes = kwargs.get("sweep_regimes", "4,64,2048,8192")
        self.domains = kwargs.get("domains", "ep_local,global")
        self.validity = kwargs.get("validity", "dense,ep8,ep4,ep2")


class _FakeSuite:
    source_revision = "test"
    observed_revision = "test"
    source_digest = "audit-only"
    device_name = "TEST-DEVICE"
    torch_version = "test-torch"


class TestTableRoundTrip(CustomTestCase):
    """Emitter -> validator, using ONLY production builders.

    9th review: the previous preflight built each synthetic source table
    from the consumer's own declaration and injected fields production
    never sent, so it validated a request that did not exist and hid both
    the ownership mismatch and the missing-field failure. These cases
    execute no CUDA work; the module still requires the normal
    Triton/SGLang benchmark import environment.
    """

    def setUp(self):
        from benchmark.kernels.lora_moe import bench_lora_b as main_b
        from benchmark.kernels.lora_moe.timing import kernel_fingerprint

        self.main_b = main_b
        self.args = _Args()
        self.suite = _FakeSuite()
        self.suite.torch_version = str(torch.__version__)
        self.device = torch.device("cpu")
        self.kernel_digest = kernel_fingerprint()
        self.sweep_checkpoint_digest = f"sha256:{'3' * 64}"
        self.sweep_skips_digest = f"sha256:{'4' * 64}"
        # 10th review: derive the table's config strings from the
        # PRODUCTION serializer (bench_common.config_key) over configs the
        # actually yields. Hand-written strings used shared-down's
        # a second token order, so nothing in the suite ever proved
        # that what Path A writes is what Path C can parse.
        from benchmark.kernels.lora_moe.bench_common import (
            exhaustive_grouped_lora_b_grid,
        )

        self.best = {}
        for rank in (16, 32, 64, 128):
            for family in self.main_b.MAIN_TABLE_REQUIRED_FAMILIES:
                if family == "indexed":
                    first = self.main_b._sweep_grid(family, rank)[0]
                else:
                    first = next(
                        iter(
                            exhaustive_grouped_lora_b_grid(
                                rank=rank, stock=family == "stock"
                            )
                        )
                    )
                key = config_key(first)
                for site in ("gate_up", "down"):
                    for regime in (
                        "decode_tiny",
                        "decode",
                        "prefill",
                        "prefill_xl",
                    ):
                        self.best[(site, rank, family, regime)] = key
            rank_split_grid = self.main_b._sweep_grid("rank_split", rank)
            if rank_split_grid:
                key = config_key(rank_split_grid[0])
                for site in ("gate_up", "down"):
                    for regime in self.main_b.RANK_SPLIT_REQUIRED_REGIMES:
                        self.best[(site, rank, "rank_split", regime)] = key
        self.table = self._build_table()
        # Avoid a CUDA device query; metadata validation itself performs no
        # GPU work.
        cuda_name = mock.patch.object(
            torch.cuda, "get_device_name", return_value="TEST-DEVICE"
        )
        cuda_name.start()
        self.addCleanup(cuda_name.stop)

    def _build_table(self, *, best=None, arguments=None):
        return self.main_b.build_main_table(
            best=self.best if best is None else best,
            arguments=self.args if arguments is None else arguments,
            suite=self.suite,
            kernel_digest=self.kernel_digest,
            sweep_checkpoint_digest=self.sweep_checkpoint_digest,
            sweep_skips_digest=self.sweep_skips_digest,
        )

    def test_emitted_table_passes_its_own_validator(self):
        # main -> main. This is the round trip the 9th review found broken:
        # the emitter hand-rolled a configs-only digest while the validator
        # bound semantic metadata, so every table failed its first consumer.
        require_table_provenance(
            self.table,
            torch.device("cpu"),
            workload=self.main_b.build_transfer_request(self.args),
            locally_retuned=self.main_b.LOCALLY_RETUNED,
        )

    def test_emitted_table_is_bound_to_sweep_suite_and_skip_ledger(self):
        self.assertEqual(
            self.table["_meta"]["sweep_checkpoint_digest"],
            self.sweep_checkpoint_digest,
        )
        self.assertEqual(
            self.table["_meta"]["sweep_skips_digest"],
            self.sweep_skips_digest,
        )
        recorded = self.table["_meta"]["table_content_digest"]
        # 12th review: EVERY bound identity field individually — including
        # source_digest, which was propagated into consumer records as
        # table_source_digest yet excluded from the binding, so a
        # post-emission forgery passed provenance and got stamped into
        # published evidence.
        mutations = {
            "sweep_checkpoint_digest": f"sha256:{'5' * 64}",
            "sweep_skips_digest": f"sha256:{'6' * 64}",
            "source_digest": "files:FORGED-AFTER-EMISSION",
        }
        for field, forged in mutations.items():
            with self.subTest(field=field):
                table = self._build_table()
                self.assertEqual(table["_meta"]["table_content_digest"], recorded)
                table["_meta"][field] = forged
                self.assertNotEqual(table_content_digest(table), recorded)
                with self.assertRaisesRegex(ValueError, "modified after emission"):
                    require_table_provenance(
                        table,
                        self.device,
                        workload=self.main_b.build_transfer_request(self.args),
                        locally_retuned=self.main_b.LOCALLY_RETUNED,
                    )

    def test_actual_sweep_artifacts_round_trip_into_table_identity(self):
        import hashlib

        from benchmark.kernels.lora_moe import timing
        from benchmark.kernels.lora_moe.bench_common import write_skip_sidecar

        suite = timing.TimingSuite(
            suite="lora_b_schedules",
            device_name=self.suite.device_name,
            source_revision=self.suite.source_revision,
            observed_revision=self.suite.observed_revision,
            source_digest=self.suite.source_digest,
            kernel_digest=self.kernel_digest,
            torch_version=self.suite.torch_version,
            host="test-host",
            execution_digest="exec:test",
            producer_files=("producer.py",),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            timing,
            "execution_fingerprint",
            return_value="exec:test",
        ):
            output = str(Path(directory) / "result.json")
            checkpoint_path, checkpoint_digest = self.main_b.write_sweep_checkpoint(
                suite, output
            )
            skips_path, skips_digest = write_skip_sidecar(
                output,
                [{"stage": "compile", "config": "test"}],
                content_addressed=True,
            )
            table = self.main_b.build_main_table(
                best=self.best,
                arguments=self.args,
                suite=suite,
                kernel_digest=self.kernel_digest,
                sweep_checkpoint_digest=f"sha256:{checkpoint_digest}",
                sweep_skips_digest=f"sha256:{skips_digest}",
            )
            self.assertEqual(
                hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                table["_meta"]["sweep_checkpoint_digest"].removeprefix("sha256:"),
            )
            self.assertEqual(
                hashlib.sha256(Path(skips_path).read_bytes()).hexdigest(),
                table["_meta"]["sweep_skips_digest"].removeprefix("sha256:"),
            )
            self.assertEqual(
                table_content_digest(table),
                table["_meta"]["table_content_digest"],
            )
            require_table_provenance(
                table,
                self.device,
                workload=self.main_b.build_transfer_request(self.args),
                locally_retuned=self.main_b.LOCALLY_RETUNED,
            )

    def test_main_section_preflight_uses_each_sections_real_surface(self):
        self.main_b.require_main_table_for_sections(
            self.table,
            ranks=(16, 32, 64, 128),
            sections={"decided", "leg"},
        )

        decided_missing = self._build_table()
        del decided_missing["gate_up"]["32"]["lean_two_launch"]["decode"]
        with self.assertRaisesRegex(RuntimeError, "lean_two_launch"):
            self.main_b.require_main_table_for_sections(
                decided_missing,
                ranks=(32,),
                sections={"decided"},
            )

        rank_split_missing = self._build_table()
        del rank_split_missing["down"]["32"]["rank_split"]
        with self.assertRaisesRegex(RuntimeError, "rank_split"):
            self.main_b.require_main_table_for_sections(
                rank_split_missing,
                ranks=(32,),
                sections={"decided"},
            )

        malformed = self._build_table()
        malformed["gate_up"]["32"]["one_launch"]["decode"] += "-bogus1"
        with self.assertRaisesRegex(ValueError, "invalid table config"):
            self.main_b.require_main_table_for_sections(
                malformed,
                ranks=(32,),
                sections={"decided"},
            )

        outside_grid = self._build_table()
        outside_grid["gate_up"]["32"]["indexed"]["decode"] = "bn1024-bk16-w4-s2"
        with self.assertRaisesRegex(ValueError, "outside the declared grid"):
            self.main_b.require_main_table_for_sections(
                outside_grid,
                ranks=(32,),
                sections={"decided"},
            )

        leg_missing = self._build_table()
        del leg_missing["down"]["16"]["indexed"]["prefill"]
        with self.assertRaisesRegex(RuntimeError, "indexed"):
            self.main_b.require_main_table_for_sections(
                leg_missing,
                # Leg deliberately uses its own fixed rank surface.
                ranks=(32,),
                sections={"leg"},
            )

    def test_shared_down_declares_ownership_that_forbids_main_table_transfer(self):
        # Shared-down is shared_outer; the main table is per_expert, and
        # ownership is REQUIRED and non-retunable. Rather than paper over
        # that, the bench consumes NO table — assert both facts.
        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb
        from benchmark.kernels.lora_moe.bench_common import (
            TABLE_LOCALLY_RETUNED_WORKLOAD,
        )

        self.assertEqual(sdb.WEIGHT_OWNERSHIP, "shared_outer")
        self.assertEqual(
            self.table["_meta"]["workload"]["weight_ownership"], "per_expert"
        )
        self.assertNotIn("weight_ownership", TABLE_LOCALLY_RETUNED_WORKLOAD)
        # and the transfer it would imply is refused
        with self.assertRaisesRegex(ValueError, "weight_ownership"):
            require_table_provenance(
                self.table,
                torch.device("cpu"),
                workload={
                    **self.main_b.build_transfer_request(self.args),
                    "weight_ownership": sdb.WEIGHT_OWNERSHIP,
                },
                locally_retuned=self.main_b.LOCALLY_RETUNED,
            )

    def test_mutating_a_promoted_config_breaks_the_round_trip(self):
        self.table["down"]["16"]["stock"]["decode"] = "bn512-bk16-m16-g8-w4-s2"
        with self.assertRaisesRegex(ValueError, "modified after emission"):
            require_table_provenance(
                self.table,
                torch.device("cpu"),
                workload=self.main_b.build_transfer_request(self.args),
                locally_retuned=self.main_b.LOCALLY_RETUNED,
            )

    def test_mutating_bound_metadata_breaks_the_round_trip(self):
        # The digest binds workload identity, so retagging a table as a
        # different model must not survive validation either.
        self.table["_meta"]["workload"]["model_preset"] = "some_other_model"
        with self.assertRaisesRegex(ValueError, "modified after emission"):
            require_table_provenance(
                self.table,
                torch.device("cpu"),
                workload=self.main_b.build_transfer_request(self.args),
                locally_retuned=self.main_b.LOCALLY_RETUNED,
            )

    def test_nondefault_request_fields_propagate_through_real_builders(self):
        partial_args = _Args(
            model_preset="ragged_gated_176",
            ranks="8,48",
            sweep_regimes="4,8192",
        )
        main_request = self.main_b.build_transfer_request(partial_args)
        self.assertEqual(main_request["model_preset"], "ragged_gated_176")
        self.assertEqual(main_request["ranks"], "8,48")
        self.assertEqual(main_request["sweep_regimes"], "4,8192")
        with self.assertRaisesRegex(ValueError, "every regime"):
            self.main_b.require_table_emission_regimes(partial_args.sweep_regimes)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.main_b.require_table_emission_regimes("1,8,64,2048,8192")

        full_args = _Args(model_preset="ragged_gated_176", ranks="8,48")
        nondefault_best = {}
        for rank in (8, 48):
            for family in self.main_b.MAIN_TABLE_REQUIRED_FAMILIES:
                config = self.main_b._sweep_grid(family, rank)[0]
                key = config_key(config)
                for site in ("gate_up", "down"):
                    for regime in self.main_b.MAIN_TABLE_REQUIRED_REGIMES:
                        nondefault_best[(site, rank, family, regime)] = key
            rank_split_grid = self.main_b._sweep_grid("rank_split", rank)
            if rank_split_grid:
                key = config_key(rank_split_grid[0])
                for site in ("gate_up", "down"):
                    for regime in self.main_b.RANK_SPLIT_REQUIRED_REGIMES:
                        nondefault_best[(site, rank, "rank_split", regime)] = key
        emitted = self._build_table(best=nondefault_best, arguments=full_args)
        self.assertEqual(
            emitted["_meta"]["workload"],
            self.main_b.build_transfer_request(full_args),
        )

    def test_accumulate_gate_admits_bf16_base_rounding_at_the_floor(self):
        """13th finding (GB300 smoke): accumulate arms write
        bf16(base + delta), so each element carries base-magnitude
        rounding; at the 12.5%% validity boundary the achievable rel-L2
        floor (~3.1e-2) exceeded the fixed 1e-2 gate — unsatisfiable for
        a PERFECT kernel. The gate now derives the floor from the
        measured base/delta L2 ratio. A dropped delta must still fail."""
        from benchmark.kernels.lora_moe.bench_common import (
            require_delta_close_chunked,
        )

        generator = torch.Generator().manual_seed(7)
        base = torch.randn(64, 256, generator=generator, dtype=torch.float64)
        delta = 0.15 * torch.randn(64, 256, generator=generator, dtype=torch.float64)
        # what a perfect accumulate kernel produces: bf16(base + delta)
        observed = (base + delta).to(torch.bfloat16).to(torch.float64)
        require_delta_close_chunked(
            observed,
            delta,
            gate_dtype=torch.bfloat16,
            label="perfect-kernel-at-floor",
            observed_base=base,
        )
        with self.assertRaises(AssertionError):
            require_delta_close_chunked(
                base.to(torch.bfloat16).to(torch.float64),  # delta DROPPED
                delta,
                gate_dtype=torch.bfloat16,
                label="dropped-delta-at-floor",
                observed_base=base,
            )

    def test_accumulate_gate_floor_is_bounded_and_sparse_deltas_degenerate(self):
        """13th review (fail-open repro, now permanent): the derived floor
        scaled with L2(base)/L2(delta) UNBOUNDED — a dense base plus a
        SPARSE delta (peaks fine, tiny L2) let 10,000 one-quantum errors
        pass at rel_l2 = 6.25. The gate now carries a hard ceiling and
        raises DegenerateSignalError when the floor would exceed it."""
        from benchmark.kernels.lora_moe.bench_common import (
            ACCUMULATE_REL_L2_CEILING,
            require_delta_close_chunked,
        )
        from benchmark.kernels.lora_moe.signal_gates import (
            DegenerateSignalError,
        )

        rows, cols = 128, 256
        base = torch.full((rows, cols), 1.0, dtype=torch.float64)
        # sparse delta: max amplitude passes the peak validity rule, but
        # its L2 is drowned by base storage noise across 32768 elements
        delta = torch.zeros(rows, cols, dtype=torch.float64)
        delta[0, 0] = 0.125
        corrupted = base.clone()
        corrupted += torch.full_like(base, 2.0**-8)  # 32k one-quantum errors
        corrupted[0, 0] += 0.125
        with self.assertRaises((DegenerateSignalError, AssertionError)):
            require_delta_close_chunked(
                corrupted,
                delta,
                gate_dtype=torch.bfloat16,
                label="sparse-delta-dense-corruption",
                observed_base=base,
            )
        self.assertLessEqual(ACCUMULATE_REL_L2_CEILING, 0.2)

    def test_decide_cell_records_boundary_scope(self):
        """12th review: SGMV-vs-grouped compared a prepared-input arm to a
        route-inclusive arm and printed a verdict indistinguishable from a
        same-boundary one. The scope is now part of the decision."""
        from benchmark.kernels.lora_moe.crossover_ledger import decide_cell

        charged = decide_cell(
            arm_a="a",
            samples_a=[2.0, 2.0],
            arm_b="b",
            samples_b=[1.0, 1.0],
            boundary_a="route_inclusive",
            boundary_b="route_inclusive",
        )
        self.assertEqual(charged.scope, "charged")
        self.assertEqual(charged.winner, "b")
        ceiling = decide_cell(
            arm_a="a",
            samples_a=[2.0, 2.0],
            arm_b="b",
            samples_b=[1.0, 1.0],
            boundary_a="route_inclusive",
            boundary_b="prepared_input",
        )
        self.assertEqual(ceiling.scope, "ceiling")
        undeclared = decide_cell(
            arm_a="a",
            samples_a=[2.0, 2.0],
            arm_b="b",
            samples_b=[1.0, 1.0],
        )
        self.assertEqual(undeclared.scope, "undeclared")
        # every S4 producer declares boundaries at its decide site — none
        # may emit an undeclared decision
        for module_path in (
            "benchmark/kernels/lora_moe/bench_lora_b.py",
            "benchmark/kernels/lora_moe/bench_shared_down_b.py",
            "benchmark/kernels/lora_moe/bench_perexpert_sgmv_b.py",
        ):
            source = (Path(__file__).resolve().parents[3] / module_path).read_text()
            calls = source.count("decide_cell(")
            declared = source.count("boundary_a=")
            self.assertEqual(
                calls, declared, f"{module_path}: undeclared decide_cell call"
            )

    def test_grouped_arms_get_the_same_exhaustive_grid_as_challengers(self):
        """12th review: per-expert re-tuned its grouped baselines over a
        one-step neighborhood of a dense/ep_local table winner while the
        sgmv challengers swept a full grid. The neighborhood helper is
        deleted; both secondary producers now sweep the SHARED exhaustive
        grid on their own geometry."""
        import benchmark.kernels.lora_moe.bench_common as bench_common

        self.assertFalse(hasattr(bench_common, "grouped_neighborhood"))
        for module_path in (
            "benchmark/kernels/lora_moe/bench_perexpert_sgmv_b.py",
            "benchmark/kernels/lora_moe/bench_shared_down_b.py",
        ):
            source = (Path(__file__).resolve().parents[3] / module_path).read_text()
            self.assertIn("exhaustive_grouped_lora_b_grid(", source)
            self.assertNotIn("grouped_neighborhood", source)
            # the argparse literal, not docstring history mentions
            self.assertNotIn('"--config-table"', source)

    def test_rank_split_grid_explores_beyond_the_old_bn64_ceiling(self):
        """12th review: every archived rank_split winner sat AT the old
        BN=64 ceiling, so 'uniformly rejected' was only established for a
        hobbled variant. The grid must now span the same BN/GM/warp space
        as the grouped families it is adjudicated against."""
        grid = self.main_b._sweep_grid("rank_split", 64)
        bns = {config["BLOCK_SIZE_N"] for config in grid}
        gms = {config["GROUP_SIZE_M"] for config in grid}
        warps = {config["num_warps"] for config in grid}
        self.assertGreaterEqual(max(bns), 512)
        self.assertEqual(gms, {1, 4, 8, 16})
        self.assertIn(8, warps)

    def test_aliased_or_directory_output_paths_fail_at_startup(self):
        import tempfile as _tempfile

        from benchmark.kernels.lora_moe.bench_common import (
            require_distinct_paths,
            require_writable_destination,
        )

        with _tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "alias"):
                require_distinct_paths(
                    f"{directory}/x.json", None, f"{directory}/x.json"
                )
            with self.assertRaisesRegex(ValueError, "DIRECTORY"):
                require_writable_destination(directory)
            arguments = self.main_b.build_parser().parse_args(
                [
                    "--output",
                    f"{directory}/same.json",
                    "--sections",
                    "sweep",
                    "--emit-config-table",
                    f"{directory}/same.json",
                ]
            )
            with self.assertRaisesRegex(ValueError, "alias"):
                self.main_b.validate_run_arguments(arguments)

    def test_main_rank_axis_is_canonical(self):
        self.assertEqual(self.main_b.parse_rank_axis("8,48,128"), (8, 48, 128))
        for invalid in ("", "0", "-8", "8,8", "8,bad"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.main_b.parse_rank_axis(invalid)

        spaced = _Args(
            ranks=" 8, +48 ",
            sweep_regimes=" 4, 64, 2048, +8192 ",
        )
        request = self.main_b.build_transfer_request(spaced)
        self.assertEqual(request["ranks"], "8,48")
        self.assertEqual(request["sweep_regimes"], "4,64,2048,8192")

    def test_shipped_defaults_survive_startup_validation(self):
        """12th review: parse_sections runs before CUDA init and raises on
        any name not in SECTION_ORDER, so a default/SECTION_ORDER drift
        would crash EVERY default invocation with the suite green. Drive
        production's own parser defaults through production's own
        validator (build_parser + validate_run_arguments)."""
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as directory:
            arguments = self.main_b.build_parser().parse_args(
                ["--output", f"{directory}/out.json"]
            )
            sections = self.main_b.validate_run_arguments(arguments)
            self.assertTrue(sections)
            self.assertLessEqual(sections, set(self.main_b.SECTION_ORDER))
            # the two new startup refusals (12th review)
            emit_no_sweep = self.main_b.build_parser().parse_args(
                [
                    "--output",
                    f"{directory}/out.json",
                    "--emit-config-table",
                    f"{directory}/t.json",
                    "--sections",
                    "decided",
                ]
            )
            with self.assertRaisesRegex(ValueError, "requires the sweep"):
                self.main_b.validate_run_arguments(emit_no_sweep)
            combined = self.main_b.build_parser().parse_args(
                [
                    "--output",
                    f"{directory}/out.json",
                    "--sections",
                    "sweep,decided",
                    "--emit-config-table",
                    f"{directory}/t.json",
                ]
            )
            with self.assertRaisesRegex(ValueError, "separate invocation"):
                self.main_b.validate_run_arguments(combined)
            missing_dir = self.main_b.build_parser().parse_args(
                ["--output", f"{directory}/no/such/dir/out.json"]
            )
            with self.assertRaisesRegex(ValueError, "does not exist"):
                self.main_b.validate_run_arguments(missing_dir)

    def test_rank_split_grid_joint_block_k_split_constraint(self):
        """12th review: the set of BLOCK_K values alone is identical under
        the old and new caps; the discriminating contract is the JOINT
        constraint every emitted config must satisfy: bk*2 <= rank (a
        split must have at least two K tiles) and rank // split >= bk
        (every split's share covers a whole tile)."""
        for rank in (16, 32, 48, 64, 96, 128):
            grid = self.main_b._sweep_grid("rank_split", rank)
            for config in grid:
                bk, split = config["BLOCK_SIZE_K"], config["SPLIT_K"]
                with self.subTest(rank=rank, bk=bk, split=split):
                    self.assertLessEqual(bk * 2, rank)
                    self.assertGreaterEqual(rank // split, bk)
            if rank >= 32:
                self.assertTrue(grid, f"rank {rank} must have rank_split configs")

    def test_main_section_axis_is_known_unique_and_canonical(self):
        parsed = self.main_b.parse_sections("leg,sweep,decided")
        self.assertEqual(
            tuple(
                section for section in self.main_b.SECTION_ORDER if section in parsed
            ),
            ("sweep", "decided", "leg"),
        )
        for invalid in ("", "sweep,sweep", "sweep,unknown"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.main_b.parse_sections(invalid)

    def test_emitted_config_keys_parse_back_to_the_same_config(self):
        """Path A's key vocabulary must be readable by Path C's parser.

        10th review: no test fed a production-emitted key through
        ``parse_table_config``, so a rename or reordering in
        ``config_key`` would make every table unreadable with the suite
        still green.
        """
        from benchmark.kernels.lora_moe.bench_common import (
            exhaustive_grouped_lora_b_grid,
            parse_table_config,
        )

        for rank in (16, 32, 64, 128):
            for stock in (True, False):
                configs = list(exhaustive_grouped_lora_b_grid(rank=rank, stock=stock))
                # Token vocabulary and ordering are invariant across the
                # Cartesian grid; cover both ends without thousands of
                # duplicate subtests.
                for config in (configs[0], configs[-1]):
                    key = config_key(config)
                    with self.subTest(rank=rank, stock=stock, key=key):
                        self.assertEqual(parse_table_config(key), config)

    def test_default_main_sweep_covers_every_per_expert_regime(self):
        """Pin the T -> regime-CLASS mapping, not just the T values.

        10th review: comparing only the "4,64,2048,8192" strings left the
        real coupling untested — the table's KEYS come from
        ``bench_common.regime_of(T)``, governed by DECODE_TINY_T_MAX /
        DECODE_T_MAX / PREFILL_XL_T_MIN. Retuning any threshold kept this
        test green while Path C died on a missing regime key after a
        multi-hour sweep. Derive the classes the way production does.
        """
        from benchmark.kernels.lora_moe import bench_perexpert_sgmv_b as pe

        emitted_classes = {
            regime_of(int(t)) for t in self.main_b.DEFAULT_SWEEP_REGIMES.split(",")
        }
        self.assertEqual(
            emitted_classes,
            set(pe.SWEEP_T),
            "Path A's default anchors must emit exactly Path C's regime keys",
        )
        # and each consumer T must land in the class it is named for
        for regime, num_tokens in pe.SWEEP_T.items():
            with self.subTest(regime=regime):
                self.assertEqual(regime_of(num_tokens), regime)
                self.assertEqual(regime_of(num_tokens), regime)

    def test_redigested_table_missing_required_source_key_is_rejected(self):
        table = self._build_table()
        del table["_meta"]["workload"]["route_generator"]
        table["_meta"]["table_content_digest"] = table_content_digest(table)
        caller = {
            **self.main_b.build_transfer_request(self.args),
            "route_generator": None,
        }
        with self.assertRaisesRegex(ValueError, "lacks required invariant"):
            require_table_provenance(
                table,
                torch.device("cpu"),
                workload=caller,
                locally_retuned=self.main_b.LOCALLY_RETUNED,
            )


class TestPerExpertSetup(CustomTestCase):
    def test_setup_canonicalizes_axes_once(self):
        from benchmark.kernels.lora_moe import bench_perexpert_sgmv_b as pe

        arguments = _Args(ranks=" 8, 48 ", domains="ep_local, global")
        self.assertEqual(
            pe.build_run_setup(arguments),
            ((8, 48), ("ep_local", "global")),
        )

    def test_setup_rejects_invalid_axes_before_cuda(self):
        from benchmark.kernels.lora_moe import bench_perexpert_sgmv_b as pe

        invalid = (
            ("", "ep_local"),
            ("0", "ep_local"),
            ("x", "ep_local"),
            ("8,8", "ep_local"),
            ("8,,16", "ep_local"),
            ("8", ""),
            ("8", "ep_local,ep_local"),
            ("8", "ep_local,,global"),
            ("8", "unknown"),
        )
        for ranks, domains in invalid:
            with self.subTest(ranks=ranks, domains=domains):
                with self.assertRaises(ValueError):
                    pe.build_run_setup(_Args(ranks=ranks, domains=domains))

    def test_record_identity_contains_the_resolved_domain_and_proxy_scope(self):
        from benchmark.kernels.lora_moe import bench_perexpert_sgmv_b as pe

        case = SimpleNamespace(
            model_preset="qwen35_35b",
            active_adapters=4,
            slot_capacity=8,
            include_base_rows=True,
            route_generator="iid",
            tp_size=8,
            ep_size=8,
            moe_dp_size=1,
            ep_rank=0,
            expert_id_domain="global",
        )
        workload = pe.record_workload_identity(case)
        self.assertEqual(workload["topology"], "tp8_ep8_moedp1")
        self.assertEqual(workload["ep_rank"], 0)
        self.assertEqual(workload["expert_id_domain"], "global")
        self.assertEqual(workload["evidence_scope"], "single_gpu_local_shape_proxy")
        self.assertEqual(workload["weight_ownership"], "per_expert")


class TestSharedDownSetup(CustomTestCase):
    def test_production_parser_and_setup_preserve_nondefault_axes(self):
        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb

        arguments = sdb._build_parser().parse_args(
            [
                "--output",
                "unused.json",
                "--ranks",
                "8,48,128",
                "--validity",
                "ep2,dense",
            ]
        )
        self.assertEqual(
            sdb.build_run_setup(arguments), ((8, 48, 128), ("ep2", "dense"))
        )
        self.assertNotIn(
            "config_table", {action.dest for action in sdb._build_parser()._actions}
        )

    def test_setup_rejects_invalid_axes_before_cuda(self):
        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb

        invalid = (
            ("", "dense"),
            ("0", "dense"),
            ("16,16", "dense"),
            ("16,,32", "dense"),
            ("16,", "dense"),
            ("16", ""),
            ("16", "dense,dense"),
            ("16", "dense,,ep2"),
            ("16", "dense,"),
            ("16", "unknown"),
        )
        for ranks, validity in invalid:
            with self.subTest(ranks=ranks, validity=validity):
                with self.assertRaises(ValueError):
                    sdb.build_run_setup(SimpleNamespace(ranks=ranks, validity=validity))

    def test_every_producer_and_neighbourhood_share_one_block_k_cap(self):
        """One cap formula, used everywhere (11th review).

        The grouped/sgmv grids moved to padded_block_k_cap while the
        indexed branch and grouped_neighborhood stayed on max(rank, 32).
        At rank 48 the grouped grid offered BLOCK_K {16,32,64} while
        indexed saw only {16,32} — so indexed was adjudicated against
        families tuned over a wider axis — and the neighbourhood could not
        reach a config the producer's own grid contains.
        """
        from benchmark.kernels.lora_moe import bench_lora_b as main_b
        from benchmark.kernels.lora_moe.bench_common import (
            exhaustive_grouped_lora_b_grid,
            exhaustive_sgmv_grid,
        )

        expected_by_rank = {
            1: {16, 32},
            2: {16, 32},
            4: {16, 32},
            8: {16, 32},
            12: {16, 32},
            16: {16, 32},
            24: {16, 32},
            32: {16, 32},
            48: {16, 32, 64},
            64: {16, 32, 64},
            96: {16, 32, 64, 128},
            128: {16, 32, 64, 128},
            192: {16, 32, 64, 128},
        }
        rank_split_expected_by_rank = {
            1: set(),
            2: set(),
            4: set(),
            8: set(),
            12: set(),
            16: set(),
            24: set(),
            32: {16},
            48: {16},
            64: {16, 32},
            96: {16, 32},
            128: {16, 32, 64},
            192: {16, 32, 64},
        }
        for rank, expected in expected_by_rank.items():
            with self.subTest(rank=rank):
                grouped = {
                    c["BLOCK_SIZE_K"]
                    for c in exhaustive_grouped_lora_b_grid(rank=rank, stock=False)
                }
                # the sgmv grid names its K axis BLOCK_K, not BLOCK_SIZE_K
                sgmv = {
                    c["BLOCK_K"]
                    for c in exhaustive_sgmv_grid(rank=rank, n_columns=2048)
                }
                indexed = {
                    c["BLOCK_SIZE_K"] for c in main_b._sweep_grid("indexed", rank)
                }
                rank_split = {
                    c["BLOCK_SIZE_K"] for c in main_b._sweep_grid("rank_split", rank)
                }
                indexed_extra = (
                    {rank} if 0 < rank < 16 and rank & (rank - 1) == 0 else set()
                )
                self.assertEqual(grouped, expected)
                self.assertEqual(sgmv, expected)
                self.assertEqual(indexed, expected | indexed_extra)
                self.assertEqual(rank_split, rank_split_expected_by_rank[rank])

    def test_grouped_grid_never_emits_a_block_k_triton_cannot_compile(self):
        """Every BLOCK_SIZE_K must be a power of two >= 16, at ANY rank.

        A ``BLOCK_SIZE_K = rank`` insertion used to add 8 for rank 8 and 48
        for rank 48. Both are uncompilable, and because a Triton
        CompilationError matches no skip signature, the fail-closed sweep
        aborted the entire run on its first config.
        """
        from benchmark.kernels.lora_moe.bench_common import (
            exhaustive_grouped_lora_b_grid,
        )

        for rank in (1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
            for stock in (True, False):
                block_ks = {
                    config["BLOCK_SIZE_K"]
                    for config in exhaustive_grouped_lora_b_grid(rank=rank, stock=stock)
                }
                with self.subTest(rank=rank, stock=stock):
                    self.assertTrue(block_ks)
                    self.assertTrue(
                        all(
                            block_k >= 16 and block_k & (block_k - 1) == 0
                            for block_k in block_ks
                        )
                    )

    def test_indexed_and_sgmv_rank8_grids_keep_their_valid_padded_tiles(self):
        from benchmark.kernels.lora_moe import bench_lora_b as main_b
        from benchmark.kernels.lora_moe.bench_common import exhaustive_sgmv_grid

        indexed_block_ks = {
            config["BLOCK_SIZE_K"] for config in main_b._sweep_grid("indexed", 8)
        }
        self.assertEqual(indexed_block_ks, {8, 16, 32})
        self.assertNotIn(
            48,
            {config["BLOCK_SIZE_K"] for config in main_b._sweep_grid("indexed", 48)},
        )
        sgmv_block_ks = {
            config["BLOCK_K"] for config in exhaustive_sgmv_grid(rank=8, n_columns=64)
        }
        self.assertEqual(sgmv_block_ks, {16, 32})

        rank48_block_ks = {
            config["BLOCK_K"] for config in exhaustive_sgmv_grid(rank=48, n_columns=64)
        }
        self.assertEqual(rank48_block_ks, {16, 32, 64})

    def test_rank_split_dispatch_matches_its_sweep_surface(self):
        from benchmark.kernels.lora_moe import bench_lora_b as main_b

        self.assertFalse(main_b._rank_split_is_swept(16, "decode"))
        self.assertTrue(main_b._rank_split_is_swept(32, "decode_tiny"))
        self.assertTrue(main_b._rank_split_is_swept(32, "decode"))
        self.assertFalse(main_b._rank_split_is_swept(32, "prefill"))

    def test_grouped_grid_is_complete_and_shared_with_main(self):
        from benchmark.kernels.lora_moe import bench_lora_b as main_b
        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb

        # 10th review: BLOCK_SIZE_K must ALWAYS be a power of two >= 16.
        # The previous expectation pinned rank 8 -> {8,16,32} and rank 48 ->
        # {16,32,48}; Triton cannot compile either (tl.dot needs every dim
        # >= 16, tl.arange needs a power-of-two length) and the resulting
        # CompilationError is not a skip signature, so the fail-closed
        # sweep aborted the whole run on its first config. Short and
        # non-power-of-two ranks are measured with masked K through the
        # next power-of-two tile, which also covers the one-iteration
        # candidate rather than silently under-tuning rank 48.
        expected_block_ks = {
            8: {16, 32},
            16: {16, 32},
            48: {16, 32, 64},
            64: {16, 32, 64},
            128: {16, 32, 64, 128},
        }
        for rank, expected in expected_block_ks.items():
            for family in ("stock", "one_launch"):
                with self.subTest(rank=rank, family=family):
                    configs = list(sdb._grouped_grid(rank, family))
                    self.assertEqual(
                        {config["BLOCK_SIZE_K"] for config in configs}, expected
                    )
                    self.assertEqual(len(configs), 104 * len(expected))
                    self.assertEqual(
                        len({config_key(config) for config in configs}),
                        len(configs),
                    )
                    self.assertEqual(configs, main_b._sweep_grid(family, rank))
                    if family == "stock":
                        self.assertTrue(
                            all(config["BLOCK_SIZE_M"] == 16 for config in configs)
                        )
                    else:
                        self.assertTrue(
                            all("BLOCK_SIZE_M" not in config for config in configs)
                        )

    def test_measurement_params_include_resolved_local_proxy_identity(self):
        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb

        case = SimpleNamespace(
            model_preset="qwen35_35b",
            active_adapters=4,
            slot_capacity=8,
            include_base_rows=True,
            route_generator="iid",
            tp_size=8,
            ep_size=4,
            moe_dp_size=1,
            ep_rank=0,
            expert_id_domain="global",
        )
        params = sdb._measurement_params(case, phase="sweep")
        self.assertEqual(sdb.SUITE_NAME, "shared_down_b_v7")
        self.assertEqual(params["workload"]["topology"], "tp8_ep4_moedp1")
        self.assertEqual(params["workload"]["ep_rank"], 0)
        self.assertEqual(params["workload"]["expert_id_domain"], "global")
        self.assertEqual(
            params["workload"]["evidence_scope"],
            "single_gpu_local_shape_proxy",
        )
        self.assertEqual(params["workload"]["weight_ownership"], "shared_outer")
        self.assertEqual(params["phase"], "sweep")


class TestRank8KernelAdmission(CustomTestCase):
    """Compile and correctness-gate the rank-8 tiles changed by this review."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _case(self, *, shared_factor_signature="per_expert"):
        from benchmark.kernels.lora_moe.cases import (
            AdapterCell,
            Topology,
            build_case,
        )

        return build_case(
            device=str(self.device),
            model_preset="tiny_smoke",
            topology=Topology(tp_size=1, ep_size=1),
            adapter_cell=AdapterCell(
                active_adapters=2,
                include_base_rows=True,
                slot_capacity=4,
            ),
            route_generator="iid",
            num_tokens=4,
            active_rank=8,
            shared_factor_signature=shared_factor_signature,
            seed=17,
            source_revision="rank8-smoke",
        )

    def test_indexed_bk8_compiles_and_matches_the_trusted_reference(self):
        from benchmark.kernels.lora_moe import bench_lora_b as main_b

        fixture = main_b._BFixture(self._case(), self.device)
        config = next(
            candidate
            for candidate in main_b._sweep_grid("indexed", 8)
            if candidate
            == {
                "BLOCK_SIZE_N": 16,
                "BLOCK_SIZE_K": 8,
                "num_warps": 2,
                "num_stages": 2,
            }
        )
        for site in ("gate_up", "down"):
            main_b._admit(
                fixture,
                site,
                "indexed",
                config,
                None,
                f"indexed BK8 {site}",
            )

    def test_sgmv_and_csgmv_padded_tiles_compile_at_rank8(self):
        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb

        fixture = sdb._DownBFixture(
            self._case(shared_factor_signature="shared_down_b"),
            self.device,
        )
        reference = fixture.reference_delta("rank8 SGMV")
        for arm, grid in (
            ("sgmv_accum", sdb._sgmv_grid(8, fixture.case.moe_hidden_size)),
            (
                "csgmv_accum_padded",
                sdb._csgmv_grid(8, fixture.case.moe_hidden_size),
            ),
        ):
            configs = {}
            for config in grid:
                configs.setdefault(config["BLOCK_K"], config)
            self.assertEqual(set(configs), {16, 32})
            for block_k, config in configs.items():
                sdb._admit(
                    fixture,
                    arm,
                    {arm: config},
                    reference,
                    f"{arm} BK{block_k} rank8",
                )


class TestCanonicalPairs(CustomTestCase):
    """The producers' OWN pair lists, plus a real decide_cell adjudication."""

    def test_main_b_pairs_include_rank_split_against_the_promoted_kernel(self):
        # 8th review: this list used to live inside main() and was only
        # grepped for, so the exact prior regression could return green.
        from benchmark.kernels.lora_moe.bench_lora_b import DECIDED_PAIRS

        pairs = set(DECIDED_PAIRS)
        self.assertIn(("one_launch", "rank_split"), pairs)
        self.assertIn(("one_launch", "indexed"), pairs)
        self.assertIn(("one_launch", "lean_matched"), pairs)
        self.assertIn(("stock", "one_launch"), pairs)

    def test_shared_down_pairs_cover_every_arm_and_the_padded_showdown(self):
        from benchmark.kernels.lora_moe import bench_shared_down_b as sdb

        pairs = set(sdb.DECIDED_PAIRS)
        self.assertIn(("sgmv_accum_padded", "csgmv_accum_padded"), pairs)
        self.assertIn(("sgmv_accum", "sgmv_accum_padded"), pairs)
        for arm in sdb.ARMS[1:]:
            self.assertIn(("stock_charged", arm), pairs)
            if arm != "one_launch_charged":
                self.assertIn(("one_launch_charged", arm), pairs)

    def test_perexpert_pairs_include_the_promoted_baseline(self):
        from benchmark.kernels.lora_moe import bench_perexpert_sgmv_b as pe

        pairs = set(pe.DECIDED_PAIRS)
        for arm in pe.ARMS[2:]:
            self.assertIn(("one_launch", arm), pairs)
        self.assertIn(("sgmv_pe_padded", "csgmv_pe_padded"), pairs)

    def test_decide_cell_adjudicates_a_known_ordering(self):
        # Proves the adjudication rule the producers call actually resolves
        # a clear winner and a clear tie, rather than trusting the label.
        from benchmark.kernels.lora_moe.crossover_ledger import decide_cell

        faster = [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5]
        slower = [2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5]
        decision = decide_cell(arm_a="a", samples_a=slower, arm_b="b", samples_b=faster)
        self.assertEqual(decision.winner, "b")
        tie = decide_cell(
            arm_a="a", samples_a=faster, arm_b="b", samples_b=list(faster)
        )
        self.assertIsNone(tie.winner)


if __name__ == "__main__":
    unittest.main()
