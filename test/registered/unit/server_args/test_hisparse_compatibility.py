import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.hisparse_hook import validate_hisparse
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ClassificationBombSpec(CustomSpecAlgo):
    def is_none(self) -> bool:
        raise AssertionError("custom classification must not run")

    def is_eagle(self) -> bool:
        raise AssertionError("custom classification must not run")


def _make_early_args(raw_algorithm):
    args = object.__new__(ServerArgs)
    args.enable_hisparse = True
    args.speculative_algorithm = raw_algorithm
    return args


def _make_late_args(raw_algorithm):
    return SimpleNamespace(
        enable_hisparse=True,
        speculative_algorithm=raw_algorithm,
        disaggregation_decode_enable_offload_kvcache=True,
    )


class TestHiSparseEarlyCompatibility(unittest.TestCase):
    def test_explicit_none_spellings_canonicalize_without_registry_lookup(self) -> None:
        """Explicit NONE spellings canonicalize before speculative registry work."""
        for raw_algorithm in ("NONE", "none", "NoNe"):
            args = _make_early_args(raw_algorithm)
            with patch(
                "sglang.srt.server_args.is_npu",
                return_value=False,
            ), patch.object(
                SpeculativeAlgorithm,
                "from_string",
                side_effect=AssertionError("registry lookup must not run"),
            ):
                args._validate_hisparse_early_compatibility()
            self.assertIsNone(args.speculative_algorithm)

    def test_builtin_none_canonicalizes_by_identity(self) -> None:
        """Builtin NONE canonicalizes without invoking enum parsing."""
        args = _make_early_args(SpeculativeAlgorithm.NONE)
        with patch("sglang.srt.server_args.is_npu", return_value=False), patch.object(
            SpeculativeAlgorithm,
            "from_string",
            side_effect=AssertionError("enum parsing must not run"),
        ):
            args._validate_hisparse_early_compatibility()
        self.assertIsNone(args.speculative_algorithm)

    def test_non_none_algorithm_rejects_before_registry_or_model_work(self) -> None:
        """Non-NONE raw algorithms fail in the model-independent early gate."""
        args = _make_early_args("EAGLE")
        with patch("sglang.srt.server_args.is_npu", return_value=False), patch.object(
            SpeculativeAlgorithm,
            "from_string",
            side_effect=AssertionError("registry lookup must not run"),
        ):
            with self.assertRaisesRegex(ValueError, "speculative decoding"):
                args._validate_hisparse_early_compatibility()

    def test_npu_rejects_before_raw_algorithm_handling(self) -> None:
        """NPU rejects HiSparse before any speculative algorithm handling."""
        args = _make_early_args(None)
        with patch("sglang.srt.server_args.is_npu", return_value=True):
            with self.assertRaisesRegex(ValueError, "NPU"):
                args._validate_hisparse_early_compatibility()


class TestHiSparseLateCompatibility(unittest.TestCase):
    def test_custom_object_rejects_without_classification_methods(self) -> None:
        """Concrete custom algorithms reject without duck-typed classification."""
        custom_algorithm = _ClassificationBombSpec(
            "BOMB",
            lambda _server_args: object,
        )
        args = _make_late_args(custom_algorithm)
        with patch("sglang.srt.utils.is_npu", return_value=False), patch.object(
            SpeculativeAlgorithm,
            "from_string",
            side_effect=AssertionError("concrete objects must not be reparsed"),
        ):
            with self.assertRaisesRegex(ValueError, "speculative decoding"):
                validate_hisparse(args)

    def test_unsupported_raw_type_reports_type_and_value(self) -> None:
        """Unsupported raw values produce a clear defensive validation error."""
        args = _make_late_args(17)
        with patch("sglang.srt.utils.is_npu", return_value=False):
            with self.assertRaisesRegex(ValueError, "type=int, value=17"):
                validate_hisparse(args)

    def test_builtin_none_is_preserved_without_reparsing(self) -> None:
        """Concrete builtin NONE passes the spec gate without enum parsing."""
        args = _make_late_args(SpeculativeAlgorithm.NONE)
        with patch("sglang.srt.utils.is_npu", return_value=False), patch.object(
            SpeculativeAlgorithm,
            "from_string",
            side_effect=AssertionError("concrete enums must not be reparsed"),
        ):
            with self.assertRaisesRegex(ValueError, "decode-side KV offload"):
                validate_hisparse(args)


if __name__ == "__main__":
    unittest.main()
