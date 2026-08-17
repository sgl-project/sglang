import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.server_fixtures.standalone_fixture import StandaloneServerBase
from sglang.test.test_utils import CustomTestCase, is_in_ci

# V2 standalone speculative decoding. CI runs only fa3 (the backend this is
# deployed on); triton / flashinfer stay runnable locally, and their spec verify
# numerics live in attention/unittests/dense/test_{triton,flashinfer}.py.
# Non-V2 backends moved to test_spec_standalone_extra.py.
register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-large")


class TestStandaloneV2SpeculativeDecodingBase(
    StandaloneServerBase, CustomTestCase, RegexConstrainedMixin, JSONConstrainedMixin
):
    # Hosts the constrained mixins: overlap is on, so they exercise the
    # grammar barrier path.
    attention_backend = "fa3"


@unittest.skipIf(is_in_ci(), "CI covers fa3 only; run locally for triton.")
class TestStandaloneV2SpeculativeDecodingTriton(StandaloneServerBase, CustomTestCase):
    attention_backend = "triton"


@unittest.skipIf(is_in_ci(), "CI covers fa3 only; run locally for flashinfer.")
class TestStandaloneV2SpeculativeDecodingFlashinfer(
    StandaloneServerBase, CustomTestCase
):
    attention_backend = "flashinfer"


if __name__ == "__main__":
    unittest.main()
