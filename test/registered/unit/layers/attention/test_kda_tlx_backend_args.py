import unittest

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKDATlxBackendArgs(unittest.TestCase):
    def _resolve(self, **kwargs):
        args = ServerArgs(model_path="dummy", **kwargs)
        with (
            override_platform(is_sm100=False),
            override_platform(is_cuda=False),
        ):
            handle_linear_attn_backend(args)
        return args

    def test_replayssm_spec_accepts_tlx_decode(self):
        self._resolve(
            linear_attn_decode_backend="tlx",
            linear_attn_prefill_backend="tlx",
            enable_linear_replayssm_spec=True,
        )

    def test_replayssm_spec_still_rejects_other_decode_backends(self):
        with self.assertRaisesRegex(ValueError, "triton, flashinfer or tlx"):
            self._resolve(
                linear_attn_decode_backend="cutedsl",
                enable_linear_replayssm_spec=True,
            )

    def test_replayssm_decode_rejects_tlx(self):
        # Buffered ReplaySSM decode needs a decode kernel that writes the ring.
        with self.assertRaisesRegex(ValueError, "Triton, or Helion"):
            self._resolve(
                linear_attn_decode_backend="tlx",
                enable_linear_replayssm=True,
            )


if __name__ == "__main__":
    unittest.main()
