"""CPU unit tests for triton DCP extend dispatch.

Non-zero-prefix extend under DCP is forced to MHA-mode chunked prefill
(prefix KV is sharded across dcp ranks and only the MHA chunked path gathers
it at the model level); zero-prefix extend keeps plain MHA and decode keeps
the absorbed-MLA subtype, both unchanged from non-DCP.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.models.deepseek_common import attention_backend_handler as abh
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HANDLER_MODULE = "sglang.srt.models.deepseek_common.attention_backend_handler"


def _forward_batch(mode: str, prefix_lens):
    fb = MagicMock()
    fb.forward_mode.is_extend_without_speculative.return_value = mode == "extend"
    fb.forward_mode.is_decode.return_value = mode == "decode"
    fb.extend_prefix_lens_cpu = prefix_lens
    return fb


class TestTritonDcpExtendDispatch(CustomTestCase):
    def _dispatch(self, dcp: bool, mode: str, prefix_lens):
        server_args = MagicMock()
        server_args.enable_deterministic_inference = False
        with (
            patch(f"{HANDLER_MODULE}.is_in_tc_piecewise_cuda_graph", lambda: False),
            patch(f"{HANDLER_MODULE}.dcp_enabled", lambda: dcp),
            patch(f"{HANDLER_MODULE}.get_global_server_args", lambda: server_args),
            # MagicMock attn has every attribute, which would otherwise
            # satisfy the intel-amx fused-rope probe in _dispatch_mla_subtype.
            patch(f"{HANDLER_MODULE}.use_intel_amx_backend", lambda attn: False),
        ):
            return abh.handle_attention_triton(
                MagicMock(), _forward_batch(mode, prefix_lens)
            )

    def test_dcp_nonzero_prefix_extend_forces_mha_chunked(self):
        # The new behavior: dcp + non-zero prefix routes to the chunked path.
        got = self._dispatch(dcp=True, mode="extend", prefix_lens=[0, 7, 3])
        self.assertEqual(got, AttnForwardMethod.MHA_CHUNKED_KV)

    def test_no_dcp_nonzero_prefix_extend_keeps_mla_subtype(self):
        # Regression guard: without dcp the pre-existing extend path is intact.
        got = self._dispatch(dcp=False, mode="extend", prefix_lens=[0, 7, 3])
        self.assertEqual(got, AttnForwardMethod.MLA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
