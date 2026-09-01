"""PPProxyTensors tolerates a pipeline stage that sends no residual — CPU-only.

Regression guard for sglang#36906. An mHC model (``hc_hidden_size is not None``,
e.g. GLM-5.3-Flash / DeepSeek-V4) folds the residual into the widened hidden
state, so a non-first pipeline stage receives a proxy that carries only
``hidden_states``. ``Glm5NextModel.forward`` previously read
``pp_proxy_tensors["residual"]`` unconditionally and died with
``KeyError: residual`` on every non-first stage; it now reads
``pp_proxy_tensors.tensors.get("residual")`` and re-inserts the key on the send
side only when a residual actually exists (a ``None`` cannot travel over PP
IPC).

These tests pin the container contract that fix relies on. They need no GPU and
no model weights.
"""

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _forward_pp_residual(pp_proxy_tensors):
    """The residual read/re-send pattern of ``Glm5NextModel.forward``, isolated
    from the (GPU-only) model body.

    Returns the residual read from the incoming proxy (``None`` when the stage
    sent none) and the proxy this stage would forward on.
    """
    hidden_states = pp_proxy_tensors["hidden_states"]
    residual = pp_proxy_tensors.tensors.get("residual")

    proxy = {"hidden_states": hidden_states}
    if residual is not None:
        proxy["residual"] = residual
    return residual, PPProxyTensors(proxy)


class TestPPProxyMissingResidual(CustomTestCase):
    def test_absent_residual_reads_as_none(self):
        """The read path the fix uses: no residual key -> None, not KeyError."""
        pp = PPProxyTensors({"hidden_states": torch.zeros(3, 2)})
        self.assertIsNone(pp.tensors.get("residual"))

    def test_indexing_absent_residual_raises(self):
        """Documents the pre-fix crash: __getitem__ still raises KeyError, so
        the fix had to stop indexing the key unconditionally."""
        pp = PPProxyTensors({"hidden_states": torch.zeros(3, 2)})
        with self.assertRaises(KeyError):
            pp["residual"]

    def test_absent_residual_forward_omits_key(self):
        """An mHC-style stage (hidden_states only) does not KeyError and does
        not put a None residual back on the wire."""
        hidden = torch.arange(6.0).reshape(3, 2)
        residual, out = _forward_pp_residual(PPProxyTensors({"hidden_states": hidden}))
        self.assertIsNone(residual)
        self.assertNotIn("residual", out.tensors)
        self.assertIs(out.tensors["hidden_states"], hidden)

    def test_present_residual_passes_through_unchanged(self):
        """A non-mHC stage still forwards the residual it received, by identity,
        so every pp_size > 1, non-mHC run is unaffected."""
        hidden = torch.arange(6.0).reshape(3, 2)
        res = torch.ones(3, 2)
        residual, out = _forward_pp_residual(
            PPProxyTensors({"hidden_states": hidden, "residual": res})
        )
        self.assertIs(residual, res)
        self.assertIs(out.tensors["residual"], res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
