"""Guard the MPS early-return in ``ModelRunner.init_attention_backends``.

``GDNAttnBackend.__init__`` dereferences
``model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]``
on hybrid GDN models, but the MLX stub leaves ``mamba_cache = None``
(MLX stores mamba state in its own runner). The MPS branch must short-
circuit before any CUDA-style backend constructor runs, so the
``None.conv`` crash never fires.

The check is a behaviour guard: we patch the constructors that would
crash and assert they are never called on the MPS path.
"""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx (mlx_lm import path on this branch)"


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxAttentionBackendSkip(unittest.TestCase):
    """``init_attention_backends`` must skip on MPS."""

    def test_mps_runner_does_not_call_init_attention_backend(self):
        """Stand up a fake ModelRunner with device='mps' and verify the
        early-return path skips init_attention_backend entirely."""
        from sglang.srt.model_executor.model_runner import ModelRunner

        # Build a ModelRunner-shaped object without running __init__.
        # We only need the methods the test exercises.
        runner = ModelRunner.__new__(ModelRunner)
        runner.device = "mps"

        init_attention_backend = MagicMock()
        init_aux_hidden_state_capture = MagicMock()
        runner.init_attention_backend = init_attention_backend
        runner.init_aux_hidden_state_capture = init_aux_hidden_state_capture

        # Should return early before any of the platform branches.
        ModelRunner.init_attention_backends(runner)

        init_aux_hidden_state_capture.assert_called_once_with()
        init_attention_backend.assert_not_called()

    def test_mps_runner_does_not_instantiate_gdn_backend(self):
        """If init_attention_backends accidentally reaches the CUDA path
        on MPS, GDNAttnBackend.__init__ raises AttributeError. We patch
        the constructor to record the call and assert it never fires."""
        from sglang.srt.model_executor import model_runner as mr_mod

        runner = mr_mod.ModelRunner.__new__(mr_mod.ModelRunner)
        runner.device = "mps"
        runner.init_attention_backend = MagicMock()
        runner.init_aux_hidden_state_capture = MagicMock()

        with patch(
            "sglang.srt.layers.attention.attention_registry.attn_backend_wrapper"
        ) as wrapper:
            mr_mod.ModelRunner.init_attention_backends(runner)
            wrapper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
