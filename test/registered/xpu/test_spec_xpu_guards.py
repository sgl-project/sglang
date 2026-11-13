"""EAGLE-family speculative-decoding guards that only Intel XPU hardware can prove.

Two regressions live here:

1. Draft initialization. For a NEXTN/MTP draft the worker shares the target
   embedding and head through ``set_embed_and_head`` / ``set_embed``, whose bodies
   used to end in ``torch.cuda.synchronize()``. On a torch+xpu build that raises
   "Torch not compiled with CUDA enabled", so the scheduler died before the server
   was ready. ``TestSetEmbedRunsOnXpu`` calls those real methods with XPU tensors,
   which executes the statement that crashed; the companion source scan in
   ``test/registered/unit/spec/test_set_embed_device_neutral.py`` only reads text on
   CPU and so cannot catch it.
2. Verify sampling. ``--speculative-use-rejection-sampling`` has no effect on XPU
   because verify commits the target argmax, so the launcher must refuse it rather
   than accept it silently.

The greedy-fallback warning needs a target that can actually run verify on XPU, so
it is asserted against a live EAGLE server in
``test/registered/e2e/xpu/test_spec_eagle_intel_xpu.py`` instead of here.

Usage:
python3 -m unittest test_spec_xpu_guards.TestSetEmbedRunsOnXpu
"""

import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")

# Tiny random checkpoint; only its config is read, no weights are loaded.
_ARG_CHECK_MODEL = "yujiepan/glm-moe-dsa-tiny-random"


def _embed_head_stub(device):
    """The attribute shape ``set_embed_and_head`` / ``set_embed`` mutate.

    Both bodies ``del`` and reassign ``self.model.embed_tokens.weight`` and
    ``self.lm_head.weight``, so a stub carrying those is enough to run the real
    method without building a model.
    """
    return SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.zeros(1, device=device))
        ),
        lm_head=SimpleNamespace(weight=torch.zeros(1, device=device)),
        config=SimpleNamespace(hidden_size=8, target_hidden_size=8),
    )


class TestSetEmbedRunsOnXpu(CustomTestCase):
    """Sharing the target embedding must not raise on XPU.

    Each case calls the production method for one NEXTN/MTP draft family. A
    ``torch.cuda.*`` call in any of these bodies raises on torch+xpu, which is
    what killed NEXTN draft init.
    """

    def _draft_classes(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
        from sglang.srt.models.glm4_moe import Glm4MoeForCausalLM
        from sglang.srt.models.mimo_v2_nextn import MiMoV2MTP
        from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM

        # One per distinct set_embed_and_head body a NextN draft resolves to.
        return {
            "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
            "Glm4MoeForCausalLM": Glm4MoeForCausalLM,
            "Qwen3NextForCausalLM": Qwen3NextForCausalLM,
            "MiMoV2MTP": MiMoV2MTP,
        }

    def test_set_embed_and_head_on_xpu(self):
        device = "xpu"
        embed = torch.zeros(4, 8, device=device)
        head = torch.zeros(4, 8, device=device)
        for name, cls in self._draft_classes().items():
            with self.subTest(model=name):
                stub = _embed_head_stub(device)
                cls.set_embed_and_head(stub, embed, head)
                self.assertIs(stub.model.embed_tokens.weight, embed)
                self.assertIs(stub.lm_head.weight, head)

    def test_set_embed_on_xpu(self):
        """EAGLE3 drafts keep their own head and only take the embedding."""
        device = "xpu"
        embed = torch.zeros(4, 8, device=device)
        for name, cls in self._draft_classes().items():
            set_embed = getattr(cls, "set_embed", None)
            if set_embed is None:
                continue
            with self.subTest(model=name):
                stub = _embed_head_stub(device)
                set_embed(stub, embed)
                self.assertIs(stub.model.embed_tokens.weight, embed)


class TestRejectionSamplingRefusedOnXpu(CustomTestCase):
    """XPU must refuse --speculative-use-rejection-sampling, not ignore it.

    The flag hits a device check while arguments resolve, so the launcher exits
    before any weights load.
    """

    def test_launch_is_refused(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                _ARG_CHECK_MODEL,
                "--device",
                "xpu",
                "--trust-remote-code",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--speculative-use-rejection-sampling",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertNotEqual(proc.returncode, 0, "the launch should have been refused")
        self.assertIn("not supported on XPU", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
