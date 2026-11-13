"""intel_xpu attention backend (EAGLE3 topk=1 chain + EAGLE/Llama-2 spec)."""

import tempfile
import unittest
from pathlib import Path

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecHiddenStatesKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_xpu_ci(est_time=1800, suite="nightly-xpu-kernel-main-1-gpu", nightly=True)


class TestEagle3IntelXPU(
    Eagle3Base,
    MatchedStopMixin,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE3 spec v2 on the intel_xpu attention backend (kits listed in bases)."""

    attention_backend = "intel_xpu"
    max_running_requests = 24
    gsm8k_num_examples = 300
    gsm8k_check_accept_len = True
    mem_fraction_static = 0.95
    extra_args = ("--max-total-tokens", "16384", "--disable-decode-cuda-graph")
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2IntelXPU(
    EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit, SpecHiddenStatesKit
):
    """EAGLE/Llama-2 on intel_xpu using the supported topk = 1 paged config."""

    attention_backend = "intel_xpu"
    spec_topk = 1
    page_size = 64
    gsm8k_check_accept_len = True
    gsm8k_num_examples = 300
    enable_return_hidden_states = True
    mem_fraction_static = 0.95
    max_running_requests = 6
    chunked_prefill_size = 512
    extra_args = ("--max-total-tokens", "16384", "--disable-decode-cuda-graph")


_GREEDY_FALLBACK_WARNING = "Speculative verify falls back to the target argmax"

_SERVER_LOG = Path(tempfile.gettempdir()) / "xpu_eagle_greedy_fallback.log"


class TestGreedyFallbackSurfacedIntelXPU(Eagle3Base):
    """A sampled request must surface the greedy-verify fallback once.

    XPU has no speculative sampling-verify kernel, so verify commits the target
    argmax instead of honoring `temperature` / `top_p` / `top_k`. That warning is
    the only signal an operator gets, and it used to be absent.

    The fallback covers verification only, so this does not assert that a sampled
    request reproduces the greedy continuation: the first token still comes from
    the ordinary prefill sampler, which does honor the sampling parameters.
    """

    attention_backend = "intel_xpu"
    max_running_requests = 4
    mem_fraction_static = 0.95
    chunked_prefill_size = 512
    extra_args = ("--max-total-tokens", "8192", "--disable-decode-cuda-graph")
    server_log_path = str(_SERVER_LOG)

    def _generate(self, **sampling_params):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "[INST] Name the capital of France. [/INST]",
                "sampling_params": {"max_new_tokens": 16, **sampling_params},
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["text"]

    def test_sampled_request_warns_once(self):
        # A greedy request must stay silent: the fallback changes nothing for it.
        self.assertTrue(self._generate(temperature=0))
        self.assertNotIn(_GREEDY_FALLBACK_WARNING, _SERVER_LOG.read_text())

        self.assertTrue(self._generate(temperature=0.9, top_p=0.8, top_k=40))
        log = _SERVER_LOG.read_text()
        self.assertIn(_GREEDY_FALLBACK_WARNING, log)
        self.assertEqual(
            log.count(_GREEDY_FALLBACK_WARNING), 1, "the warning must be logged once"
        )

        # A second sampled request must not repeat it.
        self.assertTrue(self._generate(temperature=0.9, top_p=0.8, top_k=40))
        self.assertEqual(
            _SERVER_LOG.read_text().count(_GREEDY_FALLBACK_WARNING),
            1,
            "the warning must be logged once",
        )


if __name__ == "__main__":
    unittest.main()
