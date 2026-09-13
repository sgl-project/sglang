"""NPU accuracy test mixin.

Wraps :class:`~sglang.test.kits.eval_accuracy_kit.GSM8KMixin` with:

* 1% accuracy tolerance (threshold * 0.99)
* Automatic retry (up to 3 attempts)

Delegates to ``super().test_gsm8k()`` so community-side changes to eval
setup, backend selection, or parameters are inherited automatically.

Usage::

    from sglang.test.ascend.npu_eval_accuracy_kit import NPUGSM8KMixin

    class TestFoo(CustomTestCase, NPUGSM8KMixin):
        gsm8k_accuracy_thres = 0.6
"""

import os

import requests

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin

_NPU_ACCURACY_TOLERANCE = 0.99
_NPU_MAX_ACCURACY_ATTEMPTS = 3

_is_pr_pipeline = os.environ.get("GITHUB_EVENT_NAME") == "pull_request"


def run_npu_pr_smoke(base_url):
    """Send a single inference request for PR pipeline smoke test."""
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        },
    )
    assert response.status_code == 200
    assert "Paris" in response.text


class NPUGSM8KMixin(GSM8KMixin):
    """NPU GSM8K accuracy mixin — 1% tolerance + up to 3 retries."""

    def test_gsm8k(self):
        if _is_pr_pipeline:
            run_npu_pr_smoke(self.base_url)
            return
        threshold = self.gsm8k_score_threshold
        if threshold != threshold:  # NaN → legacy alias
            threshold = self.gsm8k_accuracy_thres
        relaxed = threshold * _NPU_ACCURACY_TOLERANCE

        # Patch both so the parent picks up the relaxed threshold.
        self.gsm8k_score_threshold = relaxed
        self.gsm8k_accuracy_thres = relaxed

        for attempt in range(_NPU_MAX_ACCURACY_ATTEMPTS):
            try:
                super().test_gsm8k()
                return
            except AssertionError:
                if attempt == _NPU_MAX_ACCURACY_ATTEMPTS - 1:
                    raise
                print(
                    f"[{type(self).__name__}] GSM8K attempt "
                    f"{attempt + 1}/{_NPU_MAX_ACCURACY_ATTEMPTS} failed, "
                    f"retrying..."
                )
