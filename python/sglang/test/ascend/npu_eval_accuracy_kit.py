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

import math
import os

import requests

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin

_NPU_ACCURACY_TOLERANCE = 0.99
_NPU_MAX_ACCURACY_ATTEMPTS = 3

# Nightly is also PR-triggered, so exclude it by workflow ref; every other caller
# (full test, version bump, release cut) still gets the full run.
_is_pr_pipeline = os.environ.get("GITHUB_EVENT_NAME") == "pull_request" and (
    "nightly-test-npu.yml" not in os.environ.get("GITHUB_WORKFLOW_REF", "")
)


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
    """NPU GSM8K accuracy mixin — 1% tolerance + up to 3 retries.

    ``gsm8k_backend`` selects the harness:

    * ``"run_eval"`` (default) / ``"sgl_eval"``: delegated to the parent.
    * ``"evalscope"``: the NPU historical baseline harness (4-shot chat,
      temperature 1.0). Use it to keep thresholds calibrated against the
      legacy evalscope numbers (e.g. Moonlight bf16 tp2 = 0.837).
    """

    # evalscope request concurrency; 64 matches the harness the 0.83
    # Moonlight baseline was calibrated with.
    gsm8k_eval_batch_size = 64

    def _test_gsm8k_evalscope(self, threshold):
        from urllib.parse import urlparse

        from sglang.test.ascend.e2e.test_npu_accuracy_utils import run_evalscope

        parsed = urlparse(self.base_url)
        # Match the historical evalscope baseline sampling: temperature 1.0
        # with a generous token budget (gsm8k_max_tokens defaults to None).
        generation_config = {
            "temperature": (
                self.gsm8k_temperature if self.gsm8k_temperature is not None else 1.0
            ),
            "max_tokens": self.gsm8k_max_tokens if self.gsm8k_max_tokens else 7168,
        }
        limit = (
            self.gsm8k_num_examples if self.gsm8k_num_examples is not None else 100000
        )
        best = 0.0
        for attempt in range(_NPU_MAX_ACCURACY_ATTEMPTS):
            metrics = run_evalscope(
                host=parsed.hostname,
                port=parsed.port,
                model=os.path.basename(self.model),
                datasets=["gsm8k"],
                limit=limit,
                eval_batch_size=self.gsm8k_eval_batch_size,
                generation_config=generation_config,
            )
            best = max(best, float(metrics.get("accuracy", 0.0)))
            if best >= threshold:
                print(f"gsm8k evalscope score={best:.4f} >= {threshold:.4f}")
                return
            if attempt < _NPU_MAX_ACCURACY_ATTEMPTS - 1:
                print(
                    f"[{type(self).__name__}] evalscope GSM8K attempt "
                    f"{attempt + 1}/{_NPU_MAX_ACCURACY_ATTEMPTS} failed "
                    f"(best={best:.4f} < {threshold:.4f}), retrying..."
                )
        raise AssertionError(
            f"GSM8K evalscope accuracy {best} below threshold {threshold} "
            f"after {_NPU_MAX_ACCURACY_ATTEMPTS} attempts"
        )

    def _resolve_relaxed_threshold(self) -> float:
        """Return the GSM8K threshold with the NPU 1% tolerance applied.

        ``gsm8k_score_threshold`` wins; the legacy ``gsm8k_accuracy_thres``
        alias is used when it is unset (NaN). The relaxed value is cached, so
        re-entering the test cannot compound the tolerance, and it is written
        back to both attributes because the parent mixin reads it from there.
        """
        cached = getattr(self, "_npu_relaxed_threshold", None)
        if cached is not None:
            return cached

        threshold = self.gsm8k_score_threshold
        if math.isnan(threshold):  # canonical knob unset -> legacy alias
            threshold = self.gsm8k_accuracy_thres
        relaxed = threshold * _NPU_ACCURACY_TOLERANCE

        self.gsm8k_score_threshold = relaxed
        self.gsm8k_accuracy_thres = relaxed
        self._npu_relaxed_threshold = relaxed
        return relaxed

    def test_gsm8k(self):
        if _is_pr_pipeline:
            run_npu_pr_smoke(self.base_url)
            return

        relaxed = self._resolve_relaxed_threshold()

        if self.gsm8k_backend == "evalscope":
            self._test_gsm8k_evalscope(relaxed)
            return

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
