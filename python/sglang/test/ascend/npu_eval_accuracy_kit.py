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

# Nightly is also PR-triggered, so key on GITHUB_WORKFLOW_REF; empty locally = non-PR.
_workflow_ref = os.environ.get("GITHUB_WORKFLOW_REF", "")
_is_pr_pipeline = bool(_workflow_ref) and (
    "/.github/workflows/nightly-test-npu.yml" not in _workflow_ref
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


def honor_hf_endpoint_for_sgl_eval():
    """Route sgl-eval dataset archive downloads through HF_ENDPOINT.

    sgl-eval pins its archive URLs to https://huggingface.co and never consults
    HF_ENDPOINT, so the NPU runners (whose egress only allows the mirror) fail
    every prepare-backed benchmark with "ConnectError: [Errno 101] Network is
    unreachable". Rewrite the host at download time; sgl-eval still verifies the
    archive digest. No-op when HF_ENDPOINT is unset, and inert once sgl-eval
    honors it upstream.
    """
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if not endpoint or endpoint == "https://huggingface.co":
        return

    from sgl_eval.evals import _loader

    original = getattr(_loader, "_download_verified_archive", None)
    if original is None or getattr(original, "_sglang_hf_endpoint_shim", False):
        return

    def _download_verified_archive(url, expected_sha256, directory):
        if url.startswith("https://huggingface.co/"):
            url = endpoint + url[len("https://huggingface.co") :]
        return original(url, expected_sha256, directory)

    _download_verified_archive._sglang_hf_endpoint_shim = True
    _loader._download_verified_archive = _download_verified_archive


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
