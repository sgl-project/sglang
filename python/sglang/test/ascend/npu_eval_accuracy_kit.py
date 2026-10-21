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

import hashlib
import os
import shutil
from pathlib import Path

import requests

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin

_NPU_ACCURACY_TOLERANCE = 0.99
_NPU_MAX_ACCURACY_ATTEMPTS = 3

_is_pr_pipeline = os.environ.get("GITHUB_EVENT_NAME") == "pull_request"

# sgl_eval fetches benchmark archives (the MMLU data.tar among them) with its own
# httpx client: it ignores HF_ENDPOINT, prints nothing while downloading and
# never retries, so a blocked or slow mirror is indistinguishable from a hang.
# CI prefetches the archive through HF_ENDPOINT into this directory, and the shim
# below hands the local copy to sgl_eval after re-checking the sha256 it pins --
# a stale copy falls back to the original download instead of grading other data.
_SGL_EVAL_ARCHIVE_DIR_ENV = "SGL_EVAL_ARCHIVE_DIR"
_SGL_EVAL_ARCHIVE_DIR_DEFAULT = Path.home() / ".cache" / "sgl_eval" / "archives"


def _sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def honor_local_sgl_eval_archive():
    """Serve sgl_eval archives from a prefetched directory when possible."""
    archive_dir = Path(
        os.environ.get(_SGL_EVAL_ARCHIVE_DIR_ENV) or _SGL_EVAL_ARCHIVE_DIR_DEFAULT
    )
    if not archive_dir.is_dir():
        return
    try:
        from sgl_eval.evals import _loader
    except ImportError:
        return
    original = getattr(_loader, "_download_verified_archive", None)
    if original is None or getattr(original, "_sglang_local_archive_shim", False):
        return

    def _download_verified_archive(url, expected_sha256, directory):
        local = archive_dir / url.rsplit("/", 1)[-1]
        expected = (expected_sha256 or "").lower()
        if expected and local.is_file() and _sha256_of(local) == expected:
            # sgl_eval unlinks the path it gets back, so hand it a private copy.
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            staged = directory / local.name
            shutil.copyfile(local, staged)
            return staged
        return original(url, expected_sha256, directory)

    _download_verified_archive._sglang_local_archive_shim = True
    _loader._download_verified_archive = _download_verified_archive


# Applied on import so every NPU accuracy case benefits; a no-op without the dir.
honor_local_sgl_eval_archive()


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
