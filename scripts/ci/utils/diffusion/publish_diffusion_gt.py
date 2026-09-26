"""
Publish diffusion CI ground-truth images to sgl-project/ci-data-diffusion
via the GitHub API (same pattern as publish_traces.py).

The target branch is chosen by ci_data_branch.py; see its docstring.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageFilter

# Reuse GitHub API helpers from publish_traces.
# Support both direct script execution and package-style imports.
if __package__:
    from ..publish_traces import (
        create_blobs,
        create_commit,
        create_tree,
        get_tree_sha,
        is_permission_error,
        is_rate_limit_error,
        make_github_request,
        update_branch_ref,
        verify_token_permissions,
    )
    from .ci_data_branch import MAIN_BRANCH, resolve_publish_branch
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ci_data_branch import MAIN_BRANCH, resolve_publish_branch
    from publish_traces import (
        create_blobs,
        create_commit,
        create_tree,
        get_tree_sha,
        is_permission_error,
        is_rate_limit_error,
        make_github_request,
        update_branch_ref,
        verify_token_permissions,
    )

REPO_OWNER = "sgl-project"
REPO_NAME = "ci-data-diffusion"
DEFAULT_TARGET_DIR = "diffusion-ci/consistency_gt/sglang_generated"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
QUALITY_MAX_SIDE = 256
LOW_DETAIL_STD_THRESHOLD = 0.075
LOW_DETAIL_ENTROPY_THRESHOLD = 0.55
LOW_DETAIL_BLUR_RESIDUAL_THRESHOLD = 0.035
LOW_DETAIL_GRADIENT_P95_THRESHOLD = 0.045
RANDOM_NOISE_CORRELATION_THRESHOLD = 0.55
RANDOM_NOISE_LOW_FREQUENCY_THRESHOLD = 0.20
RANDOM_NOISE_BLUR_RESIDUAL_THRESHOLD = 0.045
OLD_NEW_MIN_SSIM = 0.20
OLD_NEW_MAX_MEAN_ABS_DIFF = 45.0


@dataclass(frozen=True)
class ImageQualityMetrics:
    luminance_std: float
    entropy: float
    blur_residual: float
    gradient_p95: float
    neighbor_correlation: float
    low_frequency_ratio: float


@dataclass(frozen=True)
class OldNewMetrics:
    ssim: float
    mean_abs_diff: float


def collect_images(source_dir, target_dir):
    """Collect image files from source_dir and return list of (repo_path, content) tuples."""
    files = []
    for entry in sorted(os.listdir(source_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        full_path = os.path.join(source_dir, entry)
        if not os.path.isfile(full_path):
            continue
        with open(full_path, "rb") as f:
            content = f.read()
        repo_path = f"{target_dir}/{entry}"
        files.append((repo_path, content))
    return files


def git_blob_sha(content):
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def get_branch_head(repo_owner, repo_name, branch, token):
    """Head SHA of ``branch``, or None if it does not exist.

    Uses the singular ``git/ref`` endpoint: the plural ``git/refs/heads/<name>``
    prefix-matches, returning a list for ``gt/foo`` when only ``gt/foo-rc1`` exists.
    """
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/ref/heads/"
        f"{quote(branch, safe='/')}"
    )
    try:
        response = make_github_request(url, token)
    except HTTPError as e:
        if e.code == 404:
            return None
        raise
    return json.loads(response)["object"]["sha"]


def get_base_sha(repo_owner, repo_name, branch, token):
    """The commit to compare against and build on: ``branch``, else ``main`` before it exists."""
    head = get_branch_head(repo_owner, repo_name, branch, token)
    if head is not None:
        return head, True
    base = get_branch_head(repo_owner, repo_name, MAIN_BRANCH, token)
    if base is None:
        raise RuntimeError(f"{repo_owner}/{repo_name} has no {MAIN_BRANCH} branch")
    return base, False


class BranchCreateRace(Exception):
    """A parallel job created the branch between our lookup and our create."""


def create_branch(repo_owner, repo_name, branch, commit_sha, token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs"
    try:
        make_github_request(
            url,
            token,
            method="POST",
            data={"ref": f"refs/heads/{branch}", "sha": commit_sha},
        )
    except HTTPError as e:
        if e.code == 422 and "already exists" in e.error_body:
            # Only a race if the branch now exists; otherwise the name collides
            # with a nested ref (gt/foo vs gt/foo/bar), and retrying cannot help.
            if get_branch_head(repo_owner, repo_name, branch, token) is not None:
                raise BranchCreateRace(branch) from e
            raise RuntimeError(
                f"cannot create branch {branch!r}: {e.error_body}. It is a path "
                "prefix of an existing branch, or has one as a prefix; pick another."
            ) from e
        if e.code == 422:
            # Also returned transiently for a just-created commit, so the publish
            # retry loop retries it.
            print(
                f"Creating branch {branch!r} returned 422; retrying. If this persists, "
                "the name may collide with an existing nested branch (gt/foo vs gt/foo/bar)."
            )
        raise
    print(f"Created {repo_owner}/{repo_name} branch {branch}")


def get_remote_image_entries(repo_owner, repo_name, target_dir, token, ref):
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
        f"{target_dir}?ref={quote(ref, safe='')}"
    )
    try:
        response = make_github_request(url, token)
    except HTTPError as e:
        if e.code == 404:
            return {}
        raise
    entries = json.loads(response)
    return {
        item["path"]: item
        for item in entries
        if item.get("type") == "file"
        and "sha" in item
        and os.path.splitext(item["path"])[1].lower() in IMAGE_EXTENSIONS
    }


def filter_changed_files(files, remote_blob_shas):
    return [
        (path, content)
        for path, content in files
        if remote_blob_shas.get(path) != git_blob_sha(content)
    ]


def get_remote_blob_content(repo_owner, repo_name, blob_sha, token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/blobs/{blob_sha}"
    response = make_github_request(url, token)
    blob = json.loads(response)
    if blob.get("encoding") != "base64":
        raise ValueError(
            f"Unexpected blob encoding for {blob_sha}: {blob.get('encoding')}"
        )
    return base64.b64decode(blob["content"])


def _load_quality_image(content):
    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGB")
        image.thumbnail((QUALITY_MAX_SIDE, QUALITY_MAX_SIDE), Image.Resampling.BICUBIC)
        return image.copy()


def _image_to_rgb_array(image):
    return np.asarray(image, dtype=np.float32)


def _luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _neighbor_correlation(luma):
    def corr(a, b):
        a = a.ravel()
        b = b.ravel()
        if a.std() < 1e-6 or b.std() < 1e-6:
            return 1.0
        return float(np.corrcoef(a, b)[0, 1])

    return (corr(luma[:, 1:], luma[:, :-1]) + corr(luma[1:, :], luma[:-1, :])) / 2


def _low_frequency_ratio(luma):
    centered = luma - luma.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    total_power = power.sum()
    if total_power < 1e-12:
        return 0.0

    height, width = luma.shape
    y, x = np.ogrid[:height, :width]
    center_y = height // 2
    center_x = width // 2
    radius = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    low_frequency_radius = min(height, width) * 0.08
    return float(power[radius <= low_frequency_radius].sum() / total_power)


def compute_image_quality_metrics(content):
    image = _load_quality_image(content)
    rgb = _image_to_rgb_array(image)
    luma = _luminance(rgb) / 255.0

    gradients = np.concatenate(
        [
            np.abs(np.diff(luma, axis=1)).ravel(),
            np.abs(np.diff(luma, axis=0)).ravel(),
        ]
    )
    histogram, _ = np.histogram(luma, bins=32, range=(0, 1))
    probabilities = histogram / histogram.sum()
    nonzero_probabilities = probabilities[probabilities > 0]
    entropy = float(
        -(nonzero_probabilities * np.log2(nonzero_probabilities)).sum() / 5.0
    )
    blurred = _image_to_rgb_array(image.filter(ImageFilter.GaussianBlur(radius=3)))

    return ImageQualityMetrics(
        luminance_std=float(luma.std()),
        entropy=entropy,
        blur_residual=float(np.mean(np.abs(rgb - blurred)) / 255.0),
        gradient_p95=float(np.percentile(gradients, 95)),
        neighbor_correlation=_neighbor_correlation(luma),
        low_frequency_ratio=_low_frequency_ratio(luma),
    )


def get_quality_failure_reasons(metrics):
    reasons = []
    low_detail_static = (
        metrics.luminance_std < LOW_DETAIL_STD_THRESHOLD
        and metrics.entropy < LOW_DETAIL_ENTROPY_THRESHOLD
        and (
            metrics.blur_residual < LOW_DETAIL_BLUR_RESIDUAL_THRESHOLD
            or metrics.gradient_p95 < LOW_DETAIL_GRADIENT_P95_THRESHOLD
        )
    )
    high_frequency_noise = (
        metrics.neighbor_correlation < RANDOM_NOISE_CORRELATION_THRESHOLD
        and metrics.low_frequency_ratio < RANDOM_NOISE_LOW_FREQUENCY_THRESHOLD
        and metrics.blur_residual > RANDOM_NOISE_BLUR_RESIDUAL_THRESHOLD
    )
    if low_detail_static:
        reasons.append("low-contrast low-detail output")
    if high_frequency_noise:
        reasons.append("high-frequency random noise")
    return reasons


def _resize_for_old_new_compare(content, size=None):
    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGB")
        if size is None:
            image.thumbnail(
                (QUALITY_MAX_SIDE, QUALITY_MAX_SIDE), Image.Resampling.BICUBIC
            )
        else:
            image = image.resize(size, Image.Resampling.BICUBIC)
        return _image_to_rgb_array(image)


def compute_old_new_metrics(old_content, new_content):
    old_rgb = _resize_for_old_new_compare(old_content)
    new_rgb = _resize_for_old_new_compare(
        new_content, size=(old_rgb.shape[1], old_rgb.shape[0])
    )
    old_luma = _luminance(old_rgb) / 255.0
    new_luma = _luminance(new_rgb) / 255.0

    old_mean = old_luma.mean()
    new_mean = new_luma.mean()
    old_variance = old_luma.var()
    new_variance = new_luma.var()
    covariance = ((old_luma - old_mean) * (new_luma - new_mean)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (
        (2 * old_mean * new_mean + c1)
        * (2 * covariance + c2)
        / ((old_mean**2 + new_mean**2 + c1) * (old_variance + new_variance + c2))
    )

    return OldNewMetrics(
        ssim=float(ssim),
        mean_abs_diff=float(np.abs(old_rgb - new_rgb).mean()),
    )


def _format_quality_metrics(metrics):
    return (
        f"std={metrics.luminance_std:.4f}, entropy={metrics.entropy:.4f}, "
        f"blur_residual={metrics.blur_residual:.4f}, "
        f"gradient_p95={metrics.gradient_p95:.4f}, "
        f"neighbor_corr={metrics.neighbor_correlation:.4f}, "
        f"low_freq={metrics.low_frequency_ratio:.4f}"
    )


def _format_old_new_metrics(metrics):
    return f"ssim={metrics.ssim:.4f}, mean_abs_diff={metrics.mean_abs_diff:.2f}"


def validate_gt_files(files_to_upload, changed_files, remote_image_entries, token):
    failures = []
    for path, content in files_to_upload:
        quality_metrics = compute_image_quality_metrics(content)
        quality_reasons = get_quality_failure_reasons(quality_metrics)
        if quality_reasons:
            failures.append(
                f"{path}: {', '.join(quality_reasons)} "
                f"({_format_quality_metrics(quality_metrics)})"
            )

    for path, content in changed_files:
        remote_entry = remote_image_entries.get(path)
        if not remote_entry:
            continue

        old_content = get_remote_blob_content(
            REPO_OWNER, REPO_NAME, remote_entry["sha"], token
        )
        old_quality_metrics = compute_image_quality_metrics(old_content)
        old_quality_reasons = get_quality_failure_reasons(old_quality_metrics)
        if old_quality_reasons:
            print(
                f"Skipping old/new drift check for {path} because existing GT is "
                f"already suspicious: {', '.join(old_quality_reasons)} "
                f"({_format_quality_metrics(old_quality_metrics)})"
            )
            continue

        old_new_metrics = compute_old_new_metrics(old_content, content)
        if (
            old_new_metrics.ssim < OLD_NEW_MIN_SSIM
            and old_new_metrics.mean_abs_diff > OLD_NEW_MAX_MEAN_ABS_DIFF
        ):
            failures.append(
                f"{path}: changed too far from existing GT "
                f"({_format_old_new_metrics(old_new_metrics)})"
            )

    if not failures:
        print(
            f"GT quality gate passed for {len(files_to_upload)} generated image(s) "
            f"and {len(changed_files)} changed image(s)."
        )
        return

    print("GT quality gate failed; refusing to publish suspicious image updates:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)


def check_quality(source_dir, target_dir, branch):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    files_to_upload = collect_images(source_dir, target_dir)
    if not files_to_upload:
        print(f"No image files found in {source_dir}")
        return

    base_sha, branch_exists = get_base_sha(REPO_OWNER, REPO_NAME, branch, token)
    read_ref = branch if branch_exists else MAIN_BRANCH
    print(f"Comparing against existing GT on {REPO_OWNER}/{REPO_NAME}@{read_ref}")
    remote_image_entries = get_remote_image_entries(
        REPO_OWNER, REPO_NAME, target_dir, token, base_sha
    )
    remote_blob_shas = {
        path: item["sha"] for path, item in remote_image_entries.items()
    }
    changed_files = filter_changed_files(files_to_upload, remote_blob_shas)
    validate_gt_files(files_to_upload, changed_files, remote_image_entries, token)


def build_commit_message(target_dir, num_files):
    """Commit message naming who produced this GT, from what, and with which torch.

    Every commit is authored by the PAT owner, so without this the ci-data history
    cannot tell a torch-upgrade experiment from a routine new-model refresh.
    """
    lines = [
        f"diffusion-ci: update images in {target_dir} ({num_files} files) [automated]",
        "",
    ]
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if repo and run_id:
        lines.append(f"Run: {server}/{repo}/actions/runs/{run_id}")
    if os.environ.get("GITHUB_TRIGGERING_ACTOR"):
        lines.append(f"Triggered-by: {os.environ['GITHUB_TRIGGERING_ACTOR']}")
    if os.environ.get("GT_SOURCE_REF"):
        lines.append(f"Source-ref: {os.environ['GT_SOURCE_REF']}")
    # The cwd is the sglang checkout the GT was generated from.
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if rev.returncode == 0:
        lines.append(f"Source-sha: {rev.stdout.strip()}")
    try:
        lines.append(f"Torch: {version('torch')}")
    except PackageNotFoundError:
        pass
    return "\n".join(lines).rstrip() + "\n"


def report_published_commit(branch, commit_sha, target_dir):
    # Other jobs of the run commit to the same branch; the SHA to pin is the head
    # printed by the workflow's final report job, not this intermediate commit.
    line = (
        f"Published GT to {REPO_OWNER}/{REPO_NAME}@{branch} ({target_dir}) "
        f"as {commit_sha}. Pin the branch head from the run's final report job."
    )
    print(line)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def publish(source_dir, target_dir, branch):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    files_to_upload = collect_images(source_dir, target_dir)
    if not files_to_upload:
        print(f"No image files found in {source_dir}")
        return

    print(
        f"Found {len(files_to_upload)} image(s) to upload to {REPO_OWNER}/{REPO_NAME}/{target_dir}"
    )

    # Verify token
    perm = verify_token_permissions(REPO_OWNER, REPO_NAME, token)
    if perm == "rate_limited":
        print("GitHub API rate-limited, skipping upload.")
        return
    if not perm:
        print("Token permission verification failed.")
        sys.exit(1)

    print(f"Publishing to {REPO_OWNER}/{REPO_NAME} branch {branch}")

    # Commit with retry (handle concurrent pushes)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            base_sha, branch_exists = get_base_sha(REPO_OWNER, REPO_NAME, branch, token)
            tree_sha = get_tree_sha(REPO_OWNER, REPO_NAME, base_sha, token)
            remote_image_entries = get_remote_image_entries(
                REPO_OWNER, REPO_NAME, target_dir, token, base_sha
            )
            remote_blob_shas = {
                path: item["sha"] for path, item in remote_image_entries.items()
            }
            changed_files = filter_changed_files(files_to_upload, remote_blob_shas)
            validate_gt_files(
                files_to_upload, changed_files, remote_image_entries, token
            )
            if not changed_files:
                print("No image changes to publish.")
                return

            try:
                tree_items = create_blobs(REPO_OWNER, REPO_NAME, changed_files, token)
            except Exception as e:
                if is_rate_limit_error(e):
                    print("Rate-limited during blob creation, skipping.")
                    return
                if is_permission_error(e):
                    print(
                        f"ERROR: Token lacks write permission to {REPO_OWNER}/{REPO_NAME}. "
                        "Update GH_PAT_FOR_NIGHTLY_CI_DATA with a token that has contents:write."
                    )
                    sys.exit(1)
                raise

            new_tree_sha = create_tree(
                REPO_OWNER, REPO_NAME, tree_sha, tree_items, token
            )
            if new_tree_sha == tree_sha:
                print("No tree changes to publish.")
                return

            commit_msg = build_commit_message(target_dir, len(changed_files))
            commit_sha = create_commit(
                REPO_OWNER, REPO_NAME, new_tree_sha, base_sha, commit_msg, token
            )
            if branch_exists:
                update_branch_ref(REPO_OWNER, REPO_NAME, branch, commit_sha, token)
            else:
                # Born with its first commit, so an empty run leaves no branch behind.
                create_branch(REPO_OWNER, REPO_NAME, branch, commit_sha, token)
            print(
                f"Successfully pushed {len(changed_files)} changed images (commit {commit_sha[:10]})"
            )
            report_published_commit(branch, commit_sha, target_dir)
            return
        except BranchCreateRace:
            if attempt < max_retries - 1:
                print(f"Branch {branch} was created concurrently, retrying...")
                continue
            raise
        except Exception as e:
            if is_rate_limit_error(e):
                print("Rate-limited, skipping.")
                return
            if is_permission_error(e):
                print(f"ERROR: permission denied to {REPO_OWNER}/{REPO_NAME}")
                sys.exit(1)

            retryable = False
            if hasattr(e, "error_body"):
                if "Update is not a fast forward" in e.error_body:
                    retryable = True
                elif "Object does not exist" in e.error_body:
                    retryable = True

            if isinstance(e, HTTPError) and e.code in [422, 500, 502, 503, 504]:
                retryable = True

            if retryable and attempt < max_retries - 1:
                import time

                wait = 2**attempt
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed, retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                print(f"Failed after {attempt + 1} attempts: {e}")
                raise


def _resolve_target_dir(target_dir: str | None, per_platform: bool) -> str:
    """Resolve the effective remote target dir.

    With ``per_platform`` set, append the consistency platform subdir
    (``h100``/``b200``/``5090``) so the publish path matches how the consistency
    tests resolve GT: ``<target>/<platform>/<file>`` first, bare ``<target>/<file>``
    only as fallback. Reuses the SAME resolver the tests use so read and write can
    never drift. Lazy import keeps this script light when the flag is off.
    """
    target_dir = target_dir or DEFAULT_TARGET_DIR
    if per_platform:
        from sglang.multimodal_gen.test.test_utils import get_consistency_platform

        target_dir = f"{target_dir}/{get_consistency_platform()}"
    return target_dir


def main():
    parser = argparse.ArgumentParser(
        description="Publish diffusion GT images to GitHub"
    )
    parser.add_argument(
        "--source-dir", required=True, help="Directory containing GT images"
    )
    parser.add_argument(
        "--target-dir",
        required=False,
        default=None,
        help=f"Target directory in the remote repo (default: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("CI_DATA_BRANCH", ""),
        help=(
            f"{REPO_NAME} branch: {MAIN_BRANCH}, isolated, or a branch name "
            "(default: $CI_DATA_BRANCH; see ci_data_branch.py)"
        ),
    )
    parser.add_argument(
        "--source-ref",
        default=os.environ.get("GT_SOURCE_REF", ""),
        help="The sglang ref the GT was generated from (default: $GT_SOURCE_REF)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate generated GT images without publishing them",
    )
    args = parser.parse_args()

    per_platform = os.environ.get("SGLANG_DIFFUSION_GT_PER_PLATFORM") == "1"
    target_dir = _resolve_target_dir(args.target_dir, per_platform)
    if args.check_only:
        # Read-only, so an unchosen branch just compares against main.
        branch = (
            resolve_publish_branch(args.branch, args.source_ref)
            if args.branch
            else MAIN_BRANCH
        )
        check_quality(args.source_dir, target_dir, branch)
    else:
        branch = resolve_publish_branch(args.branch, args.source_ref)
        publish(args.source_dir, target_dir, branch)


if __name__ == "__main__":
    main()
