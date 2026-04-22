import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests
from github import Auth, Github

# Configuration
PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"


def find_workflow_run_url(
    gh_repo,
    workflow_id,
    ref,
    target_stage,
    token,
    dispatch_time,
    pr_head_sha=None,
    max_wait=30,
    test_command=None,
):
    """
    Poll for the workflow run URL after dispatch.

    Uses the dynamic run-name feature to identify runs:
    - Fork PRs: display_title = "[stage-name] sha"
    - Non-fork PRs: display_title = "[stage-name]"

    Args:
        gh_repo: PyGithub repository object
        workflow_id: ID of the workflow that was dispatched
        ref: Branch/ref the workflow was dispatched on
        target_stage: The stage name we're looking for
        token: GitHub API token
        dispatch_time: Unix timestamp when dispatch was triggered
        pr_head_sha: PR head SHA (for fork PRs, used to match display_title)
        max_wait: Maximum seconds to wait for the run to appear

    Returns:
        The workflow run URL if found, None otherwise.
    """
    # Build expected display_title based on workflow's run-name.
    # rerun-test includes test_command: "[rerun-test] <test_command> [<sha>]"
    # Other workflows: "[stage-name] [<sha>]"
    suffix = f" {test_command}" if test_command else ""
    if pr_head_sha:
        expected_title = f"[{target_stage}]{suffix} {pr_head_sha}"
    else:
        expected_title = f"[{target_stage}]{suffix}"

    print(f"Looking for workflow run with display_title: {expected_title}")

    for attempt in range(max_wait // 5):
        time.sleep(5)

        # Get recent workflow_dispatch runs for this workflow
        runs_url = f"https://api.github.com/repos/{gh_repo.full_name}/actions/workflows/{workflow_id}/runs"
        runs_resp = requests.get(
            runs_url,
            params={"event": "workflow_dispatch", "branch": ref, "per_page": 10},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )

        if runs_resp.status_code != 200:
            print(f"Failed to fetch workflow runs: {runs_resp.status_code}")
            continue

        for run in runs_resp.json().get("workflow_runs", []):
            # Skip runs created before our dispatch (with 10s tolerance)
            run_created = datetime.fromisoformat(
                run["created_at"].replace("Z", "+00:00")
            ).timestamp()
            if run_created < dispatch_time - 10:
                continue

            # Match by display_title (set by workflow's run-name directive)
            # This is immediately available, unlike job names which require waiting
            display_title = run.get("display_title", "")
            if display_title == expected_title:
                print(
                    f"Found matching workflow run: {run['id']} with title '{display_title}'"
                )
                return run["html_url"]

    print(f"Could not find workflow run after {max_wait} seconds")
    return None


def get_env_var(name):
    val = os.getenv(name)
    if not val:
        print(f"Error: Environment variable {name} not set.")
        sys.exit(1)
    return val


def load_permissions(user_login):
    """
    Reads the permissions JSON from the local file system and returns
    the permissions dict for the specific user.
    """
    try:
        print(f"Loading permissions from {PERMISSIONS_FILE_PATH}...")
        if not os.path.exists(PERMISSIONS_FILE_PATH):
            print(f"Error: Permissions file not found at {PERMISSIONS_FILE_PATH}")
            return None

        with open(PERMISSIONS_FILE_PATH, "r") as f:
            data = json.load(f)

        user_perms = data.get(user_login)

        if not user_perms:
            print(f"User '{user_login}' not found in permissions file.")
            return None

        return user_perms

    except Exception as e:
        print(f"Failed to load or parse permissions file: {e}")
        sys.exit(1)


def has_sgl_kernel_changes(pr):
    """
    Check if the PR has changes to the sgl-kernel directory.
    This is used to determine if we need a full workflow rerun
    (to rebuild the kernel) vs just rerunning failed jobs.
    """
    try:
        files = pr.get_files()
        for f in files:
            if f.filename.startswith("sgl-kernel/"):
                return True
        return False
    except Exception as e:
        print(f"Warning: Could not check PR files for sgl-kernel changes: {e}")
        # Default to False to avoid unnecessary full reruns
        return False


def handle_tag_run_ci(gh_repo, pr, comment, user_perms, react_on_success=True):
    """
    Handles the /tag-run-ci-label command.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_tag_run_ci_label", False):
        print("Permission denied: can_tag_run_ci_label is false.")
        return False

    print("Permission granted. Adding 'run-ci' label.")
    pr.add_to_labels("run-ci")

    if react_on_success:
        comment.create_reaction("+1")
        print("Label added and comment reacted.")
    else:
        print("Label added (reaction suppressed).")

    return True


def handle_rerun_failed_ci(gh_repo, pr, comment, user_perms, react_on_success=True):
    """
    Handles the /rerun-failed-ci command.
    Reruns workflows with 'failure' or 'skipped' conclusions.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        return False

    print("Permission granted. Triggering rerun of failed or skipped workflows.")

    # Check if PR has sgl-kernel changes - if so, we may need full reruns
    # to ensure sgl-kernel-build-wheels runs and produces fresh artifacts.
    # However, if the wheel already built successfully for this commit,
    # we can just rerun failed jobs — the artifact is already there.
    sgl_kernel_changes = has_sgl_kernel_changes(pr)
    if sgl_kernel_changes:
        print("PR has sgl-kernel changes - checking if kernel wheel already built")

    # Get the SHA of the latest commit in the PR
    head_sha = pr.head.sha
    print(f"Checking workflows for commit: {head_sha}")

    # If PR has sgl-kernel changes, check whether ALL wheel builds already
    # succeeded for this commit (CUDA + ARM). If so, we can use
    # rerun_failed_jobs and avoid retriggering all tests. If any wheel
    # build is pending/failed, a dependent job could fail for missing
    # artifacts, so fall back to full rerun.
    # Check-runs display names: "Build Wheel (<python>, <cuda>)" (CUDA) and
    # "Build Wheel Arm (<python>, <cuda>)" (ARM). The YAML job ids
    # sgl-kernel-build-wheels{,-arm} are NOT what the check-runs API
    # returns — it returns the job's `name:` field.
    kernel_wheel_built = False
    if sgl_kernel_changes:
        try:
            wheel_builds = [
                cr
                for cr in gh_repo.get_commit(head_sha).get_check_runs()
                if cr.name.startswith("Build Wheel")
            ]
            kernel_wheel_built = bool(wheel_builds) and all(
                cr.conclusion == "success" for cr in wheel_builds
            )
            print(
                f"All {len(wheel_builds)} kernel wheel build(s) passed - using rerun_failed_jobs"
                if kernel_wheel_built
                else f"Kernel wheel not fully built "
                f"({sum(1 for c in wheel_builds if c.conclusion == 'success')}"
                f"/{len(wheel_builds)} success) - will use full rerun"
            )
        except Exception as e:
            print(
                f"Failed to check kernel wheel status: {e} - falling back to full rerun"
            )

    # Rerun workflows with conclusion=failure or conclusion=skipped.
    #
    # - failure: use rerun_failed_jobs() which reruns failed jobs *and their
    #   dependent jobs* (GitHub API). Fast-fail cascades call
    #   core.setFailed(...) so their conclusion is "failure" and are covered.
    # - skipped: the entire run was skipped (no jobs ran), so there are no
    #   failed jobs for rerun_failed_jobs() to target. Use run.rerun().
    # - kernel wheel escape: if the PR touches sgl-kernel and not all wheel
    #   builds are success yet, full-rerun failure runs too — Build Wheel
    #   lives in pr-test-sgl-kernel.yml, consumers in pr-test.yml, and
    #   rerun_failed_jobs() is scoped to a single workflow run.
    runs = gh_repo.get_workflow_runs(head_sha=head_sha)

    rerun_count = 0
    for run in runs:
        if run.status != "completed":
            continue
        if run.conclusion not in ("failure", "skipped"):
            continue

        print(f"Processing {run.conclusion} workflow: {run.name} (ID: {run.id})")
        try:
            if run.conclusion == "skipped" or (
                sgl_kernel_changes and not kernel_wheel_built
            ):
                print("  Full rerun")
                run.rerun()
            else:
                print("  rerun_failed_jobs")
                run.rerun_failed_jobs()
            rerun_count += 1
        except Exception as e:
            print(f"Failed to rerun workflow {run.id}: {e}")

    if rerun_count > 0:
        print(f"Triggered rerun for {rerun_count} workflows.")
        if react_on_success:
            comment.create_reaction("+1")
        return True
    else:
        print("No failed or skipped workflows found to rerun.")
        return False


def handle_rerun_stage(
    gh_repo, pr, comment, user_perms, stage_name, token, react_on_success=True
):
    """
    Handles the /rerun-stage <stage-name> command.
    Triggers a workflow_dispatch to run only the specified stage, skipping dependencies.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_rerun_stage", False):
        print("Permission denied: can_rerun_stage is false.")
        return False

    if not stage_name:
        print("Error: No stage name provided")
        comment.create_reaction("confused")
        pr.create_issue_comment(
            f"❌ Please specify a stage name: `/rerun-stage <stage-name>`\n\n"
            f"Examples: `/rerun-stage unit-test-backend-4-gpu`, `/rerun-stage accuracy-test-1-gpu`"
        )
        return False

    print(f"Permission granted. Triggering workflow_dispatch for stage '{stage_name}'.")

    # Valid NVIDIA stage names that support target_stage
    nvidia_stages = [
        "stage-a-test-1-gpu-small",
        "stage-a-test-cpu",
        "stage-b-test-1-gpu-small",
        "stage-b-test-1-gpu-large",
        "stage-b-test-2-gpu-large",
        "stage-b-test-4-gpu-b200",
        "stage-c-test-4-gpu-h100",
        "stage-c-test-8-gpu-h200",
        "stage-c-test-8-gpu-h20",
        "stage-c-test-4-gpu-b200",
        "stage-c-test-4-gpu-b200-small",
        "stage-c-test-4-gpu-gb200",
        "stage-c-test-deepep-4-gpu-h100",
        "stage-c-test-deepep-8-gpu-h200",
        "multimodal-gen-test-1-gpu",
        "multimodal-gen-test-2-gpu",
        "multimodal-gen-component-accuracy",
        "multimodal-gen-component-accuracy-1-gpu",
        "multimodal-gen-component-accuracy-2-gpu",
        "multimodal-gen-test-1-b200",
    ]

    # Valid AMD stage names that support target_stage
    amd_stages = [
        "sgl-kernel-unit-test-amd",
        "sgl-kernel-unit-test-2-gpu-amd",
        "stage-a-test-1-gpu-small-amd",
        "stage-b-test-1-gpu-small-amd",
        "stage-b-test-1-gpu-small-amd-nondeterministic",
        "stage-b-test-1-gpu-small-amd-mi35x",
        "stage-b-test-1-gpu-large-amd",
        "stage-b-test-2-gpu-large-amd",
        "multimodal-gen-test-1-gpu-amd",
        "multimodal-gen-test-2-gpu-amd",
        "stage-c-test-large-8-gpu-amd",
        "stage-c-test-large-8-gpu-amd-mi35x",
    ]

    valid_stages = nvidia_stages + amd_stages
    is_amd_stage = stage_name in amd_stages

    if stage_name not in valid_stages:
        comment.create_reaction("confused")
        pr.create_issue_comment(
            f"❌ Stage `{stage_name}` doesn't support isolated runs yet.\n\n"
            f"**NVIDIA stages:**\n"
            + "\n".join(f"- `{s}`" for s in nvidia_stages)
            + "\n\n**AMD stages:**\n"
            + "\n".join(f"- `{s}`" for s in amd_stages)
            + "\n\nOther stages will be added soon. For now, use `/rerun-failed-ci` for those stages."
        )
        return False

    try:
        # Get the appropriate workflow based on stage type
        workflow_name = "PR Test (AMD)" if is_amd_stage else "PR Test"
        workflows = gh_repo.get_workflows()
        target_workflow = None
        for wf in workflows:
            if wf.name == workflow_name:
                target_workflow = wf
                break

        if not target_workflow:
            print(f"Error: {workflow_name} workflow not found")
            return False

        # Check if PR is from a fork by comparing repo owners
        # Handle case where fork repo may have been deleted (pr.head.repo is None)
        is_fork = (
            pr.head.repo is None or pr.head.repo.owner.login != gh_repo.owner.login
        )
        print(f"PR is from fork: {is_fork}")

        # pr_head_sha is used for fork PRs (passed to workflow and used for URL lookup)
        pr_head_sha = None

        if is_fork:
            # For fork PRs: dispatch on main and pass SHA as input
            # This is needed because fork branch names don't exist in the main repo
            ref = "main"
            pr_head_sha = pr.head.sha
            print(
                f"Triggering {workflow_name} workflow on ref: {ref}, PR head SHA: {pr_head_sha}"
            )
            if is_amd_stage:
                inputs = {
                    "target_stage": stage_name,
                    "pr_head_sha": pr_head_sha,
                }
            else:
                inputs = {
                    "target_stage": stage_name,
                    "pr_head_sha": pr_head_sha,
                }
        else:
            # For non-fork PRs: dispatch on the PR branch directly
            # This allows testing workflow changes before merge
            ref = pr.head.ref
            print(f"Triggering {workflow_name} workflow on branch: {ref}")
            if is_amd_stage:
                inputs = {"target_stage": stage_name}
            else:
                inputs = {"target_stage": stage_name}

        # Record dispatch time before triggering
        dispatch_time = time.time()

        # Use requests directly as PyGithub's create_dispatch only accepts HTTP 204
        dispatch_url = f"https://api.github.com/repos/{gh_repo.full_name}/actions/workflows/{target_workflow.id}/dispatches"
        dispatch_resp = requests.post(
            dispatch_url,
            json={"ref": ref, "inputs": inputs},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        success = dispatch_resp.status_code in (200, 204)
        if not success:
            print(f"Dispatch failed: {dispatch_resp.status_code} {dispatch_resp.text}")

        if success:
            print(f"Successfully triggered workflow for stage '{stage_name}'")
            if react_on_success:
                comment.create_reaction("+1")

                run_url = find_workflow_run_url(
                    gh_repo,
                    target_workflow.id,
                    ref,
                    stage_name,
                    token,
                    dispatch_time,
                    pr_head_sha=pr_head_sha,
                    max_wait=30,
                )
                if run_url:
                    pr.create_issue_comment(
                        f"✅ Triggered `{stage_name}` to run independently"
                        f" (skipping dependencies)."
                        f" [View workflow run]({run_url})"
                    )
                else:
                    pr.create_issue_comment(
                        f"✅ Triggered `{stage_name}` to run independently"
                        f" (skipping dependencies).\n"
                        f"⚠️ Could not retrieve workflow run URL. "
                        f"Check the [Actions tab](https://github.com/{gh_repo.full_name}/actions) for progress."
                    )
            return True
        else:
            print("Failed to trigger workflow_dispatch")
            return False

    except Exception as e:
        print(f"Error triggering workflow_dispatch: {e}")
        comment.create_reaction("confused")
        pr.create_issue_comment(
            f"❌ Failed to trigger workflow: {str(e)}\n\n"
            f"Please check the logs or contact maintainers."
        )
        return False


CUDA_SUITE_TO_RUNNER = {
    # PR test suites
    "stage-a-test-1-gpu-small": "1-gpu-5090",
    "stage-a-test-cpu": "ubuntu-latest",
    "stage-b-test-1-gpu-small": "1-gpu-5090",
    "stage-b-test-1-gpu-large": "1-gpu-h100",
    "stage-b-test-2-gpu-large": "2-gpu-h100",
    "stage-b-test-4-gpu-b200": "4-gpu-b200",
    "stage-c-test-4-gpu-h100": "4-gpu-h100",
    "stage-c-test-8-gpu-h200": "8-gpu-h200",
    "stage-c-test-8-gpu-h20": "8-gpu-h20",
    "stage-c-test-4-gpu-b200": "4-gpu-b200",
    "stage-c-test-4-gpu-b200-small": "4-gpu-b200-low-disk",
    "stage-c-test-deepep-4-gpu-h100": "4-gpu-h100",
    "stage-c-test-deepep-8-gpu-h200": "8-gpu-h200",
    # Nightly test suites (NVIDIA)
    "nightly-1-gpu": "1-gpu-h100",
    "nightly-4-gpu": "4-gpu-h100",
    "nightly-4-gpu-b200": "4-gpu-b200",
    "nightly-8-gpu-common": "8-gpu-h200",
    "nightly-8-gpu-h200": "8-gpu-h200",
    "nightly-8-gpu-h20": "8-gpu-h20",
    "nightly-8-gpu-b200": "8-gpu-b200",
    "nightly-eval-text-2-gpu": "2-gpu-h100",
    "nightly-eval-vlm-2-gpu": "2-gpu-h100",
    "nightly-perf-text-2-gpu": "2-gpu-h100",
    "nightly-perf-vlm-2-gpu": "2-gpu-h100",
    "nightly-kernel-1-gpu": "1-gpu-h100",
    "nightly-kernel-8-gpu-h200": "8-gpu-h200",
    # Weekly test suites
    "weekly-8-gpu-h200": "8-gpu-h200",
}

DEEPEP_SUITES = {
    "stage-c-test-8-gpu-h20",
    "stage-c-test-deepep-4-gpu-h100",
    "stage-c-test-deepep-8-gpu-h200",
}


MULTIMODAL_TEST_DIR = "python/sglang/multimodal_gen/test"

MULTIMODAL_PATH_TO_RUNNER = {
    "2_gpu": "2-gpu-h100",
    "2-gpu": "2-gpu-h100",
}
MULTIMODAL_DEFAULT_RUNNER = "1-gpu-h100"


def resolve_test_file(file_part):
    """
    Resolve a user-provided file path to a path relative to test/ or full path for multimodal.

    Supports:
    - Full path: test/registered/core/test_srt_endpoint.py
    - Relative to test/: registered/core/test_srt_endpoint.py
    - Bare filename: test_srt_endpoint.py (glob-matched, must be unique)
    - Multimodal paths: python/sglang/multimodal_gen/test/server/test_server_a.py

    Returns (resolved_path, is_multimodal, error_message). On success error_message is None.
    """
    # Check if it's explicitly a multimodal path
    multimodal_prefixes = [
        "python/sglang/multimodal_gen/test/",
        "sglang/multimodal_gen/test/",
        "multimodal_gen/test/",
    ]
    for prefix in multimodal_prefixes:
        if file_part.startswith(prefix):
            full_path = (
                file_part
                if file_part.startswith("python/")
                else f"python/sglang/multimodal_gen/test/{file_part[len(prefix):]}"
            )
            if not os.path.isfile(full_path):
                return None, False, f"File not found: `{full_path}`"
            return full_path, True, None

    # Existing logic for test/registered/ paths
    if file_part.startswith("test/"):
        file_part = file_part[len("test/") :]

    if "/" not in file_part:
        # Try test/registered/ first
        matches = glob.glob(f"test/registered/**/{file_part}", recursive=True)

        # Try multimodal test directory
        mm_matches = glob.glob(f"{MULTIMODAL_TEST_DIR}/**/{file_part}", recursive=True)
        # Filter to only test files
        mm_matches = [m for m in mm_matches if os.path.basename(m).startswith("test_")]

        if len(matches) == 1 and len(mm_matches) == 0:
            return matches[0][len("test/") :], False, None
        if len(matches) == 0 and len(mm_matches) == 1:
            return mm_matches[0], True, None

        all_matches = matches + mm_matches
        if len(all_matches) == 0:
            return (
                None,
                False,
                f"No test file found matching `{file_part}` under `test/registered/` or `{MULTIMODAL_TEST_DIR}/`.",
            )
        if len(all_matches) > 1:
            match_list = "\n".join(f"- `{m}`" for m in sorted(all_matches))
            return (
                None,
                False,
                (
                    f"Ambiguous filename `{file_part}` — matched {len(all_matches)} files:\n\n"
                    f"{match_list}\n\n"
                    f"Please provide the full path, e.g. `/rerun-test {all_matches[0]}`"
                ),
            )
        # Shouldn't reach here, but handle gracefully
        if mm_matches:
            return mm_matches[0], True, None
        return matches[0][len("test/") :], False, None

    # Path with directory - check test/ location
    full_path = f"test/{file_part}"
    if os.path.isfile(full_path):
        return file_part, False, None

    return None, False, f"File not found: `{full_path}`"


def detect_multimodal_suite(file_path):
    """
    Determine runner for a multimodal gen test file based on its path.

    Returns (runner_label, error_message).
    """
    # Check path components and basename for GPU count hints
    for pattern, runner in MULTIMODAL_PATH_TO_RUNNER.items():
        if pattern in file_path:
            return runner, None
    return MULTIMODAL_DEFAULT_RUNNER, None


def detect_suite(file_path_from_test):
    """
    Read a test file and extract the suite from register_cuda_ci or register_cpu_ci.

    Returns (suite_name, runner_label, use_deepep, is_cpu, error_message).
    """
    full_path = f"test/{file_path_from_test}"
    with open(full_path, "r") as f:
        content = f.read()

    # Try CUDA first
    match = re.search(
        r'^[^#\n]*register_cuda_ci\([^)]*suite\s*=\s*["\']([^"\']+)["\']',
        content,
        re.MULTILINE,
    )
    if match:
        suite = match.group(1)
        runner = CUDA_SUITE_TO_RUNNER.get(suite)
        if not runner:
            known = ", ".join(f"`{s}`" for s in sorted(CUDA_SUITE_TO_RUNNER))
            return (
                suite,
                None,
                False,
                False,
                (
                    f"Unknown CUDA suite `{suite}` in `{full_path}`.\n\n"
                    f"Known suites: {known}"
                ),
            )
        use_deepep = suite in DEEPEP_SUITES
        return suite, runner, use_deepep, False, None

    # Try CPU
    match = re.search(
        r'^[^#\n]*register_cpu_ci\([^)]*suite\s*=\s*["\']([^"\']+)["\']',
        content,
        re.MULTILINE,
    )
    if match:
        suite = match.group(1)
        return suite, "ubuntu-latest", False, True, None

    return (
        None,
        None,
        False,
        False,
        (
            f"No `register_cuda_ci()` or `register_cpu_ci()` found in `{full_path}`.\n\n"
            f"This file may not be a registered CI test."
        ),
    )


def _resolve_test_spec(test_spec):
    """
    Resolve a single test spec into its components without dispatching.

    Returns a dict with keys: spec, resolved_path, test_command, suite,
    runner_label, use_deepep, is_cpu, error.
    """
    if "::" in test_spec:
        file_part, test_selector = test_spec.split("::", 1)
    else:
        file_part = test_spec
        test_selector = None

    file_part = file_part.strip()
    if test_selector:
        test_selector = test_selector.strip()

    resolved_path, is_multimodal, err = resolve_test_file(file_part)
    if err:
        return {"spec": test_spec, "error": err}

    if is_multimodal:
        runner_label, err = detect_multimodal_suite(resolved_path)
        if err:
            return {"spec": test_spec, "error": err}

        # For multimodal pytest tests, use :: separator for test selection
        test_command = resolved_path
        if test_selector:
            test_command = f"{resolved_path}::{test_selector}"

        print(
            f"Resolved (multimodal): file={resolved_path}, selector={test_selector}, "
            f"runner={runner_label}, command='{test_command}'"
        )
        return {
            "spec": test_spec,
            "test_command": test_command,
            "suite": "multimodal",
            "runner_label": runner_label,
            "use_deepep": False,
            "is_cpu": False,
            "install_diffusion": True,
            "error": None,
        }

    suite, runner_label, use_deepep, is_cpu, err = detect_suite(resolved_path)
    if err:
        return {"spec": test_spec, "error": err}

    test_command = resolved_path
    if test_selector:
        test_command = f"{resolved_path} {test_selector}"

    print(
        f"Resolved: file={resolved_path}, selector={test_selector}, "
        f"suite={suite}, runner={runner_label}, deepep={use_deepep}, "
        f"cpu={is_cpu}, command='{test_command}'"
    )
    return {
        "spec": test_spec,
        "test_command": test_command,
        "suite": suite,
        "runner_label": runner_label,
        "use_deepep": use_deepep,
        "is_cpu": is_cpu,
        "install_diffusion": False,
        "error": None,
    }


def _dispatch_batch(gh_repo, pr, batch, token):
    """
    Dispatch a single workflow run for a batch of resolved test specs
    that share the same (runner_label, use_deepep, is_cpu).

    Returns a dict with keys: specs, success, test_commands, runner_label, run_url, error.
    """
    test_commands = [r["test_command"] for r in batch]
    runner_label = batch[0]["runner_label"]
    use_deepep = batch[0]["use_deepep"]
    is_cpu = batch[0]["is_cpu"]
    install_diffusion = batch[0].get("install_diffusion", False)

    # Join multiple commands with newlines for the workflow to iterate over
    combined_command = "\n".join(test_commands)

    try:
        workflow_name = "Rerun Test"
        workflows = gh_repo.get_workflows()
        target_workflow = None
        for wf in workflows:
            if wf.name == workflow_name:
                target_workflow = wf
                break

        if not target_workflow:
            return {
                "specs": [r["spec"] for r in batch],
                "success": False,
                "error": f"{workflow_name} workflow not found",
            }

        is_fork = (
            pr.head.repo is None or pr.head.repo.owner.login != gh_repo.owner.login
        )

        pr_head_sha = None
        inputs = {
            "test_command": combined_command,
            "runner_label": runner_label,
            "use_deepep": str(use_deepep).lower(),
            "is_cpu": str(is_cpu).lower(),
            "install_diffusion": str(install_diffusion).lower(),
        }
        if is_fork:
            ref = "main"
            pr_head_sha = pr.head.sha
            inputs["pr_head_sha"] = pr_head_sha
        else:
            ref = pr.head.ref

        dispatch_time = time.time()

        dispatch_url = f"https://api.github.com/repos/{gh_repo.full_name}/actions/workflows/{target_workflow.id}/dispatches"
        dispatch_resp = requests.post(
            dispatch_url,
            json={"ref": ref, "inputs": inputs},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        success = dispatch_resp.status_code in (200, 204)
        if not success:
            print(f"Dispatch failed: {dispatch_resp.status_code} {dispatch_resp.text}")
            return {
                "specs": [r["spec"] for r in batch],
                "success": False,
                "error": f"Dispatch failed: {dispatch_resp.status_code}",
            }

        print(f"Successfully triggered rerun-test: {combined_command}")

        run_url = find_workflow_run_url(
            gh_repo,
            target_workflow.id,
            ref,
            "rerun-test",
            token,
            dispatch_time,
            pr_head_sha=pr_head_sha,
            max_wait=30,
            test_command=combined_command,
        )
        return {
            "specs": [r["spec"] for r in batch],
            "success": True,
            "test_commands": test_commands,
            "runner_label": runner_label,
            "run_url": run_url,
        }

    except Exception as e:
        print(f"Error triggering rerun-test for batch: {e}")
        return {
            "specs": [r["spec"] for r in batch],
            "success": False,
            "error": str(e),
        }


def handle_rerun_test(gh_repo, pr, comment, user_perms, test_specs, token):
    """
    Handles the /rerun-test command. Resolves all test specs, groups them by
    (runner_label, use_deepep, is_cpu), and dispatches one workflow per group.
    """
    # SECURITY: For fork PRs, only allow /rerun-test if the commenter has write+ permission.
    # This command checks out and executes code from the PR branch on self-hosted GPU
    # runners, so we must ensure the commenter is a trusted collaborator.
    is_fork = pr.head.repo is None or pr.head.repo.owner.login != gh_repo.owner.login
    if is_fork:
        commenter = comment.user.login
        perm = gh_repo.get_collaborator_permission(commenter)
        if perm not in ("admin", "write"):
            print(f"Permission denied: /rerun-test on fork PR by {commenter}.")
            comment.create_reaction("confused")
            pr.create_issue_comment(
                "❌ `/rerun-test` is not available for fork PRs unless the commenter "
                "has write permission on the repo.\n\n"
                "Please ask a maintainer to run this command, or use the normal CI flow."
            )
            return False
        print(f"Fork PR, but commenter {commenter} has write+ permission. Proceeding.")

    if not (
        user_perms.get("can_rerun_test", False)
        or user_perms.get("can_rerun_stage", False)
    ):
        print("Permission denied: neither can_rerun_test nor can_rerun_stage is true.")
        return False

    if not test_specs:
        comment.create_reaction("confused")
        pr.create_issue_comment(
            "❌ Please specify a test: `/rerun-test <file>::<TestClass.test_method>`\n\n"
            "Examples:\n"
            "- `/rerun-test test/registered/core/test_srt_endpoint.py::TestSRTEndpoint.test_simple_decode`\n"
            "- `/rerun-test registered/core/test_srt_endpoint.py::TestSRTEndpoint`\n"
            "- `/rerun-test test_srt_endpoint.py`\n"
            "- `/rerun-test test_a.py test_b.py test_c.py` (multiple tests)"
        )
        return False

    # Phase 1: Resolve all specs
    resolved = []
    resolve_failures = []
    for spec in test_specs:
        r = _resolve_test_spec(spec)
        if r.get("error"):
            resolve_failures.append(r)
        else:
            resolved.append(r)

    # Phase 2: Group by (runner_label, use_deepep, is_cpu, install_diffusion)
    groups = {}
    for r in resolved:
        key = (
            r["runner_label"],
            r["use_deepep"],
            r["is_cpu"],
            r.get("install_diffusion", False),
        )
        groups.setdefault(key, []).append(r)

    # Phase 3: Dispatch one workflow per group
    dispatch_results = []
    for batch in groups.values():
        dispatch_results.append(_dispatch_batch(gh_repo, pr, batch, token))

    # Build consolidated comment
    lines = []
    for dr in dispatch_results:
        if dr["success"]:
            install_diff = any(
                r.get("install_diffusion", False)
                for r in resolved
                if r["spec"] in dr["specs"]
            )
            if install_diff:
                cmds = "\n".join(
                    f"python3 -m pytest {cmd} -x" for cmd in dr["test_commands"]
                )
            else:
                cmds = "\n".join(
                    f"cd test/ && python3 {cmd}" for cmd in dr["test_commands"]
                )
            if dr.get("run_url"):
                lines.append(
                    f"✅ `{dr['runner_label']}` ({len(dr['test_commands'])} test{'s' if len(dr['test_commands']) > 1 else ''}): "
                    f"[View workflow run]({dr['run_url']})\n"
                    f"```\n{cmds}\n```"
                )
            else:
                lines.append(
                    f"✅ `{dr['runner_label']}` ({len(dr['test_commands'])} test{'s' if len(dr['test_commands']) > 1 else ''}):\n"
                    f"```\n{cmds}\n```\n"
                    f"⚠️ Could not retrieve workflow run URL. "
                    f"Check the [Actions tab](https://github.com/{gh_repo.full_name}/actions) for progress."
                )
        else:
            specs_str = ", ".join(f"`{s}`" for s in dr["specs"])
            lines.append(f"❌ {specs_str}: {dr['error']}")

    for r in resolve_failures:
        lines.append(f"❌ `{r['spec']}`: {r['error']}")

    body = "\n\n".join(lines)

    successes = [dr for dr in dispatch_results if dr["success"]]
    if successes:
        comment.create_reaction("+1")
    if not successes and (resolve_failures or dispatch_results):
        comment.create_reaction("confused")

    pr.create_issue_comment(body)
    return len(successes) > 0


def main():
    # 1. Load Environment Variables
    token = get_env_var("GITHUB_TOKEN")
    repo_name = get_env_var("REPO_FULL_NAME")
    pr_number = int(get_env_var("PR_NUMBER"))
    comment_id = int(get_env_var("COMMENT_ID"))
    comment_body = get_env_var("COMMENT_BODY").strip()
    user_login = get_env_var("USER_LOGIN")

    # 2. Load Permissions (local file check first to avoid unnecessary API calls)
    user_perms = load_permissions(user_login)

    # 3. Initialize GitHub API with Auth
    auth = Auth.Token(token)
    g = Github(auth=auth)

    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    comment = repo.get_issue(pr_number).get_comment(comment_id)

    # PR authors can always rerun failed CI and rerun individual UTs on their own PRs,
    # even if they are not listed in CI_PERMISSIONS.json.
    # Note: /tag-run-ci-label and /rerun-stage still require CI_PERMISSIONS.json.
    # Note: /rerun-test is blocked entirely for fork PRs in handle_rerun_test() itself.
    if pr.user.login == user_login:
        if user_perms is None:
            print(
                f"User {user_login} is the PR author (not in CI_PERMISSIONS.json). "
                "Granting CI rerun permissions."
            )
            user_perms = {}
        else:
            print(
                f"User {user_login} is the PR author and has existing CI permissions."
            )
        user_perms["can_rerun_failed_ci"] = True
        user_perms["can_rerun_test"] = True

    if not user_perms:
        print(f"User {user_login} does not have any configured permissions. Exiting.")
        return

    # 4. Parse Command and Execute
    first_line = comment_body.split("\n")[0].strip()

    if first_line.startswith("/tag-run-ci-label"):
        handle_tag_run_ci(repo, pr, comment, user_perms)

    elif first_line.startswith("/rerun-failed-ci"):
        handle_rerun_failed_ci(repo, pr, comment, user_perms)

    elif first_line.startswith("/tag-and-rerun-ci"):
        # Perform both actions, but suppress individual reactions
        print("Processing combined command: /tag-and-rerun-ci")

        tagged = handle_tag_run_ci(
            repo, pr, comment, user_perms, react_on_success=False
        )

        # Wait for the label to propagate before triggering rerun
        if tagged:
            print("Waiting 5 seconds for label to propagate...")
            time.sleep(5)

        rerun = handle_rerun_failed_ci(
            repo, pr, comment, user_perms, react_on_success=False
        )

        # If at least one action was successful, add the reaction here
        if tagged or rerun:
            comment.create_reaction("+1")
            print("Combined command processed successfully; reaction added.")
        else:
            print("Combined command finished, but no actions were taken.")

    elif first_line.startswith("/rerun-stage"):
        # Extract stage name from command
        parts = first_line.split(maxsplit=1)
        stage_name = parts[1].strip() if len(parts) > 1 else None
        handle_rerun_stage(repo, pr, comment, user_perms, stage_name, token)

    elif first_line.startswith("/rerun-test"):
        test_specs = first_line.split()[1:]
        handle_rerun_test(repo, pr, comment, user_perms, test_specs or None, token)

    else:
        print(f"Unknown or ignored command: {first_line}")


if __name__ == "__main__":
    main()
