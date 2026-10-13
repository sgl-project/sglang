# PR `run-ci` Label Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require the `run-ci` label for PR PPU Full CI, let authorized repository collaborators add it through `/tag-run-ci-label`, and keep subsequent PR pushes automatically rerunning Full CI.

**Architecture:** `ci.yaml` remains the PR orchestrator and performs the event-aware label gate before calling `pr-test-ppu.yml`. The slash-command workflow always executes trusted default-branch code; the Python handler merges explicit `CI_PERMISSIONS.json` grants with GitHub repository permissions and fails closed when role lookup fails.

**Tech Stack:** GitHub Actions YAML, Python 3.10, PyGithub, Python `unittest`, PyYAML/actionlint-compatible workflow validation.

---

## File Structure

- Modify `.github/workflows/ci.yaml`: add `labeled` handling and move Full CI from Human Review gating to event-aware `run-ci` gating after Registration Validation.
- Modify `.github/workflows/pr-test-ppu.yml`: update obsolete comments while retaining the reusable workflow's defense-in-depth label check.
- Modify `.github/workflows/slash-command-handler.yml`: always checkout the repository default branch for privileged comment handling.
- Modify `scripts/ci/utils/slash_command_handler.py`: resolve effective tag permission from explicit configuration or repository role.
- Create `scripts/ci/utils/tests/test_run_ci_permissions.py`: unit-test permission resolution without network calls.
- Create `scripts/ci/utils/tests/test_run_ci_workflows.py`: statically test trigger, dependency, label-condition, and trusted-checkout contracts.

### Task 1: Repository-role authorization

**Files:**
- Create: `scripts/ci/utils/tests/test_run_ci_permissions.py`
- Modify: `scripts/ci/utils/slash_command_handler.py`

- [ ] **Step 1: Write failing permission tests**

Create `scripts/ci/utils/tests/test_run_ci_permissions.py` with import stubs for optional CI dependencies and these cases:

```python
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
HANDLER_PATH = ROOT / "scripts/ci/utils/slash_command_handler.py"

requests_stub = types.ModuleType("requests")
github_stub = types.ModuleType("github")


class _Auth:
    @staticmethod
    def Token(token):
        return token


github_stub.Auth = _Auth
github_stub.Github = object
sys.modules.setdefault("requests", requests_stub)
sys.modules.setdefault("github", github_stub)
sys.modules.setdefault("runner_configs", types.ModuleType("runner_configs"))

spec = importlib.util.spec_from_file_location("slash_command_handler", HANDLER_PATH)
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)


class ResolveTagRunCiPermissionsTest(unittest.TestCase):
    def test_explicit_permission_skips_repository_lookup(self):
        repo = Mock()

        result = handler.resolve_tag_run_ci_permissions(
            repo, "configured-user", {"can_tag_run_ci_label": True}
        )

        self.assertTrue(result["can_tag_run_ci_label"])
        repo.get_collaborator_permission.assert_not_called()

    def test_write_or_higher_repository_role_is_authorized(self):
        for permission in ("write", "maintain", "admin"):
            with self.subTest(permission=permission):
                repo = Mock()
                repo.get_collaborator_permission.return_value = permission

                result = handler.resolve_tag_run_ci_permissions(
                    repo, "maintainer", None
                )

                self.assertTrue(result["can_tag_run_ci_label"])

    def test_lower_repository_role_is_not_authorized(self):
        for permission in ("none", "read", "triage"):
            with self.subTest(permission=permission):
                repo = Mock()
                repo.get_collaborator_permission.return_value = permission

                result = handler.resolve_tag_run_ci_permissions(
                    repo, "external-user", None
                )

                self.assertNotIn("can_tag_run_ci_label", result)

    def test_lookup_failure_fails_closed(self):
        repo = Mock()
        repo.get_collaborator_permission.side_effect = RuntimeError("API unavailable")

        result = handler.resolve_tag_run_ci_permissions(
            repo, "unknown-user", None
        )

        self.assertNotIn("can_tag_run_ci_label", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 scripts/ci/utils/tests/test_run_ci_permissions.py
```

Expected: four tests error because `resolve_tag_run_ci_permissions` is not defined.

- [ ] **Step 3: Implement the minimal permission resolver**

Add near `load_permissions` in `scripts/ci/utils/slash_command_handler.py`:

```python
_TAG_RUN_CI_REPOSITORY_PERMISSIONS = {"write", "maintain", "admin"}


def resolve_tag_run_ci_permissions(gh_repo, user_login, user_perms):
    """Merge explicit tag permission with the user's repository role."""
    effective_permissions = dict(user_perms or {})
    if effective_permissions.get("can_tag_run_ci_label", False):
        return effective_permissions

    try:
        repository_permission = gh_repo.get_collaborator_permission(user_login)
    except Exception as e:
        print(
            "Failed to query repository permission for "
            f"{user_login}: {e}. Tag permission remains denied."
        )
        return effective_permissions

    print(f"User {user_login} repository permission: {repository_permission}")
    if repository_permission in _TAG_RUN_CI_REPOSITORY_PERMISSIONS:
        effective_permissions["can_tag_run_ci_label"] = True

    return effective_permissions
```

Move first-line parsing in `main()` to immediately after loading `COMMENT_BODY` and `USER_LOGIN`:

```python
first_line = comment_body.split("\n")[0].strip()
```

After `repo`, `pr`, and `comment` are initialized, resolve role permission only for commands that can add the label:

```python
if first_line.startswith(("/tag-run-ci-label", "/tag-and-rerun-ci")):
    user_perms = resolve_tag_run_ci_permissions(repo, user_login, user_perms)
```

Delete the later duplicate `first_line` assignment. Update the nearby comment so `/tag-run-ci-label` is documented as requiring either explicit configuration or repository `write` permission and above.

- [ ] **Step 4: Run permission tests and verify GREEN**

Run:

```bash
python3 scripts/ci/utils/tests/test_run_ci_permissions.py
```

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Commit authorization changes**

```bash
git add scripts/ci/utils/slash_command_handler.py scripts/ci/utils/tests/test_run_ci_permissions.py
git commit -m "feat(ci): authorize run-ci label by repository role"
```

### Task 2: Trusted slash-command checkout

**Files:**
- Create: `scripts/ci/utils/tests/test_run_ci_workflows.py`
- Modify: `.github/workflows/slash-command-handler.yml`

- [ ] **Step 1: Write the failing trusted-checkout test**

Create `scripts/ci/utils/tests/test_run_ci_workflows.py`:

```python
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


class RunCiWorkflowContractTest(unittest.TestCase):
    def read_workflow(self, name):
        return (ROOT / ".github/workflows" / name).read_text()

    def test_slash_handler_executes_default_branch_code(self):
        workflow = self.read_workflow("slash-command-handler.yml")

        self.assertIn(
            "ref: ${{ github.event.repository.default_branch }}", workflow
        )
        self.assertNotIn("refs/pull/", workflow)
        self.assertNotIn("headRefName", workflow)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 scripts/ci/utils/tests/test_run_ci_workflows.py
```

Expected: failure because the workflow still computes and checks out PR refs.

- [ ] **Step 3: Replace PR-head checkout with trusted default-branch checkout**

In `.github/workflows/slash-command-handler.yml`, delete `Get PR details` and `Check commenter permission for fork PRs`. Replace the current checkout block with:

```yaml
      # SECURITY: issue_comment runs with pull-requests/actions write access.
      # Always execute handler code from the trusted default branch; never run
      # a PR head with this token, including same-repository PR heads.
      - name: Checkout trusted handler
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.repository.default_branch }}
```

Leave command matching, workflow permissions, dependency installation, and handler environment unchanged.

- [ ] **Step 4: Run the workflow contract test and verify GREEN**

Run:

```bash
python3 scripts/ci/utils/tests/test_run_ci_workflows.py
```

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Commit trusted-checkout changes**

```bash
git add .github/workflows/slash-command-handler.yml scripts/ci/utils/tests/test_run_ci_workflows.py
git commit -m "fix(ci): run slash handler from trusted branch"
```

### Task 3: Event-aware `run-ci` Full CI gate

**Files:**
- Modify: `scripts/ci/utils/tests/test_run_ci_workflows.py`
- Modify: `.github/workflows/ci.yaml`
- Modify: `.github/workflows/pr-test-ppu.yml`

- [ ] **Step 1: Add failing orchestration contract tests**

Add these methods to `RunCiWorkflowContractTest`:

```python
    def test_ci_pipeline_uses_event_aware_run_ci_gate(self):
        workflow = self.read_workflow("ci.yaml")

        self.assertIn(
            "types: [opened, synchronize, reopened, ready_for_review, labeled]",
            workflow,
        )
        self.assertIn("needs: registration-validation", workflow)
        self.assertIn(
            "contains(github.event.pull_request.labels.*.name, 'run-ci')",
            workflow,
        )
        self.assertIn(
            "github.event.action != 'labeled' || "
            "github.event.label.name == 'run-ci'",
            workflow,
        )
        self.assertNotIn("require_run_ci_label: false", workflow)
        self.assertIn("github.event.action == 'labeled'", workflow)
        self.assertIn("github.event.label.name != 'run-ci'", workflow)

    def test_ppu_workflow_retains_defense_in_depth_label_gate(self):
        workflow = self.read_workflow("pr-test-ppu.yml")

        self.assertIn(
            "contains(github.event.pull_request.labels.*.name, 'run-ci')",
            workflow,
        )
        self.assertNotIn("ci.yaml passes false", workflow)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 scripts/ci/utils/tests/test_run_ci_workflows.py
```

Expected: orchestration tests fail because `labeled` is absent, Full CI needs Human Review, and the caller bypasses the label gate.

- [ ] **Step 3: Implement `ci.yaml` trigger and Full CI condition**

Change the PR trigger to:

```yaml
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review, labeled]
```

Extend the concurrency suffix condition so an unrelated label event cannot cancel an active Full CI run:

```yaml
concurrency:
  group: >-
    ci-${{ github.event.pull_request.number }}-${{
    ((github.event_name == 'pull_request_review' &&
    (github.event.review.state != 'approved' || github.event.review.user.type == 'Bot')) ||
    (github.event_name == 'pull_request' && github.event.action == 'labeled' &&
    github.event.label.name != 'run-ci')) && github.run_id || 'active' }}
  cancel-in-progress: true
```

Replace the Full CI section with:

```yaml
  # Stage 4: Full PPU CI. The run-ci label authorizes PPU consumption;
  # Registration Validation remains the technical prerequisite. Human Review
  # is an independent merge gate and does not delay PPU feedback.
  # For labeled events, only adding run-ci starts Full CI; adding another label
  # to a PR that already has run-ci must not create a duplicate run.
  ci:
    name: Full CI
    needs: registration-validation
    if: >-
      github.event_name == 'pull_request' &&
      contains(github.event.pull_request.labels.*.name, 'run-ci') &&
      (github.event.action != 'labeled' || github.event.label.name == 'run-ci')
    uses: ./.github/workflows/pr-test-ppu.yml
    secrets: inherit
```

- [ ] **Step 4: Update reusable-workflow comments without changing its gate**

In `.github/workflows/pr-test-ppu.yml`, update the opening and `check-changes` comments to state:

- PR execution is orchestrated by `ci.yaml`; direct `pull_request` remains absent to avoid duplicate runs.
- The inherited `pull_request` event still requires `run-ci` by default.
- `workflow_dispatch` remains available for manual verification and bypasses the PR-event gate.
- `require_run_ci_label` remains an optional reusable-workflow input, but `ci.yaml` no longer disables it.

Do not change Stage A/B/C, runner selection, secrets, check names, or finish-job aggregation.

- [ ] **Step 5: Run workflow contract tests and verify GREEN**

Run:

```bash
python3 scripts/ci/utils/tests/test_run_ci_workflows.py
```

Expected: `Ran 3 tests` and `OK`.

- [ ] **Step 6: Commit orchestration changes**

```bash
git add .github/workflows/ci.yaml .github/workflows/pr-test-ppu.yml scripts/ci/utils/tests/test_run_ci_workflows.py
git commit -m "feat(ci): gate PPU full CI on run-ci label"
```

### Task 4: Complete local validation

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run both focused test files**

```bash
python3 scripts/ci/utils/tests/test_run_ci_permissions.py
python3 scripts/ci/utils/tests/test_run_ci_workflows.py
```

Expected: permission suite reports 4 passing tests; workflow suite reports 3 passing tests.

- [ ] **Step 2: Run Python syntax and repository lint checks**

```bash
python3 -m py_compile scripts/ci/utils/slash_command_handler.py scripts/ci/utils/tests/test_run_ci_permissions.py scripts/ci/utils/tests/test_run_ci_workflows.py
pre-commit run --files scripts/ci/utils/slash_command_handler.py scripts/ci/utils/tests/test_run_ci_permissions.py scripts/ci/utils/tests/test_run_ci_workflows.py .github/workflows/ci.yaml .github/workflows/pr-test-ppu.yml .github/workflows/slash-command-handler.yml
```

Expected: both commands exit 0. If `pre-commit` reformats files, rerun focused tests and the same pre-commit command.

- [ ] **Step 3: Parse all changed workflows**

Use the existing local verification environment that contains PyYAML:

```bash
../../../.local-verify-venv/bin/python -c 'import yaml; [yaml.safe_load(open(path)) for path in [".github/workflows/ci.yaml", ".github/workflows/pr-test-ppu.yml", ".github/workflows/slash-command-handler.yml"]]'
```

Expected: exit 0 with no parser errors.

- [ ] **Step 4: Validate the effective authorization rule against GitHub**

Use the authenticated maintainer identity, without changing repository state:

```bash
gh api repos/flytiger-eco/sglang-for-sail/collaborators/$(gh api user --jq .login)/permission --jq .permission
```

Expected: `write`, `maintain`, or `admin`. This confirms the live API response is one of the values accepted by the resolver; it does not add a label or trigger CI.

- [ ] **Step 5: Review the final branch diff**

```bash
git status --short
git diff v0.5.18...HEAD --check
git diff v0.5.18...HEAD -- .github/workflows/ci.yaml .github/workflows/pr-test-ppu.yml .github/workflows/slash-command-handler.yml scripts/ci/utils/slash_command_handler.py scripts/ci/utils/tests docs/superpowers
```

Expected: only the design, plan, workflows, handler, and focused tests are changed; `git diff --check` exits 0.

- [ ] **Step 6: Commit the implementation plan if still uncommitted**

```bash
git add docs/superpowers/plans/2026-10-13-run-ci-label-trigger.md
git commit -m "docs(ci): add run-ci trigger implementation plan"
```
