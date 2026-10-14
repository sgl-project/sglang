import re
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


class RunCiWorkflowContractTest(unittest.TestCase):
    def read_workflow(self, name):
        return (ROOT / ".github/workflows" / name).read_text()

    def concurrency_group(
        self,
        event_name="pull_request",
        action="synchronize",
        run_id=100,
        **event_fields,
    ):
        github = SimpleNamespace(
            event_name=event_name,
            run_id=run_id,
            event=SimpleNamespace(
                pull_request=SimpleNamespace(
                    number=42, labels=[SimpleNamespace(name="run-ci")]
                ),
                action=action,
                **event_fields,
            ),
        )
        workflow = self.read_workflow("ci.yaml")
        self.assertIn("  cancel-in-progress: true", workflow)
        group = re.search(r"(?m)^  group: >-\n((?:    .+\n)+)", workflow)
        self.assertIsNotNone(group)
        template = " ".join(line.strip() for line in group[1].splitlines())

        def evaluate(match):
            # Evaluate only the limited expressions in the trusted local workflow.
            expression = match[1].strip().replace("&&", "and").replace("||", "or")
            return str(eval(expression, {"__builtins__": {}}, {"github": github}))

        return re.sub(r"\$\{\{(.*?)\}\}", evaluate, template)

    def test_reviews_cannot_cancel_active_pr_ci(self):
        active = self.concurrency_group()
        for state in ("approved", "commented", "changes_requested"):
            for reviewer_type in ("User", "Bot"):
                with self.subTest(state=state, reviewer_type=reviewer_type):
                    group = self.concurrency_group(
                        event_name="pull_request_review",
                        action="submitted",
                        run_id=101,
                        review=SimpleNamespace(
                            state=state, user=SimpleNamespace(type=reviewer_type)
                        ),
                    )
                    self.assertNotEqual(group, active)
                    self.assertEqual(group, "ci-42-101")

    def test_unrelated_labels_cannot_cancel_active_pr_ci(self):
        active = self.concurrency_group()
        for run_id in (201, 202):
            with self.subTest(run_id=run_id):
                group = self.concurrency_group(
                    action="labeled",
                    run_id=run_id,
                    label=SimpleNamespace(name="documentation"),
                )
                self.assertNotEqual(group, active)
                self.assertEqual(group, f"ci-42-{run_id}")

    def test_active_pr_events_share_concurrency_group(self):
        active = self.concurrency_group()
        self.assertEqual(active, "ci-42-active")
        for run_id, action in enumerate(
            ("opened", "synchronize", "reopened", "ready_for_review", "labeled"),
            start=301,
        ):
            with self.subTest(action=action):
                event_fields = (
                    {"label": SimpleNamespace(name="run-ci")}
                    if action == "labeled"
                    else {}
                )
                self.assertEqual(
                    self.concurrency_group(
                        action=action, run_id=run_id, **event_fields
                    ),
                    active,
                )

    def test_slash_handler_executes_default_branch_code(self):
        workflow = self.read_workflow("slash-command-handler.yml")

        self.assertIn("ref: ${{ github.event.repository.default_branch }}", workflow)
        self.assertNotIn("refs/pull/", workflow)
        self.assertNotIn("headRefName", workflow)

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
            "github.event.action != 'labeled' || github.event.label.name == 'run-ci'",
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


if __name__ == "__main__":
    unittest.main()
