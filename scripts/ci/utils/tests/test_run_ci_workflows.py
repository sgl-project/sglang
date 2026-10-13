import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


class RunCiWorkflowContractTest(unittest.TestCase):
    def read_workflow(self, name):
        return (ROOT / ".github/workflows" / name).read_text()

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


if __name__ == "__main__":
    unittest.main()
