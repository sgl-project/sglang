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
