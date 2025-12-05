"""
Unit tests for slash_command_handler.py

These tests verify the /rerun-stage command functionality, including:
- Permission checks
- Stage name validation
- Correct API calls (pr.create_issue_comment vs comment.create_comment)
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock the github module before importing slash_command_handler
sys.modules["github"] = MagicMock()
sys.modules["github.Auth"] = MagicMock()
sys.modules["github.Github"] = MagicMock()

from slash_command_handler import handle_rerun_stage


class TestHandleRerunStage(unittest.TestCase):
    """Tests for the handle_rerun_stage function."""

    def setUp(self):
        """Set up mock objects for each test."""
        self.mock_gh_repo = MagicMock()
        self.mock_pr = MagicMock()
        self.mock_comment = MagicMock()

        # Set up workflow mock
        self.mock_workflow = MagicMock()
        self.mock_workflow.name = "PR Test"
        self.mock_workflow.create_dispatch.return_value = True
        self.mock_gh_repo.get_workflows.return_value = [self.mock_workflow]
        self.mock_gh_repo.full_name = "sgl-project/sglang"

        # Set up PR mock
        self.mock_pr.head.ref = "test-branch"

    def test_permission_denied(self):
        """Test that function returns False when user lacks permission."""
        user_perms = {"can_rerun_stage": False}

        result = handle_rerun_stage(
            self.mock_gh_repo,
            self.mock_pr,
            self.mock_comment,
            user_perms,
            "unit-test-backend-4-gpu",
        )

        self.assertFalse(result)
        # Should not create any reactions or comments
        self.mock_comment.create_reaction.assert_not_called()
        self.mock_pr.create_issue_comment.assert_not_called()

    def test_no_stage_name_provided(self):
        """Test error handling when no stage name is provided."""
        user_perms = {"can_rerun_stage": True}

        result = handle_rerun_stage(
            self.mock_gh_repo,
            self.mock_pr,
            self.mock_comment,
            user_perms,
            None,  # No stage name
        )

        self.assertFalse(result)
        self.mock_comment.create_reaction.assert_called_once_with("confused")
        # Verify pr.create_issue_comment is called (not comment.create_comment)
        self.mock_pr.create_issue_comment.assert_called_once()
        call_args = self.mock_pr.create_issue_comment.call_args[0][0]
        self.assertIn("Please specify a stage name", call_args)

    def test_invalid_stage_name(self):
        """Test error handling when an invalid stage name is provided."""
        user_perms = {"can_rerun_stage": True}

        result = handle_rerun_stage(
            self.mock_gh_repo,
            self.mock_pr,
            self.mock_comment,
            user_perms,
            "invalid-stage-name",
        )

        self.assertFalse(result)
        self.mock_comment.create_reaction.assert_called_once_with("confused")
        # Verify pr.create_issue_comment is called (not comment.create_comment)
        self.mock_pr.create_issue_comment.assert_called_once()
        call_args = self.mock_pr.create_issue_comment.call_args[0][0]
        self.assertIn("doesn't support isolated runs", call_args)

    def test_valid_stage_success(self):
        """Test successful workflow dispatch for a valid stage."""
        user_perms = {"can_rerun_stage": True}

        result = handle_rerun_stage(
            self.mock_gh_repo,
            self.mock_pr,
            self.mock_comment,
            user_perms,
            "unit-test-backend-4-gpu",
        )

        self.assertTrue(result)
        self.mock_comment.create_reaction.assert_called_once_with("+1")
        # Verify pr.create_issue_comment is called with success message
        self.mock_pr.create_issue_comment.assert_called_once()
        call_args = self.mock_pr.create_issue_comment.call_args[0][0]
        self.assertIn("Triggered", call_args)
        self.assertIn("unit-test-backend-4-gpu", call_args)

        # Verify workflow dispatch was called correctly
        self.mock_workflow.create_dispatch.assert_called_once_with(
            ref="test-branch",
            inputs={"version": "release", "target_stage": "unit-test-backend-4-gpu"},
        )

    def test_valid_stage_no_reaction(self):
        """Test successful workflow dispatch without reaction."""
        user_perms = {"can_rerun_stage": True}

        result = handle_rerun_stage(
            self.mock_gh_repo,
            self.mock_pr,
            self.mock_comment,
            user_perms,
            "unit-test-backend-4-gpu",
            react_on_success=False,
        )

        self.assertTrue(result)
        # Should not create reaction or comment when react_on_success=False
        self.mock_comment.create_reaction.assert_not_called()
        self.mock_pr.create_issue_comment.assert_not_called()

    def test_workflow_dispatch_failure(self):
        """Test handling when workflow dispatch fails."""
        user_perms = {"can_rerun_stage": True}
        self.mock_workflow.create_dispatch.return_value = False

        result = handle_rerun_stage(
            self.mock_gh_repo,
            self.mock_pr,
            self.mock_comment,
            user_perms,
            "unit-test-backend-4-gpu",
        )

        self.assertFalse(result)

    def test_workflow_dispatch_exception(self):
        """Test handling when workflow dispatch raises an exception."""
        user_perms = {"can_rerun_stage": True}
        self.mock_workflow.create_dispatch.side_effect = Exception("API Error")

        result = handle_rerun_stage(
            self.mock_gh_repo,
            self.mock_pr,
            self.mock_comment,
            user_perms,
            "unit-test-backend-4-gpu",
        )

        self.assertFalse(result)
        self.mock_comment.create_reaction.assert_called_once_with("confused")
        # Verify pr.create_issue_comment is called with error message
        self.mock_pr.create_issue_comment.assert_called_once()
        call_args = self.mock_pr.create_issue_comment.call_args[0][0]
        self.assertIn("Failed to trigger workflow", call_args)

    def test_all_valid_stages(self):
        """Test that all documented valid stages are accepted."""
        user_perms = {"can_rerun_stage": True}

        valid_stages = [
            "stage-a-test-1",
            "multimodal-gen-test-1-gpu",
            "multimodal-gen-test-2-gpu",
            "quantization-test",
            "unit-test-backend-1-gpu",
            "unit-test-backend-2-gpu",
            "unit-test-backend-4-gpu",
            "unit-test-backend-8-gpu-h200",
            "unit-test-backend-8-gpu-h20",
            "performance-test-1-gpu-part-1",
            "performance-test-1-gpu-part-2",
            "performance-test-1-gpu-part-3",
            "performance-test-2-gpu",
            "accuracy-test-1-gpu",
            "accuracy-test-2-gpu",
            "unit-test-deepep-4-gpu",
            "unit-test-deepep-8-gpu",
            "unit-test-backend-4-gpu-b200",
            "unit-test-backend-4-gpu-gb200",
        ]

        for stage in valid_stages:
            # Reset mocks for each iteration
            self.mock_comment.reset_mock()
            self.mock_pr.reset_mock()
            self.mock_workflow.create_dispatch.return_value = True

            result = handle_rerun_stage(
                self.mock_gh_repo,
                self.mock_pr,
                self.mock_comment,
                user_perms,
                stage,
                react_on_success=False,  # Don't check reactions, just success
            )

            self.assertTrue(
                result, f"Stage '{stage}' should be valid but returned False"
            )


class TestCommentAPIUsage(unittest.TestCase):
    """
    Tests to verify the correct API is used for creating comments.

    The fix in this PR changes from:
        comment.create_comment(...)  # WRONG - IssueComment doesn't have this method
    To:
        pr.create_issue_comment(...)  # CORRECT - PullRequest has this method
    """

    def test_error_comment_uses_pr_api(self):
        """Verify error comments use pr.create_issue_comment, not comment.create_comment."""
        mock_gh_repo = MagicMock()
        mock_pr = MagicMock()
        mock_comment = MagicMock()

        user_perms = {"can_rerun_stage": True}

        # Trigger an error by providing an invalid stage
        handle_rerun_stage(
            mock_gh_repo,
            mock_pr,
            mock_comment,
            user_perms,
            "invalid-stage",
        )

        # Verify pr.create_issue_comment was called
        mock_pr.create_issue_comment.assert_called_once()

        # Verify comment.create_comment was NOT called (this was the bug)
        # Note: create_comment doesn't exist on IssueComment, so this checks
        # that we're not trying to call a non-existent method
        if hasattr(mock_comment, "create_comment"):
            mock_comment.create_comment.assert_not_called()

    def test_success_comment_uses_pr_api(self):
        """Verify success comments use pr.create_issue_comment."""
        mock_gh_repo = MagicMock()
        mock_pr = MagicMock()
        mock_comment = MagicMock()

        mock_workflow = MagicMock()
        mock_workflow.name = "PR Test"
        mock_workflow.create_dispatch.return_value = True
        mock_gh_repo.get_workflows.return_value = [mock_workflow]
        mock_gh_repo.full_name = "sgl-project/sglang"
        mock_pr.head.ref = "test-branch"

        user_perms = {"can_rerun_stage": True}

        handle_rerun_stage(
            mock_gh_repo,
            mock_pr,
            mock_comment,
            user_perms,
            "unit-test-backend-4-gpu",
        )

        # Verify pr.create_issue_comment was called
        mock_pr.create_issue_comment.assert_called_once()


if __name__ == "__main__":
    unittest.main()
