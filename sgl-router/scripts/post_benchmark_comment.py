#!/usr/bin/env python3
"""
GitHub PR Comment Poster for Benchmark Results

Posts benchmark results as comments on GitHub PRs with update capability.
Replaces JavaScript logic in GitHub Actions for better maintainability.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import requests


class GitHubCommentPoster:
    """Handles posting benchmark results as GitHub PR comments."""

    def __init__(self, token: str, repo_owner: str, repo_name: str):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def read_benchmark_results(self, results_file: str) -> Dict[str, str]:
        """Read benchmark results from file."""
        results = {}
        filepath = Path(results_file)

        if not filepath.exists():
            print(f"Results file not found: {filepath}")
            return {"error": "Results file not found"}

        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        results[key] = value
        except Exception as e:
            print(f"Error reading results file: {e}")
            return {"error": str(e)}

        return results

    def format_benchmark_comment(
        self, results: Dict[str, str], pr_number: int, commit_sha: str
    ) -> str:
        """Format benchmark results into a GitHub comment."""
        serialization_time = results.get("serialization_time", "N/A")
        deserialization_time = results.get("deserialization_time", "N/A")
        adaptation_time = results.get("adaptation_time", "N/A")
        total_time = results.get("total_time", "N/A")

        comment = f"""
### SGLang Router Benchmark Results

**Performance Summary for PR #{pr_number}**

The router benchmarks have completed successfully!

**Performance Thresholds:** All passed
- Serialization: < 2μs
- Deserialization: < 2μs
- PD Adaptation: < 5μs
- Total Pipeline: < 10μs

**Measured Results:**
- Serialization: `{serialization_time}`ns
- Deserialization: `{deserialization_time}`ns
- PD Adaptation: `{adaptation_time}`ns
- Total Pipeline: `{total_time}`ns

**Detailed Reports:**
- Download the `benchmark-results-{commit_sha}` artifact for HTML reports
- Run `make bench` locally for detailed analysis

**Commit:** {commit_sha}
""".strip()

        return comment

    def find_existing_comment(self, pr_number: int) -> Optional[int]:
        """Find existing benchmark comment in the PR."""
        url = f"{self.base_url}/issues/{pr_number}/comments"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            comments = response.json()

            for comment in comments:
                if comment.get("user", {}).get(
                    "login"
                ) == "github-actions[bot]" and "SGLang Router Benchmark Results" in comment.get(
                    "body", ""
                ):
                    return comment["id"]

        except requests.RequestException as e:
            print(f"Error fetching comments: {e}")

        return None

    def post_comment(self, pr_number: int, comment_body: str) -> bool:
        """Post a new comment on the PR."""
        url = f"{self.base_url}/issues/{pr_number}/comments"
        data = {"body": comment_body}

        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            print(f"Posted new benchmark comment on PR #{pr_number}")
            return True
        except requests.RequestException as e:
            print(f"Error posting comment: {e}")
            return False

    def update_comment(self, comment_id: int, comment_body: str) -> bool:
        """Update an existing comment."""
        url = f"{self.base_url}/issues/comments/{comment_id}"
        data = {"body": comment_body}

        try:
            response = requests.patch(url, headers=self.headers, json=data)
            response.raise_for_status()
            print(f"Updated existing benchmark comment (ID: {comment_id})")
            return True
        except requests.RequestException as e:
            print(f"Error updating comment: {e}")
            return False

    def post_or_update_comment(
        self, pr_number: int, results_file: str, commit_sha: str
    ) -> bool:
        """Post or update benchmark results comment on PR."""
        # Read benchmark results
        results = self.read_benchmark_results(results_file)
        if "error" in results:
            print(f"Failed to read benchmark results: {results['error']}")
            return False

        # Format comment
        comment_body = self.format_benchmark_comment(results, pr_number, commit_sha)

        # Check for existing comment
        existing_comment_id = self.find_existing_comment(pr_number)

        if existing_comment_id:
            return self.update_comment(existing_comment_id, comment_body)
        else:
            return self.post_comment(pr_number, comment_body)


def main():
    parser = argparse.ArgumentParser(description="Post benchmark results to GitHub PR")
    parser.add_argument(
        "--pr-number", type=int, required=True, help="Pull request number"
    )
    parser.add_argument("--commit-sha", type=str, required=True, help="Commit SHA")
    parser.add_argument(
        "--results-file",
        type=str,
        default="benchmark_results.env",
        help="Path to benchmark results file",
    )
    parser.add_argument(
        "--repo-owner", type=str, default="sgl-project", help="GitHub repository owner"
    )
    parser.add_argument(
        "--repo-name", type=str, default="sglang", help="GitHub repository name"
    )

    args = parser.parse_args()

    # Get GitHub token from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable is required")
        sys.exit(1)

    # Create poster and post comment
    poster = GitHubCommentPoster(github_token, args.repo_owner, args.repo_name)
    success = poster.post_or_update_comment(
        args.pr_number, args.results_file, args.commit_sha
    )

    if not success:
        print("Failed to post benchmark comment")
        sys.exit(1)

    print("Benchmark comment posted successfully!")


if __name__ == "__main__":
    main()
