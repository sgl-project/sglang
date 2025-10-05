#!/usr/bin/env python3
"""
SGLang CI Flaky Jobs Analyzer
Detects jobs that fail but succeed on retry with the same commit
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import requests


class FlakyJobAnalyzer:
    """Analyzer for detecting flaky CI jobs"""

    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"
        self.repo = "sgl-project/sglang"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SGLang-CI-Analyzer/1.0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_recent_runs(self, limit: int = 100) -> List[Dict]:
        """Get recent CI run data"""
        print(f"Fetching {limit} recent CI runs...")

        all_runs = []
        page = 1
        per_page = 100

        while len(all_runs) < limit:
            url = f"{self.base_url}/repos/{self.repo}/actions/runs"
            params = {"per_page": min(per_page, limit - len(all_runs)), "page": page}

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("workflow_runs"):
                    break

                all_runs.extend(data["workflow_runs"])
                print(f"Fetched {len(all_runs)} runs so far...")

                if len(data["workflow_runs"]) < per_page:
                    break

                page += 1
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching CI data: {e}")
                break

        return all_runs[:limit]

    def _get_job_details(self, run_id: int) -> List[Dict]:
        """Get job details for a specific run"""
        url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/jobs"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json().get("jobs", [])
        except:
            return []

    def analyze_flaky_jobs(self, runs: List[Dict]) -> Dict:
        """Analyze jobs to find flaky patterns (fail then succeed on retry)"""
        print("Analyzing flaky job patterns...")

        # Group runs by commit SHA
        runs_by_sha = defaultdict(list)
        for run in runs:
            sha = run.get("head_sha")
            if sha:
                runs_by_sha[sha].append(run)

        # Track job outcomes per SHA
        job_outcomes_by_sha = defaultdict(lambda: defaultdict(list))
        job_run_info = defaultdict(lambda: defaultdict(list))

        total_runs = len(runs)
        for i, run in enumerate(runs, 1):
            if i % max(1, min(50, total_runs // 10)) == 0 or i == total_runs:
                progress = (i / total_runs) * 100
                print(f"Progress: {i}/{total_runs} ({progress:.1f}%)")

            sha = run.get("head_sha")
            run_id = run.get("id")
            run_number = run.get("run_number")
            run_attempt = run.get("run_attempt", 1)
            created_at = run.get("created_at")
            run_url = f"https://github.com/{self.repo}/actions/runs/{run_id}"

            jobs = self._get_job_details(run_id)

            for job in jobs:
                job_name = job.get("name", "Unknown")
                job_conclusion = job.get("conclusion")

                # Filter out non-specific CI jobs
                if job_name in [
                    "check-changes",
                    "pr-test-finish",
                    "pr-test-h20-finish",
                    "lint",
                ]:
                    continue

                if job_conclusion in ["success", "failure"]:
                    job_outcomes_by_sha[sha][job_name].append(job_conclusion)
                    job_run_info[sha][job_name].append(
                        {
                            "run_id": run_id,
                            "run_number": run_number,
                            "run_attempt": run_attempt,
                            "conclusion": job_conclusion,
                            "created_at": created_at,
                            "url": run_url,
                        }
                    )

            time.sleep(0.1)

        # Find flaky jobs
        flaky_stats = {
            "flaky_jobs": defaultdict(
                lambda: {"flaky_count": 0, "total_count": 0, "examples": []}
            ),
            "total_shas_analyzed": len(runs_by_sha),
        }

        for sha, jobs_dict in job_outcomes_by_sha.items():
            for job_name, outcomes in jobs_dict.items():
                # Track total runs (success + failure) for this job
                flaky_stats["flaky_jobs"][job_name]["total_count"] += 1

                # Check if this job has both failures and successes for the same SHA
                has_failure = "failure" in outcomes
                has_success = "success" in outcomes

                if has_failure and has_success:
                    flaky_stats["flaky_jobs"][job_name]["flaky_count"] += 1

                    # Store example (limit to 5 per job)
                    if len(flaky_stats["flaky_jobs"][job_name]["examples"]) < 5:
                        run_infos = job_run_info[sha][job_name]
                        flaky_stats["flaky_jobs"][job_name]["examples"].append(
                            {
                                "sha": sha,
                                "outcomes": outcomes,
                                "runs": sorted(
                                    run_infos, key=lambda x: x["created_at"]
                                ),
                            }
                        )

        return flaky_stats

    def generate_report(self, stats: Dict):
        """Generate flaky jobs report"""
        print("\n" + "=" * 60)
        print("SGLang CI Flaky Jobs Report")
        print("=" * 60)

        print(f"\nTotal commits analyzed: {stats['total_shas_analyzed']}")

        flaky_jobs = stats["flaky_jobs"]
        if not flaky_jobs:
            print("\nNo flaky jobs detected!")
            return

        # Filter to only jobs that have flaky runs
        flaky_only = {k: v for k, v in flaky_jobs.items() if v["flaky_count"] > 0}

        print(f"\nFlaky Jobs Detected: {len(flaky_only)}")
        print("\nJobs that failed but succeeded on retry (sorted by flaky percentage):")

        # Sort by flakiness percentage
        sorted_flaky = sorted(
            flaky_only.items(),
            key=lambda x: x[1]["flaky_count"] / x[1]["total_count"],
            reverse=True,
        )

        for i, (job_name, data) in enumerate(sorted_flaky, 1):
            flaky_count = data["flaky_count"]
            total_count = data["total_count"]
            flaky_percentage = (
                (flaky_count / total_count * 100) if total_count > 0 else 0
            )
            print(f"\n{i:2d}. {job_name}")
            print(
                f"    Flaky on {flaky_count}/{total_count} commits ({flaky_percentage:.1f}%)"
            )

            # Show examples
            print("    Examples:")
            for example in data["examples"]:
                sha_short = example["sha"][:7]
                print(f"      Commit {sha_short}:")

                for run_info in example["runs"]:
                    created_at = datetime.fromisoformat(
                        run_info["created_at"].replace("Z", "+00:00")
                    )
                    conclusion_emoji = (
                        "✓" if run_info["conclusion"] == "success" else "✗"
                    )
                    print(
                        f"        {conclusion_emoji} Run #{run_info['run_number']} (attempt {run_info['run_attempt']}) - "
                        f"{run_info['conclusion']} ({created_at.strftime('%Y-%m-%d %H:%M')}): {run_info['url']}"
                    )

        print("\n" + "=" * 60)

    def save_report(self, stats: Dict, output_file: str = "flaky_jobs_analysis.json"):
        """Save detailed report to file"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="SGLang CI Flaky Jobs Analyzer")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of runs to analyze (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="flaky_jobs_analysis.json",
        help="Output file (default: flaky_jobs_analysis.json)",
    )

    args = parser.parse_args()

    analyzer = FlakyJobAnalyzer(args.token)

    try:
        runs = analyzer.get_recent_runs(args.limit)

        if not runs:
            print("No CI run data found")
            return

        stats = analyzer.analyze_flaky_jobs(runs)
        analyzer.generate_report(stats)
        analyzer.save_report(stats, args.output)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
