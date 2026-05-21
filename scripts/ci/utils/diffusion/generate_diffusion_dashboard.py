"""Generate a Markdown dashboard for diffusion cross-framework comparisons.

Reads current comparison results + historical data from sgl-project/ci-data repo
and produces a Markdown report with tables and trend charts saved as PNG files.

Usage:
    python3 scripts/ci/utils/diffusion/generate_diffusion_dashboard.py \
        --results comparison-results.json \
        --output dashboard.md \
        --charts-dir comparison-charts/ \
        --history-dir history/           # optional, local history JSONs
        --fetch-history                  # fetch from GitHub API instead
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# History fetching (from sgl-project/ci-data repo via GitHub API)
# ---------------------------------------------------------------------------

CI_DATA_REPO_OWNER = "sgl-project"
CI_DATA_REPO_NAME = "ci-data"
CI_DATA_BRANCH = "main"
HISTORY_PREFIX = "diffusion-comparisons"
MAX_HISTORY_RUNS = 14

# Base URL for chart images pushed to sgl-project/ci-data
CHARTS_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{CI_DATA_REPO_OWNER}/{CI_DATA_REPO_NAME}"
    f"/{CI_DATA_BRANCH}/{HISTORY_PREFIX}/charts"
)


def _github_get(url: str, token: str) -> dict | list | None:
    """Simple GET to GitHub API."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"  Warning: GitHub API request failed ({e.code}): {url}")
        return None
    except Exception as e:
        print(f"  Warning: GitHub API request error: {e}")
        return None


def fetch_history_from_github(token: str) -> list[dict]:
    """Fetch recent comparison result JSONs from sgl-project/ci-data repo."""
    print("Fetching historical comparison data from GitHub...")
    url = (
        f"https://api.github.com/repos/{CI_DATA_REPO_OWNER}/{CI_DATA_REPO_NAME}"
        f"/contents/{HISTORY_PREFIX}?ref={CI_DATA_BRANCH}"
    )
    listing = _github_get(url, token)
    if not listing or not isinstance(listing, list):
        print("  No historical data found.")
        return []

    # Filter JSON files and sort by name (date prefix) descending
    json_files = sorted(
        [f for f in listing if f["name"].endswith(".json")],
        key=lambda f: f["name"],
        reverse=True,
    )[:MAX_HISTORY_RUNS]

    history = []
    for entry in json_files:
        raw_url = entry.get("download_url")
        if not raw_url:
            continue
        data = _github_get(raw_url, token)
        if data and isinstance(data, dict):
            history.append(data)
    print(f"  Loaded {len(history)} historical run(s).")
    return history


def load_history_from_dir(history_dir: str) -> list[dict]:
    """Load historical JSONs from a local directory."""
    if not os.path.isdir(history_dir):
        return []
    files = sorted(
        [f for f in os.listdir(history_dir) if f.endswith(".json")],
        reverse=True,
    )[:MAX_HISTORY_RUNS]
    history = []
    for fname in files:
        try:
            with open(os.path.join(history_dir, fname)) as f:
                history.append(json.load(f))
        except Exception:
            pass
    return history


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------


def _fmt_latency(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def _fmt_speedup(sglang_lat: float | None, other_lat: float | None) -> str:
    if sglang_lat is None or other_lat is None or sglang_lat <= 0:
        return "N/A"
    ratio = other_lat / sglang_lat
    return f"{ratio:.2f}x"


def _short_date(ts: str) -> str:
    """Extract short date from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d")
    except Exception:
        return ts[:10]


def _short_sha(sha: str) -> str:
    return sha[:7] if sha and sha != "unknown" else "?"


def _assess_risk(
    cid: str,
    current_cases: dict[str, dict[str, float | None]],
    history: list[dict],
    other_frameworks: list[str],
) -> tuple[str, str]:
    """Assess risk for a given case, returning (emoji, reason).

    Rules (checked in order):
    - N/A latency → ❌ broken
    - History exists: SGLang latency >5% vs avg of last 3 runs → ⚠️ regression
    - Competitor exists & SGLang slower → 🔴 competitive risk
    - SGLang faster than all competitors by >20% → 🟢 strong advantage
    - SGLang faster than all competitors by ≤20% → 🟡 moderate advantage
    - Default → ✅ stable
    """
    sg_lat = current_cases.get(cid, {}).get("sglang")

    # Broken: sglang latency is N/A
    if sg_lat is None:
        return "❌", f"{cid}: SGLang latency is N/A (broken)"

    # Check regression against 3-run historical average
    if history:
        hist_lats: list[float] = []
        for run in history[:3]:
            run_cases = _extract_case_results(run)
            h_lat = run_cases.get(cid, {}).get("sglang")
            if h_lat is not None:
                hist_lats.append(h_lat)
        if hist_lats:
            avg_3 = sum(hist_lats) / len(hist_lats)
            if avg_3 > 0 and (sg_lat - avg_3) / avg_3 > 0.05:
                pct = (sg_lat - avg_3) / avg_3 * 100
                return (
                    "⚠️",
                    f"{cid}: SGLang regression +{pct:.1f}% vs 3-run avg "
                    f"({sg_lat:.2f}s vs {avg_3:.2f}s)",
                )

    # Check competitive risk
    if other_frameworks:
        competitor_lats: dict[str, float] = {}
        for ofw in other_frameworks:
            olat = current_cases.get(cid, {}).get(ofw)
            if olat is not None:
                competitor_lats[ofw] = olat

        if competitor_lats:
            # SGLang slower than any competitor?
            for ofw, olat in competitor_lats.items():
                if sg_lat > olat:
                    return (
                        "🔴",
                        f"{cid}: SGLang slower than {ofw} "
                        f"({sg_lat:.2f}s vs {olat:.2f}s)",
                    )

            # SGLang faster — check margin
            min_competitor = min(competitor_lats.values())
            advantage = (min_competitor - sg_lat) / min_competitor
            if advantage > 0.20:
                return "🟢", ""
            else:
                return "🟡", ""

    # Default: stable
    return "✅", ""


def _trend_emoji(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return ""
    diff_pct = (current - previous) / previous * 100
    if diff_pct < -2:
        return " :arrow_down:"  # faster (good)
    elif diff_pct > 2:
        return " :arrow_up:"  # slower (bad)
    return " :left_right_arrow:"


def _extract_case_results(run_data: dict) -> dict[str, dict[str, float | None]]:
    """Extract {case_id: {framework: latency}} from a run."""
    mapping: dict[str, dict[str, float | None]] = {}
    for r in run_data.get("results", []):
        cid = r["case_id"]
        fw = r["framework"]
        if cid not in mapping:
            mapping[cid] = {}
        mapping[cid][fw] = r.get("latency_s")
    return mapping


def _sanitize_filename(name: str) -> str:
    """Sanitize a case ID to be a safe filename."""
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")


def generate_dashboard(
    current: dict,
    history: list[dict],
    charts_dir: str | None = None,
) -> tuple[str, list[str]]:
    """Generate full markdown dashboard.

    Returns (markdown_string, alert_reasons) where alert_reasons is a list of
    human-readable strings for cases that need attention (empty if all is well).

    If charts_dir is provided, saves chart PNGs as files to that directory
    and references them via raw.githubusercontent URLs. Otherwise, charts
    are omitted.

    Returns the markdown string.
    """
    lines: list[str] = []
    lines.append("# Diffusion Cross-Framework Performance Dashboard\n")
    ts = current.get("timestamp", datetime.now(timezone.utc).isoformat())
    sha = current.get("commit_sha", "unknown")
    lines.append(f"*Generated: {_short_date(ts)} | Commit: `{_short_sha(sha)}`*\n")

    current_cases = _extract_case_results(current)
    case_ids = list(current_cases.keys())

    # ---- Regression detection ----
    REGRESSION_THRESHOLD = 0.05  # 5%
    regressions: list[str] = []
    if history:
        prev_cases = _extract_case_results(history[0])
        for cid in case_ids:
            for fw in ("sglang", "vllm-omni"):
                cur = current_cases.get(cid, {}).get(fw)
                prev = prev_cases.get(cid, {}).get(fw)
                if cur and prev and prev > 0:
                    pct = (cur - prev) / prev
                    if pct > REGRESSION_THRESHOLD:
                        regressions.append(
                            f"**{cid}** ({fw}): {prev:.2f}s -> {cur:.2f}s "
                            f"(+{pct*100:.1f}%)"
                        )

    if regressions:
        lines.append("> [!WARNING]\n> **Performance Regression Detected**\n>")
        for reg in regressions:
            lines.append(f"> - {reg}")
        lines.append("\n")

    # Discover all frameworks present in results
    all_frameworks = []
    seen_fw = set()
    for r in current.get("results", []):
        fw = r["framework"]
        if fw not in seen_fw:
            all_frameworks.append(fw)
            seen_fw.add(fw)
    # Ensure sglang is first
    if "sglang" in all_frameworks:
        all_frameworks.remove("sglang")
        all_frameworks.insert(0, "sglang")
    other_frameworks = [fw for fw in all_frameworks if fw != "sglang"]

    # ---- Section 1: Cross-Framework Comparison (current run) ----
    lines.append("## Cross-Framework Performance Comparison\n")

    # Compute risk assessments for all cases
    risk_map: dict[str, tuple[str, str]] = {}
    for cid in case_ids:
        risk_map[cid] = _assess_risk(cid, current_cases, history, other_frameworks)

    # Dynamic header
    header = "| Model | Risk |"
    sep = "|-------|------|"
    for fw in all_frameworks:
        header += f" {fw} (s) |"
        sep += "---------|"
    for ofw in other_frameworks:
        header += f" vs {ofw} |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    # One row per case (deduplicated by case_id)
    seen_cases = set()
    for r in current.get("results", []):
        cid = r["case_id"]
        if cid in seen_cases:
            continue
        seen_cases.add(cid)

        case_fws = current_cases.get(cid, {})
        sg_lat = case_fws.get("sglang")

        risk_emoji, _ = risk_map.get(cid, ("✅", ""))
        row = f"| {r['model'].split('/')[-1]} | {risk_emoji} |"
        # Latency columns -- bold the fastest
        lats = {fw: case_fws.get(fw) for fw in all_frameworks}
        valid_lats = [v for v in lats.values() if v is not None]
        min_lat = min(valid_lats) if valid_lats else None
        for fw in all_frameworks:
            lat = lats[fw]
            if lat is not None and min_lat is not None and lat == min_lat:
                row += f" **{_fmt_latency(lat)}** |"
            else:
                row += f" {_fmt_latency(lat)} |"
        # Speedup columns
        for ofw in other_frameworks:
            row += f" {_fmt_speedup(sg_lat, case_fws.get(ofw))} |"
        lines.append(row)

    # ---- Section 2: Cross-Framework Speedup Trend (only if multiple frameworks) ----
    if history and other_frameworks:
        lines.append("\n## SGLang vs vLLM-Omni Speedup Over Time\n")

        header = "| Date |"
        sep = "|------|"
        for cid in case_ids:
            header += f" {cid} |"
            sep += "---------|"
        lines.append(header)
        lines.append(sep)

        all_runs = [current] + history
        for run in all_runs:
            run_cases = _extract_case_results(run)
            date = _short_date(run.get("timestamp", ""))
            row = f"| {date} |"
            for cid in case_ids:
                sg = run_cases.get(cid, {}).get("sglang")
                vl = run_cases.get(cid, {}).get("vllm-omni")
                row += f" {_fmt_speedup(sg, vl)} |"
            lines.append(row)

    # ---- Section 4: Matplotlib Trend Charts (saved as PNG files) ----
    if history and charts_dir:
        all_runs = list(reversed([current] + history))  # chronological order

        def _chart_label(run: dict) -> str:
            d = _short_date(run.get("timestamp", ""))
            s = _short_sha(run.get("commit_sha", ""))
            return f"{d}\n({s})"

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(charts_dir, exist_ok=True)

            # Per-case latency trend charts
            for cid in case_ids:
                labels = []
                sg_vals = []
                vl_vals = []
                for run in all_runs:
                    run_cases = _extract_case_results(run)
                    sg = run_cases.get(cid, {}).get("sglang")
                    vl = run_cases.get(cid, {}).get("vllm-omni")
                    if sg is None:
                        continue
                    labels.append(_chart_label(run))
                    sg_vals.append(sg)
                    vl_vals.append(vl)

                if not sg_vals:
                    continue

                has_vl = any(v is not None for v in vl_vals)
                fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))

                # SGLang line
                ax.plot(
                    range(len(sg_vals)),
                    sg_vals,
                    "o-",
                    color="#2563eb",
                    linewidth=2,
                    markersize=6,
                    label="SGLang",
                )
                for i, v in enumerate(sg_vals):
                    ax.annotate(
                        f"{v:.2f}s",
                        (i, v),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                        color="#2563eb",
                    )

                # vLLM-Omni line (if data exists)
                if has_vl:
                    vl_clean = [v if v is not None else float("nan") for v in vl_vals]
                    ax.plot(
                        range(len(vl_clean)),
                        vl_clean,
                        "s--",
                        color="#dc2626",
                        linewidth=2,
                        markersize=5,
                        label="vLLM-Omni",
                    )
                    for i, v in enumerate(vl_vals):
                        if v is not None:
                            ax.annotate(
                                f"{v:.2f}s",
                                (i, v),
                                textcoords="offset points",
                                xytext=(0, -14),
                                ha="center",
                                fontsize=8,
                                color="#dc2626",
                            )

                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=7)
                ax.set_ylabel("Latency (s)")
                ax.set_title(f"Latency Trend -- {cid}", fontsize=11, fontweight="bold")
                ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
                ax.grid(True, alpha=0.3)
                all_vals = sg_vals + [v for v in vl_vals if v is not None]
                y_min = min(all_vals)
                y_max = max(all_vals)
                y_range = y_max - y_min if y_max > y_min else max(y_max * 0.1, 0.1)
                ax.set_ylim(
                    bottom=max(0, y_min - y_range * 0.3),
                    top=y_max + y_range * 0.3,
                )

                filename = f"latency_{_sanitize_filename(cid)}.png"
                chart_path = os.path.join(charts_dir, filename)
                fig.savefig(chart_path, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved chart: {chart_path}")

                chart_url = f"{CHARTS_RAW_BASE_URL}/{filename}"
                lines.append(f"\n### Latency Trend: {cid}\n")
                lines.append(f"![Latency Trend {cid}]({chart_url})\n")

            # Speedup trend chart (only if multiple frameworks)
            if other_frameworks:
                fig, ax = plt.subplots(figsize=(max(6, len(all_runs) * 1.2), 4))
                colors = ["#2563eb", "#dc2626", "#16a34a", "#ea580c"]
                for ci_idx, cid in enumerate(case_ids):
                    speedups = []
                    run_labels = []
                    for run in all_runs:
                        run_cases = _extract_case_results(run)
                        sg = run_cases.get(cid, {}).get("sglang")
                        vl = run_cases.get(cid, {}).get("vllm-omni")
                        if sg and vl and sg > 0:
                            speedups.append(vl / sg)
                        else:
                            speedups.append(None)
                        run_labels.append(_chart_label(run))
                    clean = [v if v is not None else float("nan") for v in speedups]
                    ax.plot(
                        range(len(clean)),
                        clean,
                        "o-",
                        color=colors[ci_idx % len(colors)],
                        linewidth=2,
                        markersize=5,
                        label=cid,
                    )

                ax.set_xticks(range(len(run_labels)))
                ax.set_xticklabels(run_labels, fontsize=7)
                ax.set_ylabel("Speedup (x)")
                ax.set_title(
                    "SGLang Speedup Over vLLM-Omni", fontsize=11, fontweight="bold"
                )
                ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
                ax.legend(loc="upper left", fontsize=7)
                ax.grid(True, alpha=0.3)

                filename = "speedup_trend.png"
                chart_path = os.path.join(charts_dir, filename)
                fig.savefig(chart_path, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved chart: {chart_path}")

                chart_url = f"{CHARTS_RAW_BASE_URL}/{filename}"
                lines.append("\n### Speedup Trend (SGLang vs vLLM-Omni)\n")
                lines.append(f"![Speedup Trend]({chart_url})\n")

        except ImportError:
            lines.append("\n*Charts unavailable (matplotlib not installed)*\n")

    # ---- SGLang Performance Trend (raw data table, at the end) ----
    if history:
        lines.append(f"\n## SGLang Performance Trend (Last {len(history) + 1} Runs)\n")

        header = "| Date | Commit |"
        sep = "|------|--------|"
        for cid in case_ids:
            header += f" {cid} (s) |"
            sep += "---------|"
        header += " Trend |"
        sep += "-------|"
        lines.append(header)
        lines.append(sep)

        all_runs = [current] + history
        for i, run in enumerate(all_runs):
            run_cases = _extract_case_results(run)
            date = _short_date(run.get("timestamp", ""))
            sha_s = _short_sha(run.get("commit_sha", ""))
            row = f"| {date} | `{sha_s}` |"
            for cid in case_ids:
                lat = run_cases.get(cid, {}).get("sglang")
                row += f" {_fmt_latency(lat)} |"
            if i + 1 < len(all_runs):
                prev_cases = _extract_case_results(all_runs[i + 1])
                emojis = []
                for cid in case_ids:
                    cur = run_cases.get(cid, {}).get("sglang")
                    prev = prev_cases.get(cid, {}).get("sglang")
                    emojis.append(_trend_emoji(cur, prev))
                row += " ".join(emojis) + " |"
            else:
                row += " -- |"
            lines.append(row)

    # ---- Risk Notification ----
    alert_cases = [
        (cid, emoji, reason)
        for cid, (emoji, reason) in risk_map.items()
        if emoji in ("⚠️", "🔴", "❌")
    ]
    if alert_cases:
        lines.append("\n> [!CAUTION]")
        lines.append("> **Action Required — Performance Alert**")
        lines.append(">")
        lines.append("> The following cases need attention:")
        for _cid, _emoji, reason in alert_cases:
            lines.append(f"> - {reason}")
        lines.append("")

    # Footer
    lines.append("\n---")
    lines.append(
        "*Generated by `generate_diffusion_dashboard.py` in SGLang nightly CI.*"
    )

    alert_reasons = [reason for _, _, reason in alert_cases]
    return "\n".join(lines) + "\n", alert_reasons


ALERT_ASSIGNEES = ["mickqian", "bbuf", "yhyang201"]
ALERT_LABEL = "perf-regression"


ALERT_ISSUE_TITLE = "[Diffusion CI] Performance regression tracker"


def _find_alert_issue(repo: str) -> tuple[str | None, bool]:
    """Find the perf-regression tracker issue (open OR closed).

    Returns (issue_number, is_open).  Prefers an open issue; if none,
    returns the most recent closed one so it can be reopened.
    """
    import subprocess

    for state in ("open", "closed"):
        result = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                repo,
                "--label",
                ALERT_LABEL,
                "--state",
                state,
                "--json",
                "number",
                "--limit",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            continue
        issues = json.loads(result.stdout)
        if issues:
            return str(issues[0]["number"]), state == "open"
    return None, False


def _create_alert_issue(alert_reasons: list[str]) -> None:
    """Create or update the single perf-regression tracker issue.

    Logic:
    - If an open issue exists  → add a comment with the new alert.
    - If a closed issue exists → reopen it, then add a comment.
    - If no issue exists       → create one.

    This guarantees at most one tracker issue ever exists.

    Uses `gh` (GitHub CLI) which is available in all GitHub Actions runners.
    Falls back silently outside CI.
    """
    import subprocess

    run_url = ""
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "sgl-project/sglang")
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    if run_id:
        run_url = f"{server_url}/{repo}/actions/runs/{run_id}"

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    body_lines = [
        f"## Performance Alert — {date}",
        "",
        "The nightly diffusion benchmark detected the following issue(s):",
        "",
    ]
    for reason in alert_reasons:
        body_lines.append(f"- {reason}")
    if run_url:
        body_lines += ["", f"**CI Run:** {run_url}"]
    body = "\n".join(body_lines)

    try:
        existing, is_open = _find_alert_issue(repo)

        if existing:
            # Reopen if closed
            if not is_open:
                subprocess.run(
                    [
                        "gh",
                        "issue",
                        "reopen",
                        existing,
                        "--repo",
                        repo,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                print(f"Reopened alert issue #{existing}")

            # Add comment
            result = subprocess.run(
                [
                    "gh",
                    "issue",
                    "comment",
                    existing,
                    "--repo",
                    repo,
                    "--body",
                    body,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print(f"Commented on alert issue #{existing}")
            else:
                print(
                    f"Warning: failed to comment on issue #{existing} "
                    f"(rc={result.returncode}): {result.stderr.strip()}"
                )
        else:
            # Create a new issue
            cmd = [
                "gh",
                "issue",
                "create",
                "--repo",
                repo,
                "--title",
                ALERT_ISSUE_TITLE,
                "--body",
                body,
                "--label",
                ALERT_LABEL,
            ]
            for user in ALERT_ASSIGNEES:
                cmd += ["--assignee", user]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"Created alert issue: {result.stdout.strip()}")
            else:
                print(
                    f"Warning: failed to create alert issue "
                    f"(rc={result.returncode}): {result.stderr.strip()}"
                )
    except FileNotFoundError:
        print("Warning: `gh` CLI not found — skipping alert issue creation")
    except Exception as e:
        print(f"Warning: failed to create/update alert issue: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate diffusion cross-framework comparison dashboard"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to comparison-results.json from current run",
    )
    parser.add_argument(
        "--output",
        default="dashboard.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--charts-dir",
        default="comparison-charts",
        help="Directory to save chart PNG files (default: comparison-charts/)",
    )
    parser.add_argument(
        "--history-dir",
        default=None,
        help="Local directory containing historical comparison JSONs",
    )
    parser.add_argument(
        "--fetch-history",
        action="store_true",
        help="Fetch history from ci-data GitHub repo",
    )
    parser.add_argument(
        "--step-summary",
        action="store_true",
        help="Also write to $GITHUB_STEP_SUMMARY",
    )

    args = parser.parse_args()

    # Load current results
    with open(args.results) as f:
        current = json.load(f)
    print(f"Loaded current results: {len(current.get('results', []))} entries")

    # Load history
    history: list[dict] = []
    if args.fetch_history:
        token = os.environ.get("GH_PAT_FOR_NIGHTLY_CI_DATA") or os.environ.get(
            "GITHUB_TOKEN"
        )
        if token:
            history = fetch_history_from_github(token)
        else:
            print("Warning: No GitHub token available, skipping history fetch")
    elif args.history_dir:
        history = load_history_from_dir(args.history_dir)
        print(f"Loaded {len(history)} historical run(s) from {args.history_dir}")

    # Generate dashboard
    markdown, alert_reasons = generate_dashboard(
        current, history, charts_dir=args.charts_dir
    )

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(markdown)
    print(f"Dashboard written to {args.output}")

    # Write to GitHub Step Summary
    if args.step_summary:
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a") as f:
                f.write(markdown)
            print("Dashboard appended to $GITHUB_STEP_SUMMARY")
        else:
            print("Warning: $GITHUB_STEP_SUMMARY not set, skipping")

    # Create GitHub Issue for performance alerts (so assignees get notified)
    if alert_reasons:
        _create_alert_issue(alert_reasons)
    else:
        print("No performance alerts — skipping issue creation.")


if __name__ == "__main__":
    main()
