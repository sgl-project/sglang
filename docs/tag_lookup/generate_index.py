import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

# Short hash length for commits (7 is git's default short hash)
SHORT_HASH_LEN = 8
COMMIT_CHUNK_SIZE = 1000


def run_git(cmd):
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return output.decode("utf-8", errors="replace").strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running cmd: {cmd}\n{e.output.decode('utf-8', errors='replace')}")
        sys.exit(1)


def get_tags():
    # Get tags sorted by creator date
    cmd = [
        "git",
        "tag",
        "--list",
        "v*",
        "gateway-v*",
        "--sort=creatordate",
        "--format=%(refname:short)|%(creatordate:iso8601)|%(objectname)",
    ]
    raw = run_git(cmd)
    tags = []
    if not raw:
        return []
    for line in raw.split("\n"):
        parts = line.split("|")
        if len(parts) >= 3:
            name, date, commit = parts[0], parts[1], parts[2]
            tag_type = "gateway" if name.startswith("gateway-") else "main"
            tags.append(
                {"name": name, "date": date, "commit": commit, "type": tag_type}
            )
    return tags


def extract_pr_num(message):
    lines = message.strip().split("\n")
    first_line = lines[0]

    m = re.search(r"\(#(\d+)\)$", first_line)
    if m:
        return m.group(1)

    m = re.search(r"Merge pull request #(\d+)", message)
    if m:
        return m.group(1)

    return None


def process_tag_line(tags, commit_map, pr_map, tag_type, tag_to_idx):
    """Process a single release line (main or gateway) independently."""
    seen_commits = set()

    for tag in tags:
        tag_name = tag["name"]
        print(f"Processing {tag_name}...")

        commits = run_git(["git", "rev-list", tag_name]).split("\n")

        new_commits = []
        for c in commits:
            c = c.strip()
            if not c:
                continue
            if c in seen_commits:
                continue
            new_commits.append(c)
            seen_commits.add(c)

        if not new_commits:
            continue

        for i in range(0, len(new_commits), COMMIT_CHUNK_SIZE):
            chunk = new_commits[i : i + COMMIT_CHUNK_SIZE]

            cmd = ["git", "show", "-s", "--format=%H|%B%n--END-COMMIT--"] + chunk
            raw_logs = run_git(cmd)

            entries = raw_logs.split("--END-COMMIT--\n")
            for log_entry in entries:
                if not log_entry.strip():
                    continue
                parts = log_entry.split("|", 1)
                if len(parts) < 2:
                    continue
                sha = parts[0].strip()
                msg = parts[1].strip()

                # Use short hash as key
                short_sha = sha[:SHORT_HASH_LEN]
                tag_idx = tag_to_idx[tag_name]

                # Store tag index for this release line
                if short_sha not in commit_map:
                    commit_map[short_sha] = {}
                commit_map[short_sha][tag_type] = tag_idx

                pr = extract_pr_num(msg)
                if pr:
                    if pr not in pr_map:
                        pr_map[pr] = {}
                    if tag_type not in pr_map[pr]:
                        pr_map[pr][tag_type] = tag_idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate lookup index for sglang tags"
    )
    parser.add_argument("--output", default="tag_index.json", help="Output JSON file")
    args = parser.parse_args()

    tags = get_tags()
    print(f"Found {len(tags)} tags.")

    main_tags = [t for t in tags if t["type"] == "main"]
    gateway_tags = [t for t in tags if t["type"] == "gateway"]

    print(f"  - {len(main_tags)} main tags")
    print(f"  - {len(gateway_tags)} gateway tags")

    # Build tag list and index mapping
    # Tags array: [name, date, type] for each tag
    tag_list = []
    tag_to_idx = {}

    for tag in tags:
        tag_to_idx[tag["name"]] = len(tag_list)
        # Compact format: [name, date, type (0=main, 1=gateway)]
        tag_list.append(
            [tag["name"], tag["date"], 1 if tag["type"] == "gateway" else 0]
        )

    pr_map = {}
    commit_map = {}

    process_tag_line(main_tags, commit_map, pr_map, "m", tag_to_idx)
    process_tag_line(gateway_tags, commit_map, pr_map, "g", tag_to_idx)

    # Compact output format:
    # - tags: array of [name, date, type]
    # - prs: {pr_num: tag_idx} or {pr_num: {m: idx, g: idx}}
    # - commits: {short_hash: tag_idx} or {short_hash: {m: idx, g: idx}}

    # Simplify single-entry dicts to just the value
    def simplify_map(m):
        result = {}
        for k, v in m.items():
            if len(v) == 1:
                # Single entry: just store the index directly with type prefix
                key_type, idx = list(v.items())[0]
                result[k] = f"{key_type}{idx}"
            else:
                # Multiple entries: keep as dict
                result[k] = v
        return result

    output_data = {
        "t": tag_list,  # tags
        "p": simplify_map(pr_map),  # prs
        "c": simplify_map(commit_map),  # commits
        "g": datetime.now().isoformat(),  # generated_at
    }

    # Write minified JSON
    json_str = json.dumps(output_data, separators=(",", ":"))

    with open(args.output, "w") as f:
        f.write(json_str)

    json_size = os.path.getsize(args.output)

    print(f"Index generated at {args.output}")
    print(f"Stats: {len(tag_list)} tags, {len(pr_map)} PRs, {len(commit_map)} commits.")
    print(f"Size: {json_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
