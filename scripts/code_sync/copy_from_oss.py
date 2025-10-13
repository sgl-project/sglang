"""
Sync code from OSS repo to the local repo and open a PR if changes exist.

NOTE:
1. You need to execute this script in the git root folder.
2. A GH_TOKEN environment variable is required to create the pull request.
  - see also https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

This script will:
1. Clone the sgl-project/sglang repository (or use a local copy).
2. Sync specified files and directories using rsync.
3. Check if the sync operation resulted in any changes.
4. If there are changes:
   a. Create a new branch.
   b. Commit and push the changes.
   c. Open a pull request using the GitHub CLI (gh).

Usage:
# Run the full sync and PR creation process
python3 scripts/copy_from_oss.py

# Perform a dry run without making any actual changes
python3 scripts/copy_from_oss.py --dry-run

# Use a local directory as the source instead of cloning
python3 scripts/copy_from_oss.py --local-dir ~/projects/sglang
"""

import argparse
import datetime
import os
import shutil
import subprocess
import tempfile

# --- Configuration Begin ---
# List of folders and files to copy from the OSS repo.
# Changes outside these paths will be ignored.
folder_names = [
    "3rdparty",
    "assets",
    "benchmark",
    "docker",
    "docs",
    "examples",
    "python/sglang/lang",
    "python/sglang/srt",
    "python/sglang/test",
    "python/sglang/utils.py",
    "python/sglang/README.md",
    "sgl-kernel",
    "test/lang",
    "test/srt",
    "test/README.md",
    "README.md",
]

private_repo = "your-org/sglang-private-repo"
# --- Configuration End ---


def write_github_step_summary(content):
    if not os.environ.get("GITHUB_STEP_SUMMARY"):
        return

    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(content)


def check_dependencies():
    """Check for required command-line tools."""
    if not shutil.which("git"):
        raise EnvironmentError("git is not installed or not in PATH.")
    if not shutil.which("gh"):
        raise EnvironmentError("GitHub CLI (gh) is not installed or not in PATH.")
    print("✅ All dependencies (git, gh) are available.")


def checkout_main(dry_run):
    """Checkout to the main branch."""
    commands = [
        "git checkout main",
        "git reset --hard",
    ]
    for cmd in commands:
        print(f"Run: {cmd}")
        if not dry_run:
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Git command failed: {e.stderr.decode()}")
                raise
    print("✅ Checkout the main branch.")


def get_source_folder(args):
    """
    Prepare the source repository, either by cloning from GitHub or using a local directory.
    Returns the path to the source repo root, a temporary directory path (if created),
    and the short commit hash.
    """
    temp_dir = None
    if args.local_dir:
        oss_root = os.path.expanduser(args.local_dir)
        if not os.path.exists(oss_root):
            raise FileNotFoundError(
                f"Specified local directory {oss_root} does not exist."
            )
        print(f"Using local directory as the source: {oss_root}")
    else:
        temp_dir = tempfile.mkdtemp()
        oss_root = temp_dir
        print(f"Created temporary directory: {oss_root}")

        repo_url = "https://github.com/sgl-project/sglang.git"
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--single-branch",
                    "--branch",
                    "main",
                    repo_url,
                    temp_dir,
                ],
                check=True,
                capture_output=True,
            )
            print(f"Successfully cloned repository to {temp_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr.decode()}")
            raise

    commit_hash = subprocess.run(
        ["git", "-C", oss_root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()[:8]
    print(f"✅ Get source OSS code at commit: {commit_hash}")
    return oss_root, temp_dir, commit_hash


def sync_directories(oss_root, folder_names, dry_run):
    """Sync specified directories from oss_root to current working directory."""
    rsync_commands = []
    for folder_name in folder_names:
        target_name = f"{oss_root}/{folder_name}"
        src_name = "./" + "/".join(folder_name.split("/")[:-1])
        cmd = f"rsync -r --delete {target_name} {src_name}"
        rsync_commands.append(cmd)

    for cmd in rsync_commands:
        try:
            print(f"Run: {cmd}")
            if not dry_run:
                subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command '{cmd}': {e}")
            raise
    print(f"✅ Sync all folders.")


def check_for_changes():
    """Check if there are any uncommitted git changes."""
    # This command exits with 1 if there are changes, 0 otherwise.
    result = subprocess.run(["git", "diff", "--quiet"])
    return result.returncode != 0


def create_and_push_branch(branch_name, commit_message, dry_run):
    """Create a new branch, commit all changes, and push to origin."""
    commands = [
        f"git checkout -b {branch_name}",
        "git config user.name 'github-actions[bot]'",
        "git config user.email 'github-actions[bot]@users.noreply.github.com'",
        "git add .",
        f"git commit -m '{commit_message}'",
        f"git push origin {branch_name} --force",
    ]
    print("\nCreating and pushing git branch...")
    for cmd in commands:
        print(f"Run: {cmd}")
        if not dry_run:
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Git command failed: {e.stderr.decode()}")
                raise


def create_pull_request(branch_name, title, body, dry_run):
    """Create a pull request using the GitHub CLI."""
    gh_token = os.getenv("GH_TOKEN")
    if not gh_token:
        print(
            "\n⚠️ Warning: GH_TOKEN environment variable not set. Skipping PR creation."
        )
        if not dry_run:
            return

    print("\nCreating pull request...")
    command = [
        "gh",
        "pr",
        "create",
        "--base",
        "main",
        "--head",
        branch_name,
        "--repo",
        private_repo,
        "--title",
        title,
        "--body",
        body,
    ]
    print(f"Run: {' '.join(command)}")
    if not dry_run:
        env = os.environ.copy()
        env["GH_TOKEN"] = gh_token
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, env=env
            )
            pr_url = result.stdout.strip()
            msg = f"✅ Successfully created pull request: {pr_url}"
            print(msg)
            write_github_step_summary(msg)
        except subprocess.CalledProcessError as e:
            print(f"Error creating pull request: {e.stderr}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Copy code from OSS and open a PR if changes are detected."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        help="Path to local SGLang directory to use instead of cloning from GitHub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the script without executing git, rsync, or gh commands.",
    )
    args = parser.parse_args()

    check_dependencies()
    checkout_main(args.dry_run)

    oss_root, temp_dir, oss_commit = get_source_folder(args)

    try:
        # Sync directories
        sync_directories(oss_root, folder_names, args.dry_run)

        # Check for changes and create PR if necessary
        if not check_for_changes():
            msg = "😴 No changes detected. The code is already in sync."
            print(msg)
            write_github_step_summary(msg)
            return

        print("✅ Changes detected. Proceeding to create a PR.")

        current_date = datetime.datetime.now().strftime("%Y%m%d")
        branch_name = f"copy-from-oss-{oss_commit}-{current_date}"
        commit_message = f"Copy OSS code from {oss_commit} on {current_date}"
        pr_title = (
            f"[Automated PR] Copy OSS code from commit {oss_commit} on {current_date}"
        )
        pr_body = (
            f"Copy OSS code from https://github.com/sgl-project/sglang/commit/{oss_commit} on {current_date}."
            "\n\n---\n\n"
            "*This is an automated PR created by scripts/copy_from_oss.py.*"
        )

        create_and_push_branch(branch_name, commit_message, args.dry_run)
        create_pull_request(branch_name, pr_title, pr_body, args.dry_run)

    finally:
        # Remove temporary directory if it was created
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
                print(f"\nRemoved temporary directory: {temp_dir}")
            except OSError as e:
                print(f"Error removing temporary directory {temp_dir}: {e}")


if __name__ == "__main__":
    main()
