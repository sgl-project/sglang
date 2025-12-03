"""
Sync a specific commit from the local private repo to the OSS upstream and open a PR.

NOTE:
1. You need to execute this script in the git root folder.
2. A GH_TOKEN environment variable is required to create the pull request.
  - see also https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

This script will:
1. Take a commit hash as an argument (or use the latest commit by default).
2. Create a patch for that commit.
3. Filter the patch to only include changes in specified directories.
4. Clone the sgl-project/sglang repository.
5. Create a new branch in the OSS repo.
6. Apply the filtered patch, commit, and force push.
7. Open a pull request to the OSS repo using the GitHub CLI (gh).

Usage:
# Sync the latest commit from the current branch
python3 scripts/copy_to_oss.py

# Run the full sync and PR creation process for a given commit
python3 scripts/copy_to_oss.py --commit <commit_hash>

# Perform a dry run without making any actual changes
python3 scripts/copy_to_oss.py --commit <commit_hash> --dry-run
"""

import argparse
import datetime
import os
import shutil
import subprocess
import tempfile

# --- Configuration Begin ---
# List of folders and files to copy to the OSS repo.
# Changes outside these paths will be ignored.
folder_names = [
    "3rdparty",
    "assets",
    "benchmark",
    "docker",
    "docs",
    "examples",
    "python/sglang/lang",
    "python/sglang/jit_kernel",
    "python/sglang/srt",
    "python/sglang/test",
    "python/sglang/utils.py",
    "python/sglang/README.md",
    "sgl-kernel",
    "test/manual",
    "test/nightly",
    "test/registered",
    "test/srt",
    "test/README.md",
    "README.md",
]

# --- Configuration End ---


def write_github_step_summary(content):
    if not os.environ.get("GITHUB_STEP_SUMMARY"):
        return

    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(content)


def get_commit_info(commit_ref):
    """
    Retrieves the hash and message of a specific commit.

    Args:
        commit_ref (str): The commit hash, tag, or branch to inspect (e.g., 'HEAD').

    Returns:
        A tuple containing the (commit_hash, commit_message),
        or (None, None) if an error occurs.
    """
    try:
        # Use a custom format to get the hash (%H) and the full message (%B)
        # separated by a null character for safe parsing.
        command = ["git", "log", "-1", f"--pretty=%H%x00%B", commit_ref]
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )

        # Split the output by the null character separator
        commit_hash, commit_message = result.stdout.strip().split("\x00", 1)
        return commit_hash, commit_message

    except FileNotFoundError:
        print("❌ Error: 'git' command not found. Is Git installed and in your PATH?")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting commit info for '{commit_ref}': {e.stderr.strip()}")
        print(
            "Hint: Make sure you are running this from within a Git repository and the commit exists."
        )

    return None, None


def check_dependencies():
    """Check for required command-line tools."""
    if not shutil.which("git"):
        raise EnvironmentError("git is not installed or not in PATH.")
    if not shutil.which("gh"):
        raise EnvironmentError("GitHub CLI (gh) is not installed or not in PATH.")
    print("✅ All dependencies (git, gh) are available.")


def create_filtered_patch(commit_hash, dry_run):
    """
    Create a patch file for the given commit, containing only changes
    to files and directories specified in `folder_names`.
    """
    print(f"Creating a filtered patch for commit {commit_hash}")

    try:
        # Get the list of all files changed in the commit
        changed_files_raw = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        changed_files = changed_files_raw.strip().split("\n")

        # Filter the list of files
        relevant_files = [
            f for f in changed_files if any(f.startswith(path) for path in folder_names)
        ]

        if not relevant_files:
            msg = "\n😴 No relevant file changes found in this commit. Exiting."
            print(msg)
            write_github_step_summary(msg)
            return None, None

        print("Found relevant changes in the following files:")
        for f in relevant_files:
            print(f"  - {f}")

        # Create a patch containing only the changes for the relevant files
        patch_command = [
            "git",
            "format-patch",
            "--stdout",
            f"{commit_hash}^..{commit_hash}",
            "--",
        ] + relevant_files

        print(f"Run: {' '.join(patch_command)}")

        patch_content = subprocess.run(
            patch_command, capture_output=True, text=True, check=True
        ).stdout

        # Save the patch to a temporary file
        patch_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".patch", encoding="utf-8"
        )
        patch_file.write(patch_content)
        patch_file.close()

        print(f"✅ Filtered patch created successfully at: {patch_file.name}")
        return patch_file.name, relevant_files

    except subprocess.CalledProcessError as e:
        print(f"Error creating patch: {e.stderr}")
        raise


def get_oss_repo(dry_run):
    """
    Clones the OSS repository into a temporary directory.
    Returns the path to the repo root and the temp directory itself.
    """
    gh_token = os.getenv("GH_TOKEN")
    if not gh_token:
        print("⚠️ Warning: GH_TOKEN environment variable not set. Skipping PR creation.")
        if not dry_run:
            return

    temp_dir = tempfile.mkdtemp()
    oss_root = os.path.join(temp_dir, "sglang")
    print(f"\nCreated temporary directory for OSS repo: {temp_dir}")

    repo_url = f"https://{gh_token}@github.com/sgl-project/sglang.git"
    command = ["git", "clone", "--branch", "main", repo_url, oss_root]

    print(f"Run: {' '.join(command)}")
    if not dry_run:
        try:
            subprocess.run(command, check=True, capture_output=True)
            print(f"✅ Successfully cloned repository to {oss_root}")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr.decode()}")
            shutil.rmtree(temp_dir)
            raise

    return oss_root, temp_dir


def apply_patch_and_push(oss_root, patch_file, branch_name, commit_message, dry_run):
    """
    In the OSS repo, create a branch, apply the patch, commit, and push.
    """
    print("\nApplying patch and pushing to OSS repo...")

    original_cwd = os.getcwd()
    if not dry_run:
        os.chdir(oss_root)

    try:
        # Define commands as lists to avoid shell injection issues
        commands_to_run = [
            ["git", "checkout", "-b", branch_name],
            ["git", "apply", patch_file],
            ["git", "config", "user.name", "github-actions[bot]"],
            [
                "git",
                "config",
                "user.email",
                "github-actions[bot]@users.noreply.github.com",
            ],
            ["git", "add", "."],
        ]

        for cmd_list in commands_to_run:
            print(f"Run: {' '.join(cmd_list)}")
            if not dry_run:
                subprocess.run(cmd_list, check=True, capture_output=True, text=True)

        # Handle commit separately to pass multi-line message safely via stdin
        commit_cmd = ["git", "commit", "-F", "-"]
        print(f"Run: {' '.join(commit_cmd)}")
        if not dry_run:
            print(f"Commit Message:\n---\n{commit_message}\n---")
            subprocess.run(
                commit_cmd,
                input=commit_message,
                text=True,
                check=True,
                capture_output=True,
            )

        # Push the changes
        push_cmd = ["git", "push", "origin", branch_name, "--force"]
        print(f"Run: {' '.join(push_cmd)}")
        if not dry_run:
            subprocess.run(push_cmd, check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e.stderr}")
        raise
    finally:
        if not dry_run:
            os.chdir(original_cwd)

    print("✅ Branch created, patch applied, and pushed successfully.")


def create_pull_request(oss_root, branch_name, title, body, dry_run):
    """Create a pull request in the OSS repo using the GitHub CLI."""
    gh_token = os.getenv("GH_TOKEN")
    if not gh_token:
        print("⚠️ Warning: GH_TOKEN environment variable not set. Skipping PR creation.")
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
        "sgl-project/sglang",
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
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=oss_root,
            )
            msg = f"✅ Successfully created pull request: {result.stdout.strip()}"
            print(msg)
            write_github_step_summary(msg)
        except subprocess.CalledProcessError as e:
            print(f"Error creating pull request: {e.stderr}")
            # Check if a PR already exists
            if "A pull request for" in e.stderr and "already exists" in e.stderr:
                print("ℹ️ A PR for this branch likely already exists.")
            else:
                raise


def get_commit_author(commit_hash):
    """Get the author name and email of a commit."""
    try:
        author_name = subprocess.run(
            ["git", "show", "-s", "--format=%an", commit_hash],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        author_email = subprocess.run(
            ["git", "show", "-s", "--format=%ae", commit_hash],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return author_name, author_email
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit author for {commit_hash}: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Copy a commit from the private repo to OSS and open a PR."
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="LAST",
        help="The commit hash to sync. Defaults to 'LAST' to use the latest commit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the script without executing git, rsync, or gh commands.",
    )
    args = parser.parse_args()

    check_dependencies()

    commit_ref = "HEAD" if args.commit == "LAST" else args.commit
    commit_hash, original_commit_message = get_commit_info(commit_ref)

    if not commit_hash:
        return  # Exit if we couldn't get commit info

    # Display the details of the commit being processed
    if args.commit == "LAST":
        summary = (
            f"\nℹ️ No commit specified. Using the last commit:\n"
            f"  - **Hash:** `{commit_hash}`\n"
            f"  - **Message:** {original_commit_message}\n\n"
        )
    else:
        summary = (
            f"\nℹ️ Using specified commit:\n"
            f"  - **Hash:** `{commit_hash}`\n"
            f"  - **Message:** {original_commit_message}\n\n"
        )
    print(summary)
    write_github_step_summary(summary)

    short_hash = commit_hash[:8]

    patch_file = None
    temp_dir = None
    try:
        # 1. Create a filtered patch from the local repo
        patch_file, relevant_files = create_filtered_patch(commit_hash, args.dry_run)
        if not patch_file:
            return

        # 2. Get the OSS repo
        oss_root, temp_dir = get_oss_repo(args.dry_run)

        # 3. Get original commit author for the co-author line
        author_name, author_email = get_commit_author(commit_hash)

        # 4. Prepare content for the commit and PR based on changed files
        file_list_str = "\n".join([f"- {f}" for f in relevant_files])
        filename_list_str = ", ".join([f.split("/")[-1] for f in relevant_files])
        if len(filename_list_str) > 40:
            filename_list_str = filename_list_str[:40] + "..."
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        pr_title = f"[Auto Sync] Update {filename_list_str} ({current_date})"
        pr_body = (
            f"Sync changes from commit `{short_hash}`.\n\n"
            f"**Files Changed:**\n{file_list_str}\n\n"
            f"Author: {author_name} <{author_email}>"
            f"\n\n---\n\n"
            f"*This is an automated PR created by scripts/copy_from_oss.py.*"
        )

        # 5. Create branch, apply patch, and push
        branch_name = f"sync-{short_hash}-{current_date}"
        co_author_line = f"Co-authored-by: {author_name} <{author_email}>"
        commit_message = f"{pr_title}\n\n{co_author_line}"
        apply_patch_and_push(
            oss_root, patch_file, branch_name, commit_message, args.dry_run
        )

        # 6. Create Pull Request
        create_pull_request(oss_root, branch_name, pr_title, pr_body, args.dry_run)

    finally:
        # Cleanup temporary files
        if patch_file and os.path.exists(patch_file):
            os.remove(patch_file)
            print(f"\nRemoved temporary patch file: {patch_file}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
