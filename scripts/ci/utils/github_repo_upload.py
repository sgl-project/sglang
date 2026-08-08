"""GitHub API helpers for committing files into a storage repository.

Used by the diffusion CI publishers to push images and comparison results into
sgl-project/ci-data. Performance traces no longer go through here; they are
uploaded to S3 by publish_traces.py.
"""

import base64
import json
import time
import warnings
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def is_rate_limit_error(e):
    """Check if an exception is a GitHub rate limit error (not permission error)"""
    if not isinstance(e, HTTPError):
        return False
    if e.code == 429:
        return True
    if e.code == 403:
        # 403 can be rate limit OR permission error - check the message
        error_body = getattr(e, "error_body", "")
        if isinstance(error_body, str):
            # Rate limit errors contain specific phrases
            rate_limit_phrases = [
                "rate limit",
                "abuse detection",
                "secondary rate limit",
            ]
            return any(phrase in error_body.lower() for phrase in rate_limit_phrases)
    return False


def is_permission_error(e):
    """Check if an exception is a GitHub permission error"""
    if not isinstance(e, HTTPError) or e.code != 403:
        return False
    error_body = getattr(e, "error_body", "")
    if isinstance(error_body, str):
        permission_phrases = [
            "resource not accessible",
            "must have push access",
            "permission",
            "denied",
        ]
        return any(phrase in error_body.lower() for phrase in permission_phrases)
    return False


def make_github_request(url, token, method="GET", data=None):
    """Make authenticated request to GitHub API"""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        # "User-Agent": "sglang-ci",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if data:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode("utf-8")

    req = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(req) as response:
            return response.read().decode("utf-8")
    except HTTPError as e:
        print(f"GitHub API request failed: {e}")
        try:
            error_body = e.read().decode("utf-8")
            print(f"Error response body: {error_body}")
            e.error_body = error_body  # Attach for later inspection
        except Exception:
            e.error_body = ""
        raise
    except Exception as e:
        print(f"GitHub API request failed with a non-HTTP error: {e}")
        raise


def verify_token_permissions(repo_owner, repo_name, token):
    """Verify that the token has necessary permissions for the repository"""
    print("Verifying token permissions...")

    checks = [
        (
            f"https://api.github.com/repos/{repo_owner}/{repo_name}",  # Check if we can access the repository
            "Repository access verified",
        ),
        (
            f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents",  # Check if we can read the repository contents
            "Repository contents access verified",
        ),
    ]

    for url, success_message in checks:
        try:
            response = make_github_request(url, token)
            if success_message == "Repository access verified":
                repo_data = json.loads(response)
                print(f"{success_message}: {repo_data['full_name']}")
            else:
                print(success_message)
        except Exception as e:
            if is_rate_limit_error(e):
                warnings.warn(
                    "GitHub API rate limit exceeded during token verification."
                )
                return "rate_limited"
            print(f"Failed to verify permissions for {url}: {e}")
            return False

    return True


def get_branch_sha(repo_owner, repo_name, branch, token):
    """Get SHA of the branch head"""
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs/heads/{branch}"
    )
    response = make_github_request(url, token)
    data = json.loads(response)
    return data["object"]["sha"]


def get_tree_sha(repo_owner, repo_name, commit_sha, token):
    """Get tree SHA from commit"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/commits/{commit_sha}"
    response = make_github_request(url, token)
    data = json.loads(response)
    return data["tree"]["sha"]


def create_blob(repo_owner, repo_name, content, token, max_retries=3):
    """Create a blob with file content"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/blobs"

    # Encode content as base64 for GitHub API
    content_b64 = base64.b64encode(content).decode("utf-8")

    data = {"content": content_b64, "encoding": "base64"}

    for attempt in range(max_retries):
        try:
            response = make_github_request(url, token, method="POST", data=data)
            return json.loads(response)["sha"]
        except Exception as e:
            # Don't retry on rate limit errors - fail fast
            if is_rate_limit_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                print(
                    f"Blob creation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise


def create_blobs(repo_owner, repo_name, files, token):
    """Create blobs for all files and return tree items with blob SHAs"""
    tree_items = []
    for i, (file_path, content) in enumerate(files):
        # Create blob first to get SHA
        blob_sha = create_blob(repo_owner, repo_name, content, token)
        tree_items.append(
            {
                "path": file_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha,
            }
        )
        # Progress indicator for large uploads
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"Created {i + 1}/{len(files)} blobs...")
    return tree_items


def create_tree(repo_owner, repo_name, base_tree_sha, tree_items, token, max_retries=3):
    """Create a new tree from pre-created blob SHAs"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees"

    data = {"base_tree": base_tree_sha, "tree": tree_items}

    for attempt in range(max_retries):
        try:
            response = make_github_request(url, token, method="POST", data=data)
            return json.loads(response)["sha"]
        except Exception as e:
            # Don't retry on rate limit errors - fail fast
            if is_rate_limit_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"Tree creation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise


def create_commit(
    repo_owner, repo_name, tree_sha, parent_sha, message, token, max_retries=3
):
    """Create a new commit"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/commits"

    data = {"tree": tree_sha, "parents": [parent_sha], "message": message}

    for attempt in range(max_retries):
        try:
            response = make_github_request(url, token, method="POST", data=data)
            commit_sha = json.loads(response)["sha"]

            # Verify the commit was actually created
            verify_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/commits/{commit_sha}"
            verify_response = make_github_request(verify_url, token)
            verify_data = json.loads(verify_response)
            if verify_data["sha"] != commit_sha:
                raise Exception(
                    f"Commit verification failed: expected {commit_sha}, got {verify_data['sha']}"
                )

            return commit_sha
        except Exception as e:
            # Don't retry on rate limit errors - fail fast
            if is_rate_limit_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"Commit creation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise


def update_branch_ref(repo_owner, repo_name, branch, commit_sha, token, max_retries=3):
    """Update branch reference to point to new commit"""
    url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs/heads/{branch}"
    )

    data = {"sha": commit_sha}

    for attempt in range(max_retries):
        try:
            make_github_request(url, token, method="PATCH", data=data)
            return
        except HTTPError as e:
            # Don't retry on rate limit errors - fail fast
            if is_rate_limit_error(e):
                raise

            # Check if this is an "Object does not exist" error
            is_object_not_exist = False
            if hasattr(e, "error_body"):
                try:
                    error_data = json.loads(e.error_body)
                    if "Object does not exist" in error_data.get("message", ""):
                        is_object_not_exist = True
                except Exception:
                    pass

            if is_object_not_exist and attempt < max_retries - 1:
                # This might be a transient consistency issue - wait and retry
                wait_time = 2**attempt
                print(
                    f"Branch update failed with 'Object does not exist' (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s for consistency..."
                )
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            # Don't retry on rate limit errors - fail fast
            if is_rate_limit_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"Branch update failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise
