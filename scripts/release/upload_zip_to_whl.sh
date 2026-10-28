#!/usr/bin/env bash

set -euo pipefail

readonly REPOSITORY="sgl-project/whl"
readonly TARGET_BRANCH="gh-pages"
readonly MAX_ASSET_BYTES=2147483648
readonly INDEX_URL="https://docs.sglang.io/whl/others/"
readonly MAX_INDEX_PUSH_ATTEMPTS=3

API_OUTPUT=""
RELEASE_URL=""
ASSET_URL=""
TEMP_ROOT=""

usage() {
  cat >&2 <<'EOF'
Usage: scripts/release/upload_zip_to_whl.sh <zip-path> <version> [release-tag] [release-title]

Example:
  scripts/release/upload_zip_to_whl.sh ~/Downloads/model-cache.zip v1.2.0
  scripts/release/upload_zip_to_whl.sh archive.zip 20260617 custom-tag "Custom release"
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'Warning: %s\n' "$*" >&2
}

cleanup() {
  if [[ -n "${TEMP_ROOT:-}" && -d "$TEMP_ROOT" ]]; then
    rm -rf -- "$TEMP_ROOT"
  fi
}

api_get_optional() {
  local endpoint="$1"
  local lookup_status

  API_OUTPUT=""
  if API_OUTPUT=$(gh api "$endpoint" 2>&1); then
    return 0
  else
    lookup_status=$?
  fi

  if [[ "$API_OUTPUT" == *"HTTP 404"* ]]; then
    API_OUTPUT=""
    return 1
  fi

  printf '%s\n' "$API_OUTPUT" >&2
  warn "GitHub API request failed for ${endpoint} (gh exited ${lookup_status})"
  return 2
}

compute_sha256() {
  local file_path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file_path" | awk '{print $1}'
    return
  fi

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file_path" | awk '{print $1}'
    return
  fi

  die "Neither sha256sum nor shasum is installed"
}

load_and_validate_release() {
  local endpoint="$1"
  local expected_name="$2"
  local expected_size="$3"
  local expected_checksum="$4"
  local release_body
  local asset_rows
  local remote_name
  local remote_size
  local remote_url
  local remote_digest
  local matched_size=""
  local matched_url=""
  local matched_digest=""
  local match_count=0
  local checksum_marker

  RELEASE_URL=$(gh api "$endpoint" --jq '.html_url') ||
    die "Unable to read the Release URL"
  release_body=$(gh api "$endpoint" --jq '.body // ""') ||
    die "Unable to read the Release body"
  asset_rows=$(
    gh api "$endpoint" \
      --jq '.assets[] | [.name, .size, .browser_download_url, (.digest // "")] | @tsv'
  ) || die "Unable to read the Release assets"

  while IFS=$'\t' read -r remote_name remote_size remote_url remote_digest; do
    if [[ "$remote_name" == "$expected_name" ]]; then
      ((match_count += 1))
      matched_size="$remote_size"
      matched_url="$remote_url"
      matched_digest="$remote_digest"
    fi
  done <<<"$asset_rows"

  checksum_marker="SHA256: \`${expected_checksum}\`"
  [[ "$release_body" == *"$checksum_marker"* ]] ||
    die "Existing Release checksum does not match the local ZIP"
  [[ "$match_count" -eq 1 ]] ||
    die "Release must contain exactly one asset named ${expected_name}"
  [[ "$matched_size" == "$expected_size" ]] ||
    die "Existing Release asset size does not match the local ZIP"
  [[ "$matched_digest" == "sha256:${expected_checksum}" ]] ||
    die "Existing Release asset digest does not match the local ZIP"
  [[ -n "$matched_url" ]] ||
    die "Existing Release asset has no download URL"

  ASSET_URL="$matched_url"
}

publish_indexes() {
  local helper_path="$1"
  local asset_url="$2"
  local filename="$3"
  local release_tag="$4"
  local checksum="$5"
  local github_login
  local attempt
  local attempt_dir
  local index_status

  github_login=$(gh api user --jq '.login') ||
    die "Unable to read the authenticated GitHub username"
  [[ -n "$github_login" ]] ||
    die "GitHub returned an empty authenticated username"

  for ((attempt = 1; attempt <= MAX_INDEX_PUSH_ATTEMPTS; attempt += 1)); do
    attempt_dir="${TEMP_ROOT}/whl-${attempt}"

    if ! git clone --quiet --depth 1 --branch "$TARGET_BRANCH" \
      "https://github.com/${REPOSITORY}.git" "$attempt_dir"; then
      warn "Index clone attempt ${attempt}/${MAX_INDEX_PUSH_ATTEMPTS} failed"
      continue
    fi

    python3 "$helper_path" \
      --repo-dir "$attempt_dir" \
      --asset-url "$asset_url" \
      --filename "$filename" \
      --tag "$release_tag" \
      --sha256 "$checksum" >/dev/null ||
      die "Unable to update the local whl indexes"

    index_status=$(
      git -C "$attempt_dir" status --porcelain -- index.html others/index.html
    )
    if [[ -z "$index_status" ]]; then
      return 0
    fi

    git -C "$attempt_dir" config user.name "$github_login"
    git -C "$attempt_dir" config \
      user.email "${github_login}@users.noreply.github.com"
    git -C "$attempt_dir" add -- index.html others/index.html
    git -C "$attempt_dir" commit --quiet \
      -m "Add ${filename} to others index for ${release_tag}" ||
      die "Unable to commit the local whl index update"

    if git -c credential.helper= \
      -c credential.helper='!gh auth git-credential' \
      -C "$attempt_dir" push --quiet origin "HEAD:${TARGET_BRANCH}"; then
      return 0
    fi

    warn "Index push attempt ${attempt}/${MAX_INDEX_PUSH_ATTEMPTS} failed; retrying from the latest branch tip"
  done

  die "Release is available, but the whl indexes could not be pushed after ${MAX_INDEX_PUSH_ATTEMPTS} attempts; rerun the same command to resume"
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 2
fi

input_path="$1"
version="$2"

[[ -e "$input_path" ]] || die "File does not exist: ${input_path}"
[[ -f "$input_path" ]] || die "Path is not a regular file: ${input_path}"
[[ -s "$input_path" ]] || die "ZIP file is empty: ${input_path}"

zip_name=$(basename -- "$input_path")
zip_dir=$(dirname -- "$input_path")
zip_dir=$(cd "$zip_dir" && pwd -P) || die "Cannot resolve ZIP directory"
zip_path="${zip_dir}/${zip_name}"

case "$zip_name" in
  *.[zZ][iI][pP]) ;;
  *) die "File must have a .zip extension: ${zip_name}" ;;
esac

if [[ "$zip_name" == *\\* ]] ||
  printf '%s' "$zip_name" | LC_ALL=C grep -q '[[:cntrl:]]'; then
  die "ZIP filename cannot contain backslashes or control characters"
fi

file_size=$(wc -c <"$zip_path")
file_size=${file_size//[[:space:]]/}
[[ "$file_size" =~ ^[0-9]+$ ]] ||
  die "Unable to determine ZIP file size: ${zip_path}"
((file_size < MAX_ASSET_BYTES)) ||
  die "ZIP file must be smaller than 2 GiB (${MAX_ASSET_BYTES} bytes)"

[[ -n "$version" ]] || die "Version cannot be empty"
[[ "$version" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "Version may contain only ASCII letters, digits, '.', '_', and '-'"

release_tag="${3:-zip-${version}}"
release_title="${4:-ZIP ${version}: ${zip_name}}"
[[ "$release_tag" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "Release tag may contain only ASCII letters, digits, '.', '_', and '-'"
[[ -n "$release_title" ]] || die "Release title cannot be empty"
if printf '%s' "$release_title" | LC_ALL=C grep -q '[[:cntrl:]]'; then
  die "Release title cannot contain control characters"
fi

command -v gh >/dev/null 2>&1 ||
  die "GitHub CLI is required; install it from https://cli.github.com/"
command -v git >/dev/null 2>&1 || die "Git is required"
command -v python3 >/dev/null 2>&1 || die "Python 3 is required"
gh auth status --hostname github.com >/dev/null 2>&1 ||
  die "GitHub CLI is not authenticated; run: gh auth login"

push_permission=$(gh api "repos/${REPOSITORY}" --jq '.permissions.push // false') ||
  die "Unable to read repository permissions for ${REPOSITORY}"
[[ "$push_permission" == "true" ]] ||
  die "The authenticated GitHub account needs write access to ${REPOSITORY}"

checksum=$(compute_sha256 "$zip_path")
[[ "$checksum" =~ ^[0-9A-Fa-f]{64}$ ]] ||
  die "Unable to compute a valid SHA256 checksum"

release_endpoint="repos/${REPOSITORY}/releases/tags/${release_tag}"
tag_endpoint="repos/${REPOSITORY}/git/ref/tags/${release_tag}"
release_exists=false
tag_exists=false

if api_get_optional "$release_endpoint"; then
  release_exists=true
else
  api_status=$?
  [[ "$api_status" -eq 1 ]] ||
    die "Unable to determine whether Release ${release_tag} exists"
fi

if api_get_optional "$tag_endpoint"; then
  tag_exists=true
else
  api_status=$?
  [[ "$api_status" -eq 1 ]] ||
    die "Unable to determine whether tag ${release_tag} exists"
fi

if [[ "$release_exists" == "false" && "$tag_exists" == "false" ]]; then
  release_notes=$(
    printf 'Uploaded asset: `%s`\n\nSHA256: `%s`' "$zip_name" "$checksum"
  )

  gh release create "$release_tag" "$zip_path" \
    --repo "$REPOSITORY" \
    --target "$TARGET_BRANCH" \
    --title "$release_title" \
    --notes "$release_notes" \
    --latest=false >/dev/null ||
    die "GitHub failed to create Release ${release_tag}"
elif [[ "$release_exists" == "true" && "$tag_exists" == "true" ]]; then
  warn "Release ${release_tag} already exists; validating it before resuming the index update"
else
  die "Release/tag state for ${release_tag} is inconsistent; choose a new version or repair it manually"
fi

load_and_validate_release \
  "$release_endpoint" "$zip_name" "$file_size" "$checksum"

script_dir=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) ||
  die "Cannot resolve the script directory"
index_helper="${script_dir}/update_others_whl_index.py"
[[ -f "$index_helper" ]] || die "Missing index helper: ${index_helper}"

TEMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/sglang-whl-upload.XXXXXX") ||
  die "Unable to create a temporary directory"
[[ -n "$TEMP_ROOT" && -d "$TEMP_ROOT" ]] ||
  die "mktemp did not create a valid temporary directory"
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

publish_indexes "$index_helper" "$ASSET_URL" "$zip_name" "$release_tag" "$checksum"

printf 'Release: %s\n' "$RELEASE_URL"
printf 'Asset:   %s\n' "$ASSET_URL"
printf 'SHA256:  %s\n' "$checksum"
printf 'Index:   %s\n' "$INDEX_URL"
printf 'wget %q\n' "$ASSET_URL"
