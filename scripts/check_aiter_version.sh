#!/bin/bash
#
# Check if AITER wheel needs to be rebuilt based on docker/rocm.Dockerfile
#
# Usage:
#   check_aiter_version.sh <rocm_version>
#   Example: check_aiter_version.sh 700
#
# Returns:
#   - Sets REBUILD_NEEDED=true if rebuild needed
#   - Sets REBUILD_NEEDED=false if existing wheel can be reused
#   - Sets AITER_VERSION to the detected version
#
# Exit codes:
#   0 - Success (outputs can be parsed)
#   1 - Error (missing Dockerfile, invalid arguments, etc.)

set -e

# Parse command-line arguments
if [[ $# -ne 1 ]]; then
  echo "Error: Expected exactly one ROCm version argument" >&2
  echo "Usage: $0 <rocm_version>" >&2
  echo "Example: $0 700" >&2
  exit 1
fi

ROCM_VERSION="$1"

# Validate ROCm version
if [[ "$ROCM_VERSION" != "700" ]] && [[ "$ROCM_VERSION" != "720" ]]; then
  echo "Error: Invalid ROCm version '$ROCM_VERSION'. Must be 700 or 720" >&2
  exit 1
fi

echo "Checking AITER for ROCm $ROCM_VERSION" >&2

DOCKERFILE="docker/rocm.Dockerfile"

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Error: $DOCKERFILE not found" >&2
  exit 1
fi

# Extract AITER_COMMIT_DEFAULT from Dockerfile for gfx950 sections
# The Dockerfile has 4 build stages:
# - gfx942 (rocm 7.0)
# - gfx942-rocm720 (rocm 7.2)
# - gfx950 (rocm 7.0)
# - gfx950-rocm720 (rocm 7.2)
# We parse gfx950 and gfx950-rocm720 sections

# Parse Dockerfile to extract AITER_COMMIT_DEFAULT for each build stage
# Output format: stage_name=aiter_value
parse_aiter_versions() {
  awk '
    /^FROM .* AS / {
      # Extract stage name from "FROM ... AS <stage_name>"
      stage = $NF
    }
    /^ENV AITER_COMMIT_DEFAULT=/ {
      # Extract AITER_COMMIT_DEFAULT value
      # Remove ENV AITER_COMMIT_DEFAULT= prefix and quotes
      value = $0
      sub(/.*AITER_COMMIT_DEFAULT=/, "", value)
      gsub(/["'\'']/, "", value)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      if (stage != "" && value != "") {
        print stage "=" value
      }
    }
  ' "$1"
}

# Parse the Dockerfile
echo "Parsing AITER versions from $DOCKERFILE..." >&2
PARSED_VERSIONS=$(parse_aiter_versions "$DOCKERFILE")

if [[ -z "$PARSED_VERSIONS" ]]; then
  echo "Error: Could not parse AITER_COMMIT_DEFAULT from $DOCKERFILE" >&2
  exit 1
fi

echo "Parsed AITER versions:" >&2
echo "$PARSED_VERSIONS" >&2

# Extract gfx950 value for the specified ROCm version
declare -A AITER_COMMIT_MAP
while IFS='=' read -r stage value; do
  AITER_COMMIT_MAP["$stage"]="$value"
done <<< "$PARSED_VERSIONS"

# Map ROCm version to stage name
if [[ "$ROCM_VERSION" == "700" ]]; then
  STAGE="gfx950"
elif [[ "$ROCM_VERSION" == "720" ]]; then
  STAGE="gfx950-rocm720"
fi

# Validate that we have a value for this ROCm version
if [[ -z "${AITER_COMMIT_MAP[$STAGE]}" ]]; then
  echo "Error: AITER_COMMIT_DEFAULT not found for stage '$STAGE' (ROCm $ROCM_VERSION)" >&2
  exit 1
fi

AITER_COMMIT="${AITER_COMMIT_MAP[$STAGE]}"
echo "ROCm ${ROCM_VERSION}: stage=${STAGE}, AITER_COMMIT=${AITER_COMMIT}" >&2

# Function to get AITER version for a specific commit
get_aiter_version() {
  local commit="$1"

  if [[ $commit =~ ^v[0-9] ]]; then
    # Version tag (e.g., v0.1.12.post1)
    echo "$commit"
  else
    # Commit SHA - need to trace git history to find version
    echo "AITER commit $commit is a SHA, tracing git history..." >&2

    local AITER_REPO="https://github.com/ROCm/aiter.git"
    local TEMP_DIR=$(mktemp -d)

    git clone --quiet "$AITER_REPO" "$TEMP_DIR/aiter" >&2 || {
      echo "Error: Failed to clone AITER repository" >&2
      rm -rf "$TEMP_DIR"
      return 1
    }

    cd "$TEMP_DIR/aiter"
    local version=$(git describe --tags "$commit" 2>/dev/null || echo "unknown")
    cd - > /dev/null
    rm -rf "$TEMP_DIR"

    if [[ "$version" == "unknown" ]]; then
      echo "Warning: Could not determine version for commit $commit, using SHA" >&2
      echo "$commit"
    else
      echo "Traced commit $commit to version: $version" >&2
      echo "$version"
    fi
  fi
}

# Get AITER version for this commit
AITER_VERSION=$(get_aiter_version "$AITER_COMMIT")
echo "AITER_VERSION=${AITER_VERSION}" >&2
echo "" >&2

# Check S3 for existing wheel
REBUILD_NEEDED=false
S3_BUCKET="${AMD_S3_BUCKET_NAME:-aioss-pypi-prod}"

echo "Checking S3 bucket: $S3_BUCKET" >&2

# Verify AWS CLI is available
if ! command -v aws &> /dev/null; then
  echo "Warning: AWS CLI not found. Assuming rebuild is needed." >&2
  REBUILD_NEEDED=true
  echo "REBUILD_NEEDED=$REBUILD_NEEDED"
  echo "AITER_VERSION=$AITER_VERSION"
  echo "AITER_COMMIT=$AITER_COMMIT"
  exit 0
fi

# Build AWS CLI command prefix with profile support
# The aws-actions/configure-aws-credentials@v4 action sets AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
# For local usage, users can set AWS_PROFILE environment variable (e.g., export AWS_PROFILE=sglang)
AWS_CMD="aws"
if [[ -n "$AWS_PROFILE" ]]; then
  AWS_CMD="aws --profile $AWS_PROFILE"
  echo "Using AWS profile: $AWS_PROFILE" >&2
fi

# Verify AWS credentials are configured
if [[ -z "$AWS_ACCESS_KEY_ID" ]] && ! $AWS_CMD sts get-caller-identity &> /dev/null; then
  echo "Warning: AWS credentials not configured. Assuming rebuild is needed." >&2
  echo "Hint: Set AWS_PROFILE environment variable (e.g., export AWS_PROFILE=sglang) if using named profiles" >&2
  REBUILD_NEEDED=true
  echo "REBUILD_NEEDED=$REBUILD_NEEDED"
  echo "AITER_VERSION=$AITER_VERSION"
  echo "AITER_COMMIT=$AITER_COMMIT"
  exit 0
fi

echo "AWS CLI and credentials verified" >&2

# Check S3 for this specific version
S3_PATH="s3://${S3_BUCKET}/sglang/rocm${ROCM_VERSION}/packages/amd-aiter/"
echo "Checking: $S3_PATH" >&2

# List wheels in S3 (requires AWS credentials)
if ! WHEELS=$($AWS_CMD s3 ls "$S3_PATH" 2>/dev/null); then
  echo "Warning: Could not list S3 path $S3_PATH (AWS credentials or path may not exist)" >&2
  # If we can't list S3, assume rebuild is needed
  REBUILD_NEEDED=true
else
  # Check if this AITER version exists
  # Expected pattern: amd_aiter-{VERSION}-cp310-cp310-linux_x86_64.whl
  # Version might be like 0.1.12 or v0.1.12.post1 (no +rocm suffix)
  VERSION_PATTERN="${AITER_VERSION#v}"  # Remove 'v' prefix if present

  if ! echo "$WHEELS" | grep -q "amd_aiter-${VERSION_PATTERN}-"; then
    echo "amd-aiter version $AITER_VERSION not found for rocm${ROCM_VERSION}" >&2
    REBUILD_NEEDED=true
  else
    echo "Found existing amd-aiter wheel for rocm${ROCM_VERSION}" >&2
  fi
fi

# Output results (can be parsed by GitHub Actions)
echo "REBUILD_NEEDED=$REBUILD_NEEDED"
echo "AITER_VERSION=$AITER_VERSION"
echo "AITER_COMMIT=$AITER_COMMIT"

exit 0
