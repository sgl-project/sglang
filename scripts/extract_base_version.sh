#!/bin/bash
# Extract base version from a tag by removing pre-release suffixes
# Example: v0.10.0rc1 -> v0.10.0
# Example: v1.2.3-alpha2 -> v1.2.3
# Example: v2.0.0.dev5 -> v2.0.0
#
# Usage: extract_base_version.sh <tag>
#   tag: The version tag to process (e.g., v0.10.0rc1)
#
# Returns: The base version without pre-release suffix

if [ -z "$1" ]; then
  echo "Error: No tag provided" >&2
  echo "Usage: $0 <tag>" >&2
  exit 1
fi

TAG="$1"

# Remove pre-release suffixes (rc, alpha, beta, pre, dev followed by optional numbers)
# Matches: -rc1, .rc1, rc1, -alpha, .beta2, dev5, etc.
echo "$TAG" | sed -E 's/[\.-]?(rc|alpha|beta|pre|dev)[0-9]*$//'
