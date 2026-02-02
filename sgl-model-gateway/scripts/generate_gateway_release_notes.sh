#!/bin/bash
# Generate release notes for SGLang Gateway/Router
# Only includes commits that affect gateway-related paths

set -e

# Configuration
GATEWAY_PATHS=(
    "sgl-model-gateway"
    "python/sglang/srt/grpc"
    "python/sglang/srt/entrypoints/grpc_server.py"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 <previous-tag> <current-tag>"
    echo ""
    echo "Example: $0 gateway-v0.2.2 gateway-v0.2.3"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE    Save output to file (default: stdout)"
    echo "  -f, --format FORMAT  Output format: markdown|github|plain (default: markdown)"
    echo "  --create-release     Create GitHub release using gh CLI (default: draft)"
    echo "  --draft              Create as draft release (default when using --create-release)"
    echo "  --no-draft           Publish release immediately (skip draft)"
    echo "  -h, --help          Show this help message"
    exit 1
}

# Parse arguments
OUTPUT_FILE=""
FORMAT="markdown"
CREATE_RELEASE=false
DRAFT_RELEASE="default"  # Default to draft unless explicitly disabled
PREV_TAG=""
CURR_TAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        --create-release)
            CREATE_RELEASE=true
            shift
            ;;
        --draft)
            DRAFT_RELEASE=true
            shift
            ;;
        --no-draft)
            DRAFT_RELEASE=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [[ -z "$PREV_TAG" ]]; then
                PREV_TAG="$1"
            elif [[ -z "$CURR_TAG" ]]; then
                CURR_TAG="$1"
            else
                echo "Error: Too many arguments"
                usage
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "$PREV_TAG" ]] || [[ -z "$CURR_TAG" ]]; then
    echo "Error: Both previous and current tags are required"
    usage
fi

# Navigate to repo root (main sglang repo, not sgl-model-gateway)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo -e "${BLUE}Generating Gateway/Router release notes...${NC}" >&2
echo -e "${BLUE}Previous: $PREV_TAG${NC}" >&2
echo -e "${BLUE}Current:  $CURR_TAG${NC}" >&2
echo "" >&2

# Verify tags exist
if ! git rev-parse "$PREV_TAG" >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Tag $PREV_TAG not found, using initial commit${NC}" >&2
    PREV_TAG=$(git rev-list --max-parents=0 HEAD)
fi

if ! git rev-parse "$CURR_TAG" >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Tag $CURR_TAG not found, using HEAD${NC}" >&2
    CURR_TAG="HEAD"
fi

# Build path filter arguments
PATH_ARGS=()
for path in "${GATEWAY_PATHS[@]}"; do
    PATH_ARGS+=("--" "$path")
done

# Get filtered commit list
COMMITS=$(git log "$PREV_TAG..$CURR_TAG" --oneline --no-merges "${PATH_ARGS[@]}")

if [[ -z "$COMMITS" ]]; then
    echo -e "${YELLOW}No commits found for gateway paths between $PREV_TAG and $CURR_TAG${NC}" >&2
    exit 0
fi

COMMIT_COUNT=$(echo "$COMMITS" | wc -l | tr -d ' ')
echo -e "${GREEN}Found $COMMIT_COUNT gateway-related commits${NC}" >&2

# Get contributors
echo -e "${BLUE}Analyzing contributors...${NC}" >&2

# Get all contributors in this release (with commit count)
CONTRIBUTORS=$(git log "$PREV_TAG..$CURR_TAG" --format='%aN <%aE>' --no-merges "${PATH_ARGS[@]}" | sort | uniq -c | sort -rn)

# Get all contributors before this release (from initial commit up to PREV_TAG)
# Using $(git rev-list --max-parents=0 HEAD) to get initial commit ensures we check entire history
INITIAL_COMMIT=$(git rev-list --max-parents=0 HEAD | tail -1)
PREV_CONTRIBUTORS=$(git log "$INITIAL_COMMIT..$PREV_TAG" --format='%aN <%aE>' --no-merges "${PATH_ARGS[@]}" | sort | uniq)

# Find new contributors
NEW_CONTRIBUTORS=""
while IFS= read -r line; do
    contributor=$(echo "$line" | sed 's/^[[:space:]]*[0-9]*[[:space:]]*//')
    if [[ -n "$contributor" ]] && ! echo "$PREV_CONTRIBUTORS" | grep -Fxq "$contributor"; then
        NEW_CONTRIBUTORS="$NEW_CONTRIBUTORS$contributor"$'\n'
    fi
done <<< "$CONTRIBUTORS"

CONTRIBUTOR_COUNT=$(echo "$CONTRIBUTORS" | grep -c '^' || echo 0)
NEW_CONTRIBUTOR_COUNT=$(echo "$NEW_CONTRIBUTORS" | grep -c '^' || echo 0)

echo -e "${GREEN}Found $CONTRIBUTOR_COUNT contributors ($NEW_CONTRIBUTOR_COUNT new)${NC}" >&2
echo "" >&2

# Generate release notes based on format
generate_notes() {
    case $FORMAT in
        markdown|github)
            echo "## What's Changed in Gateway"
            echo ""
            echo "### Gateway Changes ($COMMIT_COUNT commits)"
            echo ""

            # Categorize commits with author attribution
            echo "$COMMITS" | while IFS= read -r line; do
                commit_hash=$(echo "$line" | awk '{print $1}')
                commit_msg=$(echo "$line" | cut -d' ' -f2-)

                # Get PR number from commit message
                pr_num=$(echo "$commit_msg" | grep -o '(#[0-9]*' | grep -o '[0-9]*' | head -1)

                # Try to get GitHub username from PR if gh CLI is available
                gh_user=""
                if [[ -n "$pr_num" ]] && command -v gh &> /dev/null; then
                    gh_user=$(gh pr view "$pr_num" --json author --jq '.author.login' 2>/dev/null || echo "")
                fi

                # Fallback: try to extract from email (works for users.noreply.github.com emails)
                if [[ -z "$gh_user" ]]; then
                    email=$(git show -s --format='%aE' "$commit_hash")
                    gh_user=$(echo "$email" | sed 's/@users\.noreply\.github\.com$//' | sed 's/^[0-9]*+//')
                    # If still contains @, it's not a GitHub username
                    if [[ "$gh_user" == *"@"* ]]; then
                        gh_user=""
                    fi
                fi

                # Format author link
                if [[ -n "$gh_user" ]]; then
                    author_link="by @$gh_user"
                else
                    # Final fallback: use full name
                    author=$(git show -s --format='%aN' "$commit_hash")
                    author_link="by $author"
                fi

                # Format PR link
                if [[ -n "$pr_num" ]]; then
                    pr_link="in https://github.com/sgl-project/sglang/pull/$pr_num"
                else
                    pr_link=""
                fi

                echo "- $commit_msg $author_link $pr_link"
            done

            # New Contributors section
            if [[ -n "$NEW_CONTRIBUTORS" ]] && [[ "$NEW_CONTRIBUTOR_COUNT" -gt 0 ]]; then
                echo ""
                echo "### New Contributors"
                echo ""
                while IFS= read -r contributor; do
                    if [[ -n "$contributor" ]]; then
                        # Extract name and email
                        name=$(echo "$contributor" | sed 's/ <.*//')
                        email=$(echo "$contributor" | sed 's/.*<\(.*\)>/\1/')

                        # Get their first commit
                        first_commit=$(git log "$PREV_TAG..$CURR_TAG" --author="$contributor" --format='%h' --reverse --no-merges "${PATH_ARGS[@]}" | head -1)

                        # Try to get GitHub username from first commit's PR
                        gh_user=""
                        if command -v gh &> /dev/null; then
                            commit_msg=$(git log --format=%s -n 1 "$first_commit")
                            pr_num=$(echo "$commit_msg" | grep -o '(#[0-9]*' | grep -o '[0-9]*' | head -1)
                            if [[ -n "$pr_num" ]]; then
                                gh_user=$(gh pr view "$pr_num" --json author --jq '.author.login' 2>/dev/null || echo "")
                            fi
                        fi

                        # Fallback: try to get GitHub username from email
                        if [[ -z "$gh_user" ]]; then
                            gh_user=$(echo "$email" | sed 's/@users\.noreply\.github\.com$//' | sed 's/^[0-9]*+//')
                            # If still contains @, it's not a GitHub username
                            if [[ "$gh_user" == *"@"* ]]; then
                                gh_user=""
                            fi
                        fi

                        if [[ -n "$gh_user" ]]; then
                            echo "* @$gh_user made their first contribution in https://github.com/sgl-project/sglang/commit/$first_commit"
                        else
                            echo "* $name made their first contribution in https://github.com/sgl-project/sglang/commit/$first_commit"
                        fi
                    fi
                done <<< "$NEW_CONTRIBUTORS"
            fi

            echo ""
            echo "### Paths Included"
            echo ""
            for path in "${GATEWAY_PATHS[@]}"; do
                echo "- \`$path\`"
            done
            echo ""
            echo "**Full Changelog**: https://github.com/sgl-project/sglang/compare/$PREV_TAG...$CURR_TAG"
            ;;
        plain)
            echo "Gateway/Router Release Notes: $CURR_TAG"
            echo "=========================================="
            echo ""
            echo "$COMMITS"
            echo ""
            echo "Contributors: $CONTRIBUTOR_COUNT ($NEW_CONTRIBUTOR_COUNT new)"
            ;;
    esac
}

# Output release notes
if [[ -n "$OUTPUT_FILE" ]]; then
    generate_notes > "$OUTPUT_FILE"
    echo -e "${GREEN}Release notes saved to: $OUTPUT_FILE${NC}" >&2
else
    generate_notes
fi

# Create GitHub release if requested
if [[ "$CREATE_RELEASE" == true ]]; then
    if ! command -v gh &> /dev/null; then
        echo -e "${YELLOW}Error: gh CLI not found. Install from https://cli.github.com/${NC}" >&2
        exit 1
    fi

    NOTES_FILE=$(mktemp)
    generate_notes > "$NOTES_FILE"

    # Default to draft if not explicitly set to false
    if [[ "$DRAFT_RELEASE" == "default" ]]; then
        DRAFT_RELEASE=true
    fi

    echo "" >&2
    if [[ "$DRAFT_RELEASE" == true ]]; then
        echo -e "${BLUE}Creating GitHub DRAFT release...${NC}" >&2
    else
        echo -e "${BLUE}Creating GitHub release (publishing immediately)...${NC}" >&2
    fi

    # Build gh command with optional --draft flag
    GH_ARGS=("$CURR_TAG" --title "Gateway/Router $CURR_TAG" --notes-file "$NOTES_FILE" --repo sgl-project/sglang)
    if [[ "$DRAFT_RELEASE" == true ]]; then
        GH_ARGS+=(--draft)
    fi

    gh release create "${GH_ARGS[@]}"

    rm -f "$NOTES_FILE"
    if [[ "$DRAFT_RELEASE" == true ]]; then
        echo -e "${GREEN}Draft release created successfully!${NC}" >&2
        echo -e "${YELLOW}Visit https://github.com/sgl-project/sglang/releases to review and publish${NC}" >&2
    else
        echo -e "${GREEN}Release published successfully!${NC}" >&2
    fi
fi
