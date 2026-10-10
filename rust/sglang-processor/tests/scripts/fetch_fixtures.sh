#!/bin/sh
# Downloads the parity fixtures into tests/fixtures. They live on Hugging Face,
# not in git, because recorded engine output is large. After regenerating them,
# upload, then pin REVISION to `hf datasets info $REPO --expand sha`:
#   hf upload --repo-type dataset --exclude '.cache/*' --delete '*.json' $REPO tests/fixtures .
set -e
REPO=sgl-project/sglang-processor-parity
REVISION=main
cd "$(dirname "$0")/../.."
# Keep fixtures already at REVISION, so regenerated ones survive until uploaded.
if [ "$(cat tests/fixtures/.cache/revision 2>/dev/null)" = "$REVISION" ]; then
    exit 0
fi
fixtures_tmp=$(mktemp -d tests/fixtures.XXXXXX)
trap 'rm -rf "$fixtures_tmp"' EXIT
hf download --repo-type dataset --revision "$REVISION" --local-dir "$fixtures_tmp" "$REPO"
# Replace only after a successful download, including removal of obsolete files.
rm -rf tests/fixtures
mv "$fixtures_tmp" tests/fixtures
echo "$REVISION" > tests/fixtures/.cache/revision
