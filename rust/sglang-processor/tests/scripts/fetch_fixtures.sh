#!/bin/sh
# Downloads the parity fixtures into tests/fixtures. They live on Hugging Face,
# not in git, because recorded engine output is large. After regenerating them,
# upload, then pin REVISION to `hf datasets info $REPO --expand sha`:
#   hf upload --repo-type dataset --exclude '.cache/*' --delete '*.json' $REPO tests/fixtures .
set -e
REPO=sgl-project/sglang-processor-parity
REVISION=main
cd "$(dirname "$0")/../.."
hf download --repo-type dataset --revision "$REVISION" --local-dir tests/fixtures "$REPO"
