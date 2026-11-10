#!/usr/bin/env bash
# Official LFS SHA256 check for DeepSeek-V4.1-Flash shards.
exec python3 "$(dirname "$0")/verify_dsv41_flash_sha256.py" "$@"
