#!/bin/bash
# Launch sglang with the gfx1151-appropriate defaults, then hand off to the
# user's arguments. Exists because the defaults that matter here are CLI flags,
# not env vars, so they cannot be baked in with ENV:
#
#   --attention-backend triton
#       ServerArgs picks "aiter" on ROCm (server_args.py). aiter's attention
#       kernels are CDNA-only, so gfx1151 needs the Triton backend.
#
# Any flag the caller passes wins: the flag is only injected when absent.
# Bypass this wrapper entirely by invoking python3 -m sglang.launch_server.
set -euo pipefail

args=("$@")

has_flag() {
    local needle="$1"
    # ${args[@]+...} keeps `set -u` from treating an empty array as unbound,
    # which is what a bare `sglang_launch` would otherwise hit on bash < 4.4.
    for a in ${args[@]+"${args[@]}"}; do
        [[ "$a" == "$needle" || "$a" == "$needle="* ]] && return 0
    done
    return 1
}

has_flag --attention-backend || args+=(--attention-backend triton)

exec python3 -m sglang.launch_server "${args[@]}"
