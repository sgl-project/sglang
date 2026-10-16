#!/bin/bash
export ETCD_ENDPOINTS=http://<P0_HOST>:2379
export NATS_SERVER=nats://<P0_HOST>:4222
export DYN_REQUEST_PLANE=tcp
export DYN_SKIP_SGLANG_LOG_FORMATTING=1
export PIP_BREAK_SYSTEM_PACKAGES=1
DYN_LOCK_DIR="$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "${HOME:-/root}")" && ( flock -x 200; if [ -f "$DYN_LOCK_DIR/.srtctl_dynamo_install.complete" ]; then echo 'dynamo install already completed in this environment, skipping'; else { D=/configs/dynamo-wheels/61c37bc33c3e88da3facaa39fa2b5caf616740ba-e18e53fa && [ -f $D/.complete ] && echo "Installing dynamo from source (61c37bc33c3e88da3facaa39fa2b5caf616740ba + upstream e18e53fa3c attn_dp_size fix, $D)..." && pip install --break-system-packages --force-reinstall $D/ai_dynamo_runtime-*.whl && rm -rf /tmp/dynamo-src && mkdir -p /tmp/dynamo-src && tar -xzf $D/dynamo-src.tar.gz -C /tmp/dynamo-src && pip install --break-system-packages -e /tmp/dynamo-src/dynamo && grep -q sglang_dp_layout /tmp/dynamo-src/dynamo/components/src/dynamo/sglang/capacity.py && echo 'Dynamo installed from source (61c37bc33c3e88da3facaa39fa2b5caf616740ba + e18e53fa)' ; } && touch "$DYN_LOCK_DIR/.srtctl_dynamo_install.complete"; fi ) 200>"$DYN_LOCK_DIR/.srtctl_dynamo_install.lock" || exit 97
exec python3 -m dynamo.frontend --http-port=8180 --router-mode kv --router-session-affinity-ttl-secs 3600 --active-decode-blocks-threshold None --active-prefill-tokens-threshold None --active-prefill-tokens-threshold-frac None --router-replica-sync --router-queue-threshold None --router-temperature 1.0
