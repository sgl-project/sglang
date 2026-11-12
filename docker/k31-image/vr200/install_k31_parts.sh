#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# Install the FlashInfer DynBlock parts built by build_fi_sm107.py into the image's site-packages:
#   - the patched RoutingCustomPolicy.cuh (DynBlockKernelMaxNumExperts = 1024 for K3.1's 896 experts);
#   - fused_moe_trtllm_sm107.so and its k31-dynblock.json marker, over the flashinfer-jit-cache sm107a copy.
# Optional second argument: a directory holding a flashmla_ops.abi3.so to install over sgl_kernel's; the K3.1 points of
# docker/k31-vr200.Dockerfile pass "none" (they ran the image's FlashMLA).
# Prints the header line, the sha256 prefixes of both modules and a K31_PARTS_OK line (checked by the image build).
# Usage: install_k31_parts.sh FI_DIR [FLASHMLA_DIR|none]
set -euo pipefail
FI=${1:?usage: install_k31_parts.sh FI_DIR [FLASHMLA_DIR|none]}; FM=${2:-none}
SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cp "$FI/RoutingCustomPolicy.cuh" "$SITE/flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
n=0
for d in "$SITE"/flashinfer_jit_cache/providers/sm107a/jit_cache/fused_moe_trtllm_sm107; do
  [ -d "$d" ] || continue
  cp "$FI/fused_moe_trtllm_sm107.so" "$d/fused_moe_trtllm_sm107.so"; cp "$FI/k31-dynblock.json" "$d/"; n=$((n + 1))
done
[ "$n" -gt 0 ] || { echo "install_k31_parts: no sm107a jit-cache copy of fused_moe_trtllm_sm107 under $SITE" >&2; exit 1; }
if [ "$FM" != none ]; then cp "$FM/flashmla_ops.abi3.so" "$SITE/sgl_kernel/flashmla_ops.abi3.so"; fi
grep -o "DynBlockKernelMaxNumExperts = [0-9]*" "$SITE/flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
sha256sum "$SITE"/flashinfer_jit_cache/providers/sm107a/jit_cache/fused_moe_trtllm_sm107/fused_moe_trtllm_sm107.so "$SITE/sgl_kernel/flashmla_ops.abi3.so" | cut -c1-16
echo "K31_PARTS_OK fi=$(tr -d '\n ' < "$FI/k31-dynblock.json" | cut -c1-80) flashmla=$( [ "$FM" != none ] && cat "$FM/SOURCE" 2>/dev/null || echo image-pypi)"
