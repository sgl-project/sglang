# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# Adapted from FlashInfer gvr_2 top-k (flashinfer/topk_varlen/kernels/gvr2_topk_{host,decode}.py) at flashinfer-ai/flashinfer
# main 449a35734 (Apache-2.0, FlashInfer). Additions for K31HT_DSA_SELECT are marked "SGLang:"; the index-output paths are
# unchanged. Used only under K31HT_DSA_SELECT (ops/attention/dsa/k31ht_select.py), default off.
# Standalone import: FlashInfer's JIT cache / availability helpers fall back on ImportError (in-process compile).
# FlashInfer NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA / Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
