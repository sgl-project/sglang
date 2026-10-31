# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 persistent FlyDSL decode kernels for MI355X (gfx950), TP8.

Ported from vLLM (vllm/models/kimi_k3/amd/mono, Apache-2.0). ``common/`` is in
turn adapted from ROCm/ATOM ``atom/mono`` at a526f0d (MIT); AMD's notice is
kept in those files.

Nothing is imported eagerly: pulling in a submodule pulls in ``flydsl`` and
``aiter``, which a caller only probing availability should not pay for.
"""
