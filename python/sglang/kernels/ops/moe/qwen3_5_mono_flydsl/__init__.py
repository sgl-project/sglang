# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 persistent FlyDSL decode kernels for MI355X (gfx950), TP8.

Ported from vLLM (vllm-project/vllm#60833, Apache-2.0). The kernels build on
the Kimi-K3 mono framework in ``k3_mono_flydsl`` (``common/``, ``stages/``).

Nothing is imported eagerly: pulling in a submodule pulls in ``flydsl`` and
``aiter``, which a caller only probing availability should not pay for.
"""
