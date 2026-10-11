# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash mono decode layer: a decoder layer of a decode step in
two persistent FlyDSL launches, K1 (the attention seam and front) and the FFN
half, its TP all-reduces in-kernel. The host side is ``runner``; SGLang docks
it in ``..dsv41_mono_decode``."""
