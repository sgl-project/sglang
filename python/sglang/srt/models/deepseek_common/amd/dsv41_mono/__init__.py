# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash mono decode layer: the FFN half of a decoder layer of a
decode step in one persistent FlyDSL launch, its TP all-reduces in-kernel. The
host side is ``runner``; SGLang docks it in ``..dsv41_mono_decode``."""
