# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from vllm-project/vllm @ 193922d6ff: vllm/models/deepseek_v41/amd/mono/__init__.py
"""DeepSeek-V4.1-Flash mono decode layer: one decoder layer of a decode step in
two persistent FlyDSL launches, its TP all-reduces in-kernel (README.md). The
host side is ``runner``; vLLM docks it in ``..mono_decode``."""
