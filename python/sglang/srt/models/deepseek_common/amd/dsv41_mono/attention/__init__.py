# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono layer's attention front (K1: wqkv, q / kv norms and the KV insert,
wq_b): ``front`` and its ``device`` primitives, ``gemv`` and ``plan``."""
