"""Ref/real-independent invariant assertions for kv_canary kernel tests.

Each invariant only looks at the kernel's inputs and outputs (shape relationships, monotonicity, tail
positions, etc.) — it must never re-implement the reference algorithm. Hand and fuzz tests both call
into this module so a single contract violation surfaces consistently.

This file is intentionally left as a skeleton; concrete invariants are introduced in a follow-up.
"""

from __future__ import annotations
