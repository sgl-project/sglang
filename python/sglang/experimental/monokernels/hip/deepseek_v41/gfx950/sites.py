# SPDX-License-Identifier: Apache-2.0
"""Every guarded wait of the two launches: the mailbox regions, the epoch hand-off and K1's front flags. The kernels
register them in this order (a site's id is its index + 1); SGLANG_TEST_MONO_FAULT_INJECT names one of them. No
imports, so the config hook can read it."""

WAIT_SITES = (
    "attn",
    "pmix",
    "lin",
    "xrdy",
    "logit",
    "x8rdy",
    "smid",
    "route",
    "smq",
    "ugf",
    "ffn",
    "epoch",
    "front_flags",
    "front_qkvp",
)
