# SPDX-License-Identifier: Apache-2.0
"""Prepared TP8 ordinary-decode rows, fixed before SGLang sizes the KV pool."""

DECODE_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64)
MAX_ROWS = 64


def prepared_widths(offered, capture_widths=None):
    """Merge the defaults and configured request buckets within the offered rows.

    This profile is q1: a request bucket is already the execution row count.
    Widths above 64 retain SGLang's native path. Unknown post-KV pool-clamp bins
    do too; this function is used only during the pre-KV resource build.
    """
    desired = set(DECODE_WIDTHS)
    if capture_widths is not None:
        if any(type(rows) is not int or rows <= 0 for rows in capture_widths):
            raise ValueError("Kimi capture request widths must be positive integers")
        desired.update(capture_widths)
    return tuple(
        sorted(
            {
                rows
                for rows in offered
                if type(rows) is int and 1 <= rows <= MAX_ROWS and rows in desired
            }
        )
    )
