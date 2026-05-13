# SPDX-License-Identifier: Apache-2.0
"""Serve-time omni runtime bindings.

`sglang.omni.runtime` wires core omni requests into the running SRT server:
tokenizer-manager transport, scheduler-thread task state, and persistent omni
session records. It does not own ModelRunner KV/session execution; that lives in
`sglang.srt.omni_session.runtime`.
"""

