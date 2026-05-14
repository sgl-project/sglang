# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic omni contracts and orchestration.

`sglang.omni.core` describes what happens in an omni request: common protocol
objects, interleaved AR/media boundaries, and the coordinator loop. It must not
own SRT scheduler state, HTTP transport details, or model-specific token grammar.
"""
