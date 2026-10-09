// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Which HTTP request surface a proxied generation request arrived on. The
//! OpenAI chat surface and sglang's native `/generate` share one handler and
//! differ in where the prompt lives and which worker path they forward to.
//! The API profile's rules are applied per surface before the handler runs
//! (`ApiProfile::apply` / `ApiProfile::apply_generate`).

/// The request surface a generation request arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Surface {
    /// `POST /v1/chat/completions` — OpenAI chat shape.
    Chat,
    /// `POST /generate` — sglang-native shape (`text` / `input_ids` +
    /// `sampling_params`).
    Generate,
}

impl Surface {
    /// The path the request is proxied to on the worker. Today each surface
    /// forwards to its own spelling — the engine serves both.
    pub(crate) fn upstream_path(self) -> &'static str {
        match self {
            Surface::Chat => "/v1/chat/completions",
            Surface::Generate => "/generate",
        }
    }
}
