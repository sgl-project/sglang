// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Component-scope integration tests.
//!
//! Each submodule exercises a single library component (policy, registry,
//! discovery, health, tokenizer) via the crate's public API. None of these
//! tests spin up the full HTTP router; for those see `tests/proxy/`.

mod discovery;
mod health;
mod policies;
mod tokenizer;
mod workers;
