// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Stub proxy — Task 7 replaces this with the full reqwest-backed implementation.

pub struct Proxy {
    pub worker_url: String,
}

impl Proxy {
    pub fn new(worker_url: String) -> Self {
        Self { worker_url }
    }
}
