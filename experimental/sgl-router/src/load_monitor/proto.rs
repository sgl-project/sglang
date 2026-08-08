// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Generated protobuf and gRPC types for engine load reporting.

// tonic 0.12 emits fully-qualified marker and result paths. Keep the exception
// scoped to generated bindings so handwritten Router code retains the lint.
#![allow(unused_qualifications)]

tonic::include_proto!("sglang.router.loadmonitor.v1");
