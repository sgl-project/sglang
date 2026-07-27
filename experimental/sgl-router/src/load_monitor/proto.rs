// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Generated protobuf and gRPC types for engine load reporting.

// tonic 0.12 emits fully-qualified marker and result paths. Keep the exception
// scoped to generated bindings so handwritten Router code retains the lint.
#![allow(unused_qualifications)]

tonic::include_proto!("router.loadmonitor.v1");

#[cfg(test)]
mod tests {
    /// Verifies that the Router-local schema stays byte-for-byte aligned with
    /// the Python engine schema in the same SGLang checkout.
    #[test]
    fn router_proto_matches_python_engine_proto() {
        let router = include_str!("../../proto/load_monitor.proto");
        let engine =
            include_str!("../../../../python/sglang/srt/load_reporter/proto/load_monitor.proto");
        assert_eq!(
            router, engine,
            "Router and engine load-monitor protos diverged"
        );
        assert!(router.contains("package router.loadmonitor.v1;"));
        assert!(
            !router.contains("go_package"),
            "open-source load-monitor proto must not carry a Go package"
        );
    }
}
