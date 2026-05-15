// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

// Stub — Task 5 fills this in.
use crate::discovery::DiscoveryEvent;
use crate::workers::WorkerRegistry;
use std::sync::Arc;
use tokio::sync::mpsc;

pub async fn run(_rx: mpsc::Receiver<DiscoveryEvent>, _registry: Arc<WorkerRegistry>) {
    // Task 5 — drives registry mutations from event stream.
}
