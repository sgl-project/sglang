// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes EndpointSlice discovery backend — stub.
//!
//! Real implementation lands in Task 12 (kube + k8s-openapi).

use crate::config::K8sDiscoveryConfig;
use anyhow::Result;
use tokio::sync::mpsc;

pub async fn spawn(
    _cfg: K8sDiscoveryConfig,
    _tx: mpsc::Sender<super::DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    anyhow::bail!("k8s discovery not implemented yet (Task 12)")
}
