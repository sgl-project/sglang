// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType, TierType,
};

pub fn hbm() -> i32 {
    TierType::TierHbm as i32
}

pub fn dram() -> i32 {
    TierType::TierDram as i32
}

pub fn hashes(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

pub fn action(kind: ExternalKvActionType, tier: i32, values: &[&str]) -> ExternalKvAction {
    ExternalKvAction {
        r#type: kind as i32,
        tier,
        hashes: hashes(values),
        component_masks: Vec::new(),
        block_sizes: Vec::new(),
    }
}

/// A component-aware REPORT action: each hash carries its component bitmask and
/// token count, index-aligned with `values`.
#[allow(dead_code)] // used by redis_integration, not grpc_contract
pub fn component_report(
    tier: i32,
    values: &[&str],
    masks: &[u32],
    block_sizes: &[u32],
) -> ExternalKvAction {
    ExternalKvAction {
        r#type: ExternalKvActionType::ActionReport as i32,
        tier,
        hashes: hashes(values),
        component_masks: masks.to_vec(),
        block_sizes: block_sizes.to_vec(),
    }
}

pub fn apply_request(
    worker: &str,
    address: &str,
    seq: u64,
    actions: Vec<ExternalKvAction>,
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq,
        actions,
        worker_address: address.to_string(),
        cache_spec: None,
    }
}
