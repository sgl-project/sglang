// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn unavailable_indexer_degrades_to_empty_prefix_signal() {
    for error in [
        sgl_kv_indexer::PrefixIndexError::Overloaded,
        sgl_kv_indexer::PrefixIndexError::Timeout,
        sgl_kv_indexer::PrefixIndexError::Unreachable,
        sgl_kv_indexer::PrefixIndexError::QueryTooLarge,
    ] {
        assert_eq!(
            resolve_prefix_query(Err(error.clone()), "tiny").unwrap(),
            sgl_kv_indexer::PrefixOutcome::Empty,
            "{error} should degrade"
        );
    }
}

#[test]
fn rejected_indexer_query_still_fails_selection() {
    assert!(matches!(
        resolve_prefix_query(
            Err(sgl_kv_indexer::PrefixIndexError::Rejected(
                sgl_kv_indexer::RpcCode::InvalidArgument
            )),
            "tiny"
        ),
        Err(ApiError::PolicySelectionFailed { .. })
    ));
}
