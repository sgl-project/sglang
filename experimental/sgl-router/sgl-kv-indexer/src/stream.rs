// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Canonical identity for independently recoverable Worker/DP streams.

use std::fmt;

use tonic::Status;

use crate::pb::StreamId;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct StreamKey {
    pub namespace: String,
    pub worker_id: String,
    pub dp_rank: u32,
}

impl StreamKey {
    pub(crate) fn new(namespace: String, worker_id: String, dp_rank: u32) -> Result<Self, Status> {
        if worker_id.trim().is_empty() {
            return Err(Status::invalid_argument("worker_id must not be empty"));
        }
        Ok(Self {
            namespace,
            worker_id,
            dp_rank,
        })
    }

    /// New callers send StreamId. Legacy calls retain the historical
    /// namespace="", dp_rank=0 identity so old Bridge deployments remain usable.
    pub(crate) fn from_wire(
        stream_id: Option<StreamId>,
        legacy_worker_id: &str,
    ) -> Result<Self, Status> {
        match stream_id {
            Some(stream) => {
                if !legacy_worker_id.is_empty() && legacy_worker_id != stream.worker_id {
                    return Err(Status::invalid_argument(
                        "worker_id and stream_id.worker_id must match",
                    ));
                }
                Self::new(stream.namespace, stream.worker_id, stream.dp_rank)
            }
            None => Self::new(String::new(), legacy_worker_id.to_owned(), 0),
        }
    }

    pub(crate) fn to_wire(&self) -> StreamId {
        StreamId {
            namespace: self.namespace.clone(),
            worker_id: self.worker_id.clone(),
            dp_rank: self.dp_rank,
        }
    }
}

impl fmt::Display for StreamKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.namespace, self.worker_id, self.dp_rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_identity_maps_to_default_namespace_and_rank() {
        assert_eq!(
            StreamKey::from_wire(None, "worker").unwrap(),
            StreamKey {
                namespace: String::new(),
                worker_id: "worker".into(),
                dp_rank: 0,
            }
        );
    }

    #[test]
    fn rejects_conflicting_legacy_and_canonical_ids() {
        let error = StreamKey::from_wire(
            Some(StreamId {
                namespace: "ns".into(),
                worker_id: "canonical".into(),
                dp_rank: 1,
            }),
            "legacy",
        )
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }
}
