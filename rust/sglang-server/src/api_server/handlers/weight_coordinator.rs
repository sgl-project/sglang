//! The cross-rank weight updates (`update_weights_from_disk`,
//! `begin_weight_update`, `end_weight_update`) reach every DP rank through
//! one coordinator, DP rank 0's listener, whichever listener the client sent
//! them to. The coordinator holds its update lock across the whole fan-out,
//! as Python's TokenizerManager holds its writer lock, and the peer relay
//! holds each other listener's lock only under it. So no two update locks
//! are ever waited for in opposite orders, and every scheduler applies the
//! updates in the order the coordinator's lock admits them. Another listener
//! forwards the update over [`WEIGHT_UPDATE_COORDINATOR_PATH`] and takes no
//! update lock of its own.
//!
//! [`WEIGHT_UPDATE_COORDINATOR_PATH`]: crate::api_server::core::WEIGHT_UPDATE_COORDINATOR_PATH

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;

use super::handle::CoreHandle;
use super::weights::{DiskUpdateOutcome, WeightUpdateOutcome};
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout::{ControlTarget, RankId};
use crate::message::io_struct::DiskWeightUpdate;

/// The DP rank whose listener coordinates. Fixed: there is no failover, so
/// while that listener is down every cross-rank weight update fails.
const COORDINATOR_RANK: RankId = 0;

/// How long the coordinator waits for each peer relay's reply. Arbitrary
/// and generous, chosen to cover a large model's disk load; safe to change.
pub(super) const PEER_REPLY_DEADLINE: Duration = Duration::from_secs(30 * 60);

/// Longer than [`PEER_REPLY_DEADLINE`], so the coordinator gives up first
/// and the client sees its error, which names each rank that failed.
const COORDINATOR_REPLY_DEADLINE: Duration = Duration::from_secs(31 * 60);

/// A weight update that must reach every DP rank, after the origin listener
/// validated it and filled its defaults; the coordinator route's JSON body.
/// The scheduler wire messages are encode-only, so they cannot cross this hop.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum CrossRankUpdate {
    UpdateWeightsFromDisk { update: DiskWeightUpdate },
    BeginWeightUpdate { selector: String },
    EndWeightUpdate,
}

impl CrossRankUpdate {
    /// The public route that takes this update.
    fn route(&self) -> &'static str {
        match self {
            Self::UpdateWeightsFromDisk { .. } => "/update_weights_from_disk",
            Self::BeginWeightUpdate { .. } => "/begin_weight_update",
            Self::EndWeightUpdate => "/end_weight_update",
        }
    }
}

/// The public route a coordinator route `body` stands for, whose auth level
/// the update is held to; `None` if `body` is not a cross-rank weight update.
pub(crate) fn coordinated_route(body: &[u8]) -> Option<&'static str> {
    serde_json::from_slice::<CrossRankUpdate>(body)
        .ok()
        .map(|update| update.route())
}

/// Every rank's outcome merged, as the origin returns it to its client; the
/// coordinator route's JSON reply.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum CrossRankOutcome {
    UpdateWeightsFromDisk {
        outcome: DiskUpdateOutcome,
    },
    /// `begin_weight_update` and `end_weight_update`.
    Session {
        outcome: WeightUpdateOutcome,
    },
}

impl CrossRankOutcome {
    pub(super) fn into_disk(self) -> Result<DiskUpdateOutcome, CoreError> {
        match self {
            Self::UpdateWeightsFromDisk { outcome } => Ok(outcome),
            other => Err(mismatched(&other)),
        }
    }

    pub(super) fn into_session(self) -> Result<WeightUpdateOutcome, CoreError> {
        match self {
            Self::Session { outcome } => Ok(outcome),
            other => Err(mismatched(&other)),
        }
    }
}

fn mismatched(outcome: &CrossRankOutcome) -> CoreError {
    CoreError::InvalidResponse(format!(
        "the weight-update coordinator answered another kind of update: {outcome:?}"
    ))
}

impl CoreHandle {
    /// Run `update` on every DP rank through the coordinator: here when this
    /// is DP rank 0's listener or has no peers, else on DP rank 0's.
    pub(super) async fn cross_rank_update(
        &self,
        update: CrossRankUpdate,
    ) -> Result<CrossRankOutcome, CoreError> {
        match self.coordinator_link()? {
            None => self.coordinate(update).await,
            Some(link) => forward_to_coordinator(link, &update).await,
        }
    }

    /// The peer to forward cross-rank weight updates to; `None` when this
    /// listener coordinates them itself.
    fn coordinator_link(&self) -> Result<Option<&Arc<dyn ControlTarget>>, CoreError> {
        let peers = &self.inner.peer_endpoints;
        if self.inner.dp_rank == COORDINATOR_RANK || peers.is_empty() {
            return Ok(None);
        }
        match peers
            .iter()
            .find(|peer| peer.ranks().contains(&COORDINATOR_RANK))
        {
            Some(link) => Ok(Some(link)),
            None => {
                tracing::error!(
                    dp_rank = self.inner.dp_rank,
                    "the endpoint table has no listener for DP rank 0, which coordinates weight updates"
                );
                Err(CoreError::Unavailable)
            }
        }
    }

    /// Serve a cross-rank weight update another listener forwarded: `body`
    /// is the coordinator route's JSON, the reply the merged outcome's. Any
    /// listener but DP rank 0's refuses it with 421 rather than forward it
    /// again, so a wrong endpoint table cannot loop it.
    pub(crate) async fn coordinate_forwarded_weight_update(
        &self,
        body: Bytes,
    ) -> Result<Bytes, CoreError> {
        if self.inner.dp_rank != COORDINATOR_RANK {
            return Err(CoreError::from_runtime_rejection(
                format!(
                    "dp rank {} does not coordinate weight updates; dp rank {COORDINATOR_RANK} does",
                    self.inner.dp_rank
                ),
                http::StatusCode::MISDIRECTED_REQUEST.as_u16(),
                Default::default(),
            ));
        }
        let update = serde_json::from_slice(&body).map_err(|error| {
            CoreError::InvalidArgument(format!("invalid cross-rank weight update: {error}"))
        })?;
        let outcome = self.coordinate(update).await?;
        serde_json::to_vec(&outcome)
            .map(Bytes::from)
            .map_err(|error| CoreError::Internal(error.to_string()))
    }

    /// Run `update` as the coordinator, in a task of its own: a caller that
    /// goes away (a client disconnect) must not stop an update midway, with
    /// some ranks on the new weights and the rest on the old.
    async fn coordinate(&self, update: CrossRankUpdate) -> Result<CrossRankOutcome, CoreError> {
        let core = self.clone();
        tokio::spawn(async move { core.apply_cross_rank_update(update).await })
            .await
            .unwrap_or_else(|error| {
                Err(CoreError::Internal(format!(
                    "the weight update's coordinator task failed: {error}"
                )))
            })
    }
}

/// Hand `update` to DP rank 0's listener over `link` and wait for its merged
/// outcome, within [`COORDINATOR_REPLY_DEADLINE`].
async fn forward_to_coordinator(
    link: &Arc<dyn ControlTarget>,
    update: &CrossRankUpdate,
) -> Result<CrossRankOutcome, CoreError> {
    let body =
        serde_json::to_vec(update).map_err(|error| CoreError::Internal(error.to_string()))?;
    let reply = tokio::time::timeout(
        COORDINATOR_REPLY_DEADLINE,
        link.coordinate_weight_update(Bytes::from(body)),
    )
    .await
    .map_err(|_| {
        CoreError::DeadlineExceeded(format!(
            "dp rank {COORDINATOR_RANK}'s listener did not answer the weight update within \
             {COORDINATOR_REPLY_DEADLINE:?}"
        ))
    })??;
    serde_json::from_slice(&reply).map_err(|error| {
        CoreError::InvalidResponse(format!(
            "invalid weight-update coordinator response: {error}"
        ))
    })
}
