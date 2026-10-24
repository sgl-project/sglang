//! Control-request fan-out and reply collection.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::future::BoxFuture;
use serde::de::DeserializeOwned;

use super::CoreError;
use crate::message::io_struct::ControlRequest;

/// The DP rank a reply came from: `deployment_info.dp_rank`, which spans
/// `0..dp_size` whether the deployment uses attention DP or plain DP.
pub(crate) type RankId = u32;

/// One rank's answer: the scheduler's encoded control output, or why this
/// rank produced none.
#[derive(Debug)]
pub(crate) struct RankReply {
    pub(crate) rank: RankId,
    pub(crate) outcome: Result<Bytes, CoreError>,
}

/// Builds the request for one target. Called once per target, so every
/// scheduler sees its own rid: a [`ControlRequest`] owns its rid and is not
/// `Clone`, and a shared rid would make per-rank logs ambiguous.
pub(crate) type RequestBuilder<'a> = dyn Fn() -> ControlRequest + Sync + 'a;

/// A target may answer for several ranks when it collects them before replying.
pub(crate) trait ControlTarget: Send + Sync {
    /// The ranks this target answers for. The fan-out reports a rank the
    /// target leaves out as failed, so an impl need not synthesize replies.
    fn ranks(&self) -> &[RankId];

    /// Deliver one request built by `build` and collect this target's replies.
    fn control<'a>(&'a self, build: &'a RequestBuilder<'a>) -> BoxFuture<'a, Vec<RankReply>>;
}

/// Every rank's reply to one control request, in DP-rank order.
#[derive(Debug)]
pub(crate) struct Replies {
    replies: Vec<RankReply>,
}

/// Deliver one control request to every DP rank.
pub(crate) async fn fan_out(
    local: RankId,
    local_call: impl Future<Output = Result<Bytes, CoreError>>,
    targets: &[Arc<dyn ControlTarget>],
    build: &RequestBuilder<'_>,
    deadline: Option<Duration>,
) -> Replies {
    let local_ranks = [local];
    let local_call = within(
        deadline,
        async move {
            vec![RankReply {
                rank: local,
                outcome: local_call.await,
            }]
        },
        &local_ranks,
    );
    let target_calls = futures::future::join_all(targets.iter().map(|target| async move {
        let expected = target.ranks();
        let mut replies = within(deadline, target.control(build), expected).await;
        // A target that answers for fewer ranks than it claims would otherwise
        // make the operation look like it reached fewer ranks than exist.
        for &rank in expected {
            if !replies.iter().any(|reply| reply.rank == rank) {
                replies.push(RankReply {
                    rank,
                    outcome: Err(CoreError::Internal(format!(
                        "dp rank {rank}: its listener returned no reply"
                    ))),
                });
            }
        }
        replies
    }));
    let (mut replies, target_replies) = futures::join!(local_call, target_calls);
    replies.extend(target_replies.into_iter().flatten());
    // Rank order, not arrival order: `ranks[i]` means the same rank on every
    // call. Stable, so a misconfiguration that names one rank twice is visible
    // as two adjacent entries rather than hidden.
    replies.sort_by_key(|reply| reply.rank);
    Replies { replies }
}

async fn within(
    deadline: Option<Duration>,
    call: impl Future<Output = Vec<RankReply>>,
    ranks: &[RankId],
) -> Vec<RankReply> {
    let Some(deadline) = deadline else {
        return call.await;
    };
    match tokio::time::timeout(deadline, call).await {
        Ok(replies) => replies,
        Err(_) => ranks
            .iter()
            .map(|&rank| RankReply {
                rank,
                outcome: Err(CoreError::DeadlineExceeded(format!(
                    "dp rank {rank}: no reply within {deadline:?}"
                ))),
            })
            .collect(),
    }
}

impl Replies {
    /// Decode every rank's reply as `T`, in rank order. Any rank that failed
    /// or returned an undecodable payload fails the whole operation; the
    /// error names each such rank. `what` names the operation in that error.
    pub(crate) fn decode<T: DeserializeOwned>(self, what: &str) -> Result<Vec<T>, CoreError> {
        let total = self.replies.len();
        let mut values = Vec::with_capacity(total);
        let mut failures = Vec::new();
        for RankReply { rank, outcome } in self.replies {
            let decoded = outcome.and_then(|bytes| {
                rmp_serde::from_slice::<T>(&bytes).map_err(|error| {
                    CoreError::InvalidResponse(format!("invalid {what} response: {error}"))
                })
            });
            match decoded {
                Ok(value) => values.push(value),
                Err(error) => failures.push((rank, error)),
            }
        }
        if failures.is_empty() {
            Ok(values)
        } else {
            Err(collapse(total, failures))
        }
    }
}

/// One error for an operation that failed on some ranks.
///
/// A single-rank deployment's error is that rank's own, unchanged, so the
/// local path keeps its exact categories and messages (a truncated reply
/// still reaches the HTTP adapter as `ResponseTruncated`). With several
/// ranks the error names each failing rank. Its category is the first
/// failure's that carries a message, so a client sees a 400 for a rejected
/// request even when another rank also timed out; failures that all share
/// one bare category (`Unavailable`, `ResponseTruncated`) keep it, which the
/// HTTP handlers match exactly for their 503 and legacy 499.
fn collapse(total: usize, mut failures: Vec<(RankId, CoreError)>) -> CoreError {
    if total == 1 && failures.len() == 1 {
        return failures.pop().expect("one failure").1;
    }
    let bare =
        |error: &CoreError| matches!(error, CoreError::Unavailable | CoreError::ResponseTruncated);
    let lead = failures
        .iter()
        .find(|(_, error)| !bare(error))
        .map(|(_, error)| error.clone());
    let detail = failures
        .iter()
        .map(|(rank, error)| format!("dp rank {rank}: {error}"))
        .collect::<Vec<_>>()
        .join("; ");
    let message = format!(
        "control request failed on {} of {total} dp ranks: {detail}",
        failures.len()
    );
    match lead {
        Some(CoreError::InvalidArgument(_)) => CoreError::InvalidArgument(message),
        Some(CoreError::Overloaded(_)) => CoreError::Overloaded(message),
        Some(CoreError::Cancelled(_)) => CoreError::Cancelled(message),
        Some(CoreError::DeadlineExceeded(_)) => CoreError::DeadlineExceeded(message),
        Some(CoreError::InvalidResponse(_)) => CoreError::InvalidResponse(message),
        Some(CoreError::RuntimeRejected {
            kind,
            legacy_http_status,
            ..
        }) => CoreError::RuntimeRejected {
            kind,
            message,
            legacy_http_status,
        },
        Some(CoreError::Internal(_)) => CoreError::Internal(message),
        // Every failure is bare: keep the category they all share, else
        // the mix has no single category to keep.
        Some(CoreError::Unavailable | CoreError::ResponseTruncated) | None => {
            let (_, first) = &failures[0];
            if failures
                .iter()
                .all(|(_, error)| std::mem::discriminant(error) == std::mem::discriminant(first))
            {
                first.clone()
            } else {
                CoreError::Internal(message)
            }
        }
    }
}

/// Python's `FanOutCommunicator.merge_results` for the `{success, message}`
/// replies: the operation succeeded only where every rank did, and the
/// messages are joined with ` | ` in rank order. Empty messages are left out,
/// where Python keeps them as blank separators, so a one-rank message reads as
/// that message and the handlers' "no message" checks keep working.
pub(crate) fn merge_status<'a>(
    replies: impl IntoIterator<Item = (bool, &'a str)>,
) -> (bool, String) {
    let mut success = true;
    let mut messages = Vec::new();
    for (rank_success, message) in replies {
        success &= rank_success;
        if !message.is_empty() {
            messages.push(message);
        }
    }
    (success, messages.join(" | "))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A target that answers from a closure, after an optional delay.
    pub(crate) struct DummyTarget {
        ranks: Vec<RankId>,
        delay: Duration,
        reply: Arc<dyn Fn(RankId) -> Result<Bytes, CoreError> + Send + Sync>,
    }

    impl DummyTarget {
        pub(crate) fn replying(
            ranks: &[RankId],
            reply: impl Fn(RankId) -> Result<Bytes, CoreError> + Send + Sync + 'static,
        ) -> Arc<dyn ControlTarget> {
            Self::delayed(ranks, Duration::ZERO, reply)
        }

        pub(crate) fn delayed(
            ranks: &[RankId],
            delay: Duration,
            reply: impl Fn(RankId) -> Result<Bytes, CoreError> + Send + Sync + 'static,
        ) -> Arc<dyn ControlTarget> {
            Arc::new(Self {
                ranks: ranks.to_vec(),
                delay,
                reply: Arc::new(reply),
            })
        }
    }

    impl ControlTarget for DummyTarget {
        fn ranks(&self) -> &[RankId] {
            &self.ranks
        }

        fn control<'a>(&'a self, build: &'a RequestBuilder<'a>) -> BoxFuture<'a, Vec<RankReply>> {
            Box::pin(async move {
                let _request = build();
                tokio::time::sleep(self.delay).await;
                self.ranks
                    .iter()
                    .map(|&rank| RankReply {
                        rank,
                        outcome: (self.reply)(rank),
                    })
                    .collect()
            })
        }
    }

    /// The msgpack a scheduler's `{success, message}` output decodes from.
    pub(crate) fn status_bytes(success: bool, message: &str) -> Bytes {
        Bytes::from(
            rmp_serde::to_vec_named(&serde_json::json!({
                "success": success,
                "message": message,
            }))
            .unwrap(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::testing::{DummyTarget, status_bytes};
    use super::*;
    use crate::api_server::core::CoreErrorKind;
    use crate::message::ids::Rid;
    use crate::message::io_struct::GetInternalStateReq;

    fn build() -> ControlRequest {
        ControlRequest::GetInternalStateReq(GetInternalStateReq::new(Rid::new().to_string()))
    }

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Status {
        success: bool,
        #[serde(default)]
        message: String,
    }

    fn ranks_of(replies: &Replies) -> Vec<RankId> {
        replies.replies.iter().map(|reply| reply.rank).collect()
    }

    fn messages(replies: Replies) -> Vec<String> {
        replies
            .decode::<Status>("status")
            .unwrap()
            .into_iter()
            .map(|status| status.message)
            .collect()
    }

    /// Replies line up by rank however the targets are listed and whenever
    /// they answer, so `ranks[i]` is the same rank on every call.
    #[tokio::test]
    async fn replies_are_in_rank_order_not_arrival_order() {
        let peers: Vec<Arc<dyn ControlTarget>> = vec![
            DummyTarget::delayed(&[2], Duration::from_millis(30), |_| {
                Ok(status_bytes(true, "rank 2"))
            }),
            DummyTarget::delayed(&[0], Duration::from_millis(10), |_| {
                Ok(status_bytes(true, "rank 0"))
            }),
        ];
        let replies = fan_out(
            1,
            async { Ok(status_bytes(true, "local")) },
            &peers,
            &build,
            None,
        )
        .await;
        assert_eq!(ranks_of(&replies), [0, 1, 2]);
        assert_eq!(messages(replies), ["rank 0", "local", "rank 2"]);
    }

    /// Every target is in flight at once: two targets that each wait for the
    /// other finish, where a sequential fan-out would deadlock.
    #[tokio::test]
    async fn targets_are_in_flight_concurrently() {
        struct Meet(Arc<tokio::sync::Barrier>, [RankId; 1]);
        impl ControlTarget for Meet {
            fn ranks(&self) -> &[RankId] {
                &self.1
            }
            fn control<'a>(&'a self, _: &'a RequestBuilder<'a>) -> BoxFuture<'a, Vec<RankReply>> {
                Box::pin(async move {
                    self.0.wait().await;
                    vec![RankReply {
                        rank: self.1[0],
                        outcome: Ok(Bytes::new()),
                    }]
                })
            }
        }
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let peers: Vec<Arc<dyn ControlTarget>> = vec![
            Arc::new(Meet(barrier.clone(), [1])),
            Arc::new(Meet(barrier, [2])),
        ];
        let replies = tokio::time::timeout(
            Duration::from_secs(5),
            fan_out(0, async { Ok(Bytes::new()) }, &peers, &build, None),
        )
        .await
        .expect("concurrent targets must not deadlock");
        assert_eq!(ranks_of(&replies), [0, 1, 2]);
    }

    /// Each target builds its own request, so every scheduler sees a fresh rid.
    #[tokio::test]
    async fn every_target_gets_its_own_request() {
        let built = Arc::new(AtomicUsize::new(0));
        let peers: Vec<Arc<dyn ControlTarget>> = vec![
            DummyTarget::replying(&[1], |_| Ok(Bytes::new())),
            DummyTarget::replying(&[2], |_| Ok(Bytes::new())),
        ];
        let counted = built.clone();
        let build = move || {
            counted.fetch_add(1, Ordering::Relaxed);
            build()
        };
        let local_request = build();
        fan_out(
            0,
            async move {
                drop(local_request);
                Ok(Bytes::new())
            },
            &peers,
            &build,
            None,
        )
        .await;
        assert_eq!(built.load(Ordering::Relaxed), 1 + peers.len());
    }

    /// A deadline fails only the ranks of the target that missed it; the
    /// ranks that answered keep their replies.
    #[tokio::test]
    async fn a_missed_deadline_fails_only_that_target() {
        struct Silent([RankId; 2]);
        impl ControlTarget for Silent {
            fn ranks(&self) -> &[RankId] {
                &self.0
            }
            fn control<'a>(&'a self, _: &'a RequestBuilder<'a>) -> BoxFuture<'a, Vec<RankReply>> {
                Box::pin(std::future::pending())
            }
        }
        let peers: Vec<Arc<dyn ControlTarget>> = vec![
            Arc::new(Silent([1, 2])),
            DummyTarget::replying(&[3], |_| Ok(status_bytes(true, ""))),
        ];
        let replies = fan_out(
            0,
            async { Ok(status_bytes(true, "")) },
            &peers,
            &build,
            Some(Duration::from_millis(50)),
        )
        .await;
        assert_eq!(ranks_of(&replies), [0, 1, 2, 3]);
        assert!(replies.replies[0].outcome.is_ok());
        assert!(replies.replies[3].outcome.is_ok());
        for reply in &replies.replies[1..3] {
            let error = reply.outcome.as_ref().unwrap_err();
            assert_eq!(error.kind(), CoreErrorKind::DeadlineExceeded);
            assert!(
                error
                    .to_string()
                    .starts_with(&format!("dp rank {}:", reply.rank)),
                "{error}"
            );
        }
        let error = replies.decode::<Status>("status").unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::DeadlineExceeded);
        assert!(error.to_string().contains("2 of 4 dp ranks"), "{error}");
    }

    /// A target that claims a rank but returns nothing for it is reported for
    /// that rank, so the operation cannot silently cover fewer ranks.
    #[tokio::test]
    async fn a_rank_the_target_skips_is_reported_as_failed() {
        struct Partial([RankId; 2]);
        impl ControlTarget for Partial {
            fn ranks(&self) -> &[RankId] {
                &self.0
            }
            fn control<'a>(&'a self, _: &'a RequestBuilder<'a>) -> BoxFuture<'a, Vec<RankReply>> {
                Box::pin(async move {
                    vec![RankReply {
                        rank: self.0[0],
                        outcome: Ok(status_bytes(true, "")),
                    }]
                })
            }
        }
        let peers: Vec<Arc<dyn ControlTarget>> = vec![Arc::new(Partial([1, 2]))];
        let replies = fan_out(
            0,
            async { Ok(status_bytes(true, "")) },
            &peers,
            &build,
            None,
        )
        .await;
        assert_eq!(ranks_of(&replies), [0, 1, 2]);
        let error = replies.decode::<Status>("status").unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::Internal);
        assert!(error.to_string().contains("dp rank 2:"), "{error}");
    }

    /// Decoding names the rank whose payload was bad and keeps the
    /// invalid-response category.
    #[tokio::test]
    async fn an_undecodable_payload_is_attributed_to_its_rank() {
        let peers: Vec<Arc<dyn ControlTarget>> = vec![DummyTarget::replying(&[1], |_| {
            Ok(Bytes::from_static(b"\xc1"))
        })];
        let replies = fan_out(
            0,
            async { Ok(status_bytes(true, "")) },
            &peers,
            &build,
            None,
        )
        .await;
        let error = replies.decode::<Status>("flush-cache").unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::Internal);
        let text = error.to_string();
        assert!(text.contains("1 of 2 dp ranks"), "{text}");
        assert!(
            text.contains("dp rank 1: invalid flush-cache response"),
            "{text}"
        );
    }

    /// A single rank's error passes through untouched; across ranks, only
    /// wholly bare failures keep their category, a mixed set keeps the first
    /// other failure's category and names every rank.
    #[test]
    fn collapsed_errors_keep_the_most_specific_category() {
        let alone = collapse(1, vec![(0, CoreError::InvalidArgument("bad".into()))]);
        assert_eq!(alone.to_string(), "bad");
        assert!(matches!(
            collapse(1, vec![(0, CoreError::ResponseTruncated)]),
            CoreError::ResponseTruncated
        ));
        assert!(matches!(
            collapse(
                2,
                vec![(0, CoreError::Unavailable), (1, CoreError::Unavailable)]
            ),
            CoreError::Unavailable
        ));
        assert!(matches!(
            collapse(
                3,
                vec![
                    (0, CoreError::Unavailable),
                    (1, CoreError::ResponseTruncated)
                ]
            ),
            CoreError::Internal(_)
        ));
        let mixed = collapse(
            3,
            vec![
                (0, CoreError::Unavailable),
                (1, CoreError::InvalidArgument("bad timeout".into())),
                (2, CoreError::Internal("boom".into())),
            ],
        );
        assert_eq!(mixed.kind(), CoreErrorKind::InvalidArgument);
        assert_eq!(
            mixed.to_string(),
            "control request failed on 3 of 3 dp ranks: dp rank 0: service unavailable; \
             dp rank 1: bad timeout; dp rank 2: boom"
        );
        let rejected = collapse(
            2,
            vec![(1, CoreError::from_runtime_rejection("busy".into(), 412))],
        );
        assert_eq!(rejected.kind(), CoreErrorKind::FailedPrecondition);
        assert_eq!(rejected.http_status().as_u16(), 412);
    }

    #[test]
    fn merge_status_follows_python_merge_results() {
        assert_eq!(merge_status([(true, "")]), (true, String::new()));
        assert_eq!(merge_status([(true, "ok")]), (true, "ok".to_owned()));
        assert_eq!(
            merge_status([(true, ""), (false, "rank 1 busy"), (false, "rank 2 busy")]),
            (false, "rank 1 busy | rank 2 busy".to_owned())
        );
    }
}
