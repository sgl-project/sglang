//! TokenizerManager — owns the request lifecycle across two isolated threads:
//!
//!   * [`ingress`] — drives the ingress FSM (Received → Validating →
//!     Normalizing → {Tokenizing | PreSendValidating}) and pushes tokenized requests to the
//!     scheduler ring.
//!   * [`egress`] — drains the scheduler-output ring and routes each chunk to
//!     the owning detokenizer shard.
//!
//! The two run on separate pinned threads with no shared state, connected to
//! the rest of the pipeline only through `flume` channels: [`TmEvent`] into the
//! ingress loop, [`Senders`] fanning out to the pools.

mod egress;
mod ingress;

pub use egress::{ActivityCounter, Egress};
pub use ingress::{Ingress, Limits};

use crate::error::Error;
use crate::ids::Rid;
use crate::message::{DetokMsg, Request, TokenIds};

/// Blocking receive that also wakes on shutdown: returns `None` when `rx` closes
/// *or* the `shutdown` sender is dropped.
pub fn recv<T>(rx: &flume::Receiver<T>, shutdown: &flume::Receiver<()>) -> Option<T> {
    flume::Selector::new()
        .recv(rx, |r| r.ok())
        .recv(shutdown, |_| None)
        .wait()
}

/// Events into the TokenizerManager ingress loop. API server + tokenizer pool
/// share this one inbox, keeping the loop a single consumer (no `select`).
pub enum TmEvent {
    /// A freshly received request from the API server.
    Ingress(Request),
    /// A request back from the tokenizer pool: `PreSendValidating` (ids filled) on success,
    /// or `Failed` on a tokenize error. `drive` handles both.
    Tokenized(Request),
}

/// Producer-side handles, cloned into every stage that needs to emit.
/// Who asked for an abort. Both variants do the same work in
/// [`Ingress::on_abort`](crate::tokenizer_manager::ingress::Ingress) — deregister
/// the detok entry, tell the scheduler to stop — and the source is kept for
/// diagnostics.
///
/// There is no in-flight rid registry to keep consistent, and so no release
/// ordering to get wrong: [`Rid::from_client`] makes every client-supplied rid
/// internally unique, so a resubmit of the "same" rid is a different `Rid` and
/// cannot be tangled up with an abort still in flight for the original.
#[derive(Clone, Debug)]
pub enum AbortSource {
    /// From an `AbortGuard` drop. Owns the release.
    Guard(Rid),
    /// From a detokenizer terminal path. Aborts the scheduler work.
    Detok(Rid),
}

impl AbortSource {
    pub fn rid(&self) -> &Rid {
        match self {
            Self::Guard(rid) | Self::Detok(rid) => rid,
        }
    }
}

#[derive(Clone)]
pub struct Senders {
    /// → TokenizerManager ingress loop.
    pub tm: flume::Sender<TmEvent>,
    /// → the same loop, but UNBOUNDED and abort-only.
    ///
    /// Aborts cannot share the bounded inbox. `try_send` there drops them exactly
    /// when they matter most — under overload — leaving the scheduler generating
    /// for a dead connection; and the caller then faces a false choice between
    /// releasing the rid (a live entry can be overwritten by a resubmit) and
    /// holding it (a permanent leak). An unbounded lane removes the dilemma: an
    /// abort is a small `String` and is always accepted, so releases can be
    /// unconditional again. It cannot grow without bound in practice — one entry
    /// per in-flight request, each already bounded by the inbox that admitted it.
    pub abort: flume::Sender<AbortSource>,
    /// → Tokenizer pool (CPU-bound, pinned threads).
    pub tok: flume::Sender<Request>,
    /// → Detokenizer shards, indexed by `Rid::shard(detok.len())`.
    pub detok: Vec<flume::Sender<DetokMsg>>,
}

impl Senders {
    #[inline]
    pub fn detok_for(&self, rid: &Rid) -> &flume::Sender<DetokMsg> {
        &self.detok[rid.shard(self.detok.len())]
    }

    /// Decode one complete token-id sequence on a detokenizer worker.
    ///
    /// Unlike request streaming decode, this has no request registration or
    /// incremental state. The oneshot carries only the reply; callers depend on
    /// this service method rather than owning a concrete tokenizer.
    pub async fn decode_once(&self, rid: &Rid, token_ids: TokenIds) -> Result<String, Error> {
        if self.detok.is_empty() {
            return Err(Error::Internal(
                "no detokenizer worker is configured".into(),
            ));
        }
        let detok = self.detok_for(rid);
        let token_ids = token_ids
            .into_iter()
            .map(|id| {
                u32::try_from(id)
                    .map_err(|_| Error::Validation(format!("Token ID {id} is out of range")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (reply, response) = tokio::sync::oneshot::channel();
        detok
            .send_async(DetokMsg::Decode { token_ids, reply })
            .await
            .map_err(|_| Error::Internal("detokenizer worker is unavailable".into()))?;
        response
            .await
            .map_err(|_| Error::Internal("detokenizer worker dropped the decode reply".into()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn senders(detok: Vec<flume::Sender<DetokMsg>>) -> Senders {
        Senders {
            tm: flume::unbounded().0,
            abort: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok,
        }
    }

    #[tokio::test]
    async fn decode_once_round_trips_the_worker_reply() {
        let (detok_tx, detok_rx) = flume::bounded(1);
        let task = tokio::spawn(async move {
            senders(vec![detok_tx])
                .decode_once(&Rid::from("decode"), vec![1, 2])
                .await
        });

        let DetokMsg::Decode { token_ids, reply } = detok_rx.recv_async().await.unwrap() else {
            panic!("expected a one-shot decode request");
        };
        assert_eq!(token_ids, [1, 2]);
        reply.send(Ok("decoded".into())).unwrap();

        assert_eq!(task.await.unwrap().unwrap(), "decoded");
    }

    #[tokio::test]
    async fn decode_once_rejects_missing_workers_and_negative_ids() {
        let rid = Rid::from("decode");
        let error = senders(vec![])
            .decode_once(&rid, vec![1])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Internal(_)));

        let (detok_tx, _detok_rx) = flume::bounded(1);
        let error = senders(vec![detok_tx])
            .decode_once(&rid, vec![-1])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
    }

    #[tokio::test]
    async fn decode_once_uses_the_request_shard() {
        let (detok0_tx, detok0_rx) = flume::bounded(1);
        let (detok1_tx, detok1_rx) = flume::bounded(1);
        let rid = Rid::from("same-routing-key");
        let expected_shard = rid.shard(2);
        let task_rid = rid.clone();
        let task = tokio::spawn(async move {
            senders(vec![detok0_tx, detok1_tx])
                .decode_once(&task_rid, vec![7])
                .await
        });

        let (expected_rx, other_rx) = if expected_shard == 0 {
            (&detok0_rx, &detok1_rx)
        } else {
            (&detok1_rx, &detok0_rx)
        };
        let DetokMsg::Decode { token_ids, reply } = expected_rx.recv_async().await.unwrap() else {
            panic!("expected a one-shot decode request");
        };
        assert_eq!(token_ids, [7]);
        assert!(other_rx.try_recv().is_err());
        reply.send(Ok("seven".into())).unwrap();
        assert_eq!(task.await.unwrap().unwrap(), "seven");
    }
}
