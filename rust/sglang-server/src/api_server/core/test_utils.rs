//! Shared response-channel fixtures for transport-neutral API tests.

use crate::api_server::core::generate::{GeneratePlan, RequestTiming};
use crate::message::ids::Rid;
use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem};
use crate::tokenizer_manager::wiring::{AbortSource, Senders};

/// The CLOSED-inbox fixture: every receiver is dropped at construction, so the
/// senders stay movable but a submission fails with 503 (the shutdown state).
pub(crate) fn senders() -> Senders {
    Senders {
        tok_manager_tx: flume::unbounded().0,
        abort_tx: flume::unbounded().0,
        tokenizer_tx: flume::unbounded().0,
        detokenizer_tx: vec![],
    }
}

/// A `Senders` whose abort lane is observable: tests assert exactly which rids
/// a dropped guard aborted.
pub(crate) fn abort_senders() -> (Senders, flume::Receiver<AbortSource>) {
    let (abort_tx, abort_rx) = flume::unbounded();
    (
        Senders {
            abort_tx,
            ..senders()
        },
        abort_rx,
    )
}

/// The guard-aborted rids from `abort_rx`, sorted. Panics on any non-`Guard`
/// source: a misrouted `Detok` abort must not be silently filtered out of a
/// cancellation assertion.
pub(crate) fn aborted_guard_rids(abort_rx: &flume::Receiver<AbortSource>) -> Vec<String> {
    let mut rids = abort_rx
        .try_iter()
        .map(|source| {
            let AbortSource::Guard(rid) = source else {
                panic!("expected guard cancellation, got {source:?}");
            };
            rid.as_str().to_owned()
        })
        .collect::<Vec<_>>();
    rids.sort();
    rids
}

pub(crate) fn chunk(rid: &str, text: &str, done: bool) -> ResponseItem {
    let output = ChunkEvent {
        rid: rid.into(),
        text: text.into(),
        token_ids: vec![1],
        prompt_tokens: 5,
        completion_tokens: 1,
        finish_reason: done.then(|| {
            serde_json::from_value(serde_json::json!({
                "type": "stop",
                "matched": "</s>"
            }))
            .unwrap()
        }),
        ..Default::default()
    };
    if done {
        ResponseItem::Done(output)
    } else {
        ResponseItem::Frame(output)
    }
}

/// A frame carrying the per-request metadata snapshots the bridge decodes.
pub(crate) fn chunk_with_metadata(
    rid: &str,
    text: &str,
    done: bool,
    reasoning_tokens: u32,
    cached_tokens: u32,
) -> ResponseItem {
    let mut item = chunk(rid, text, done);
    match &mut item {
        ResponseItem::Frame(event) | ResponseItem::Done(event) => {
            event.extras = Some(Box::new(ChunkExtras {
                reasoning_tokens,
                cached_tokens,
                ..Default::default()
            }));
        }
        _ => unreachable!("chunk builds a frame"),
    }
    item
}

pub(crate) type TestReceiver = (
    Rid,
    tokio::sync::mpsc::Receiver<ResponseItem>,
    RequestTiming,
);

pub(crate) fn planned(rid: &str) -> (TestReceiver, tokio::sync::mpsc::Sender<ResponseItem>) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    ((Rid::from(rid), rx, RequestTiming::new()), tx)
}

pub(crate) fn plan(receivers: Vec<TestReceiver>, senders: Senders) -> GeneratePlan {
    let mut guard = crate::api_server::core::guard::AbortGuard::new_empty(senders);
    for (rid, _, _) in &receivers {
        guard.arm(rid.clone());
    }
    GeneratePlan {
        receivers,
        guard,
        is_batch: true,
        incremental: true,
    }
}
