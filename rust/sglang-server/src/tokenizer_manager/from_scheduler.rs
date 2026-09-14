//! TokenizerManager dispatcher thread — drains the from_scheduler channel and
//! routes each message to the detok shard that owns its `Rid::shard`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

use crate::message::detok::DetokMsg;
use crate::message::ids::Rid;
use crate::message::response::{
    ChunkEvent, DISPATCH_TAG_BATCH, DISPATCH_TAG_ERROR, DISPATCH_TAG_RESULT, for_each_chunk,
};
use crate::runtime::Runnable;
use crate::tokenizer_manager::channel::FromSchedulerRx;
use crate::tokenizer_manager::wiring::{Senders, recv};

/// A monotonic counter bumped once per from_scheduler frame the dispatcher drains.
/// It's the rust-native equivalent of the Python `TokenizerManager`'s
/// `last_receive_tstamp`: `/health_generate` watches it advance to confirm the
/// scheduler → detok path is alive (the value itself is meaningless).
pub type ActivityCounter = Arc<AtomicU64>;

/// Dispatcher dispatcher stage. Owns the from_scheduler consumer + the detok-shard
/// senders, so the runtime spawns it as a [`Runnable`].
pub struct Dispatcher {
    from_scheduler_rx: FromSchedulerRx,
    senders: Senders,
    activity: ActivityCounter,
    shutdown: flume::Receiver<()>,
}

impl Dispatcher {
    pub fn new(
        from_scheduler_rx: FromSchedulerRx,
        senders: Senders,
        activity: ActivityCounter,
        shutdown: flume::Receiver<()>,
    ) -> Self {
        Self {
            from_scheduler_rx,
            senders,
            activity,
            shutdown,
        }
    }
}

impl Runnable for Dispatcher {
    fn run(self) {
        // Reused across frames (`clear` keeps capacity) — steady state allocates nothing.
        let shards = self.senders.detokenizer_tx.len();
        let mut buckets: Vec<Vec<ChunkEvent>> = (0..shards).map(|_| Vec::new()).collect();

        while let Some(bytes) = recv(self.from_scheduler_rx.receiver(), &self.shutdown) {
            let Some((&tag, body)) = bytes.split_first() else {
                continue;
            };
            match tag {
                // A whole decode batch: bucket each request by the shard owning its
                // rid, then hand each shard its chunks in one send.
                DISPATCH_TAG_BATCH => {
                    for b in buckets.iter_mut() {
                        b.clear();
                    }
                    let decoded = for_each_chunk(body, |ev| {
                        // The rid picks the shard; a hash collision only co-locates two
                        // requests now, it no longer merges them.
                        buckets[ev.rid.shard(shards)].push(ev);
                    });
                    // Routing only fills the buckets; nothing is delivered until the
                    // sends below. So dropping them here makes rejection atomic for
                    // free: a frame whose columns drifted would otherwise deliver the
                    // requests decoded before the bad one, carrying another request's
                    // logprobs — the corruption the decoder's bounds checks exist to
                    // prevent. Better a lost frame than a silently wrong one.
                    if !decoded.ok {
                        // Dropping the frame keeps wrong data off the wire, but a
                        // request whose chunk was in it would otherwise wait forever:
                        // mid-stream it gets a hole, and if its FINAL chunk was here
                        // it never sees `Done` and the connection hangs — there is no
                        // server-side timeout. Fail from the HEADER's rids, not the
                        // buckets: a frame that fails at request 0 buckets nothing,
                        // so bucket-driven cleanup would leave every request in it
                        // hanging.
                        // Distinguish "failed N requests" from "named nobody": a
                        // frame whose header would not decode at all yields no rids,
                        // so nothing downstream fails and every request in it waits
                        // forever. That is the case worth paging on, and it used to
                        // log the same line as the recoverable one.
                        if decoded.rids.is_empty() {
                            tracing::error!(
                                "from_scheduler: bad batch frame named NO rids; any request in \
                                 it will hang (header undecodable, or empty rid column)"
                            );
                        } else {
                            tracing::warn!(
                                rids = decoded.rids.len(),
                                "from_scheduler: bad batch frame; failing its requests"
                            );
                        }
                        for b in buckets.iter_mut() {
                            b.clear();
                        }
                        for rid in decoded.rids {
                            // 500, not 400: the client's request was fine — the
                            // scheduler's own output frame was not.
                            let shard = rid.shard(shards);
                            let _ = self.senders.detokenizer_tx[shard].send(DetokMsg::Fail {
                                rid,
                                message: "internal error: malformed scheduler output frame".into(),
                            });
                        }
                        continue;
                    }
                    for (i, b) in buckets.iter_mut().enumerate() {
                        if b.is_empty() {
                            continue;
                        }
                        let chunks = DetokMsg::Chunks(std::mem::take(b));
                        if self.senders.detokenizer_tx[i].send(chunks).is_err() {
                            tracing::error!("from_scheduler: detok shard closed");
                        }
                    }
                    // Any frame off the ring = the scheduler produced output → alive.
                    self.activity.fetch_add(1, Ordering::Relaxed);
                }
                DISPATCH_TAG_RESULT => {
                    if let Some((rid, msg)) = decode_result(body) {
                        self.route(&rid, msg);
                    }
                }
                DISPATCH_TAG_ERROR => {
                    if let Some((rid, msg)) = decode_error(body) {
                        self.route(&rid, msg);
                    }
                }
                other => tracing::warn!(tag = other, "from_scheduler: unknown frame tag"),
            }
        }
    }
}

impl Dispatcher {
    /// Route one message to the shard owning `rid`. HOL ceiling: a slow shard stalls
    /// this thread; the fix is a per-shard from_scheduler channel.
    #[inline]
    fn route(&self, rid: &Rid, msg: DetokMsg) {
        if self.senders.detok_for(rid).send(msg).is_err() {
            tracing::error!("from_scheduler: detok shard closed");
        }
    }
}

/// Control result: `[rid, payload]` → single non-streamed delivery to the sink.
fn decode_result(body: &[u8]) -> Option<(Rid, DetokMsg)> {
    let val = rmpv::decode::read_value(&mut &body[..]).ok()?;
    let rmpv::Value::Array(arr) = val else {
        return None;
    };
    let mut items = arr.into_iter();
    let rid = Rid::from(items.next()?.as_str()?);
    // The decode already owns the payload buffer — move it out.
    let payload = match items.next()? {
        rmpv::Value::Binary(b) => Bytes::from(b),
        rmpv::Value::String(s) => Bytes::from(s.into_bytes()),
        _ => return None,
    };
    Some((rid.clone(), DetokMsg::Result { rid, payload }))
}

/// Per-request failure: `[rid, message]` → terminal `Error` to the sink (→ 400).
fn decode_error(body: &[u8]) -> Option<(Rid, DetokMsg)> {
    let val = rmpv::decode::read_value(&mut &body[..]).ok()?;
    let rmpv::Value::Array(arr) = val else {
        return None;
    };
    let mut items = arr.into_iter();
    let rid = Rid::from(items.next()?.as_str()?);
    let message = match items.next()? {
        rmpv::Value::String(s) => s.into_str()?,
        _ => return None,
    };
    Some((rid.clone(), DetokMsg::Fail { rid, message }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::detok::DetokMsg;
    use crate::message::response::frame_error;

    /// A framed error round-trips: `frame_error` → tag stripped →
    /// `decode_error` yields the rid + a `Fail` carrying the message.
    #[test]
    fn error_frame_roundtrips_to_fail() {
        let framed = frame_error("42", "invalid request: bad field");
        assert_eq!(framed[0], DISPATCH_TAG_ERROR);
        let (rid, msg) = decode_error(&framed[1..]).expect("decodes");
        let want = Rid::from("42");
        assert_eq!(rid, want);
        match msg {
            DetokMsg::Fail { rid, message } => {
                assert_eq!(rid.clone(), want);
                assert_eq!(message, "invalid request: bad field");
            }
            _ => panic!("expected Fail"),
        }
    }
}
