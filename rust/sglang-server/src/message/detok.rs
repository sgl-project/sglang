//! Messages to a Detokenizer shard.

use super::ids::Rid;
use super::response::{ChunkEvent, ResponseSink};

/// Messages to a Detokenizer shard. `Register` carries the per-request sink for
/// the shard's local `rid -> sink` map. The rid STRING is the identity: `Rid::hash`
/// picks the shard (collisions there merely co-locate, which is harmless), but two
/// distinct rids that hash alike must not be the same map entry — that evicted one
/// client's sink and delivered their tokens to the other's connection. Equal rids
/// cannot reach here from different requests: `Rid::from_client` uniquifies every
/// client-supplied one.
pub enum DetokMsg {
    Register {
        /// Client-visible rid string — kept in `DetokState` so the shard can
        /// emit `TmEvent::Abort(rid)` (the wire needs the string, not the hash).
        rid: Rid,
        sink: ResponseSink,
        /// Decode logprob token ids to text here (CPU-bound) not on the api threads.
        decode_logprob_text: bool,
        /// `SamplingParams.no_stop_trim`: keep the matched stop; default trims it.
        no_stop_trim: bool,
    },
    /// One decode step's chunks for *this shard*. Batched because `from-scheduler` blocks
    /// per send.
    Chunks(Vec<ChunkEvent>),
    /// Control result: one already-serialized payload delivered to the sink verbatim.
    Result { rid: Rid, payload: bytes::Bytes },
    /// Terminal per-request failure → an `Error` to the sink (a 400, not a crash).
    Fail { rid: Rid, message: String },
    /// Drop the `rid -> sink` entry for a request rejected before the scheduler
    /// (the rejecting stage already answered the client); else `Register` leaks one
    /// entry.
    Deregister { rid: Rid },
}
