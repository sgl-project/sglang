//! Wire codecs for the SGLang prefill/decode disaggregation data plane.
//!
//! The wire formats are **Python-owned**: every encoder here must emit
//! byte-identical frames to the builders in
//! `python/sglang/srt/disaggregation/`, and every parser must accept (and
//! reject) exactly what the Python `from_zmq` parsers accept. Bit-identity is
//! enforced by golden vectors generated from the real Python modules
//! (`scripts/gen_wire_vectors.py` -> `tests/vectors/wire.json`).
//!
//! Frame families:
//! - NIXL registration / metadata: `GUARD + 24` / `GUARD + 10` frames
//!   (`nixl/conn.py` `build_receiver_registration_message`,
//!   `build_receiver_metadata_message`).
//! - Common control plane (`common/conn.py`): KV_STATUS status messages
//!   (flavor-parameterized: NIXL tag+reason, Mooncake/Ascend untagged,
//!   Mori tag+reason), ABORT / ABORT_ACK (with optional generation),
//!   BOOTSTRAP (deferred-bootstrap control), CHUNK_READY (staging).
//! - Bootstrap `/route` HTTP bodies (`common/conn.py`
//!   `register_to_bootstrap`, `_handle_route_get`): register PUT body and
//!   the two GET response shapes (`PrefillServerInfo`, `PrefillRankInfo`).
//!
//! JSON on the wire is CPython `json.dumps`: compact separators for zmq
//! frames, default `(", ", ": ")` separators for HTTP bodies, always
//! `ensure_ascii=True`. serde_json cannot emit the default separator form,
//! so JSON is hand-rolled here with Python-exact escaping.

pub mod error;
pub mod json;
pub mod nixl;
pub mod pack;
pub mod peers;
#[cfg(feature = "python-extension")]
pub mod py;
pub mod room;
pub mod status;
pub mod topology;

pub use error::Error;

/// Room identifier: `bootstrap_room` on the wire.
pub type Room = i64;

/// `KVPoll` (base/conn.py). Declaration order is the wire order; `Ord`
/// gives the `max(current, status)` transition its Python meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum KVPoll {
    Failed = 0,
    Bootstrapping = 1,
    WaitingForInput = 2,
    Transferring = 3,
    Success = 4,
}

impl KVPoll {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Failed),
            1 => Some(Self::Bootstrapping),
            2 => Some(Self::WaitingForInput),
            3 => Some(Self::Transferring),
            4 => Some(Self::Success),
            _ => None,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
