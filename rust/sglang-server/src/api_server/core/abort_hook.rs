//! The seam a model package uses to learn which requests a client aborted by
//! name, whichever entry the abort came through.

/// Told of each abort that names a request, before the abort ends anything;
/// installed at launch with [`crate::Server::start_with_extensions`]. Every
/// entry that aborts by rid reaches it with the non-empty public rid:
/// `/abort_request`, the gRPC `Abort`, and an abort another DP rank's
/// listener relays (`/internal/peer`). An `abort_all` names no request and
/// does not.
pub trait AbortHook: Send + Sync + std::fmt::Debug {
    /// An abort names `rid`, whether or not it selects a live request here.
    fn named_abort(&self, rid: &str);
}
