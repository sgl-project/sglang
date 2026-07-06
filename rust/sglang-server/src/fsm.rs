//! Request lifecycle FSM.
//!
//! The state lives *inside* the owned request struct (see [`crate::message`]),
//! so transitions are in-place mutations on a single owner — no shared state,
//! no locks. Each pipeline stage drives the transition for its own phase and
//! then moves the request to the next stage's channel.
//!
//! Port of the design enum:
//! ```text
//! Received, Validating, Normalizing, Encoding, Tokenizing, Queued,
//! Streaming { chunks_sent }, Finalizing, Completed, Failed(Error), Aborted
//! ```

use crate::error::Error;

// `Failed(Error)` carries the cause for observability even where it isn't read
// back yet; `EncodeDone` belongs to the deferred Encoder edge.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum RequestState {
    Received,
    Validating,
    /// Generate-only: sampling params normalized + verified before routing.
    Normalizing,
    Encoding,
    Tokenizing,
    Queued,
    Streaming {
        chunks_sent: u64,
    },
    Finalizing,
    Completed,
    Failed(Error),
    Aborted,
}

/// Outcome of validation, selecting the ingress branch.
#[derive(Debug, Clone, Copy)]
pub enum ValidationOutcome {
    /// Has multimodal inputs → Encoding. Deferred: no encoder yet.
    #[allow(dead_code)]
    HasMultimodal,
    /// Plain text → Tokenizing.
    NeedsTokenize,
    /// Caller already supplied token ids → straight to Queued.
    AlreadyTokenized,
}

/// Events that drive transitions. Each variant maps 1:1 to an edge in the
/// design's transition table.
#[allow(dead_code)] // EncodeDone is the deferred Encoder edge.
#[derive(Debug)]
pub enum Event {
    // --- ingress ---
    Validated(ValidationOutcome),
    NeedsNormalize,
    EncodeDone,
    TokenizeDone,
    SchedulerPicked,
    // --- egress ---
    Chunk { finish: bool },
    FinalFrameSent,
    // --- terminal (valid from any state) ---
    Error(Error),
    Disconnect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionError {
    /// The (state, event) pair has no defined edge.
    Illegal,
}

impl RequestState {
    /// Whether this is a terminal state (no further transitions expected).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            RequestState::Completed | RequestState::Failed(_) | RequestState::Aborted
        )
    }

    /// Apply `event`, mutating in place. Returns `Err(Illegal)` for undefined
    /// edges so the caller can decide whether to log-and-drop or fail the req.
    ///
    /// Terminal events (`Error`/`Disconnect`) are accepted from *any* non-terminal
    /// state, matching `(*, Error | Disconnect) -> Failed | Aborted`.
    pub fn apply(&mut self, event: Event) -> Result<(), TransitionError> {
        use Event::*;
        use RequestState::*;
        use ValidationOutcome::*;

        // Wildcard terminal edges first.
        match &event {
            Error(e) => {
                if !self.is_terminal() {
                    *self = Failed(e.clone());
                }
                return Ok(());
            }
            Disconnect => {
                if !self.is_terminal() {
                    *self = Aborted;
                }
                return Ok(());
            }
            _ => {}
        }

        let next = match (&*self, &event) {
            // ingress
            (Received, Validated(_)) => Validating,
            // Generate requests pass through Normalizing (sampling-param
            // normalize/verify); control requests skip straight to Queued.
            (Validating, NeedsNormalize) => Normalizing,
            (Validating, Validated(AlreadyTokenized)) => Queued,
            (Normalizing, Validated(HasMultimodal)) => Encoding,
            (Normalizing, Validated(NeedsTokenize)) => Tokenizing,
            (Normalizing, Validated(AlreadyTokenized)) => Queued,
            (Encoding, EncodeDone) => Tokenizing,
            (Tokenizing, TokenizeDone) => Queued,
            (Queued, SchedulerPicked) => Streaming { chunks_sent: 0 },
            // egress
            (Streaming { chunks_sent }, Chunk { finish: false }) => Streaming {
                chunks_sent: chunks_sent + 1,
            },
            (Streaming { .. }, Chunk { finish: true }) => Finalizing,
            (Finalizing, FinalFrameSent) => Completed,
            _ => return Err(TransitionError::Illegal),
        };
        *self = next;
        Ok(())
    }
}
