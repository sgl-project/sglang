//! Request lifecycle FSM.
//!
//! The state lives *inside* the owned request struct (see [`crate::message`]),
//! so transitions are in-place mutations on a single owner — no shared state,
//! no locks. Each pipeline stage drives the transition for its own phase and
//! then moves the request to the next stage's channel.
//!
//! Port of the design enum:
//! ```text
//! Received, Validating, Normalizing, Encoding, Tokenizing, PreSendValidating,
//! Queued, Streaming { chunks_sent }, Finalizing, Completed, Failed(Error),
//! Aborted
//! ```

use crate::error::Error;

#[derive(Debug, Clone)]
pub enum RequestState {
    Received,
    Validating,
    /// Generate-only: sampling params normalized + verified before routing.
    Normalizing,
    Encoding,
    Tokenizing,
    /// Every branch converges here with its final `input_ids`, for the checks
    /// that need the tokenized length (the input + `max_new_tokens` ceiling).
    /// The last state before the request leaves Rust.
    PreSendValidating,
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
    /// Has multimodal inputs → Encoding, where an MM worker runs the native
    /// pipeline and returns the final expanded `input_ids`.
    HasMultimodal,
    /// Plain text → Tokenizing.
    NeedsTokenize,
    /// Caller already supplied token ids → straight to the pre-send checks.
    AlreadyTokenized,
}

/// Events that drive transitions. Each variant maps 1:1 to an edge in the
/// design's transition table.
#[derive(Debug)]
pub enum Event {
    // --- ingress ---
    Validated(ValidationOutcome),
    NeedsNormalize,
    EncodeDone,
    TokenizeDone,
    /// The pre-send checks passed; the request may be pushed to the ring.
    PreSendValidated,
    SchedulerPicked,
    // --- egress ---
    Chunk {
        finish: bool,
    },
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
            // normalize/verify); control requests skip it, having none.
            (Validating, NeedsNormalize) => Normalizing,
            (Validating, Validated(AlreadyTokenized)) => PreSendValidating,
            (Normalizing, Validated(HasMultimodal)) => Encoding,
            (Normalizing, Validated(NeedsTokenize)) => Tokenizing,
            (Normalizing, Validated(AlreadyTokenized)) => PreSendValidating,
            // The MM worker returns the *final* placeholder-expanded input_ids,
            // so an encoded request skips the tokenizer pool — but not the
            // pre-send checks: expanded image tokens count against the same
            // input + max_new_tokens ceiling as tokenized text.
            (Encoding, EncodeDone) => PreSendValidating,
            // Every ingress branch funnels through the pre-send checks, so they
            // run exactly once per request no matter how it got its ids.
            (Tokenizing, TokenizeDone) => PreSendValidating,
            (PreSendValidating, PreSendValidated) => Queued,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn after(mut state: RequestState, event: Event) -> RequestState {
        state.apply(event).expect("edge must exist");
        state
    }

    /// Every ingress branch — control, client-supplied ids, and text through the
    /// tokenizer pool — must land in `PreSendValidating`, because that is where
    /// the checks needing the final `input_ids` run. A branch that reached
    /// `Queued` directly would skip them silently.
    #[test]
    fn every_branch_reaches_the_ring_through_pre_send_validating() {
        for from in [
            after(
                RequestState::Validating,
                Event::Validated(ValidationOutcome::AlreadyTokenized),
            ),
            after(
                RequestState::Normalizing,
                Event::Validated(ValidationOutcome::AlreadyTokenized),
            ),
            after(RequestState::Tokenizing, Event::TokenizeDone),
        ] {
            assert!(
                matches!(from, RequestState::PreSendValidating),
                "branch bypassed the pre-send checks: {from:?}"
            );
            assert!(matches!(
                after(from, Event::PreSendValidated),
                RequestState::Queued
            ));
        }
    }

    /// The converse: `Queued` has no other in-edge, so the checks can't be skipped
    /// by emitting the wrong event, and can't run twice.
    #[test]
    fn queued_has_no_other_in_edge() {
        for mut state in [
            RequestState::Validating,
            RequestState::Normalizing,
            RequestState::Tokenizing,
            RequestState::Queued,
        ] {
            assert_eq!(
                state.apply(Event::PreSendValidated),
                Err(TransitionError::Illegal),
                "only PreSendValidating may enter Queued"
            );
        }
    }
}
