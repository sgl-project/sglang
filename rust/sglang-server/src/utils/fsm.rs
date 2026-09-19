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
//!
//! The to-scheduler stages run in one fixed order, each skipped when a request
//! has nothing for it: `Tokenizing` (text → ids) then `Encoding` (ids + media →
//! placeholder-expanded ids + features), converging on `PreSendValidating`.
//! `Tokenizing` carries what follows it ([`AfterTokenize`]) so the tokenizer
//! pool, which applies `TokenizeDone` itself, needs no routing knowledge: the
//! next state is a pure function of (state, event), decided by intake once at
//! `Normalizing`. A request that already has ids still passes through
//! `Tokenizing` — intake applies `TokenizeDone` inline instead of visiting the
//! pool — so "skip" means no pool hop, never a different route.

use super::error::Error;

#[derive(Debug, Clone)]
pub enum RequestState {
    Received,
    Validating,
    /// Generate-only: sampling params normalized + verified before routing.
    Normalizing,
    Encoding,
    Tokenizing {
        /// Where `TokenizeDone` leads. Set by intake at routing time; the pool
        /// never reads it.
        then: AfterTokenize,
    },
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

/// The stage after `Tokenizing`: the pre-send checks, or — for a multimodal
/// prompt, whose placeholders the MM worker expands in ids — `Encoding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterTokenize {
    PreSend,
    Encode,
}

/// Outcome of validation.
#[derive(Debug, Clone, Copy)]
pub enum ValidationOutcome {
    /// Has multimodal inputs → Tokenizing, then Encoding, where an MM worker
    /// runs the multimodal pipeline and returns the final expanded `input_ids`.
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
    // --- request ---
    Validated(ValidationOutcome),
    NeedsNormalize,
    EncodeDone,
    TokenizeDone,
    /// The pre-send checks passed; the request may be pushed to the ring.
    PreSendValidated,
    SchedulerPicked,
    // --- response ---
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
            // request
            (Received, Validated(_)) => Validating,
            // Generate requests pass through Normalizing (sampling-param
            // normalize/verify); control requests skip it, having none.
            (Validating, NeedsNormalize) => Normalizing,
            (Validating, Validated(AlreadyTokenized)) => PreSendValidating,
            (Normalizing, Validated(HasMultimodal)) => Tokenizing {
                then: AfterTokenize::Encode,
            },
            (Normalizing, Validated(NeedsTokenize)) => Tokenizing {
                then: AfterTokenize::PreSend,
            },
            (Normalizing, Validated(AlreadyTokenized)) => PreSendValidating,
            // A multimodal prompt has its ids (the client's, or the pool's);
            // now the MM worker expands its placeholders.
            (
                Tokenizing {
                    then: AfterTokenize::Encode,
                },
                TokenizeDone,
            ) => Encoding,
            // The MM worker returns the *final* placeholder-expanded input_ids,
            // so an encoded request never revisits the tokenizer pool — but not
            // the pre-send checks: expanded image tokens count against the same
            // input + max_new_tokens ceiling as tokenized text.
            (Encoding, EncodeDone) => PreSendValidating,
            // Every to-scheduler branch funnels through the pre-send checks, so they
            // run exactly once per request no matter how it got its ids.
            (
                Tokenizing {
                    then: AfterTokenize::PreSend,
                },
                TokenizeDone,
            ) => PreSendValidating,
            (PreSendValidating, PreSendValidated) => Queued,
            (Queued, SchedulerPicked) => Streaming { chunks_sent: 0 },
            // response
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

    /// Every to-scheduler branch — control, client-supplied ids, text through the
    /// tokenizer pool, and media through the MM pool — must land in
    /// `PreSendValidating`, because that is where the checks needing the final
    /// `input_ids` run. A branch that reached `Queued` directly would skip them
    /// silently.
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
            after(
                RequestState::Tokenizing {
                    then: AfterTokenize::PreSend,
                },
                Event::TokenizeDone,
            ),
            after(RequestState::Encoding, Event::EncodeDone),
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

    /// The stages run in one fixed order: a multimodal prompt tokenizes, then
    /// encodes — the single MM route, whether the pool or intake (ids already
    /// present) applies `TokenizeDone`. It cannot reach the ring without
    /// `Encoding`, and `Encoding` never loops back to the pool.
    #[test]
    fn multimodal_prompts_tokenize_before_encoding() {
        let mm = after(
            RequestState::Normalizing,
            Event::Validated(ValidationOutcome::HasMultimodal),
        );
        assert!(matches!(
            mm,
            RequestState::Tokenizing {
                then: AfterTokenize::Encode
            }
        ));
        assert!(matches!(
            after(mm, Event::TokenizeDone),
            RequestState::Encoding
        ));

        let mut encoded = after(RequestState::Encoding, Event::EncodeDone);
        assert!(matches!(encoded, RequestState::PreSendValidating));
        assert_eq!(
            encoded.apply(Event::TokenizeDone),
            Err(TransitionError::Illegal),
            "an encoded request never revisits the tokenizer pool"
        );
    }

    /// The converse: `Queued` has no other in-edge, so the checks can't be skipped
    /// by emitting the wrong event, and can't run twice.
    #[test]
    fn queued_has_no_other_in_edge() {
        for mut state in [
            RequestState::Validating,
            RequestState::Normalizing,
            RequestState::Tokenizing {
                then: AfterTokenize::PreSend,
            },
            RequestState::Tokenizing {
                then: AfterTokenize::Encode,
            },
            RequestState::Encoding,
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
