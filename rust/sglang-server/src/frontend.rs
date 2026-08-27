//! Protocol-neutral request submission and cancellation handle.

use std::collections::HashSet;

use tokio::sync::mpsc;

use crate::message::ids::Rid;
use crate::message::request::{Request, RequestKind};
use crate::message::response::{ResponseItem, ResponseSink};
use crate::tokenizer_manager::wiring::{AbortSource, Senders, TmEvent};
use crate::utils::fsm::RequestState;

/// The adapter seam shared by HTTP today and a future native Tonic listener.
///
/// Protocol adapters decode their wire request into a [`RequestKind`], then use
/// this handle for request identity, bounded response registration, FSM intake,
/// and abort-on-disconnect state. Transport-specific error and response framing
/// remain in the adapter.
pub(crate) struct FrontendHandle {
    senders: Senders,
    response_buf: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubmitError {
    Unavailable,
}

impl FrontendHandle {
    pub(crate) fn new(senders: Senders, response_buf: usize) -> Self {
        Self {
            senders,
            response_buf,
        }
    }

    /// Register a bounded response stream before making the request visible to
    /// the shared FSM. The request ID is already final at this boundary.
    pub(crate) async fn submit(
        &self,
        kind: RequestKind,
    ) -> Result<(Rid, mpsc::Receiver<ResponseItem>), SubmitError> {
        let rid = match &kind {
            RequestKind::Generate(generate) => generate.rid.clone(),
            RequestKind::Control(control) => control.rid().into(),
            RequestKind::Detokenize { .. } => Rid::new(),
        };
        let (sink, response) = mpsc::channel::<ResponseItem>(self.response_buf);
        let request = Request {
            rid: rid.clone(),
            state: RequestState::Received,
            sink: ResponseSink::Local(sink),
            kind,
        };
        self.senders
            .tok_manager_tx
            .send_async(TmEvent::Intake(request))
            .await
            .map_err(|_| {
                tracing::error!(%rid, "frontend FSM inbox closed; request rejected");
                SubmitError::Unavailable
            })?;
        Ok((rid, response))
    }

    pub(crate) fn abort_guard(&self, rid: Rid) -> AbortGuard {
        AbortGuard::new(self.senders.abort_tx.clone(), rid)
    }

    pub(crate) fn empty_abort_guard(&self) -> AbortGuard {
        AbortGuard::new_empty(self.senders.abort_tx.clone())
    }
}

/// Aborts still-in-flight request incarnations when a protocol writer drops.
pub(crate) struct AbortGuard {
    aborts: flume::Sender<AbortSource>,
    rids: HashSet<Rid>,
}

impl AbortGuard {
    pub(crate) fn new(aborts: flume::Sender<AbortSource>, rid: Rid) -> Self {
        Self {
            aborts,
            rids: HashSet::from([rid]),
        }
    }

    pub(crate) fn new_empty(aborts: flume::Sender<AbortSource>) -> Self {
        Self {
            aborts,
            rids: HashSet::new(),
        }
    }

    pub(crate) fn arm(&mut self, rid: Rid) {
        self.rids.insert(rid);
    }

    pub(crate) fn disarm(&mut self, rid: &Rid) {
        self.rids.remove(rid);
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        for rid in self.rids.drain() {
            let _ = self.aborts.send(AbortSource::Guard(rid));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::request::GenerateRequest;
    use crate::tokenizer_manager::wiring::Senders;

    fn senders_with_intake(
        intake: flume::Sender<TmEvent>,
        aborts: flume::Sender<AbortSource>,
    ) -> Senders {
        Senders {
            tok_manager_tx: intake,
            abort_tx: aborts,
            tokenizer_tx: flume::unbounded().0,
            detokenizer_tx: vec![],
        }
    }

    #[tokio::test]
    async fn submit_registers_response_before_fsm_intake() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let frontend = FrontendHandle::new(senders_with_intake(intake_tx, abort_tx), 1);
        let generate = GenerateRequest {
            rid: Rid::from_client("client-request"),
            text: Some("prompt".into()),
            ..Default::default()
        };

        let (rid, mut response) = frontend
            .submit(RequestKind::Generate(Box::new(generate)))
            .await
            .unwrap();
        let TmEvent::Intake(request) = intake_rx.recv().unwrap() else {
            panic!("submission must enter the shared FSM")
        };
        assert!(matches!(request.state, RequestState::Received));
        assert_eq!(request.rid, rid);
        request
            .sink
            .try_send(ResponseItem::Error(crate::utils::error::Error::Validation(
                "terminal".into(),
            )))
            .unwrap();
        assert!(matches!(
            response.recv().await,
            Some(ResponseItem::Error(_))
        ));
    }

    #[test]
    fn guard_aborts_only_the_rids_still_armed() {
        let (abort_tx, abort_rx) = flume::unbounded();
        let done: Rid = "done".into();
        let mut guard = AbortGuard::new(abort_tx, done.clone());
        guard.arm("aborted".into());
        guard.disarm(&done);
        drop(guard);

        assert!(
            matches!(abort_rx.try_recv().unwrap(), AbortSource::Guard(r) if r.as_str() == "aborted")
        );
        assert!(abort_rx.try_recv().is_err());
    }

    #[test]
    fn disarmed_guard_does_not_abort() {
        let (abort_tx, abort_rx) = flume::unbounded();
        let rid = Rid::from("r9");
        let mut guard = AbortGuard::new(abort_tx, rid.clone());
        guard.disarm(&rid);
        drop(guard);
        assert!(abort_rx.try_recv().is_err());
    }
}
