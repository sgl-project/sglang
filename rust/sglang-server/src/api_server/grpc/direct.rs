//! `GenerateTokens` injection: the RPC handler converts a pre-tokenized request
//! to the scheduler's format and pushes it straight onto the scheduler ring. No
//! `CoreHandle` and no Intake thread hop; responses return through the detok
//! shards like any other request.
//!
//! Cancellation still rides the shared abort lane, whose consumer deregisters
//! the rid and pushes `AbortReq` for rids it never saw.

use tokio::sync::mpsc;
use tonic::Status;

use crate::api_server::core::CoreCall;
use crate::message::detok::DetokMsg;
use crate::message::request::{GenerateRequest, SchedulerRequest};
use crate::message::response::ResponseSink;
use crate::tokenizer_manager::channel::ToSchedulerTx;
use crate::tokenizer_manager::to_scheduler::Limits;
use crate::tokenizer_manager::validate_input_ids;
use crate::tokenizer_manager::wiring::Senders;
use crate::utils::error::Error;

#[derive(Clone)]
pub(crate) struct DirectIngress {
    to_scheduler: ToSchedulerTx,
    senders: Senders,
    limits: Limits,
    response_capacity: usize,
}

impl DirectIngress {
    pub(crate) fn new(
        to_scheduler: ToSchedulerTx,
        senders: Senders,
        limits: Limits,
        response_capacity: usize,
    ) -> Self {
        Self {
            to_scheduler,
            senders,
            limits,
            response_capacity,
        }
    }

    /// Check, register, encode, push. Every failure happens before the scheduler
    /// sees the request, so the client gets a top-level status and nothing leaks.
    pub(crate) fn submit(&self, mut req: GenerateRequest) -> Result<CoreCall, Status> {
        self.prepare(&mut req).map_err(invalid_argument)?;
        let header = req
            .encode_header()
            .map_err(|e| Status::internal(e.to_string()))?;
        let buffers = req.take_buffers();

        let (response_tx, response_rx) = mpsc::channel(self.response_capacity);
        let detok = self.senders.detok_for(&req.rid);
        // Register before the push: the shard channel is FIFO and the first chunk
        // can only exist after the scheduler drains the ring.
        let registered = detok
            .send(DetokMsg::Register {
                rid: req.rid.clone(),
                sink: ResponseSink::Local(response_tx),
                decode_logprob_text: false,
                no_stop_trim: req.sampling_params.no_stop_trim,
            })
            .is_ok();
        if !registered {
            return Err(Status::unavailable("detok shard gone"));
        }
        if !self
            .to_scheduler
            .try_push(SchedulerRequest { header, buffers })
        {
            let _ = detok.send(DetokMsg::Deregister {
                rid: req.rid.clone(),
            });
            return Err(Status::resource_exhausted("to_scheduler channel full"));
        }
        Ok(CoreCall::pushed(
            req.rid,
            response_rx,
            self.senders.abort_tx.clone(),
        ))
    }

    /// Only what injection cannot skip. `normalize` is format conversion, not
    /// validation: the scheduler reads its outputs (greedy rewrite, `top_k`,
    /// `stop_strs`). The vocab check keeps a bad id off the GPU embedding lookup.
    /// Context length and rid length are the caller's responsibility.
    fn prepare(&self, req: &mut GenerateRequest) -> Result<(), Error> {
        validate_input_ids(req, self.limits.vocab_size)?;
        req.sampling_params
            .normalize(self.limits.skip_tokenizer_init, self.limits.vocab_size)
    }
}

fn invalid_argument(error: Error) -> Status {
    Status::invalid_argument(error.to_string())
}

#[cfg(test)]
pub(crate) struct TestRig {
    pub(crate) ingress: DirectIngress,
    pub(crate) ring: crate::tokenizer_manager::channel::ToSchedulerRx,
    pub(crate) detok_rx: flume::Receiver<DetokMsg>,
    pub(crate) abort_rx: flume::Receiver<crate::tokenizer_manager::wiring::AbortSource>,
}

/// One detok shard, a real ring, and a tiny model: vocab 1000, context 64.
#[cfg(test)]
pub(crate) fn test_rig(ring_cap: usize) -> TestRig {
    let (to_scheduler, ring) = crate::tokenizer_manager::channel::to_scheduler(ring_cap);
    let (tok_manager_tx, _) = flume::unbounded();
    let (abort_tx, abort_rx) = flume::unbounded();
    let (tokenizer_tx, _) = flume::unbounded();
    let (detok_tx, detok_rx) = flume::unbounded();
    let senders = Senders {
        tok_manager_tx,
        abort_tx,
        tokenizer_tx,
        detokenizer_tx: vec![detok_tx],
    };
    let limits = Limits {
        skip_tokenizer_init: false,
        vocab_size: 1000,
        context_len: 64,
        num_reserved_tokens: 0,
        allow_auto_truncate: false,
        enable_return_hidden_states: false,
    };
    TestRig {
        ingress: DirectIngress::new(to_scheduler, senders, limits, 8),
        ring,
        detok_rx,
        abort_rx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ids::Rid;
    use crate::message::sampling::SamplingParams;

    fn request(rid: &str, ids: Vec<i64>) -> GenerateRequest {
        GenerateRequest {
            rid: Rid::from_client(rid),
            input_ids: Some(ids),
            sampling_params: SamplingParams {
                max_new_tokens: Some(8),
                ..Default::default()
            },
            stream: true,
            logprob_start_len: -1,
            ..Default::default()
        }
    }

    #[test]
    fn registers_then_pushes_header_and_ids() {
        let rig = test_rig(4);
        let _call = rig.ingress.submit(request("d1", vec![1, 2, 3])).unwrap();

        let Ok(DetokMsg::Register { rid, .. }) = rig.detok_rx.try_recv() else {
            panic!("expected the detok registration first");
        };
        assert_eq!(rid.client_facing(), "d1");

        let pushed = rig.ring.drain(4);
        assert_eq!(pushed.len(), 1);
        let header = rmpv::decode::read_value(&mut &pushed[0].header[..]).unwrap();
        let arr = header.as_array().unwrap();
        assert_eq!(arr[0].as_str(), Some("TokenizedGenerateReqInput"));
        assert!(
            arr[4].is_nil(),
            "input_ids ride as a buffer, not the header"
        );
        assert!(!pushed[0].buffers.is_empty());
    }

    #[test]
    fn out_of_vocab_ids_never_reach_the_ring_or_the_shard() {
        let rig = test_rig(4);
        let error = rig.ingress.submit(request("d2", vec![5000])).err().unwrap();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(rig.detok_rx.try_recv().is_err());
        assert!(rig.ring.drain(4).is_empty());
    }

    /// Context length is the caller's job: an over-long prompt is injected as-is.
    #[test]
    fn over_long_prompts_are_injected_unchecked() {
        let rig = test_rig(4);
        let _call = rig.ingress.submit(request("d3", vec![1; 100])).unwrap();
        assert_eq!(rig.ring.drain(4).len(), 1);
    }

    /// `normalize` is format conversion: greedy is rewritten for the scheduler.
    #[test]
    fn temperature_zero_is_rewritten_to_greedy() {
        let rig = test_rig(4);
        let mut req = request("d7", vec![1]);
        req.sampling_params.temperature = 0.0;
        rig.ingress.prepare(&mut req).unwrap();
        assert_eq!(req.sampling_params.temperature, 1.0);
        assert_eq!(req.sampling_params.top_k, 1);
    }

    #[test]
    fn full_ring_deregisters_and_reports_resource_exhausted() {
        let rig = test_rig(1);
        let _first = rig.ingress.submit(request("d4", vec![1])).unwrap();
        let error = rig.ingress.submit(request("d5", vec![1])).err().unwrap();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);

        let msgs: Vec<_> = rig.detok_rx.try_iter().collect();
        assert!(matches!(
            msgs.last(),
            Some(DetokMsg::Deregister { rid }) if rid.client_facing() == "d5"
        ));
    }

    #[test]
    fn dropping_the_call_aborts_through_the_shared_lane() {
        let rig = test_rig(4);
        let call = rig.ingress.submit(request("d6", vec![1])).unwrap();
        drop(call);
        let abort = rig.abort_rx.try_recv().unwrap();
        assert_eq!(abort.rid().client_facing(), "d6");
    }
}
