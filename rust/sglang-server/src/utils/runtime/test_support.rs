//! A whole frontend with no scheduler behind it, for tests of a model
//! package's extensions: the test reads what reaches the scheduler and
//! answers for it.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use super::{Runtime, start};
use crate::message::config::{
    DeploymentInfo, RuntimeConfig, RustServerEndpoint, RustServerServerArgs, ServerArgs,
};
use crate::message::response::{BatchHeader, frame_decode_batch_cols};
use crate::{AbortHook, GenerateHook};

/// A frontend serving HTTP on one address, whose scheduler is the test.
pub struct SchedulerlessServer(Runtime);

/// The package extensions a [`SchedulerlessServer`] serves with.
#[derive(Default)]
pub struct Extensions {
    pub generate_hook: Option<Arc<dyn GenerateHook>>,
    pub abort_hook: Option<Arc<dyn AbortHook>>,
}

/// One message that reached the scheduler: its type (`AbortReq`,
/// `TokenizedGenerateReqInput`, ...) and the rid it carries.
#[derive(Debug)]
pub struct SchedulerMessage {
    pub kind: String,
    pub rid: String,
}

impl SchedulerlessServer {
    /// Serve on `listeners[dp_rank]` as that DP rank's listener: with more
    /// than one listener, as one of an attention-DP deployment's, which
    /// relays aborts and control requests to the others.
    pub fn start(
        listeners: &[SocketAddr],
        dp_rank: usize,
        extensions: Extensions,
    ) -> Result<Self, String> {
        let deployment_info = if listeners.len() > 1 {
            DeploymentInfo {
                dp_size: listeners.len() as u32,
                dp_rank: Some(dp_rank as u32),
                enable_dp_attention: true,
                rust_server_endpoints: listeners
                    .iter()
                    .enumerate()
                    .map(|(rank, addr)| RustServerEndpoint {
                        dp_rank: rank as u32,
                        endpoint: format!("http://{addr}"),
                    })
                    .collect(),
                ..Default::default()
            }
        } else {
            DeploymentInfo::default()
        };
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: listeners[dp_rank],
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(ServerArgs {
                skip_tokenizer_init: true,
                skip_server_warmup: true,
                deployment_info,
                ..Default::default()
            }),
            generate_hook: extensions.generate_hook,
            generate_validator: None,
            abort_hook: extensions.abort_hook,
            server_info_state_keys: Vec::new(),
        };
        start(cfg, None).map(Self)
    }

    /// The next message on the scheduler ring, waiting up to `timeout`.
    pub fn next_message(&self, timeout: Duration) -> Option<SchedulerMessage> {
        if !self.0.to_scheduler_rx.wait(timeout) {
            return None;
        }
        let request = self.0.to_scheduler_rx.drain(1).pop()?;
        let header = rmpv::decode::read_value(&mut &request.header[..]).ok()?;
        let field = |i: usize| header.as_array()?.get(i)?.as_str().map(str::to_owned);
        Some(SchedulerMessage {
            kind: field(0)?,
            rid: field(1)?,
        })
    }

    /// Answer for the scheduler with one output for `rid` (the rid it
    /// received): `output_ids` more tokens, and the request's end when
    /// `finish_reason` is given (Python's `FINISH_*.to_json()`).
    pub fn push_output(
        &self,
        rid: &str,
        output_ids: &[i64],
        finish_reason: Option<serde_json::Value>,
    ) -> Result<(), String> {
        let finish_reason = finish_reason
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| error.to_string())?;
        let header = BatchHeader {
            rids: vec![rid.to_owned()],
            finish_reasons: vec![finish_reason],
            prompt_tokens: vec![1],
            tok_lens: vec![output_ids.len() as u32],
            cached_tokens: vec![0],
            reasoning_tokens: vec![0],
            ..Default::default()
        };
        let header = rmp_serde::to_vec(&header).map_err(|error| error.to_string())?;
        let data: Vec<u8> = output_ids
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        if self
            .0
            .from_scheduler_tx
            .push(frame_decode_batch_cols(&header, &[&data]))
        {
            Ok(())
        } else {
            Err("the frontend shut down".into())
        }
    }
}
