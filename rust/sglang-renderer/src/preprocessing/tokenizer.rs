//! Tokenizer worker pool and token-budget validation for renderer requests.

use crate::{
    RendererError as Error, RendererLimits, SamplingParams, TextRequest, TokenIds, TokenIdsRequest,
};
use futures::channel::oneshot;
pub(crate) use sglang_processor::TextTokenizer;
use std::sync::{Arc, Mutex};

enum PoolJob {
    Tokenize {
        request: Box<TextRequest>,
        reply: oneshot::Sender<Result<TokenIdsRequest, Error>>,
    },
    Stop,
}

struct TokenizerPoolInner {
    jobs: flume::Sender<PoolJob>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Drop for TokenizerPoolInner {
    fn drop(&mut self) {
        let workers = self.workers.get_mut().expect("tokenizer workers mutex");
        for _ in 0..workers.len() {
            let _ = self.jobs.send(PoolJob::Stop);
        }
        for worker in workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Bounded CPU tokenizer pool owned by renderer state.
#[derive(Clone)]
pub(crate) struct PooledTokenizer {
    inner: Arc<TokenizerPoolInner>,
}

impl PooledTokenizer {
    pub fn new(
        tokenizer: Arc<dyn TextTokenizer>,
        worker_count: usize,
        queue_capacity: usize,
    ) -> Self {
        let worker_count = worker_count.max(1);
        let (jobs, rx) = flume::bounded(queue_capacity.max(1));
        let mut workers = Vec::with_capacity(worker_count);
        for index in 0..worker_count {
            let rx = rx.clone();
            let tokenizer = tokenizer.clone();
            workers.push(
                std::thread::Builder::new()
                    .name(format!("renderer-tokenizer-{index}"))
                    .spawn(move || {
                        while let Ok(job) = rx.recv() {
                            match job {
                                PoolJob::Tokenize { request, reply } => {
                                    let result =
                                        tokenize_text_request(*request, tokenizer.as_ref());
                                    let _ = reply.send(result);
                                }
                                PoolJob::Stop => break,
                            }
                        }
                    })
                    .expect("spawn renderer tokenizer worker"),
            );
        }
        Self {
            inner: Arc::new(TokenizerPoolInner {
                jobs,
                workers: Mutex::new(workers),
            }),
        }
    }
}

impl PooledTokenizer {
    pub(crate) async fn tokenize(&self, request: TextRequest) -> Result<TokenIdsRequest, Error> {
        let jobs = self.inner.jobs.clone();
        let (reply, result) = oneshot::channel();
        jobs.send_async(PoolJob::Tokenize {
            request: Box::new(request),
            reply,
        })
        .await
        .map_err(|_| Error::Unavailable)?;
        result.await.map_err(|_| Error::WorkerDropped)?
    }
}

fn resolve_stop_token_window(sampling_params: &mut SamplingParams, tokenizer: &dyn TextTokenizer) {
    // Size the scheduler's stop-match window in TOKENS, as Python's
    // `normalize(tokenizer)` does.
    if let Some(stop_tokens) = sampling_params
        .stop_strs
        .iter()
        // A stop that won't encode falls back to its byte length rather
        // than failing the request: still an over-estimate, never an
        // under-estimate, so the scheduler cannot miss that stop.
        .map(|stop| {
            tokenizer
                .encode(stop, false)
                .map_or(stop.len(), |ids| ids.len())
        })
        .max()
    {
        sampling_params.stop_str_max_len = stop_tokens;
    }
}

/// Convert a text input into the token-ID request consumed by shared
/// post-tokenization preparation.
pub fn tokenize_text_request(
    request: TextRequest,
    tokenizer: &dyn TextTokenizer,
) -> Result<TokenIdsRequest, Error> {
    let TextRequest {
        rid,
        prompt,
        add_special_tokens,
        mut options,
        metadata,
    } = request;
    resolve_stop_token_window(&mut options.sampling_params, tokenizer);
    let input_ids = match prompt.encode_segments() {
        Some(segments) => tokenizer.encode_segments(&segments, add_special_tokens)?,
        None => tokenizer.encode(prompt.as_str(), add_special_tokens)?,
    };
    Ok(TokenIdsRequest {
        rid,
        input_ids,
        options,
        metadata,
    })
}

/// Validate fields that must be safe before tokenization or engine submission.
pub fn validate_text_request(request: &TextRequest, limits: &RendererLimits) -> Result<(), Error> {
    validate_request_id(&request.rid)?;
    if request.prompt.as_str().is_empty() {
        return Err(Error::Validation("prompt cannot be empty".into()));
    }
    let options = &request.options;
    validate_completion_fields(
        None,
        options.token_ids_logprob.as_deref(),
        options.return_hidden_states,
        limits,
    )
}

/// Validate an already-tokenized request without passing it through the text
/// tokenizer path.
pub fn validate_token_ids_request(
    request: &TokenIdsRequest,
    limits: &RendererLimits,
) -> Result<(), Error> {
    validate_request_id(&request.rid)?;
    if request.input_ids.is_empty() {
        return Err(Error::Validation("input_ids cannot be empty".into()));
    }
    let options = &request.options;
    validate_completion_fields(
        Some(&request.input_ids),
        options.token_ids_logprob.as_deref(),
        options.return_hidden_states,
        limits,
    )
}

pub(crate) fn validate_request_id(rid: &str) -> Result<(), Error> {
    if rid.len() > 128 {
        return Err(Error::Validation(format!(
            "rid is {} bytes, over the 128-byte limit",
            rid.len()
        )));
    }
    Ok(())
}

/// Validate the common completion fields before tokenization or engine
/// submission. Request identity remains an enclosing host concern.
pub fn validate_completion_fields(
    input_ids: Option<&[i32]>,
    token_ids_logprob: Option<&[i32]>,
    return_hidden_states: bool,
    limits: &RendererLimits,
) -> Result<(), Error> {
    for &id in input_ids.iter().flat_map(|ids| ids.iter()) {
        if id < 0 || id as u64 >= limits.vocab_size {
            return Err(Error::Validation(format!(
                "input_ids contains out-of-vocabulary token id {id}; valid range is [0, {})",
                limits.vocab_size
            )));
        }
    }
    for &id in token_ids_logprob.iter().flat_map(|ids| ids.iter()) {
        if id < 0 || id as u64 >= limits.vocab_size {
            return Err(Error::Validation(format!(
                "token_ids_logprob contains out-of-vocabulary token id {id}; valid range is [0, {})",
                limits.vocab_size
            )));
        }
    }
    if return_hidden_states && !limits.enable_return_hidden_states {
        return Err(Error::Validation(
            "The server is not configured to return the hidden states. Please set `--enable-return-hidden-states` to enable this feature."
                .into(),
        ));
    }
    Ok(())
}

/// Enforce the model context limit after tokenization.
pub fn check_total_tokens(
    request: &mut TokenIdsRequest,
    limits: &RendererLimits,
) -> Result<(), Error> {
    let mut input_ids = Some(std::mem::take(&mut request.input_ids));
    let result =
        check_completion_token_budget(&mut input_ids, &mut request.options.sampling_params, limits);
    request.input_ids = input_ids.expect("validated token-ID request retains input_ids");
    result
}

/// Enforce the context limit over the common token-only completion fields.
pub fn check_completion_token_budget(
    input_ids: &mut Option<TokenIds>,
    sampling_params: &mut SamplingParams,
    limits: &RendererLimits,
) -> Result<(), Error> {
    let max_req_len = limits.context_len;
    let input_len = input_ids.as_ref().map_or(0, Vec::len) as u64 + limits.num_reserved_tokens;
    if input_len >= max_req_len {
        if !limits.allow_auto_truncate {
            return Err(Error::Validation(format!(
                "The input ({input_len} tokens) is longer than the model's context length ({max_req_len} tokens)."
            )));
        }
        if let Some(ids) = input_ids {
            ids.truncate(max_req_len as usize);
        }
    }
    let input_len = input_ids.as_ref().map_or(0, Vec::len) as u64 + limits.num_reserved_tokens;
    let Some(max_new_tokens) = sampling_params.max_new_tokens else {
        return Ok(());
    };
    let total = input_len.saturating_add(max_new_tokens.max(0) as u64);
    if total <= max_req_len {
        return Ok(());
    }
    if !limits.allow_auto_truncate {
        return Err(Error::Validation(format!(
            "Requested token count exceeds the model's maximum context length of {max_req_len} tokens. You requested a total of {total} tokens: {input_len} tokens from the input messages and {max_new_tokens} tokens for the completion. Please reduce the number of tokens in the input messages or the completion to fit within the limit."
        )));
    }
    let clamped = max_req_len.saturating_sub(input_len) as i64;
    if sampling_params.min_new_tokens > clamped {
        return Err(Error::Validation(format!(
            "min_new_tokens must be in [0, max_new_tokens({clamped})], got {}",
            sampling_params.min_new_tokens
        )));
    }
    sampling_params.max_new_tokens = Some(clamped);
    Ok(())
}

#[cfg(test)]
mod tests {
    use sglang_processor::dynamo_renderer::{RenderedPrompt, RenderedSegment};
    use sglang_processor::{ProcessorError, dynamo_tokenizers};

    use super::*;
    use crate::GenerationOptions;

    struct SegmentTokenizer;

    impl TextTokenizer for SegmentTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> Result<TokenIds, ProcessorError> {
            Ok(vec![9])
        }

        fn encode_segments(
            &self,
            segments: &[dynamo_tokenizers::EncodeSegment<'_>],
            _add_special_tokens: bool,
        ) -> Result<TokenIds, ProcessorError> {
            Ok(segments
                .iter()
                .map(|segment| if segment.allow_special { 1 } else { 2 })
                .collect())
        }
    }

    #[test]
    fn rendered_prompt_preserves_segment_boundaries_until_tokenization() {
        let prompt = RenderedPrompt::segmented(vec![
            RenderedSegment::new("<control>", true),
            RenderedSegment::new("user text", false),
        ]);
        let tokenized = tokenize_text_request(
            TextRequest::rendered("request", prompt, false, GenerationOptions::default()),
            &SegmentTokenizer,
        )
        .unwrap();

        assert_eq!(tokenized.input_ids, [1, 2]);
    }
}
