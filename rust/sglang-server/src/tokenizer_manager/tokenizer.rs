//! Server tokenizer workers backed by the engine-free renderer tokenizer.

use std::sync::Arc;

use crate::message::request::{GenerateRequest, RequestKind};
use crate::renderer::{PreprocessJob, prepare_direct_request};
use crate::runtime::Runnable;
use crate::tokenizer_manager::to_scheduler::Limits;
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::{error::Error, fsm::Event};

pub use sglang_renderer::{DynamoTokenizer, TextTokenizer, load_tokenizer, resolve_model_file};

/// Remove one leading run of auto-added specials — exactly what an
/// `add_special_tokens=false` encode would have produced, without a second
/// tokenizer instance (the post-processor always prepends the same prefix, so
/// a template-rendered copy of those tokens is preserved).
fn strip_auto_specials(mut ids: Vec<i32>, auto_specials: &[i32]) -> Vec<i32> {
    if ids.starts_with(auto_specials) {
        ids.drain(..auto_specials.len());
    }
    ids
}

/// Apply the text-tokenization stage to one generate request. Both the normal
/// tokenizer worker and the standalone renderer call this function so stop
/// sizing and special-token handling cannot drift between the two paths.
pub(crate) fn tokenize_generate_request(
    request: &mut GenerateRequest,
    tokenizer: &dyn TextTokenizer,
    auto_specials: &[i32],
) -> Result<(), Error> {
    // Size the scheduler's stop-match window in TOKENS, as Python's
    // `normalize(tokenizer)` does.
    if let Some(stop_tokens) = request
        .sampling_params
        .stop_strs
        .iter()
        // A stop that won't encode falls back to its byte length rather
        // than failing the request: still an over-estimate, never an
        // under-estimate, so the scheduler cannot miss that stop.
        .map(|stop| tokenizer.encode(stop).map_or(stop.len(), |ids| ids.len()))
        .max()
    {
        request.sampling_params.stop_str_max_len = stop_tokens;
    }
    let ids = tokenizer.encode(request.text.as_deref().unwrap_or(""))?;
    request.input_ids = Some(if request.skip_special_tokens {
        strip_auto_specials(ids, auto_specials)
    } else {
        ids
    });
    Ok(())
}

/// One tokenizer worker: pulls a `Request` off the shared inbox, fills
/// `input_ids`, returns it to the TokenizerManager. Pinned; backend shared.
///
/// The `auto_specials` prefix (probed once at construction, Python's
/// `encode("")` probe) is stripped from template-rendered prompts —
/// [`GenerateRequest`]'s `skip_special_tokens` — so chat prompts gain no
/// extra BOS/EOS while native text keeps the post-processor specials.
pub struct TokenizerWorker {
    rx: flume::Receiver<PreprocessJob>,
    tm: Option<flume::Sender<TmEvent>>,
    tokenizer: Arc<dyn TextTokenizer>,
    auto_specials: Vec<i32>,
    limits: Limits,
}

impl TokenizerWorker {
    pub fn new(
        rx: flume::Receiver<PreprocessJob>,
        tm: Option<flume::Sender<TmEvent>>,
        tokenizer: Arc<dyn TextTokenizer>,
        limits: Limits,
    ) -> Self {
        let auto_specials = tokenizer.auto_specials();
        Self {
            rx,
            tm,
            tokenizer,
            auto_specials,
            limits,
        }
    }
}

impl Runnable for TokenizerWorker {
    fn run(self) {
        while let Ok(job) = self.rx.recv() {
            match job {
                PreprocessJob::Inference(mut req) => {
                    // Normal inference already validated and normalized in the
                    // intake FSM. Fill ids, then return through its existing
                    // lifecycle.
                    let event = {
                        let RequestKind::Generate(g) = &mut req.kind else {
                            tracing::error!("tokenizer pool received a non-generate request");
                            continue;
                        };
                        match tokenize_generate_request(
                            g,
                            self.tokenizer.as_ref(),
                            &self.auto_specials,
                        ) {
                            Ok(()) => Event::TokenizeDone,
                            Err(err) => Event::Error(err),
                        }
                    };
                    let _ = req.state.apply(event);
                    let Some(tm) = &self.tm else {
                        tracing::error!("standalone tokenizer received an inference job");
                        continue;
                    };
                    if tm.send(TmEvent::Tokenized(req)).is_err() {
                        tracing::error!("tm inbox closed; dropping request");
                        break;
                    }
                }
                PreprocessJob::Render(job) => {
                    let result = prepare_direct_request(
                        job.request,
                        self.tokenizer.as_ref(),
                        &self.auto_specials,
                        &self.limits,
                    );
                    // The HTTP request may have been cancelled while preparing.
                    let _ = job.reply.send(result);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::config::ServerArgs;
    use crate::message::request::{GenerateRequest, Request, RequestKind};
    use crate::message::response::ResponseSink;
    use crate::message::sampling::SamplingParams;
    use crate::message::types::TokenIds;
    use crate::utils::fsm::RequestState;
    use tokio::sync::mpsc;

    /// One token per whitespace-separated word, so a stop's token count differs
    /// from its byte count and the two units cannot be confused.
    struct WordTokenizer;
    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, sglang_renderer::RendererError> {
            Ok(text.split_whitespace().map(|_| 1i32).collect())
        }
    }

    /// The scheduler's stop-match window must reach the wire as a TOKEN count, as
    /// Python's `normalize(tokenizer)` produces.
    ///
    /// `Normalizing` leaves a UTF-8 BYTE count there — a safe over-estimate, but it
    /// makes the scheduler decode a longer tail on EVERY decode step of EVERY
    /// request (14 tokens vs 6 for a typical stop set). This stage owns the
    /// tokenizer, so it is where the exact count is resolved.
    #[test]
    fn tokenizing_replaces_the_byte_window_with_a_token_count() {
        let (req_tx, req_rx) = flume::unbounded::<PreprocessJob>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();

        // 8 bytes vs 3 "tokens" under WordTokenizer — units are distinguishable.
        let sp = SamplingParams {
            stop_strs: vec!["a bb ccc".to_string(), "dd".to_string()],
            stop_str_max_len: 8, // what `normalize_stops` left: max BYTE length
            ..Default::default()
        };
        let (sink_tx, _sink_rx) = mpsc::channel(4);
        req_tx
            .send(PreprocessJob::Inference(Request {
                rid: "1".into(),
                state: RequestState::Tokenizing,
                sink: ResponseSink::Local(sink_tx),
                kind: RequestKind::Generate(Box::new(GenerateRequest {
                    rid: "1".into(),
                    text: Some("hello world".into()),
                    sampling_params: sp,
                    ..Default::default()
                })),
            }))
            .expect("send");
        drop(req_tx); // closes the loop after one request

        TokenizerWorker::new(
            req_rx,
            Some(tm_tx),
            Arc::new(WordTokenizer),
            Limits::from(&ServerArgs::default()),
        )
        .run();

        let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
            panic!("expected Tokenized");
        };
        let RequestKind::Generate(g) = &req.kind else {
            panic!("expected generate");
        };
        assert_eq!(
            g.sampling_params.stop_str_max_len, 3,
            "must be the max TOKEN count (3), not the byte count (8)"
        );
    }

    /// The strip reproduces `add_special_tokens=false`: one leading run of
    /// auto-added specials is removed, a template-rendered copy is kept, and
    /// tokenizers with no auto specials (empty probe) are untouched.
    #[test]
    fn strip_auto_specials_matches_add_special_tokens_false() {
        assert_eq!(strip_auto_specials(vec![0, 0, 1, 2], &[0]), vec![0, 1, 2]);
        assert_eq!(strip_auto_specials(vec![1, 2], &[0]), vec![1, 2]);
        assert_eq!(strip_auto_specials(vec![1, 2], &[]), vec![1, 2]);
        assert_eq!(strip_auto_specials(vec![0], &[0, 9]), vec![0]);
    }

    /// Word tokens plus a prepended BOS marker (id 0) — like an HF tokenizer
    /// whose post-processor adds specials.
    struct MarkedTokenizer;
    impl TextTokenizer for MarkedTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, sglang_renderer::RendererError> {
            Ok(vec![0, text.len() as i32])
        }
        fn auto_specials(&self) -> Vec<i32> {
            vec![0]
        }
    }

    /// `skip_special_tokens` strips the probed prefix: template-rendered
    /// prompts (chat) must not gain a BOS the template didn't render — Python's
    /// `add_special_tokens=False` at the chat-template encode site.
    #[test]
    fn skip_special_tokens_strips_the_auto_added_specials() {
        let run = |skip_special_tokens: bool| {
            let (req_tx, req_rx) = flume::unbounded::<PreprocessJob>();
            let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
            req_tx
                .send(PreprocessJob::Inference(Request {
                    rid: "1".into(),
                    state: RequestState::Tokenizing,
                    sink: ResponseSink::Local(tokio::sync::mpsc::channel(4).0),
                    kind: RequestKind::Generate(Box::new(GenerateRequest {
                        rid: "1".into(),
                        text: Some("hi".into()),
                        skip_special_tokens,
                        ..Default::default()
                    })),
                }))
                .expect("send");
            drop(req_tx);
            TokenizerWorker::new(
                req_rx,
                Some(tm_tx),
                Arc::new(MarkedTokenizer),
                Limits::from(&ServerArgs::default()),
            )
            .run();
            let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
                panic!("expected Tokenized");
            };
            let RequestKind::Generate(g) = &req.kind else {
                panic!("expected generate");
            };
            g.input_ids.clone().expect("tokenized")
        };
        assert_eq!(run(false), vec![0, 2], "native prompts keep specials");
        assert_eq!(run(true), vec![2], "rendered prompts lose the auto BOS");
    }
}
