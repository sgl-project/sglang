//! Server tokenizer workers backed by the engine-free renderer tokenizer.

use std::sync::Arc;

use crate::message::request::RequestKind;
use crate::renderer::PreprocessJob;
use crate::runtime::Runnable;
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::fsm::Event;

use sglang_renderer::tokenize_text_completion;
pub use sglang_renderer::{DynamoTokenizer, TextTokenizer, load_tokenizer};

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
}

impl TokenizerWorker {
    pub fn new(
        rx: flume::Receiver<PreprocessJob>,
        tm: Option<flume::Sender<TmEvent>>,
        tokenizer: Arc<dyn TextTokenizer>,
    ) -> Self {
        let auto_specials = tokenizer.auto_specials();
        Self {
            rx,
            tm,
            tokenizer,
            auto_specials,
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
                        match tokenize_text_completion(
                            g.text.as_deref(),
                            &mut g.input_ids,
                            g.skip_special_tokens,
                            &mut g.sampling_params,
                            self.tokenizer.as_ref(),
                            &self.auto_specials,
                        ) {
                            Ok(()) => Event::TokenizeDone,
                            Err(err) => Event::Error(err.into()),
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
                    let crate::renderer::RenderJob { mut request, reply } = *job;
                    let result = tokenize_text_completion(
                        request.text.as_deref(),
                        &mut request.input_ids,
                        request.skip_special_tokens,
                        &mut request.sampling_params,
                        self.tokenizer.as_ref(),
                        &self.auto_specials,
                    )
                    .map(|()| request);
                    // The HTTP request may have been cancelled while preparing.
                    let _ = reply.send(result);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        TokenizerWorker::new(req_rx, Some(tm_tx), Arc::new(WordTokenizer)).run();

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
            TokenizerWorker::new(req_rx, Some(tm_tx), Arc::new(MarkedTokenizer)).run();
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
