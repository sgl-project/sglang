//! Tokenizer pool — CPU-bound, runs on pinned OS threads (off the async
//! executor). Each worker pulls a `Request` from the shared `flume` receiver,
//! fills `input_ids`, and moves the request back to the TokenizerManager inbox.
//!
//! The text→ids step is `sglang_processor`'s [`TextTokenizer`], implemented by
//! [`DynamoTokenizer`] (dynamo-tokenizers: HuggingFace / tiktoken / fastokens).
//! A non-skip server requires a real tokenizer (enforced at startup); under
//! `skip_tokenizer_init` the pool isn't spawned at all.
//!
//! Mirrors the Python `_tokenize_one_request` text path: when the request
//! already carries `input_ids` it skips tokenization (handled upstream in the
//! TokenizerManager `classify`); otherwise the prompt text is encoded here.

use std::sync::Arc;

use sglang_processor::{DynamoTokenizer, ProcessorError, TextTokenizer};

use crate::message::request::{Request, RequestKind, TokenizeInput};
use crate::message::response::ResponseItem;
use crate::message::types::TokenIds;
use crate::runtime::Runnable;
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::{error::Error, fsm::Event};

/// Load the tokenizer the encode pool uses; its [`DynamoTokenizer::tokenizer`]
/// handle is shared by the detok shards. `None` under `skip_tokenizer_init`,
/// else required (missing/failed load → `Err`). `tokenizer_path` is a
/// tokenizer file, a model dir, or an HF Hub repo id (resolved from the local
/// cache — no network).
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    revision: Option<&str>,
    skip_tokenizer_init: bool,
) -> Result<Option<DynamoTokenizer>, String> {
    if skip_tokenizer_init {
        tracing::info!("skip_tokenizer_init: token ids in and out; no tokenizer/detokenizer");
        return Ok(None);
    }
    let path = tokenizer_path.ok_or_else(|| {
        "no tokenizer configured: set tokenizer_path or enable skip_tokenizer_init".to_string()
    })?;
    DynamoTokenizer::load(path, revision).map(Some)
}

/// Widened once here, in the parallel pool: the scheduler's `array("q")` is
/// int64, and the ids move into their ring buffer untouched from here.
fn widen(encoded: Result<Vec<i32>, ProcessorError>) -> Result<TokenIds, Error> {
    match encoded {
        Ok(ids) => Ok(ids.into_iter().map(i64::from).collect()),
        Err(ProcessorError::Tokenize(message)) => Err(Error::Tokenize(message)),
        Err(error) => Err(Error::Tokenize(error.to_string())),
    }
}

/// One tokenizer worker: pulls a `Request` off the shared inbox, fills
/// `input_ids`, returns it to the TokenizerManager. Pinned; backend shared.
pub struct TokenizerWorker {
    rx: flume::Receiver<Request>,
    tm: flume::Sender<TmEvent>,
    tokenizer: Arc<dyn TextTokenizer>,
}

impl TokenizerWorker {
    pub fn new(
        rx: flume::Receiver<Request>,
        tm: flume::Sender<TmEvent>,
        tokenizer: Arc<dyn TextTokenizer>,
    ) -> Self {
        Self { rx, tm, tokenizer }
    }
}

impl Runnable for TokenizerWorker {
    fn run(self) {
        while let Ok(mut req) = self.rx.recv() {
            // A tokenize request is answered here, through its own sink: it is
            // not registered with a detok shard and does not return to intake.
            if let RequestKind::Tokenize(input) = &req.kind {
                let encoded = match input {
                    TokenizeInput::Text {
                        text,
                        add_special_tokens,
                    } => widen(self.tokenizer.encode(text, *add_special_tokens)),
                    TokenizeInput::Chat(chat) => widen(chat.encode(&*self.tokenizer)),
                };
                let item = match encoded {
                    Ok(ids) => ResponseItem::TokenIds(ids),
                    Err(err) => ResponseItem::Error(err),
                };
                let _ = req.sink.try_send(item); // client may be gone
                continue;
            }
            // Every other request is a generate request. Encode,
            // then advance the FSM: `TokenizeDone` on success (-> PreSendValidating,
            // or -> Encoding for a multimodal prompt; the state's `then` says which).
            // A request is never dropped here: a kind this pool cannot serve
            // goes back as `Failed`, so intake rejects it and releases its
            // tracking entry instead of leaving the client hung.
            let event = if let RequestKind::Generate(g) = &mut req.kind {
                // Size the scheduler's stop-match window in TOKENS, as Python's
                // `normalize(tokenizer)` does.
                let stop_tokens = g
                    .sampling_params
                    .stop_strs
                    .iter()
                    // A stop that won't encode falls back to its byte length rather
                    // than failing the request: still an over-estimate, never an
                    // under-estimate, so the scheduler cannot miss that stop.
                    .map(|s| {
                        self.tokenizer
                            .encode(s, false)
                            .map_or(s.len(), |ids| ids.len())
                    })
                    .max();
                if let Some(n) = stop_tokens {
                    g.sampling_params.stop_str_max_len = n;
                }
                let encoded = match g.chat_prompt.take() {
                    Some(chat) => widen(chat.encode(&*self.tokenizer)),
                    // Match Python sglang: reject an empty prompt as a 400
                    // (`Validation`), not the misleading 500 a tokenize error
                    // would give.
                    None => match g.text.as_deref().unwrap_or("") {
                        "" => Err(Error::Validation("prompt cannot be empty".into())),
                        text => widen(self.tokenizer.encode(text, true)),
                    },
                };
                match encoded {
                    Ok(ids) => {
                        g.input_ids = Some(ids);
                        Event::TokenizeDone
                    }
                    Err(err) => Event::Error(err),
                }
            } else {
                tracing::error!(rid = %req.rid, "tokenizer pool received a non-generate request");
                Event::Error(Error::Internal(
                    "non-generate request in the tokenizer pool".into(),
                ))
            };
            let _ = req.state.apply(event);
            if self.tm.send(TmEvent::Tokenized(req)).is_err() {
                tracing::error!("tm inbox closed; dropping request");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::request::{GenerateRequest, RenderedChat, RequestKind};
    use crate::message::response::ResponseSink;
    use crate::message::sampling::SamplingParams;
    use crate::utils::fsm::{AfterTokenize, RequestState};
    use sglang_processor::dynamo_renderer::RenderedPrompt;
    use tokio::sync::mpsc;

    /// One token per whitespace-separated word, so a stop's token count differs
    /// from its byte count and the two units cannot be confused.
    struct WordTokenizer;
    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str, _: bool) -> Result<Vec<i32>, ProcessorError> {
            Ok(text.split_whitespace().map(|_| 1).collect())
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
        let (req_tx, req_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();

        // 8 bytes vs 3 "tokens" under WordTokenizer — units are distinguishable.
        let sp = SamplingParams {
            stop_strs: vec!["a bb ccc".to_string(), "dd".to_string()],
            stop_str_max_len: 8, // what `normalize_stops` left: max BYTE length
            ..Default::default()
        };
        let (sink_tx, _sink_rx) = mpsc::channel(4);
        req_tx
            .send(Request {
                rid: "1".into(),
                state: RequestState::Tokenizing {
                    then: AfterTokenize::PreSend,
                },
                sink: ResponseSink::Local(sink_tx),
                kind: RequestKind::Generate(Box::new(GenerateRequest {
                    rid: "1".into(),
                    text: Some("hello world".into()),
                    sampling_params: sp,
                    ..Default::default()
                })),
            })
            .expect("send");
        drop(req_tx); // closes the loop after one request

        TokenizerWorker::new(req_rx, tm_tx, Arc::new(WordTokenizer)).run();

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

    /// A kind the pool cannot serve is returned `Failed`, never dropped:
    /// intake still holds its sink registration and pool-tracking entry.
    #[test]
    fn non_generate_request_is_returned_failed_not_dropped() {
        let (req_tx, req_rx) = flume::unbounded::<Request>();
        let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
        let (sink_tx, _sink_rx) = mpsc::channel(4);
        req_tx
            .send(Request {
                rid: "2".into(),
                state: RequestState::Tokenizing {
                    then: AfterTokenize::PreSend,
                },
                sink: ResponseSink::Local(sink_tx),
                kind: RequestKind::Detokenize {
                    token_ids: vec![1, 2],
                    skip_special_tokens: true,
                },
            })
            .expect("send");
        drop(req_tx);

        TokenizerWorker::new(req_rx, tm_tx, Arc::new(WordTokenizer)).run();

        let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned, not dropped") else {
            panic!("expected Tokenized");
        };
        assert_eq!(req.rid.as_str(), "2");
        assert!(
            matches!(req.state, RequestState::Failed(Error::Internal(_))),
            "{:?}",
            req.state
        );
    }

    /// One id per text (its length), after a BOS (id 0) when specials are
    /// added — like an HF tokenizer whose post-processor prepends one.
    struct MarkedTokenizer;
    impl TextTokenizer for MarkedTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, ProcessorError> {
            let bos = add_special_tokens.then_some(0);
            let text = (!text.is_empty()).then_some(text.len() as i32);
            Ok(bos.into_iter().chain(text).collect())
        }
        fn bos_token_id(&self) -> Option<i32> {
            Some(0)
        }
    }

    /// Plain text keeps the tokenizer's specials; a rendered chat encodes
    /// without them — Python's `add_special_tokens=False` at the
    /// chat-template encode site — and its continuation loses its BOS.
    #[test]
    fn rendered_prompts_encode_without_specials() {
        let run = |chat_prompt: Option<Box<RenderedChat>>| {
            let (req_tx, req_rx) = flume::unbounded::<Request>();
            let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
            req_tx
                .send(Request {
                    rid: "1".into(),
                    state: RequestState::Tokenizing {
                        then: AfterTokenize::PreSend,
                    },
                    sink: ResponseSink::Local(tokio::sync::mpsc::channel(4).0),
                    kind: RequestKind::Generate(Box::new(GenerateRequest {
                        rid: "1".into(),
                        text: Some("hi".into()),
                        chat_prompt,
                        ..Default::default()
                    })),
                })
                .expect("send");
            drop(req_tx);
            TokenizerWorker::new(req_rx, tm_tx, Arc::new(MarkedTokenizer)).run();
            let TmEvent::Tokenized(req) = tm_rx.try_recv().expect("returned") else {
                panic!("expected Tokenized");
            };
            let RequestKind::Generate(g) = &req.kind else {
                panic!("expected generate");
            };
            g.input_ids.clone().expect("tokenized")
        };
        assert_eq!(run(None), vec![0, 2], "plain text prompts keep specials");
        let chat = RenderedChat {
            prompt: RenderedPrompt::text("hi".into()),
            continuation: "abc".into(),
        };
        assert_eq!(
            run(Some(Box::new(chat))),
            vec![2, 3],
            "chat and continuation lose it too"
        );
    }

    /// A tokenize request is answered through its own sink and never returns
    /// to intake. `add_special_tokens` keeps or strips the auto specials, and
    /// an empty text encodes to just those specials, as Python's
    /// `encode("")` does.
    #[test]
    fn tokenize_answers_through_the_sink_without_returning_to_intake() {
        let run = |text: &str, add_special_tokens: bool| {
            let (req_tx, req_rx) = flume::unbounded::<Request>();
            let (tm_tx, tm_rx) = flume::unbounded::<TmEvent>();
            let (sink_tx, mut sink_rx) = tokio::sync::mpsc::channel(4);
            req_tx
                .send(Request {
                    rid: "t".into(),
                    state: RequestState::Queued,
                    sink: ResponseSink::Local(sink_tx),
                    kind: RequestKind::Tokenize(TokenizeInput::Text {
                        text: text.into(),
                        add_special_tokens,
                    }),
                })
                .expect("send");
            drop(req_tx);
            TokenizerWorker::new(req_rx, tm_tx, Arc::new(MarkedTokenizer)).run();
            assert!(tm_rx.try_recv().is_err(), "answered here, not by intake");
            match sink_rx.try_recv() {
                Ok(ResponseItem::TokenIds(ids)) => ids,
                other => panic!("expected token ids, got {other:?}"),
            }
        };
        assert_eq!(run("hi", true /* add_special_tokens */), vec![0, 2]);
        assert_eq!(run("hi", false /* add_special_tokens */), vec![2]);
        assert_eq!(run("", true /* add_special_tokens */), vec![0]);
        assert_eq!(run("", false /* add_special_tokens */), Vec::<i64>::new());
    }
}
