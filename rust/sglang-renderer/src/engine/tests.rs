use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use futures::{FutureExt, StreamExt, future::BoxFuture};

use super::{
    GenerateTransport, GenerationService, TokenDecoder, TokenDelta, TokenStream,
    test_utils::{position, tiny_tokenizer},
};
use crate::{
    GenerateRequest, GenerationFinishReason, GenerationOptions, GenerationOutputExtras,
    MatchedStop, ResponseError, TokenIdsRequest,
};

struct DropNotice(Arc<AtomicUsize>);

impl Drop for DropNotice {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct MemoryTransport {
    pending_submission: bool,
    dropped: Arc<AtomicUsize>,
    requests: Mutex<Vec<GenerateRequest>>,
}

impl GenerateTransport for MemoryTransport {
    fn generate(
        &self,
        request: GenerateRequest,
    ) -> BoxFuture<'_, Result<TokenStream, ResponseError>> {
        Box::pin(async move {
            let guard = DropNotice(self.dropped.clone());
            self.requests.lock().unwrap().push(request);
            if self.pending_submission {
                futures::future::pending::<()>().await;
            }
            Ok(async_stream::stream! {
                let _guard = guard;
                for ids in [vec![104], vec![101, 108]] {
                    yield Ok(TokenDelta {
                        completion_tokens: ids.len() as u64,
                        extras: Some(Box::new(GenerationOutputExtras {
                            output_logprobs: ids.iter().map(|&id| position(id, -0.1, &[(id, -0.1)])).collect(),
                            ..Default::default()
                        })),
                        token_ids: ids,
                        prompt_tokens: 1,
                        ..Default::default()
                    });
                }
                futures::future::pending::<()>().await;
            }.boxed())
        })
    }
}

fn transport(pending_submission: bool) -> Arc<MemoryTransport> {
    Arc::new(MemoryTransport {
        pending_submission,
        dropped: Arc::new(AtomicUsize::new(0)),
        requests: Mutex::new(Vec::new()),
    })
}

fn request() -> GenerateRequest {
    TokenIdsRequest {
        rid: "generate".into(),
        input_ids: vec![65],
        options: GenerationOptions::default(),
        metadata: Default::default(),
    }
    .into()
}

#[tokio::test]
async fn shared_decoder_stops_across_chunks_and_releases_transport() {
    for no_stop_trim in [false, true] {
        let transport = transport(false);
        let service =
            GenerationService::new(transport.clone(), TokenDecoder::new(tiny_tokenizer()));
        let mut request = request();
        request.sampling_params.stop = vec!["he".into()];
        request.sampling_params.stop_token_ids = Some(vec![9]);
        request.sampling_params.no_stop_trim = no_stop_trim;
        request.return_text_in_logprobs = Some(true);
        let mut events = service.generate(request).await.unwrap();

        let first = events.next().await.unwrap().unwrap();
        assert!(first.text.is_empty());
        let last = events.next().await.unwrap().unwrap();
        assert_eq!(last.text, if no_stop_trim { "he" } else { "" });
        assert_eq!(last.token_ids, [101]);
        assert_eq!(last.completion_tokens, 1);
        assert_eq!(
            last.finish_reason,
            Some(GenerationFinishReason::Stop(Some(MatchedStop::Text(
                "he".into()
            ))))
        );
        let positions = &last.extras.unwrap().output_logprobs;
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].token.text.as_deref(), Some("e"));
        assert_eq!(positions[0].top[0].text.as_deref(), Some("e"));
        // Release upstream as soon as a local stop is emitted, even if the caller
        // keeps the completed response stream alive without polling it again.
        assert_eq!(transport.dropped.load(Ordering::SeqCst), 1);
        assert!(events.next().await.is_none());

        let sent = transport.requests.lock().unwrap();
        assert!(sent[0].sampling_params.stop.is_empty());
        assert_eq!(sent[0].sampling_params.stop_token_ids, Some(vec![9]));
        assert_eq!(sent[0].return_text_in_logprobs, Some(false));
    }
}

#[tokio::test]
async fn cancellation_releases_pending_submissions_and_unpolled_streams() {
    for pending_submission in [true, false] {
        let transport = transport(pending_submission);
        let service =
            GenerationService::new(transport.clone(), TokenDecoder::new(tiny_tokenizer()));
        let submission = service.generate_many(vec![request(), request(), request()]);
        if pending_submission {
            // Poll every submission once, then cancel the aggregate future.
            assert!(submission.now_or_never().is_none());
        } else {
            let streams = submission.await.unwrap();
            assert_eq!(transport.dropped.load(Ordering::SeqCst), 0);
            drop(streams);
        }
        assert_eq!(transport.requests.lock().unwrap().len(), 3);
        assert_eq!(transport.dropped.load(Ordering::SeqCst), 3);
    }
}
