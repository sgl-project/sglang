//! Work an [`HttpLayer`] does on a `/generate` request before the handler
//! runs, with the request's rid open to aborts meanwhile.

use std::future::Future;
use std::sync::{Arc, Mutex};

use axum::http::Extensions;
use axum::{Extension, Router};

use crate::HttpLayer;
use crate::api_server::handlers::{Aborted, CoreHandle, PendingGeneration};

/// The core, as a request extension, for [`abortable_before_generate`].
#[derive(Clone)]
struct PreGenerateCore(CoreHandle);

/// The generation [`abortable_before_generate`] held, handed to `/generate`
/// as a request extension. axum extracts extensions by clone, so the handler
/// takes it from the shared slot.
#[derive(Clone)]
pub(super) struct HeldGeneration(Arc<Mutex<Option<PendingGeneration>>>);

impl HeldGeneration {
    pub(super) fn take(&self) -> Option<PendingGeneration> {
        self.0.lock().unwrap().take()
    }
}

/// Wrap `app` in `layer`, giving the layer's middleware what
/// [`abortable_before_generate`] needs.
pub(super) fn apply(app: Router, layer: &HttpLayer, core: CoreHandle) -> Router {
    layer(app).layer(Extension(PreGenerateCore(core)))
}

/// Run `work`, part of a single `/generate` request's work that an
/// [`HttpLayer`] does before `/generate` runs, with the request registered
/// under `rid` (the rid `/generate` will read) for `/abort_request`,
/// `abort_all` and the gRPC `Abort`. `extensions` are the request's.
///
/// Once such an abort selects the request, returns at once, without waiting
/// for `work`, with how the abort ends a request and whether it aborted all
/// requests ([`Aborted`]): the layer may answer the request itself, or pass
/// it on, and `/generate` ends it as the abort says without starting it.
/// Otherwise the request stays registered until `/generate` takes it over,
/// so an abort in between ends it the same way.
/// Dropping the request before `/generate` unregisters it. Outside a server's
/// app, `work` just runs.
pub async fn abortable_before_generate<T>(
    extensions: &mut Extensions,
    rid: &str,
    work: impl Future<Output = T>,
) -> Result<T, Aborted> {
    let Some(PreGenerateCore(core)) = extensions.get::<PreGenerateCore>() else {
        return Ok(work.await);
    };
    let mut pending = core.hold_generation(rid);
    let output = tokio::select! {
        biased;
        aborted = pending.cancelled() => Err(aborted),
        output = work => Ok(output),
    };
    extensions.insert(HeldGeneration(Arc::new(Mutex::new(Some(pending)))));
    output
}

/// The app [`abortable_before_generate`] needs, for tests of an
/// [`HttpLayer`] outside a running server.
#[cfg(feature = "test-support")]
pub mod test_support {
    use super::*;
    use crate::api_server::handlers::{AbortEnding, CoreConfig, CoreMetadata};
    use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

    /// Aborts for the requests an app from [`pre_generate_app`] holds.
    pub struct PreGenerateAborts {
        core: CoreHandle,
        _intake_rx: flume::Receiver<TmEvent>,
        _abort_rx: flume::Receiver<AbortSource>,
    }

    impl PreGenerateAborts {
        /// Abort the held requests under `rid`, or all of them, as
        /// `/abort_request` (`AbortFinishReason`) or the gRPC `Abort`
        /// (`CancelledError`) does. Whether any matched.
        pub fn abort(&self, rid: Option<&str>, ending: AbortEnding) -> bool {
            self.core.abort(rid, ending)
        }
    }

    /// `app`, already wrapped in the layer under test, as a server serves it
    /// to that layer, with no runtime behind it.
    pub fn pre_generate_app(app: Router) -> (Router, PreGenerateAborts) {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 1,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                multimodal: false,
                metadata: CoreMetadata::default(),
                metrics: None,
                peer_endpoints: Vec::new(),
            },
        );
        let app = app.layer(Extension(PreGenerateCore(core.clone())));
        let aborts = PreGenerateAborts {
            core,
            _intake_rx: intake_rx,
            _abort_rx: abort_rx,
        };
        (app, aborts)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::body::{Body, to_bytes};
    use axum::extract::Request;
    use axum::http::StatusCode;
    use axum::http::header::CONTENT_TYPE;
    use axum::middleware::{self, Next};
    use futures::StreamExt;
    use tower::ServiceExt;

    use super::*;
    use crate::api_server::handlers::{AbortEnding, CoreConfig, CoreMetadata};
    use crate::api_server::http::app::AppState;
    use crate::api_server::http::{info, native_api};
    use crate::message::config::ServerArgs;
    use crate::message::request::GenerationPolicy;
    use crate::message::response::{ChunkEvent, ResponseItem};
    use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

    const RID: &str = "held-rid";

    /// A layer whose work on a `/generate` waits on a remote dependency
    /// before `/generate` runs, under `abortable_before_generate`.
    struct Wait {
        /// The work started.
        started: flume::Receiver<()>,
        /// Ends the work.
        release: flume::Sender<()>,
        /// The work's result: `Err` when an abort ended the wait. The layer
        /// passes the request on either way.
        finished: flume::Receiver<Result<(), Aborted>>,
        /// Lets `/generate` run once the work is over.
        proceed: flume::Sender<()>,
    }

    struct Server {
        app: Router,
        core: CoreHandle,
        intake_rx: flume::Receiver<TmEvent>,
        abort_rx: flume::Receiver<AbortSource>,
        wait: Wait,
    }

    fn held_server() -> Server {
        let server_args = Arc::new(ServerArgs::default());
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                multimodal: false,
                metadata: CoreMetadata::from(server_args.as_ref()),
                metrics: None,
                peer_endpoints: Vec::new(),
            },
        );
        let state = Arc::new(AppState {
            core: core.clone(),
            generation_policy: GenerationPolicy::from(server_args.as_ref()),
            server_args,
            chat_formatter: None,
            generate_hook: None,
            openai_chat: None,
        });
        let (started_tx, started) = flume::unbounded();
        let (release, release_rx) = flume::unbounded();
        let (finished_tx, finished) = flume::unbounded();
        let (proceed, proceed_rx) = flume::unbounded();
        let layer: HttpLayer = Arc::new(move |app: Router| {
            let (started_tx, release_rx, finished_tx, proceed_rx) = (
                started_tx.clone(),
                release_rx.clone(),
                finished_tx.clone(),
                proceed_rx.clone(),
            );
            app.layer(middleware::from_fn(
                move |mut request: Request, next: Next| {
                    let (started_tx, release_rx, finished_tx, proceed_rx) = (
                        started_tx.clone(),
                        release_rx.clone(),
                        finished_tx.clone(),
                        proceed_rx.clone(),
                    );
                    async move {
                        if request.uri().path() == "/generate" {
                            let work = async {
                                started_tx.send(()).unwrap();
                                release_rx.recv_async().await.unwrap();
                            };
                            let finished =
                                abortable_before_generate(request.extensions_mut(), RID, work)
                                    .await;
                            finished_tx.send(finished).unwrap();
                            proceed_rx.recv_async().await.unwrap();
                        }
                        next.run(request).await
                    }
                },
            ))
        });
        let app = native_api::routes().merge(info::routes()).with_state(state);
        Server {
            app: apply(app, &layer, core.clone()),
            core,
            intake_rx,
            abort_rx,
            wait: Wait {
                started,
                release,
                finished,
                proceed,
            },
        }
    }

    impl Server {
        fn post(
            &self,
            path: &'static str,
            body: serde_json::Value,
        ) -> tokio::task::JoinHandle<axum::response::Response> {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let app = self.app.clone();
            tokio::spawn(async move { app.oneshot(request).await.unwrap() })
        }

        /// A client's `/abort_request`.
        async fn abort(&self, body: serde_json::Value) {
            let response = self.post("/abort_request", body).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        /// The response's terminal output: a unary body, or a stream's last
        /// frame before `[DONE]`. Fails if the request reached the scheduler.
        async fn terminal(
            &self,
            response: tokio::task::JoinHandle<axum::response::Response>,
            stream: bool,
        ) -> serde_json::Value {
            let response = tokio::select! {
                response = response => response.unwrap(),
                _ = self.intake_rx.recv_async() => panic!("the request reached the scheduler"),
            };
            assert_eq!(response.status(), StatusCode::OK);
            if !stream {
                let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
                return serde_json::from_slice(&body).unwrap();
            }
            let mut data = Vec::new();
            let mut body = response.into_body().into_data_stream();
            while let Some(chunk) = body.next().await {
                let chunk = chunk.unwrap();
                data.extend(
                    std::str::from_utf8(&chunk)
                        .unwrap()
                        .split("\n\n")
                        .filter_map(|event| event.strip_prefix("data: ").map(str::to_owned)),
                );
            }
            let [last, done] = data.as_slice() else {
                panic!("expected a final frame and [DONE]: {data:?}");
            };
            assert_eq!(done, "[DONE]");
            serde_json::from_str(last).unwrap()
        }
    }

    fn generate_body(stream: bool) -> serde_json::Value {
        serde_json::json!({"text": "hi", "rid": RID, "stream": stream})
    }

    /// An `/abort_request` by rid or `abort_all` while a layer still works on
    /// a `/generate` request ends the layer's wait at once, and the request
    /// ends on the frontend abort output without reaching the scheduler, even
    /// when the work would have finished and handed the request on. So does
    /// an abort landing after the work is over but before `/generate` runs.
    #[tokio::test]
    async fn an_abort_before_generate_ends_the_request_unstarted() {
        for abort in [
            serde_json::json!({"rid": RID}),
            serde_json::json!({"abort_all": true}),
        ] {
            for stream in [false, true] {
                for during_work in [true, false] {
                    let case = format!("{abort} stream={stream} during_work={during_work}");
                    let server = held_server();
                    let response = server.post("/generate", generate_body(stream));
                    server.wait.started.recv_async().await.unwrap();
                    if during_work {
                        server.abort(abort.clone()).await;
                        server.wait.release.send(()).unwrap();
                        let finished = tokio::time::timeout(
                            Duration::from_secs(5),
                            server.wait.finished.recv_async(),
                        )
                        .await
                        .expect("the layer's work returns")
                        .unwrap();
                        let abort_all = abort["abort_all"] == true;
                        assert_eq!(
                            finished,
                            Err(Aborted {
                                ending: AbortEnding::AbortFinishReason,
                                abort_all,
                            }),
                            "{case}: the abort ends the wait"
                        );
                    } else {
                        server.wait.release.send(()).unwrap();
                        assert_eq!(server.wait.finished.recv_async().await.unwrap(), Ok(()));
                        server.abort(abort.clone()).await;
                    }
                    server.wait.proceed.send(()).unwrap();

                    let terminal = server.terminal(response, stream).await;
                    let meta = &terminal["meta_info"];
                    assert_eq!(meta["id"], RID, "{case}: {terminal}");
                    assert_eq!(meta["finish_reason"]["type"], "abort", "{case}: {terminal}");
                    assert_eq!(meta["finish_reason"]["message"], "Aborted", "{case}");
                    assert!(server.intake_rx.try_recv().is_err(), "{case}");
                    assert!(server.abort_rx.try_recv().is_err(), "{case}");
                    assert!(
                        !server.core.abort(Some(RID), AbortEnding::AbortFinishReason),
                        "{case}: nothing stays registered"
                    );
                }
            }
        }
    }

    /// A held request that is not aborted runs under its own rid, where a
    /// later `/abort_request` still finds it; a gRPC `Abort` during the
    /// layer's work reports its own ending, and `/generate` ends the request
    /// on the cancelled error; a client that disconnects during the work
    /// leaves nothing registered or submitted.
    #[tokio::test]
    async fn a_held_request_runs_or_disconnects_cleanly() {
        let server = held_server();
        let response = server.post("/generate", generate_body(false));
        server.wait.started.recv_async().await.unwrap();
        server.wait.release.send(()).unwrap();
        server.wait.proceed.send(()).unwrap();
        let TmEvent::Intake { request, admission } = server.intake_rx.recv_async().await.unwrap()
        else {
            panic!("expected the generation");
        };
        assert!(admission.try_accept());
        assert_eq!(request.rid.client_facing(), RID);
        server.abort(serde_json::json!({"rid": RID})).await;
        assert!(
            matches!(server.abort_rx.try_recv(), Ok(AbortSource::Request(rid)) if rid == request.rid)
        );
        request
            .sink
            .try_send(ResponseItem::Done(ChunkEvent {
                rid: request.rid.clone(),
                finish_reason: Some(
                    serde_json::from_value(serde_json::json!({"type": "abort"})).unwrap(),
                ),
                ..Default::default()
            }))
            .unwrap();
        let terminal = server.terminal(response, false).await;
        assert_eq!(terminal["meta_info"]["finish_reason"]["type"], "abort");

        let server = held_server();
        let response = server.post("/generate", generate_body(false));
        server.wait.started.recv_async().await.unwrap();
        assert!(server.core.abort(Some(RID), AbortEnding::CancelledError));
        assert_eq!(
            server.wait.finished.recv_async().await.unwrap(),
            Err(Aborted {
                ending: AbortEnding::CancelledError,
                abort_all: false,
            })
        );
        server.wait.proceed.send(()).unwrap();
        let response = tokio::select! {
            response = response => response.unwrap(),
            _ = server.intake_rx.recv_async() => panic!("the request reached the scheduler"),
        };
        assert_eq!(response.status().as_u16(), 499);
        assert!(server.abort_rx.try_recv().is_err());

        let server = held_server();
        let response = server.post("/generate", generate_body(false));
        server.wait.started.recv_async().await.unwrap();
        response.abort();
        assert!(response.await.unwrap_err().is_cancelled());
        assert!(!server.core.abort(Some(RID), AbortEnding::AbortFinishReason));
        assert!(server.intake_rx.try_recv().is_err());
        assert!(server.abort_rx.try_recv().is_err());
    }
}
