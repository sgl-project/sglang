//! The gRPC service over its listener: generation streams, batch items,
//! aborts, the HiCache controls and the server's launch facts, with a real
//! client on one side and the test as the scheduler on the other.

use std::net::SocketAddr;
use std::time::Duration;

use serde_json::{Value, json};
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::generate_stream_item::Item;
use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;
use sglang_server::ServerArgs;
use sglang_server_test::{
    ApiKeys, OutputDetails, PreferredSamplingParams, SchedulerlessServer, free_loopback_addrs,
};
use tonic::Code;
use tonic::transport::Channel;

use crate::common::{
    connect_grpc, fused_dir, header_json, next_scheduler_request, next_scheduler_work, piece_id,
    take_startup_freeze_gc, tokenizer_dir, word_id,
};

/// A served frontend, a connected client, and the runtime driving the client.
struct Harness {
    rt: SchedulerlessServer,
    client: SglangServiceClient<Channel>,
    client_rt: tokio::runtime::Runtime,
}

impl Harness {
    /// A server with the word tokenizer of `tokenizer_dir` (ids 1..=4 are
    /// `Hel`, `lo`, `prompt`, `ok`) after `change` to its launch arguments.
    fn start(change: impl FnOnce(&mut ServerArgs)) -> Self {
        let [http_addr, grpc_addr] = free_loopback_addrs();
        let rt = SchedulerlessServer::builder(http_addr)
            .grpc_addr(grpc_addr)
            .server_args(|args| {
                args.skip_tokenizer_init = false;
                args.tokenizer_path = tokenizer_dir().into();
                args.served_model_name = "test-model".into();
                args.model_config.context_len = 4096;
                change(args);
            })
            .start()
            .expect("start runtime");
        take_startup_freeze_gc(&rt);
        Self::connect(rt, grpc_addr)
    }

    fn connect(rt: SchedulerlessServer, grpc_addr: SocketAddr) -> Self {
        // Multi-threaded, so the client's connection task runs between `block_on`s.
        let client_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let client = client_rt.block_on(connect_grpc(grpc_addr));
        Self {
            rt,
            client,
            client_rt,
        }
    }

    fn generate(
        &mut self,
        request: api::GenerateRequest,
    ) -> tonic::Streaming<api::GenerateStreamItem> {
        self.try_generate(request).expect("Generate RPC")
    }

    fn try_generate(
        &mut self,
        request: api::GenerateRequest,
    ) -> Result<tonic::Streaming<api::GenerateStreamItem>, tonic::Status> {
        let call = self.client.generate(request);
        self.client_rt
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), call).await })
            .expect("timed out starting Generate RPC")
            .map(tonic::Response::into_inner)
    }

    /// The stream's next message, `None` at its end.
    fn next(
        &self,
        stream: &mut tonic::Streaming<api::GenerateStreamItem>,
    ) -> Result<Option<api::GenerateStreamItem>, tonic::Status> {
        self.client_rt
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(5), stream.message()).await
            })
            .expect("timed out waiting for the gRPC stream")
    }

    fn next_frame(
        &self,
        stream: &mut tonic::Streaming<api::GenerateStreamItem>,
    ) -> api::GenerateResponse {
        frame(self.next(stream).unwrap().expect("a stream item"))
    }

    fn abort(&mut self, rid: &str, abort_all: bool) -> Result<api::AbortResponse, tonic::Status> {
        let call = self.client.abort(api::AbortRequest {
            rid: rid.into(),
            abort_all,
        });
        self.client_rt
            .block_on(call)
            .map(tonic::Response::into_inner)
    }

    fn nothing_submitted(&self, what: &str) {
        assert!(
            self.rt.next_request(Duration::from_millis(200)).is_none(),
            "{what} reached the scheduler"
        );
    }
}

/// One output for `rid`: `token_id`, three prompt tokens of which two cached.
fn chunk(finished: bool) -> OutputDetails {
    OutputDetails {
        prompt_tokens: 3,
        cached_tokens: 2,
        finish_reason: finished.then(|| json!({"type": "stop"})),
        ..Default::default()
    }
}

fn one_string(value: &str) -> api::StringOrList {
    api::StringOrList {
        value: Some(api::string_or_list::Value::One(value.into())),
    }
}

fn many_strings(values: &[&str]) -> api::StringOrList {
    api::StringOrList {
        value: Some(api::string_or_list::Value::Many(api::StringList {
            items: values.iter().map(|value| (*value).to_owned()).collect(),
        })),
    }
}

fn text_request(text: &str, rid: &str) -> api::GenerateRequest {
    api::GenerateRequest {
        text: Some(one_string(text)),
        rid: Some(one_string(rid)),
        stream: Some(true),
        ..Default::default()
    }
}

fn ids_request(ids: &[i64], rid: Option<&str>) -> api::GenerateRequest {
    api::GenerateRequest {
        input_ids: Some(api::TokenIdsOrList {
            value: Some(api::token_ids_or_list::Value::One(api::TokenIds {
                ids: ids.to_vec(),
            })),
        }),
        rid: rid.map(one_string),
        stream: Some(true),
        ..Default::default()
    }
}

fn frame(item: api::GenerateStreamItem) -> api::GenerateResponse {
    match item.item {
        Some(Item::Frame(frame)) => frame,
        other => panic!("expected a frame, got {other:?}"),
    }
}

fn stream_error(item: api::GenerateStreamItem) -> api::GenerateStreamError {
    match item.item {
        Some(Item::Error(error)) => error,
        other => panic!("expected an error item, got {other:?}"),
    }
}

fn meta(frame: &api::GenerateResponse) -> &api::GenerateMetaInfo {
    frame
        .meta_info
        .as_ref()
        .expect("every frame carries meta_info")
}

fn output_ids(frame: &api::GenerateResponse) -> &[i64] {
    frame
        .output_ids
        .as_ref()
        .map_or(&[], |ids| ids.ids.as_slice())
}

#[test]
fn generate_streams_cumulative_frames_for_a_text_prompt() {
    let mut h = Harness::start(|_| {});
    let mut stream = h.generate(text_request("prompt", "request-1"));

    let request = next_scheduler_request(&h.rt);
    let header = header_json(&request);
    assert_eq!(header[0], "TokenizedGenerateReqInput");
    assert_eq!(header[3], "prompt");
    assert_eq!(request.buffer_shape("input_ids"), Some(&[1][..]));
    let rid = request.rid().expect("rid");
    assert!(rid.starts_with("request-1"), "{rid}");
    // One chunk at a time: a queued backlog of cumulative frames collapses to
    // its last, so sending both up front would yield only the terminal frame.
    h.rt.push_output_with(&rid, &[1], chunk(false)).unwrap();
    let first = h.next_frame(&mut stream);
    assert_eq!(first.text, "Hel");
    assert_eq!(output_ids(&first), [1]);
    assert_eq!(first.index, None);
    let first_meta = meta(&first);
    assert_eq!(first_meta.id, "request-1");
    assert_eq!(first_meta.prompt_tokens, 3);
    assert_eq!(first_meta.cached_tokens, 2);
    assert_eq!(first_meta.completion_tokens, 1);
    assert!(first_meta.finish_reason.is_none());
    assert!(first_meta.e2e_latency.is_none());

    h.rt.push_output_with(&rid, &[2], chunk(true)).unwrap();
    let finished = h.next_frame(&mut stream);
    assert_eq!(finished.text, "Hel lo");
    assert_eq!(output_ids(&finished), [1, 2]);
    let finished_meta = meta(&finished);
    assert_eq!(finished_meta.completion_tokens, 2);
    assert!(matches!(
        finished_meta.finish_reason,
        Some(api::FinishReason {
            kind: Some(api::finish_reason::Kind::Stop(_))
        })
    ));
    assert!(finished_meta.e2e_latency.is_some());
    assert!(h.next(&mut stream).unwrap().is_none());
    h.nothing_submitted("an abort of a finished request");
}

/// Incremental frames carry each delta with the cumulative completion
/// count and cached-token split, which a scheduler abort does not carry.
#[test]
fn incremental_frames_carry_deltas_with_cumulative_count() {
    let mut h = Harness::start(|args| args.incremental_streaming_output = true);
    let mut stream = h.generate(ids_request(&[4, 5], Some("tokens")));

    let request = next_scheduler_request(&h.rt);
    assert_eq!(request.buffer_shape("input_ids"), Some(&[2][..]));
    assert!(header_json(&request)[3].is_null(), "no text beside the ids");
    let rid = request.rid().expect("rid");
    let first = OutputDetails {
        cached_tokens_split: Some(json!({"device": 1, "host": 1, "storage": 0})),
        ..chunk(false)
    };
    h.rt.push_output_with(&rid, &[1], first).unwrap();
    let first = h.next_frame(&mut stream);
    let aborted = OutputDetails {
        abort: true,
        ..chunk(true)
    };
    h.rt.push_output_with(&rid, &[2], aborted).unwrap();
    let finished = h.next_frame(&mut stream);
    assert_eq!(output_ids(&first), [1]);
    assert_eq!(output_ids(&finished), [2]);
    assert_eq!(finished.text.trim(), "lo");
    assert_eq!(meta(&finished).completion_tokens, 2);
    let split = Some(api::CachedTokensDetails {
        device: 1,
        host: 1,
        storage: None,
        storage_backend: None,
    });
    assert_eq!(meta(&first).cached_tokens_details, split);
    assert_eq!(meta(&finished).cached_tokens_details, split);
    assert!(h.next(&mut stream).unwrap().is_none());
}

/// `stream: false` is a one-frame stream: deltas fold into the terminal
/// result, and the server's incremental policy does not split a unary reply.
#[test]
fn unary_request_yields_only_the_cumulative_terminal_frame() {
    let mut h = Harness::start(|args| args.incremental_streaming_output = true);
    let mut request = text_request("prompt", "unary");
    request.stream = Some(false);
    let mut stream = h.generate(request);

    let rid = next_scheduler_request(&h.rt).rid().expect("rid");
    h.rt.push_output_with(&rid, &[1], chunk(false)).unwrap();
    h.rt.push_output_with(&rid, &[2], chunk(true)).unwrap();

    let only = h.next_frame(&mut stream);
    assert_eq!(only.text, "Hel lo");
    assert_eq!(output_ids(&only), [1, 2]);
    assert_eq!(meta(&only).completion_tokens, 2);
    assert!(h.next(&mut stream).unwrap().is_none());
}

#[test]
fn stream_drop_aborts_admitted_request() {
    let mut h = Harness::start(|_| {});
    let mut stream = h.generate(ids_request(&[1], Some("cancel-me")));
    let rid = next_scheduler_request(&h.rt).rid().expect("rid");
    h.rt.push_output_with(&rid, &[1], chunk(false)).unwrap();
    assert!(h.next(&mut stream).unwrap().is_some());
    drop(stream);
    let abort = next_scheduler_request(&h.rt);
    assert_eq!(abort.kind().as_deref(), Some("AbortReq"));
    assert_eq!(abort.rid(), Some(rid));
    h.nothing_submitted("a second abort");
}

#[test]
fn per_chunk_timeout_after_progress_returns_deadline_exceeded_and_aborts() {
    let mut h = Harness::start(|args| args.grpc_response_timeout_secs = 1);
    let mut stream = h.generate(ids_request(&[1], Some("too-slow")));
    let rid = next_scheduler_request(&h.rt).rid().expect("rid");
    h.rt.push_output_with(&rid, &[1], chunk(false)).unwrap();
    assert!(h.next(&mut stream).unwrap().is_some());

    let error = h.next(&mut stream).unwrap_err();
    assert_eq!(error.code(), Code::DeadlineExceeded);
    drop(stream);
    let abort = next_scheduler_request(&h.rt);
    assert_eq!(abort.kind().as_deref(), Some("AbortReq"));
    assert_eq!(abort.rid(), Some(rid));
}

/// A list-form request fans out like the HTTP batch: every item is tagged
/// with its position, and one item's failure neither ends the stream nor
/// touches its siblings.
#[test]
fn batch_items_are_indexed_and_fail_independently() {
    let mut h = Harness::start(|_| {});
    let mut stream = h.generate(api::GenerateRequest {
        text: Some(many_strings(&["Hel", "lo"])),
        stream: Some(true),
        ..Default::default()
    });

    let mut items = [next_scheduler_request(&h.rt), next_scheduler_request(&h.rt)];
    items.sort_by_key(|request| header_json(request)[3].as_str().map(str::to_owned));
    let [first, second] = items;
    assert_eq!(header_json(&first)[3], "Hel");
    assert_eq!(header_json(&second)[3], "lo");
    let second_rid = second.rid().expect("rid");
    h.rt.push_error(&first.rid().expect("rid"), "first failed")
        .unwrap();
    h.rt.push_output_with(&second_rid, &[4], chunk(true))
        .unwrap();

    let mut items = Vec::new();
    while let Some(item) = h.next(&mut stream).unwrap() {
        items.push(item);
    }
    assert_eq!(items.len(), 2);
    let (errors, frames): (Vec<_>, Vec<_>) = items
        .into_iter()
        .partition(|item| matches!(item.item, Some(Item::Error(_))));
    let error = stream_error(errors.into_iter().next().unwrap());
    assert_eq!(error.index, Some(0));
    assert!(error.error.unwrap().message.contains("first failed"));
    let ok = frame(frames.into_iter().next().unwrap());
    assert_eq!(ok.index, Some(1));
    assert_eq!(ok.text, "ok");
    assert!(second_rid.starts_with(&meta(&ok).id), "{second_rid}");
}

/// Check gRPC validation, scheduler results, and the configuration returned by GET.
#[test]
fn hicache_controls_preserve_results_and_configuration() {
    // Without an admin key, attach and detach are refused before any rank
    // sees them, as on HTTP.
    let mut refusing = Harness::start(|_| {});
    let attach = refusing
        .client_rt
        .block_on(
            refusing
                .client
                .attach_hi_cache_storage(api::AttachHiCacheStorageRequest {
                    hicache_storage_backend: Some("file".into()),
                    ..Default::default()
                }),
        )
        .unwrap_err();
    let detach = refusing
        .client_rt
        .block_on(
            refusing
                .client
                .detach_hi_cache_storage(api::DetachHiCacheStorageRequest {}),
        )
        .unwrap_err();
    for error in [attach, detach] {
        assert_eq!(error.code(), Code::FailedPrecondition);
        assert_eq!(
            error.message(),
            "HiCache attach/detach is not available over gRPC; use the HTTP endpoint with --admin-api-key."
        );
    }
    refusing.nothing_submitted("a storage change without an admin key");

    let mut h = Harness::start(|args| args.api_keys = ApiKeys::new(None, Some("sk-admin".into())));
    let initial = h
        .client_rt
        .block_on(
            h.client
                .get_hi_cache_storage(api::GetHiCacheStorageRequest {}),
        )
        .unwrap()
        .into_inner();
    assert_eq!(initial, api::HiCacheStatus::default());

    let error = h
        .client_rt
        .block_on(
            h.client
                .attach_hi_cache_storage(api::AttachHiCacheStorageRequest::default()),
        )
        .unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
    h.nothing_submitted("an attach with no backend");

    // Drive the RPC and the scheduler's reply together, checking the request tag.
    macro_rules! hicache_reply {
        ($tag:expr, $success:expr, $call:expr) => {{
            let call = h.client_rt.spawn($call);
            let request = next_scheduler_work(&h.rt);
            assert_eq!(request.kind().as_deref(), Some($tag));
            h.rt.push_control_result(
                &request.rid().expect("rid"),
                &json!({"success": $success, "message": "backend detail"}),
            )
            .unwrap();
            h.client_rt
                .block_on(async { tokio::time::timeout(Duration::from_secs(5), call).await })
                .expect("HiCache RPC did not finish")
                .unwrap()
                .unwrap()
                .into_inner()
        }};
    }
    for success in [false, true] {
        let mut client = h.client.clone();
        let response = hicache_reply!("AttachHiCacheStorageReqInput", success, async move {
            client
                .attach_hi_cache_storage(api::AttachHiCacheStorageRequest {
                    hicache_storage_backend: Some("file".into()),
                    hicache_storage_backend_extra_config_json: Some(r#"{"secret_key":"s"}"#.into()),
                    hicache_write_policy: Some("write_through".into()),
                    ..Default::default()
                })
                .await
        });
        assert_eq!(response.success, success);
        assert_eq!(response.message, "backend detail");
    }
    let status = h
        .client_rt
        .block_on(
            h.client
                .get_hi_cache_storage(api::GetHiCacheStorageRequest {}),
        )
        .unwrap()
        .into_inner();
    assert_eq!(
        status.hicache_storage_backend_extra_config.as_deref(),
        Some("<redacted>")
    );
    for success in [false, true] {
        let mut client = h.client.clone();
        let response = hicache_reply!("ClearHiCacheReqInput", success, async move {
            client
                .clear_hi_cache_storage(api::ClearHiCacheStorageRequest {})
                .await
        });
        assert_eq!(response.success, success);
        assert!(response.message.is_empty());
    }
    for success in [false, true] {
        let mut client = h.client.clone();
        let response = hicache_reply!("DetachHiCacheStorageReqInput", success, async move {
            client
                .detach_hi_cache_storage(api::DetachHiCacheStorageRequest {})
                .await
        });
        assert_eq!(response.success, success);
        assert_eq!(response.message, "backend detail");
    }
    let status = h
        .client_rt
        .block_on(
            h.client
                .get_hi_cache_storage(api::GetHiCacheStorageRequest {}),
        )
        .unwrap()
        .into_inner();
    assert_eq!(
        status,
        api::HiCacheStatus {
            hicache_write_policy: Some("write_through".into()),
            ..Default::default()
        }
    );
}

/// Answer the one `GetServerInfo` call with an empty internal state.
fn server_info(h: &mut Harness) -> api::GetServerInfoResponse {
    let mut client = h.client.clone();
    let call = h
        .client_rt
        .spawn(async move { client.get_server_info(api::GetServerInfoRequest {}).await });
    let request = next_scheduler_work(&h.rt);
    h.rt.push_control_result(&request.rid().expect("rid"), &json!({"internal_state": {}}))
        .unwrap();
    h.client_rt
        .block_on(async { tokio::time::timeout(Duration::from_secs(5), call).await })
        .expect("GetServerInfo did not finish")
        .unwrap()
        .unwrap()
        .into_inner()
}

/// Out-of-process orchestrators read these launch facts to reproduce the
/// server's token admission.
#[test]
fn server_info_reports_launch_facts_for_orchestrators() {
    let [http_addr, grpc_addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(http_addr)
        .grpc_addr(grpc_addr)
        .server_args(|args| {
            args.allow_auto_truncate = true;
            args.num_reserved_tokens = 4;
        })
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let mut h = Harness::connect(rt, grpc_addr);
    let info = server_info(&mut h);
    assert!(info.allow_auto_truncate);
    assert!(info.skip_tokenizer_init);
    assert_eq!(info.num_reserved_tokens, 4);
}

/// gRPC `GetServerInfo` advertises the same nonempty owning-runtime id on
/// repeated calls against the same server.
#[test]
fn server_info_instance_id_is_stable_across_grpc_reads() {
    let mut h = Harness::start(|_| {});
    let ids = [
        server_info(&mut h).instance_id,
        server_info(&mut h).instance_id,
    ];
    let uuid = uuid::Uuid::parse_str(&ids[0]).unwrap();
    assert_eq!(uuid.get_version(), Some(uuid::Version::Random));
    assert_eq!(ids[0], ids[1]);
}

#[test]
fn list_models_uses_static_metadata_without_intake() {
    let mut h = Harness::start(|_| {});
    let models = h
        .client_rt
        .block_on(h.client.list_models(api::ListModelsRequest {}))
        .unwrap()
        .into_inner()
        .models;
    assert_eq!(
        models,
        vec![api::ModelCard {
            id: "test-model".into(),
            root: "test-model".into(),
            parent: None,
            max_model_len: 4096,
        }]
    );
    h.nothing_submitted("a model listing");
}

#[test]
fn abort_rpc_cancels_generation_with_an_in_stream_error() {
    let mut h = Harness::start(|_| {});
    let mut stream = h.generate(ids_request(&[1], Some("abort-me")));
    let rid = next_scheduler_request(&h.rt).rid().expect("rid");
    assert!(h.abort("abort-me", false).unwrap().success);
    let abort = next_scheduler_request(&h.rt);
    assert_eq!(abort.kind().as_deref(), Some("AbortReq"));
    assert_eq!(abort.rid(), Some(rid));
    let error = stream_error(h.next(&mut stream).unwrap().expect("an error item"));
    assert_eq!(error.error.unwrap().code, 499);
    assert!(h.next(&mut stream).unwrap().is_none());
    assert!(!h.abort("abort-me", false).unwrap().success);
    drop(stream);
    h.nothing_submitted("a second abort");
}

#[test]
fn abort_validates_selector_and_supports_all() {
    let mut h = Harness::start(|_| {});
    for (rid, abort_all) in [("", false), ("ambiguous", true)] {
        let error = h.abort(rid, abort_all).unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
    }
    let stream = h.generate(ids_request(&[1], Some("all")));
    let rid = next_scheduler_request(&h.rt).rid().expect("rid");
    assert!(h.abort("", true).unwrap().success);
    let abort = next_scheduler_request(&h.rt);
    assert_eq!(abort.kind().as_deref(), Some("AbortReq"));
    assert_eq!(abort.rid(), Some(rid));
    drop(stream);
    h.nothing_submitted("a second abort");
}

/// One abort cancels every live generation under a rid, and none started
/// under that rid afterwards.
fn assert_duplicate_abort_and_reuse(public_id: &str) {
    let mut h = Harness::start(|_| {});
    let mut first = h.generate(ids_request(&[1], Some(public_id)));
    let mut second = h.generate(ids_request(&[1], Some(public_id)));
    let old: std::collections::HashSet<_> =
        [next_scheduler_request(&h.rt), next_scheduler_request(&h.rt)]
            .map(|request| request.rid().expect("rid"))
            .into();
    assert!(h.abort(public_id, false).unwrap().success);
    assert!(!h.abort(public_id, false).unwrap().success);
    // The scheduler is told without either response being read.
    let aborted: std::collections::HashSet<_> =
        [next_scheduler_request(&h.rt), next_scheduler_request(&h.rt)]
            .map(|abort| {
                assert_eq!(abort.kind().as_deref(), Some("AbortReq"));
                abort.rid().expect("rid")
            })
            .into();
    assert_eq!(aborted, old);

    let new = h.generate(ids_request(&[1], Some(public_id)));
    let new_rid = next_scheduler_request(&h.rt).rid().expect("rid");
    for stream in [&mut first, &mut second] {
        let error = stream_error(h.next(stream).unwrap().expect("an error item"));
        assert_eq!(error.error.unwrap().code, 499);
        assert!(h.next(stream).unwrap().is_none());
    }
    drop((first, second));
    h.nothing_submitted("an abort of an ended call");
    assert!(h.abort(public_id, false).unwrap().success);
    let abort = next_scheduler_request(&h.rt);
    assert_eq!(abort.kind().as_deref(), Some("AbortReq"));
    assert_eq!(abort.rid(), Some(new_rid));
    drop(new);
    h.nothing_submitted("a second abort of the new call");
}

#[test]
fn explicit_abort_cancels_duplicates_but_not_later_id_reuse() {
    assert_duplicate_abort_and_reuse("same");
}

#[test]
fn long_public_id_abort_cancels_duplicates_but_not_later_reuse() {
    assert_duplicate_abort_and_reuse(&"x".repeat(200));
}

/// A call that finished, or whose client hung up, is no longer there for an
/// abort to find.
#[test]
fn completed_and_dropped_calls_are_removed_from_abort_lookup() {
    let mut h = Harness::start(|_| {});
    let mut stream = h.generate(ids_request(&[1], Some("done")));
    let rid = next_scheduler_request(&h.rt).rid().expect("rid");
    h.rt.push_output_with(&rid, &[], chunk(true)).unwrap();
    frame(h.next(&mut stream).unwrap().expect("the terminal frame"));
    assert!(h.next(&mut stream).unwrap().is_none());
    assert!(!h.abort("done", false).unwrap().success);
    drop(stream);
    h.nothing_submitted("an abort of a finished call");

    let stream = h.generate(ids_request(&[1], Some("dropped")));
    let rid = next_scheduler_request(&h.rt).rid().expect("rid");
    drop(stream);
    let abort = next_scheduler_request(&h.rt);
    assert_eq!(abort.kind().as_deref(), Some("AbortReq"));
    assert_eq!(abort.rid(), Some(rid));
    assert!(!h.abort("dropped", false).unwrap().success);
    assert!(!h.abort("", true).unwrap().success);
}

/// The sampling params a generation reaches the scheduler with:
/// `max_new_tokens` first, `temperature` fifth and `top_p` sixth.
fn submitted_sampling(h: &mut Harness, params: api::SamplingParams) -> Vec<Value> {
    let mut request = ids_request(&[1], None);
    request.sampling_params = Some(api::SamplingParamsOrList {
        value: Some(api::sampling_params_or_list::Value::One(params)),
    });
    let _stream = h.generate(request);
    // Dropping an earlier call's stream aborted it; skip that.
    let header = header_json(&next_scheduler_work(&h.rt));
    header[8].as_array().expect("sampling params").clone()
}

/// Launch-time preferred sampling params fill the fields a protobuf request
/// left unset, beneath the ones it set: the HTTP precedence at the gRPC entry.
#[test]
fn preferred_sampling_params_fill_unset_fields() {
    let mut h = Harness::start(|args| {
        args.preferred_sampling_params = Some(PreferredSamplingParams(
            json!({"temperature": 0.25, "max_new_tokens": 32}),
        ));
    });
    for (max_new_tokens, expected) in [(None, 32), (Some(8), 8)] {
        let params = api::SamplingParams {
            temperature: Some(0.8),
            max_new_tokens,
            ..Default::default()
        };
        let sampling = submitted_sampling(&mut h, params);
        assert_eq!(sampling[4], 0.8, "the request's own temperature");
        assert_eq!(sampling[0], expected);
    }
}

/// Protobuf has no null: a sampling field the client left unset takes the
/// schema default, as an absent JSON key does. `max_new_tokens` is the one
/// field whose absence would otherwise mean "unbounded".
#[test]
fn unset_protobuf_sampling_fields_take_schema_defaults() {
    let mut h = Harness::start(|_| {});
    let params = api::SamplingParams {
        temperature: Some(0.8),
        ..Default::default()
    };
    let sampling = submitted_sampling(&mut h, params);
    assert_eq!(sampling[4], 0.8);
    assert_eq!(sampling[0], 128);
    assert_eq!(sampling[5], 1.0);
}

/// A runtime failure after admission is this request's in-stream error item
/// (the SSE error frame's twin, `type` and `retryable` included), not an
/// RPC status.
#[test]
fn runtime_failure_rides_the_stream_as_an_error_item() {
    let mut h = Harness::start(|_| {});
    let mut stream = h.generate(ids_request(&[1], None));
    let rid = next_scheduler_request(&h.rt).rid().expect("rid");
    h.rt.push_error(&rid, "bad state").unwrap();

    let error = stream_error(h.next(&mut stream).unwrap().expect("an error item"));
    assert_eq!(error.index, None);
    let body = error.error.unwrap();
    assert_eq!(body.code, 500);
    assert!(body.message.contains("bad state"), "{}", body.message);
    assert_eq!(body.r#type.as_deref(), Some("internal_error"));
    assert_eq!(body.retryable, Some(true));
    assert!(h.next(&mut stream).unwrap().is_none());
}

/// Normalization failures end the RPC before anything is submitted, with the
/// status that mirrors the HTTP 400.
#[test]
fn malformed_request_is_rejected_before_submission() {
    let mut h = Harness::start(|_| {});
    let mut both = text_request("prompt", "both");
    both.input_ids = ids_request(&[1], None).input_ids;
    let error = h
        .try_generate(both)
        .expect_err("text and input_ids together must be rejected");
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("text"));

    let error = h
        .try_generate(ids_request(&[], None))
        .expect_err("empty input_ids must be rejected");
    assert_eq!(error.code(), Code::InvalidArgument);
    h.nothing_submitted("a malformed request");
}

/// Refuses requests asking for more than one top logprob.
#[derive(Debug)]
struct TopLogprobsCap;

impl sglang_server::GenerateValidator for TopLogprobsCap {
    fn validate(&self, request: &sglang_server::GenerateRequest) -> Result<(), String> {
        if request.top_logprobs_num > 1 {
            return Err(format!("top_logprobs_num {}", request.top_logprobs_num));
        }
        Ok(())
    }
}

/// The launch's generate validator refuses a batch whole, as on HTTP: no
/// item reaches the scheduler, the valid one included.
#[test]
fn generate_validator_refuses_the_whole_batch_before_submission() {
    let [http_addr, grpc_addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(http_addr)
        .grpc_addr(grpc_addr)
        .server_args(|args| {
            args.skip_tokenizer_init = false;
            args.tokenizer_path = tokenizer_dir().into();
        })
        .generate_validator(std::sync::Arc::new(TopLogprobsCap))
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let mut h = Harness::connect(rt, grpc_addr);
    let request = api::GenerateRequest {
        text: Some(many_strings(&["a", "b"])),
        top_logprobs_num: Some(api::Int64OrList {
            value: Some(api::int64_or_list::Value::Many(api::Int64List {
                items: vec![1, 2],
            })),
        }),
        ..Default::default()
    };
    let error = h
        .try_generate(request)
        .expect_err("the validator must refuse the batch");
    assert_eq!(error.code(), Code::InvalidArgument);
    assert_eq!(error.message(), "top_logprobs_num 2");
    h.nothing_submitted("a refused batch");
}

/// `OllamaChat` renders the messages with the served template and streams
/// the adapter's frames, counts on the done frame; `OllamaShow` needs no
/// scheduler.
#[test]
fn ollama_chat_streams_frames_and_show_is_static() {
    let mut h = Harness::start(|args| {
        args.tokenizer_path = fused_dir().into();
        args.chat_template = Some("chatml".into());
        args.incremental_streaming_output = true;
    });
    let request = api::OllamaChatRequest {
        model: "m".into(),
        messages: vec![api::OllamaMessage {
            role: "user".into(),
            content: "hi".into(),
            images: None,
        }],
        ..Default::default()
    };
    let call = h.client.ollama_chat(request);
    let mut stream = h.client_rt.block_on(call).unwrap().into_inner();
    let submitted = next_scheduler_request(&h.rt);
    let header = header_json(&submitted);
    // The rendered chat prompt reaches the scheduler as token ids.
    assert!(
        submitted
            .buffer_shape("input_ids")
            .is_some_and(|shape| shape[0] > 0),
        "{header}"
    );
    assert_eq!(header[8][0], 2048, "the default token limit");
    let rid = submitted.rid().expect("rid");
    let next = |h: &Harness, stream: &mut tonic::Streaming<api::OllamaChatResponse>| {
        h.client_rt
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(5), stream.message()).await
            })
            .expect("timed out waiting for the Ollama stream")
            .unwrap()
    };
    h.rt.push_output_with(&rid, &[piece_id("o")], chunk(false))
        .unwrap();
    let first = next(&h, &mut stream).unwrap();
    assert_eq!(first.message.as_ref().unwrap().content, "o");
    assert!(!first.done && first.eval_count.is_none());
    h.rt.push_output_with(&rid, &[piece_id("k")], chunk(true))
        .unwrap();
    let second = next(&h, &mut stream).unwrap();
    assert_eq!(second.message.as_ref().unwrap().content, "k");
    let done = next(&h, &mut stream).unwrap();
    assert!(done.done);
    assert_eq!(done.done_reason.as_deref(), Some("stop"));
    assert_eq!(done.message.as_ref().unwrap().content, "");
    assert_eq!(done.prompt_eval_count, Some(3));
    assert_eq!(done.eval_count, Some(2));
    assert!(next(&h, &mut stream).is_none());

    let card = h
        .client_rt
        .block_on(h.client.ollama_show(api::OllamaShowRequest {
            model: "org/Qwen-Chat".into(),
        }))
        .unwrap()
        .into_inner();
    assert_eq!(card.details.unwrap().family, "Qwen");
    assert_eq!(card.parameters, "num_ctx 4096");
}

/// `AnthropicMessages` renders through the chat core and shapes the
/// reply; `AnthropicCountTokens` counts the rendered prompt's ids.
#[test]
fn anthropic_messages_and_count_tokens_over_grpc() {
    let mut h = Harness::start(|args| {
        args.tokenizer_path = fused_dir().into();
        args.chat_template = Some("chatml".into());
    });
    let request: api::AnthropicMessagesRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let mut client = h.client.clone();
    let call = h
        .client_rt
        .spawn(async move { client.anthropic_messages(request).await });
    let submitted = next_scheduler_request(&h.rt);
    assert_eq!(header_json(&submitted)[8][0], 8);
    let prompt_ids = submitted.buffer_shape("input_ids").expect("prompt ids")[0];
    h.rt.push_output_with(
        &submitted.rid().expect("rid"),
        &[piece_id("ok")],
        chunk(true),
    )
    .unwrap();
    let message = h.client_rt.block_on(call).unwrap().unwrap().into_inner();
    assert_eq!(message.r#type, "message");
    assert_eq!(message.stop_reason.as_deref(), Some("end_turn"));
    let content = message.content.unwrap().items;
    assert_eq!(content.len(), 1);
    assert!(matches!(
        &content[0].block,
        Some(api::anthropic_content_block::Block::Text(text)) if text.text == "ok"
    ));
    assert_eq!(message.usage.unwrap().input_tokens, Some(3));

    // The same prompt, counted: as many ids as the generation was sent with.
    let request: api::AnthropicCountTokensRequest = serde_json::from_value(json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let counted = h
        .client_rt
        .block_on(h.client.anthropic_count_tokens(request))
        .unwrap()
        .into_inner();
    assert_eq!(counted.input_tokens as usize, prompt_ids);
}

#[test]
fn detokenize_returns_text_and_maps_invalid_ids() {
    let mut h = Harness::start(|_| {});
    let tokens = vec![word_id("Hel"), word_id("lo")];
    let decoded = h
        .client_rt
        .block_on(h.client.detokenize(api::DetokenizeRequest { tokens }))
        .unwrap()
        .into_inner();
    assert_eq!(decoded.text, "Hel lo");

    let error = h
        .client_rt
        .block_on(
            h.client
                .detokenize(api::DetokenizeRequest { tokens: vec![-1] }),
        )
        .unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
    h.nothing_submitted("a detokenization");
}

#[test]
fn detokenize_rejects_missing_tokenizer_without_submission() {
    let mut h = Harness::start(|args| args.skip_tokenizer_init = true);
    let error = h
        .client_rt
        .block_on(
            h.client
                .detokenize(api::DetokenizeRequest { tokens: vec![1] }),
        )
        .unwrap_err();
    assert_eq!(error.code(), Code::Unimplemented);
    h.nothing_submitted("a detokenization without a tokenizer");
}
