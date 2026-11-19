//! HTTP/2 over cleartext on the HTTP listener, behind `--enable-http2`.

use sglang_server::Http2Config;
use sglang_server_test::{SchedulerlessServer, free_loopback_addrs};

use crate::common::{
    next_scheduler_rid, push_terminal, push_tokens, read_response, send_http,
    take_startup_freeze_gc,
};

/// The connection window `h2c_server` sets: not hyper's default (1 MiB),
/// and under the server's 1 MiB stream window, so a client sees it.
const H2C_CONNECTION_WINDOW: usize = 512 * 1024;

/// A server with `--enable-http2`, a stream limit of 7 and
/// a connection window of `H2C_CONNECTION_WINDOW`, and with
/// `--incremental-streaming-output`, so a stream sends every scheduler
/// output as its own event rather than merging a backlog into one.
fn h2c_server(http_addr: std::net::SocketAddr) -> SchedulerlessServer {
    SchedulerlessServer::builder(http_addr)
        .server_args(|args| {
            args.http2 = Some(Http2Config {
                max_concurrent_streams: 7,
                initial_connection_window_size: H2C_CONNECTION_WINDOW as u32,
            });
            args.incremental_streaming_output = true;
        })
        .start()
        .expect("start runtime")
}

/// An h2c (prior knowledge) client on a fresh connection, driven by
/// `client_rt`. Its send buffer is past the server's windows, so the
/// capacity it sees is what the server granted.
fn h2c_client(
    client_rt: &tokio::runtime::Runtime,
    addr: std::net::SocketAddr,
) -> h2::client::SendRequest<bytes::Bytes> {
    client_rt.block_on(async move {
        let tcp = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let (client, connection) = h2::client::Builder::new()
            .max_send_buffer_size(4 * 1024 * 1024)
            .handshake(tcp)
            .await
            .expect("h2 handshake");
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client.ready().await.expect("h2 client ready")
    })
}

/// POST `body` to `/generate` on a new stream: the response and the
/// request's send half, whose reset cancels the stream.
fn h2c_generate(
    client: &mut h2::client::SendRequest<bytes::Bytes>,
    body: &str,
) -> (h2::client::ResponseFuture, h2::SendStream<bytes::Bytes>) {
    let request = http::Request::post("http://t/generate")
        .header("content-type", "application/json")
        .body(())
        .unwrap();
    let (response, mut stream) = client.send_request(request, false).expect("send request");
    stream
        .send_data(bytes::Bytes::copy_from_slice(body.as_bytes()), true)
        .expect("send body");
    (response, stream)
}

/// A response's status and whole body.
async fn h2c_read(response: h2::client::ResponseFuture) -> (u16, String) {
    let response = response.await.expect("h2 response");
    let status = response.status().as_u16();
    let mut body = response.into_body();
    let mut text = Vec::new();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("h2 body");
        let _ = body.flow_control().release_capacity(chunk.len());
        text.extend_from_slice(&chunk);
    }
    (status, String::from_utf8(text).unwrap())
}

/// With `--enable-http2` the listener serves h2c beside HTTP/1.1 on the
/// same port, as Python's Granian server does: it advertises the
/// configured stream limit and connection window, and a unary generation
/// and a stream of many small events (one DATA frame each) complete over
/// HTTP/2.
#[test]
fn h2c_serves_generation_beside_http1() {
    let [addr] = free_loopback_addrs();
    let rt = h2c_server(addr);
    take_startup_freeze_gc(&rt);
    let client_rt = tokio::runtime::Runtime::new().unwrap();
    let mut client = h2c_client(&client_rt, addr);

    let (response, _stream) = h2c_generate(
        &mut client,
        r#"{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":8}}"#,
    );
    let response = client_rt.spawn(h2c_read(response));
    let rid = next_scheduler_rid(&rt);
    assert!(push_terminal(&rt, &rid, &[9]));
    let (status, body) = client_rt.block_on(response).unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("[9]"), "{body}");
    assert_eq!(client.current_max_send_streams(), 7);

    let (response, _stream) = h2c_generate(
        &mut client,
        r#"{"input_ids":[1,2,3],"stream":true,"sampling_params":{"max_new_tokens":512}}"#,
    );
    let response = client_rt.spawn(h2c_read(response));
    let rid = next_scheduler_rid(&rt);
    for token in 0..300 {
        assert!(push_tokens(&rt, &rid, &[token]));
    }
    assert!(push_terminal(&rt, &rid, &[300]));
    let (status, body) = client_rt
        .block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(10), response).await
        })
        .expect("the stream never ended")
        .unwrap();
    assert_eq!(status, 200, "{body}");
    let events = body
        .strip_suffix("data: [DONE]\n\n")
        .unwrap_or_else(|| panic!("stream cut: {body}"))
        .split_terminator("\n\n")
        .map(|event| {
            let data = event.strip_prefix("data: ").expect("a data event");
            serde_json::from_str::<serde_json::Value>(data).unwrap()["output_ids"].clone()
        })
        .collect::<Vec<_>>();
    // One event per output, each with its one token.
    let expected = (0..=300).map(|token| serde_json::json!([token]));
    assert!(events.into_iter().eq(expected), "{body}");

    // A request body may take the whole connection window, no more. A
    // fresh connection, so no earlier body holds part of it.
    let mut client = h2c_client(&client_rt, addr);
    let request = http::Request::post("http://t/generate")
        .header("content-type", "application/json")
        .body(())
        .unwrap();
    let (_response, mut stream) = client.send_request(request, false).unwrap();
    stream.reserve_capacity(2 * H2C_CONNECTION_WINDOW);
    client_rt.block_on(async {
        let granted = async {
            while stream.capacity() < H2C_CONNECTION_WINDOW {
                std::future::poll_fn(|cx| stream.poll_capacity(cx))
                    .await
                    .expect("capacity")
                    .expect("h2 capacity");
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(5), granted)
            .await
            .expect("the connection window was never granted");
        // Anything past the window would have come with it.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    });
    assert_eq!(stream.capacity(), H2C_CONNECTION_WINDOW);
    stream.send_reset(h2::Reason::CANCEL);

    let response = read_response(send_http(addr, "GET", "/model_info", ""));
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    rt.shutdown();
}

/// A client's RST_STREAM aborts its request, as a hang-up does over
/// HTTP/1.1: mid-stream, and before a unary request has its response.
#[test]
fn h2c_stream_reset_aborts_the_request() {
    let [addr] = free_loopback_addrs();
    let rt = h2c_server(addr);
    take_startup_freeze_gc(&rt);
    let client_rt = tokio::runtime::Runtime::new().unwrap();
    let mut client = h2c_client(&client_rt, addr);
    for streaming in [true, false] {
        let (response, mut stream) = h2c_generate(
            &mut client,
            &format!(
                r#"{{"input_ids":[1,2,3],"stream":{streaming},"sampling_params":{{"max_new_tokens":8}}}}"#
            ),
        );
        let rid = next_scheduler_rid(&rt);
        // Held until the reset, so the stream stays open.
        let _body = streaming.then(|| {
            assert!(push_tokens(&rt, &rid, &[7]));
            client_rt.block_on(async {
                let response = response.await.expect("h2 response");
                assert_eq!(response.status().as_u16(), 200);
                let mut body = response.into_body();
                body.data().await.expect("first event").expect("h2 body");
                body
            })
        });
        stream.send_reset(h2::Reason::CANCEL);
        let abort = rt
            .next_request(std::time::Duration::from_secs(5))
            .unwrap_or_else(|| panic!("streaming={streaming}: no abort after the stream reset"));
        assert_eq!(
            abort.kind().as_deref(),
            Some("AbortReq"),
            "streaming={streaming}"
        );
        assert_eq!(abort.rid(), Some(rid.clone()), "streaming={streaming}");
    }
    rt.shutdown();
}

/// Without `--enable-http2` the listener speaks HTTP/1.1 only, as
/// Python's uvicorn server does: an h2c client gets no response.
#[test]
fn http2_is_refused_without_the_flag() {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .start()
        .expect("start runtime");
    let client_rt = tokio::runtime::Runtime::new().unwrap();
    // The server may close the connection at any step past the preface.
    let response = client_rt.block_on(async move {
        let tcp = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let (client, connection) = h2::client::handshake(tcp).await?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let mut client = client.ready().await?;
        let request = http::Request::get("http://t/model_info").body(()).unwrap();
        let (response, _stream) = client.send_request(request, true)?;
        response.await
    });
    assert!(response.is_err(), "{response:?}");
    rt.shutdown();
}
