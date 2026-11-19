//! What the integration tests share: the scheduler's side of a request and
//! plain HTTP/1.1 and gRPC clients.

use std::net::SocketAddr;
use std::time::Duration;

use sglang_api_types::api::v1 as proto;
use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;
use sglang_server_test::{RingRequest, SchedulerlessServer};

/// Take the post-startup `FreezeGCReq`, the first message on the scheduler
/// ring when the startup warmup is skipped, so a test sees its own
/// requests next.
pub fn take_startup_freeze_gc(rt: &SchedulerlessServer) {
    let first = rt
        .next_request(Duration::from_secs(5))
        .expect("no FreezeGCReq after startup");
    assert_eq!(first.kind().as_deref(), Some("FreezeGCReq"));
}

/// Wait for the next request on the scheduler ring.
pub fn next_scheduler_request(rt: &SchedulerlessServer) -> RingRequest {
    rt.next_request(Duration::from_secs(5))
        .expect("no request reached the scheduler ring")
}

/// Wait for the next request on the scheduler ring and return its rid.
pub fn next_scheduler_rid(rt: &SchedulerlessServer) -> String {
    next_scheduler_request(rt).rid().expect("request rid")
}

/// Answer `rid` with `output_ids` and end it; whether the frontend took it.
pub fn push_terminal(rt: &SchedulerlessServer, rid: &str, output_ids: &[i64]) -> bool {
    let stop = serde_json::json!({"type": "stop"});
    rt.push_output(rid, output_ids, Some(stop)).is_ok()
}

/// Answer `rid` with `output_ids`, leaving it open; whether the frontend
/// took it.
pub fn push_tokens(rt: &SchedulerlessServer, rid: &str, output_ids: &[i64]) -> bool {
    rt.push_output(rid, output_ids, None).is_ok()
}

pub fn wait_drain_complete(rt: &SchedulerlessServer) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !rt.drain_complete() {
        assert!(
            std::time::Instant::now() < deadline,
            "drain did not complete"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Write an HTTP/1.1 request on a fresh connection (closed after the
/// response), for the caller to read the response from.
pub fn send_http(addr: SocketAddr, method: &str, path: &str, body: &str) -> std::net::TcpStream {
    use std::io::Write;
    let mut conn = std::net::TcpStream::connect(addr).expect("connect");
    conn.set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write!(
        conn,
        "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    conn
}

pub fn read_response(mut conn: std::net::TcpStream) -> String {
    use std::io::Read;
    let mut response = String::new();
    conn.read_to_string(&mut response).expect("read response");
    response
}

/// An HTTP response: status, raw head and body.
pub struct HttpResponse {
    pub status: u16,
    pub head: String,
    pub body: String,
}

impl HttpResponse {
    /// The value of the header `name` (lower case), if the response has it.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.head.lines().skip(1).find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim().eq_ignore_ascii_case(name)).then_some(value.trim())
        })
    }

    /// The body as JSON.
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).unwrap_or_else(|_| panic!("not JSON: {}", self.body))
    }
}

/// Write `request` in one piece and read the response to the end. A server
/// that answers before it has read the whole body resets the connection as
/// it closes; what it sent before that is still the response.
pub fn exchange(conn: &mut std::net::TcpStream, request: &[u8]) -> Vec<u8> {
    use std::io::{Read, Write};
    // The server may already have answered and closed.
    let _ = conn.write_all(request);
    let mut response = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        match conn.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => response.extend_from_slice(&buf[..n]),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) if !response.is_empty() => {
                assert_eq!(error.kind(), std::io::ErrorKind::ConnectionReset, "{error}");
                break;
            }
            Err(error) => panic!("read response: {error}"),
        }
    }
    response
}

/// One HTTP/1.1 exchange with a JSON content type on a fresh connection;
/// `headers` are whole lines without their line ending.
pub fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[&str],
    body: &str,
) -> HttpResponse {
    let mut all = vec!["Content-Type: application/json"];
    all.extend_from_slice(headers);
    http_bytes(addr, method, path, &all, body.as_bytes())
}

/// One HTTP/1.1 exchange on a fresh connection with exactly `headers` beside
/// the framing ones, and any bytes for a body.
pub fn http_bytes(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[&str],
    body: &[u8],
) -> HttpResponse {
    let mut conn = std::net::TcpStream::connect(addr).expect("connect");
    conn.set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let headers: String = headers.iter().map(|line| format!("{line}\r\n")).collect();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: t\r\n{headers}\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let response = exchange(&mut conn, &[head.as_bytes(), body].concat());
    let response = String::from_utf8_lossy(&response);
    let (head, body) = response.split_once("\r\n\r\n").expect("response head");
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .expect("status code");
    HttpResponse {
        status,
        head: head.to_owned(),
        body: body.to_owned(),
    }
}

/// The next message on the scheduler ring that is not the frontend's abort
/// of an earlier request.
pub fn next_scheduler_work(rt: &SchedulerlessServer) -> RingRequest {
    loop {
        let request = next_scheduler_request(rt);
        if request.kind().as_deref() != Some("AbortReq") {
            return request;
        }
    }
}

/// A ring message's header as JSON: `[type, rid, ...fields]`.
pub fn header_json(request: &RingRequest) -> serde_json::Value {
    rmp_serde::from_slice(&request.header).expect("msgpack header")
}

pub async fn connect_grpc(addr: SocketAddr) -> SglangServiceClient<tonic::transport::Channel> {
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_timeout(Duration::from_secs(2));
    SglangServiceClient::new(endpoint.connect().await.expect("connect gRPC client"))
}

pub fn grpc_ids_request(ids: &[i64], rid: &str) -> proto::GenerateRequest {
    proto::GenerateRequest {
        input_ids: Some(proto::TokenIdsOrList {
            value: Some(proto::token_ids_or_list::Value::One(proto::TokenIds {
                ids: ids.to_vec(),
            })),
        }),
        rid: Some(proto::StringOrList {
            value: Some(proto::string_or_list::Value::One(rid.into())),
        }),
        stream: Some(true),
        ..Default::default()
    }
}

/// The directory of a word-level tokenizer: ids 1.. are the words of
/// [`WORDS`] in order, anything else is id 0. Decoding joins words with a
/// space.
pub const WORDS: [&str; 9] = ["Hel", "lo", "prompt", "ok", "a", "b", "c", "partial", "hi"];

/// The id of `word` in [`tokenizer_dir`]'s tokenizer.
pub fn word_id(word: &str) -> i64 {
    let index = WORDS.iter().position(|known| *known == word);
    index.unwrap_or_else(|| panic!("{word} is not in the test vocabulary")) as i64 + 1
}

pub fn tokenizer_dir() -> &'static str {
    static DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    DIR.get_or_init(|| checkpoint_dir("tokenizer", None))
}

/// [`tokenizer_dir`]'s tokenizer in a checkpoint whose `config.json` names a
/// DeepSeek V4.1 model, which the server lowers chat for.
pub fn deepseek_dir() -> &'static str {
    static DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    DIR.get_or_init(|| checkpoint_dir("deepseek", Some("deepseek_v41")))
}

/// The pieces of [`fused_dir`]'s tokenizer, ids 1.. in order.
pub const PIECES: [&str; 9] = [
    "<think>be",
    "cause</think>Par",
    "is",
    "Par",
    "<think>because Paris is famous</think>Paris",
    "Paris",
    "o",
    "k",
    "ok",
];

/// The id of `piece` in [`fused_dir`]'s tokenizer.
pub fn piece_id(piece: &str) -> i64 {
    let index = PIECES.iter().position(|known| *known == piece);
    index.unwrap_or_else(|| panic!("{piece} is not a test piece")) as i64 + 1
}

/// The directory of a tokenizer whose ids 1.. are [`PIECES`], decoded by
/// joining them with nothing between, so an output's text is exactly its
/// pieces'.
pub fn fused_dir() -> &'static str {
    static DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("sglang-it-fused-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".into(), 0.into());
        for (id, piece) in PIECES.iter().enumerate() {
            vocab.insert((*piece).into(), (id + 1).into());
        }
        let model = serde_json::json!({"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"});
        let config = serde_json::json!({
            "model": model,
            "pre_tokenizer": {"type": "Whitespace"},
            "decoder": {"type": "Fuse"},
        });
        std::fs::write(dir.join("tokenizer.json"), config.to_string()).unwrap();
        dir.to_string_lossy().into_owned()
    })
}

/// A DeepSeek V4.1 checkpoint with a byte-level tokenizer: id 0 is the lone
/// byte `0xC3`, the first half of a two-byte character, id 1 is `!`, and
/// ids 2.. are the bytes of `hi`.
pub fn byte_level_deepseek_dir() -> &'static str {
    static DIR: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("sglang-it-bytes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Byte-level vocabularies spell a byte as the character of that code point.
        let vocab = serde_json::json!({"\u{c3}": 0, "!": 1, "h": 2, "i": 3});
        let config = serde_json::json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        });
        std::fs::write(dir.join("tokenizer.json"), config.to_string()).unwrap();
        let model = serde_json::json!({"model_type": "deepseek_v41"});
        std::fs::write(dir.join("config.json"), model.to_string()).unwrap();
        dir.to_string_lossy().into_owned()
    })
}

fn checkpoint_dir(name: &str, model_type: Option<&str>) -> String {
    {
        let dir = std::env::temp_dir().join(format!("sglang-it-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(model_type) = model_type {
            let config = serde_json::json!({"model_type": model_type});
            std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        }
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".into(), 0.into());
        for (id, word) in WORDS.iter().enumerate() {
            vocab.insert((*word).into(), (id + 1).into());
        }
        let model = serde_json::json!({"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"});
        let config = serde_json::json!({"model": model, "pre_tokenizer": {"type": "Whitespace"}});
        std::fs::write(dir.join("tokenizer.json"), config.to_string()).unwrap();
        dir.to_string_lossy().into_owned()
    }
}

/// A streaming `POST` whose server-sent events are read one at a time;
/// dropping it hangs up.
pub struct Sse {
    reader: std::io::BufReader<std::net::TcpStream>,
    pub status: u16,
    pub content_type: Option<String>,
}

impl Sse {
    /// Send `body` to `path` with a JSON content type and read the response
    /// head.
    pub fn post(addr: SocketAddr, path: &str, body: &serde_json::Value) -> Self {
        Self::post_with(addr, path, &[], body)
    }

    /// [`Self::post`] with `headers` too, whole lines without their ending.
    pub fn post_with(
        addr: SocketAddr,
        path: &str,
        headers: &[&str],
        body: &serde_json::Value,
    ) -> Self {
        let headers: String = headers.iter().map(|line| format!("{line}\r\n")).collect();
        use std::io::{BufRead, Write};
        let body = body.to_string();
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        conn.set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(
            conn,
            "POST {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n{headers}\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut reader = std::io::BufReader::new(conn);
        let mut status = 0;
        let mut content_type = None;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("response head");
            if line.starts_with("HTTP/1.1 ") {
                status = line.split_whitespace().nth(1).unwrap().parse().unwrap();
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-type")
            {
                content_type = Some(value.trim().to_owned());
            }
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        Self {
            reader,
            status,
            content_type,
        }
    }

    /// The next event's data, `None` once the stream has ended.
    pub fn next(&mut self) -> Option<String> {
        use std::io::BufRead;
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line).expect("event stream") == 0 {
                return None;
            }
            if let Some(data) = line.strip_prefix("data: ") {
                return Some(data.trim_end().to_owned());
            }
        }
    }

    /// The next event's data as JSON.
    pub fn next_json(&mut self) -> serde_json::Value {
        let data = self.next().expect("the stream ended early");
        serde_json::from_str(&data).unwrap_or_else(|_| panic!("not JSON: {data}"))
    }
}

/// The listeners of an attention-DP deployment of `N` ranks, each a real
/// server whose scheduler is the test, after `change` to every rank's launch
/// arguments.
pub fn dp_deployment<const N: usize>(
    change: impl Fn(usize, &mut sglang_server::ServerArgs),
) -> [(SchedulerlessServer, SocketAddr); N] {
    let listeners: [SocketAddr; N] = sglang_server_test::free_loopback_addrs();
    std::array::from_fn(|rank| {
        let rt = SchedulerlessServer::builder(listeners[rank])
            .dp_listeners(&listeners, rank)
            .server_args(|args| change(rank, args))
            .start()
            .expect("start runtime");
        take_startup_freeze_gc(&rt);
        (rt, listeners[rank])
    })
}
