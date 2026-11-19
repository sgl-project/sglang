//! Crash-dump recording: the finished requests of the last five minutes
//! and every unfinished one.

use std::collections::{HashMap, VecDeque};
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use serde_json::{Map, Value};

use crate::api_server::core::CoreOutput;

use super::py_object::omit_custom_params;
use super::{RawBody, lock};

/// Python's `record_request_for_crash_dump` keeps a finished request this long.
const CRASH_DUMP_WINDOW_SECS: f64 = 300.0;

/// The finished requests of the last five minutes and every unfinished one,
/// as Python's `crash_dump_request_list` and `rid_to_state` hold them. One
/// lock covers both, so a request moves from one to the other atomically and
/// a snapshot sees it exactly once; JSON is written outside the lock.
#[derive(Default)]
pub(super) struct CrashRecorder {
    state: Mutex<RecorderState>,
    next_id: AtomicU64,
    /// Server-added `custom_params` keys, left out of the bodies it writes.
    omit_custom_params: Vec<String>,
}

#[derive(Default)]
struct RecorderState {
    finished: VecDeque<FinishedRequest>,
    in_flight: HashMap<u64, InFlightRequest>,
}

/// The JSON object Python rebuilds requests from: a native body, shared by
/// the items of a batch, or the fields the core knows of a request it built.
pub(crate) struct DumpBody {
    id: u64,
    json: RawBody,
}

impl DumpBody {
    /// The body's JSON. With header overrides it is the parsed body with the
    /// overrides written in, serialized: what the request decoded. The
    /// server-added `omit` keys are left out of its `custom_params`.
    fn write(&self, out: &mut Vec<u8>, omit: &[String]) {
        let RawBody { bytes, overrides } = &self.json;
        // A body none of the keys can appear in is written as it came.
        let mentions =
            |key: &String| !key.is_empty() && bytes.windows(key.len()).any(|w| w == key.as_bytes());
        if overrides.is_none() && !omit.iter().any(mentions) {
            out.extend_from_slice(bytes);
            return;
        }
        let mut body: Value = serde_json::from_slice(bytes).expect("parsed when received");
        if let Some(fields) = body.as_object_mut() {
            if let Some(overrides) = overrides {
                fields.extend(overrides.clone());
            }
            if let Some(sampling_params) = fields.get_mut("sampling_params") {
                omit_custom_params(sampling_params, omit);
            }
        }
        serde_json::to_writer(out, &body).expect("a JSON value serializes");
    }
}

/// One request of a [`DumpBody`]: its batch item and the rid the client sees.
#[derive(Clone)]
pub(super) struct DumpRequest {
    pub(super) body: Arc<DumpBody>,
    pub(super) batch_index: Option<usize>,
    pub(super) rid: String,
}

struct FinishedRequest {
    request: DumpRequest,
    out: Bytes,
    created_time: f64,
    finished_time: f64,
}

struct InFlightRequest {
    request: DumpRequest,
    created_time: f64,
    output: Arc<Mutex<CoreOutput>>,
}

impl CrashRecorder {
    pub(super) fn omitting(omit_custom_params: Vec<String>) -> Self {
        Self {
            omit_custom_params,
            ..Self::default()
        }
    }

    pub(super) fn body(&self, json: RawBody) -> Arc<DumpBody> {
        Arc::new(DumpBody {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            json,
        })
    }

    pub(super) fn register(
        &self,
        request: DumpRequest,
        created_time: f64,
        output: Arc<Mutex<CoreOutput>>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        lock(&self.state).in_flight.insert(
            id,
            InFlightRequest {
                request,
                created_time,
                output,
            },
        );
        id
    }

    pub(super) fn unregister(&self, id: u64) {
        lock(&self.state).in_flight.remove(&id);
    }

    /// Move a finished request to the window, evicting what finished five
    /// minutes before it (Python `record_request_for_crash_dump`).
    pub(super) fn finish(&self, id: u64, log_metrics: bool, out: &Map<String, Value>, now: f64) {
        let out = log_metrics
            .then(|| Bytes::from(serde_json::to_vec(out).expect("a JSON map serializes")));
        let mut state = lock(&self.state);
        let Some(entry) = state.in_flight.remove(&id) else {
            return;
        };
        let Some(out) = out else {
            return;
        };
        state.finished.push_back(FinishedRequest {
            request: entry.request,
            out,
            created_time: entry.created_time,
            finished_time: now,
        });
        while state
            .finished
            .front()
            .is_some_and(|oldest| now - oldest.finished_time >= CRASH_DUMP_WINDOW_SECS)
        {
            state.finished.pop_front();
        }
    }

    /// `{"bodies": {id: body}, "finished": [record], "unfinished": [record]}`,
    /// each record `{"body", "batch_index", "rid", "out", "created_time",
    /// "finished_time"}` and each body written once, however many items of a
    /// batch name it.
    pub(super) fn snapshot(&self, now: f64) -> Vec<u8> {
        // Capture under the lock (reference counts and the partial outputs),
        // write outside it.
        let (finished, unfinished) = {
            let state = lock(&self.state);
            let finished: Vec<(DumpRequest, Bytes, f64, f64)> = state
                .finished
                .iter()
                .map(|r| {
                    (
                        r.request.clone(),
                        r.out.clone(),
                        r.created_time,
                        r.finished_time,
                    )
                })
                .collect();
            let unfinished: Vec<(DumpRequest, Map<String, Value>, f64)> = state
                .in_flight
                .values()
                .map(|e| {
                    (
                        e.request.clone(),
                        partial_out(&lock(&e.output)),
                        e.created_time,
                    )
                })
                .collect();
            (finished, unfinished)
        };
        let mut bodies = std::collections::BTreeMap::new();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"\"finished\":[");
        for (i, (request, out, created, finished)) in finished.iter().enumerate() {
            if i > 0 {
                buf.push(b',');
            }
            bodies.insert(request.body.id, request.body.clone());
            write_record(&mut buf, request, out, *created, *finished);
        }
        buf.extend_from_slice(b"],\"unfinished\":[");
        for (i, (request, out, created)) in unfinished.iter().enumerate() {
            if i > 0 {
                buf.push(b',');
            }
            bodies.insert(request.body.id, request.body.clone());
            let out = serde_json::to_vec(out).expect("a JSON map serializes");
            write_record(&mut buf, request, &out, *created, now);
        }
        buf.extend_from_slice(b"]}");

        let mut snapshot = b"{\"bodies\":{".to_vec();
        for (i, (id, body)) in bodies.iter().enumerate() {
            if i > 0 {
                snapshot.push(b',');
            }
            write!(snapshot, "\"{id}\":").expect("writes to a Vec");
            body.write(&mut snapshot, &self.omit_custom_params);
        }
        snapshot.extend_from_slice(b"},");
        snapshot.extend_from_slice(&buf);
        snapshot
    }
}

/// Python's `ReqState.get_crash_dump_output`: what was generated so far.
fn partial_out(output: &CoreOutput) -> Map<String, Value> {
    let mut out = Map::new();
    if !output.text.is_empty() {
        out.insert("text".into(), output.text.clone().into());
    }
    if !output.token_ids.is_empty() {
        out.insert("output_ids".into(), output.token_ids.clone().into());
    }
    out
}

fn write_record(buf: &mut Vec<u8>, request: &DumpRequest, out: &[u8], created: f64, finished: f64) {
    write!(buf, "{{\"body\":{},\"batch_index\":", request.body.id).expect("writes to a Vec");
    serde_json::to_writer(&mut *buf, &request.batch_index).expect("serializes");
    buf.extend_from_slice(b",\"rid\":");
    serde_json::to_writer(&mut *buf, &request.rid).expect("serializes");
    buf.extend_from_slice(b",\"out\":");
    buf.extend_from_slice(out);
    buf.extend_from_slice(b",\"created_time\":");
    serde_json::to_writer(&mut *buf, &created).expect("serializes");
    buf.extend_from_slice(b",\"finished_time\":");
    serde_json::to_writer(&mut *buf, &finished).expect("serializes");
    buf.push(b'}');
}

#[cfg(test)]
mod tests {
    use super::super::tests::{chunk, receive_body};
    use super::super::{ReceivedAs, RequestLog};
    use super::*;
    use crate::message::ids::Rid;
    use crate::message::multimodal::MmItem;
    use crate::message::request::GenerateRequest;
    use crate::message::response::ResponseItem;

    /// A crash dump leaves the server-added `custom_params` keys out of a body
    /// that carries them, whole or per batch item, and keeps the client's own;
    /// a body without them is still written as it came.
    #[test]
    fn server_added_custom_params_are_not_dumped() {
        let log = RequestLog::for_test_omitting(None, true, &["server_only"]);
        let recorder = log.recorder.as_ref().unwrap();
        let dump = |raw: &'static [u8], rid: &str| {
            let body: Value = serde_json::from_slice(raw).unwrap();
            let raw = RawBody {
                bytes: Bytes::from_static(raw),
                overrides: None,
            };
            let snapshot = log.snapshot_body(&body, Some(raw), &http::HeaderMap::new());
            let request = DumpRequest {
                body: snapshot.unwrap().dump.unwrap(),
                batch_index: None,
                rid: rid.into(),
            };
            recorder.register(request, 1.0, Arc::default());
        };
        dump(
            br#"{"rid": ["a", "b"], "text": ["x", "y"], "sampling_params": [
                {"custom_params": {"server_only": "srv-added-1", "client": 1}},
                {"custom_params": {"server_only": "srv-added-2"}}]}"#,
            "a",
        );
        let untouched: &'static [u8] =
            br#"{ "rid" : "c", "sampling_params": {"custom_params": {"client": 2}} }"#;
        dump(untouched, "c");
        let snapshot = recorder.snapshot(2.0);
        let text = String::from_utf8(snapshot.clone()).unwrap();
        assert!(!text.contains("server_only"), "{text}");
        assert!(!text.contains("srv-added"), "{text}");
        let parsed: Value = serde_json::from_slice(&snapshot).unwrap();
        assert_eq!(
            parsed["bodies"]["0"]["sampling_params"][0]["custom_params"],
            serde_json::json!({"client": 1})
        );
        assert!(
            text.contains(std::str::from_utf8(untouched).unwrap()),
            "{text}"
        );
    }

    fn recorded(recorder: &CrashRecorder) -> (Vec<String>, Vec<Value>) {
        let snapshot: Value = serde_json::from_slice(&recorder.snapshot(0.0)).unwrap();
        let rids = |key: &str| {
            snapshot[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|record| record["rid"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        (
            rids("finished"),
            snapshot["unfinished"].as_array().unwrap().clone(),
        )
    }

    fn dump_request(recorder: &CrashRecorder, rid: &str) -> DumpRequest {
        DumpRequest {
            body: recorder.body(RawBody {
                bytes: Bytes::from_static(b"{}"),
                overrides: None,
            }),
            batch_index: None,
            rid: rid.into(),
        }
    }

    /// Python's window: a finished request stays five minutes after it finished;
    /// the health probe (`log_metrics=False`) is never kept; an unfinished one
    /// carries what it generated so far, and leaves when its call is dropped.
    #[test]
    fn recorder_keeps_five_minutes_and_the_unfinished() {
        let recorder = CrashRecorder::default();
        let output = || Arc::new(Mutex::new(CoreOutput::default()));
        let first = recorder.register(dump_request(&recorder, "first"), 1.0, output());
        let probe = recorder.register(dump_request(&recorder, "probe"), 1.0, output());
        let partial = output();
        lock(&partial).text = "so far".into();
        lock(&partial).token_ids = vec![4, 5];
        let _running = recorder.register(dump_request(&recorder, "running"), 1.0, partial);
        let dropped = recorder.register(dump_request(&recorder, "dropped"), 1.0, output());

        recorder.finish(first, true, &Map::new(), 10.0);
        recorder.finish(probe, false, &Map::new(), 10.0);
        recorder.unregister(dropped);
        let (finished, unfinished) = recorded(&recorder);
        assert_eq!(finished, ["first"]);
        assert_eq!(unfinished.len(), 1);
        assert_eq!(unfinished[0]["rid"], "running");
        assert_eq!(
            unfinished[0]["out"],
            serde_json::json!({"text": "so far", "output_ids": [4, 5]})
        );

        // Just inside the window, then out of it once a later request finishes.
        let second = recorder.register(dump_request(&recorder, "second"), 2.0, output());
        recorder.finish(second, true, &Map::new(), 309.9);
        assert_eq!(recorded(&recorder).0, ["first", "second"]);
        let third = recorder.register(dump_request(&recorder, "third"), 3.0, output());
        recorder.finish(third, true, &Map::new(), 310.0);
        assert_eq!(recorded(&recorder).0, ["second", "third"]);
    }

    /// A call dropped unfinished after admission (the client went away) is kept
    /// with what it generated, as Python records an aborted request; one dropped
    /// before the scheduler saw it (preprocessing failed) is not.
    #[test]
    fn dropped_records_leave_the_unfinished_set() {
        let log = RequestLog::for_test(None, true);
        let snapshot = |log: &RequestLog| -> Value {
            serde_json::from_slice(&log.crash_dump_snapshot().unwrap()).unwrap()
        };
        let body = serde_json::json!({"rid": "gone", "text": "hi"});
        let mut admitted = receive_body(&log, &body, &http::HeaderMap::new()).remove(0);
        admitted.admit();
        admitted.observe(&ResponseItem::Frame(chunk(
            &Rid::from_client("gone"),
            "so",
            &[4],
            None,
        )));
        let rejected = receive_body(
            &log,
            &serde_json::json!({"rid": "rejected", "text": "hi"}),
            &http::HeaderMap::new(),
        )
        .remove(0);
        assert_eq!(snapshot(&log)["unfinished"].as_array().unwrap().len(), 2);

        drop(admitted);
        drop(rejected);
        let snapshot = snapshot(&log);
        assert_eq!(snapshot["unfinished"], serde_json::json!([]));
        let finished = snapshot["finished"].as_array().unwrap();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0]["rid"], "gone");
        assert_eq!(finished[0]["out"]["text"], "so");
        assert_eq!(finished[0]["out"]["output_ids"], serde_json::json!([4]));
        assert_eq!(
            finished[0]["out"]["meta_info"]["finish_reason"],
            Value::Null
        );
    }

    /// A crash dump keeps the body's bytes as they came. With header overrides
    /// it applies them only when written, and the dump is byte for byte the one
    /// serializing the overridden body on every request gave.
    #[test]
    fn dump_keeps_the_raw_body() {
        // Spacing, escapes and number spellings that serializing changes.
        let raw = Bytes::from_static(
            br#"{ "rid" : "client", "text": "\u00e9", "input_ids": [1, 2,3],
                "sampling_params": {"temperature": 7E-1, "top_p": 1.0e0}, "stream": false }"#,
        );
        let dump = |body: &Value, raw: Option<RawBody>| -> Vec<u8> {
            let log = RequestLog::for_test(None, true);
            let snapshot = log.snapshot_body(body, raw, &http::HeaderMap::new());
            let recorder = log.recorder.as_ref().unwrap();
            let request = DumpRequest {
                body: snapshot.unwrap().dump.unwrap(),
                batch_index: None,
                rid: "r".into(),
            };
            recorder.register(request, 1.0, Arc::default());
            recorder.snapshot(2.0)
        };
        let mut body: Value = serde_json::from_slice(&raw).unwrap();
        let exact = RawBody {
            bytes: raw.clone(),
            overrides: None,
        };
        let expected = [br#"{"bodies":{"0":"#.as_slice(), &raw, b"},"].concat();
        assert!(dump(&body, Some(exact)).starts_with(&expected));

        // The native handler: one override replaces a field, one adds a field.
        let overrides = serde_json::json!({"rid": "router", "bootstrap_room": i64::MAX});
        let overrides = overrides.as_object().unwrap();
        for (field, value) in overrides {
            body.as_object_mut()
                .unwrap()
                .insert(field.clone(), value.clone());
        }
        let overridden = RawBody {
            bytes: raw,
            overrides: Some(overrides.clone()),
        };
        // `None` serializes `body`, as every overridden request used to.
        assert_eq!(
            String::from_utf8(dump(&body, Some(overridden))).unwrap(),
            String::from_utf8(dump(&body, None)).unwrap()
        );
    }

    /// A request an adapter built keeps its media in the dump, in the fields
    /// Python's request takes them in.
    #[test]
    fn built_request_keeps_its_media() {
        use crate::message::multimodal::MediaHints;
        use crate::message::request::MmData;

        let log = RequestLog::for_test(None, true);
        let request = GenerateRequest {
            rid: Rid::from_client("with-media"),
            text: Some("look".into()),
            mm: Some(Box::new(MmData {
                image_data: vec![
                    MmItem::Source("data:image/png;base64,AAAA".into()),
                    MmItem::Ref {
                        url: "file:///i.png".into(),
                        hints: MediaHints {
                            detail: Some("low".into()),
                            ..Default::default()
                        },
                    },
                ],
                video_data: vec![MmItem::Ref {
                    url: "file:///v.mp4".into(),
                    hints: MediaHints {
                        fps: Some(2.0),
                        max_frames: Some(16),
                        max_tokens_per_frame: Some(64),
                        ..Default::default()
                    },
                }],
                ..Default::default()
            })),
            ..Default::default()
        };
        let _record = log
            .receive(&[request], ReceivedAs::without_headers(), false, true)
            .remove(0);
        let snapshot: Value = serde_json::from_slice(&log.crash_dump_snapshot().unwrap()).unwrap();
        let body = &snapshot["bodies"][snapshot["unfinished"][0]["body"].to_string()];
        assert_eq!(
            body["image_data"],
            serde_json::json!([
                "data:image/png;base64,AAAA",
                {"url": "file:///i.png", "detail": "low"}
            ])
        );
        assert_eq!(
            body["video_data"],
            serde_json::json!([{
                "url": "file:///v.mp4", "fps": 2.0, "max_frames": 16, "max_tokens_per_frame": 64,
            }])
        );
        assert_eq!(body["audio_data"], Value::Null);
    }
}
