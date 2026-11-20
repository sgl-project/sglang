//! Text completions: how choices fold into a unary response and how a
//! stream's deltas, usage and end are framed.

use std::net::SocketAddr;

use serde_json::{Value, json};
use sglang_server_test::{OutputDetails, SchedulerlessServer};

use super::server_with;
use crate::common::{Sse, fused_dir, http, next_scheduler_request, piece_id};

/// A server whose output text is exactly the pieces it is sent.
fn server(cache_report: bool) -> (SchedulerlessServer, SocketAddr) {
    server_with(|args| {
        args.tokenizer_path = fused_dir().into();
        args.enable_cache_report = cache_report;
    })
}

fn frame() -> OutputDetails {
    OutputDetails {
        prompt_tokens: 5,
        ..Default::default()
    }
}

fn stop(cached_tokens: u32) -> OutputDetails {
    OutputDetails {
        cached_tokens,
        finish_reason: Some(json!({"type": "stop", "matched": "</s>"})),
        ..frame()
    }
}

#[test]
fn unary_fold_orders_choices_and_counts_each_prompt_once() {
    for cache_report in [false, true] {
        let (rt, addr) = server(cache_report);
        let body = json!({"model": "model", "prompt": "hi", "n": 2});
        let client = std::thread::spawn(move || {
            http(addr, "POST", "/v1/completions", &[], &body.to_string())
        });
        // The scheduler sees a prompt's choices under rids in choice order.
        let mut rids = [next_scheduler_request(&rt), next_scheduler_request(&rt)]
            .map(|request| request.rid().expect("rid"));
        rids.sort();
        let [rid0, rid1] = rids;
        rt.push_output_with(&rid0, &[piece_id("Par")], frame())
            .unwrap();
        rt.push_output_with(&rid0, &[piece_id("is")], stop(3))
            .unwrap();
        rt.push_output_with(&rid1, &[piece_id("is")], frame())
            .unwrap();
        rt.push_output_with(&rid1, &[piece_id("Par")], stop(4))
            .unwrap();

        let response = client.join().unwrap();
        assert_eq!(response.status, 200, "{}", response.body);
        let value = response.json();
        assert_eq!(value["choices"][0]["text"], "Paris");
        assert_eq!(value["choices"][1]["text"], "isPar");
        assert_eq!(value["choices"][0]["matched_stop"], "</s>");
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 4);
        assert_eq!(
            value["usage"]["completion_tokens_details"]["reasoning_tokens"],
            0
        );
        // A prompt's cached tokens are its first choice's, as on Python.
        let details = &value["usage"]["prompt_tokens_details"];
        match cache_report {
            true => assert_eq!(details["cached_tokens"], 3, "{value}"),
            false => assert!(details.is_null(), "{value}"),
        }
    }
}

#[test]
fn stream_uses_deltas_then_usage_and_done() {
    for cache_report in [false, true] {
        let (rt, addr) = server(cache_report);
        let body = json!({
            "model": "model",
            "prompt": "hi",
            "stream": true,
            "stream_options": {"include_usage": true, "continuous_usage_stats": cache_report},
        });
        let mut events = Sse::post(addr, "/v1/completions", &body);
        let rid = next_scheduler_request(&rt).rid().expect("rid");
        rt.push_output_with(&rid, &[piece_id("Par")], frame())
            .unwrap();
        let first = events.next_json();
        rt.push_output_with(&rid, &[piece_id("is")], stop(0))
            .unwrap();
        let mut frames = Vec::new();
        while let Some(data) = events.next() {
            frames.push(data);
        }
        assert_eq!(frames.len(), 3, "{frames:?}");
        let terminal: Value = serde_json::from_str(&frames[0]).unwrap();
        let usage: Value = serde_json::from_str(&frames[1]).unwrap();
        assert_eq!(first["choices"][0]["text"], "Par");
        assert_eq!(terminal["choices"][0]["text"], "is");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert!(usage["choices"].as_array().unwrap().is_empty());
        assert_eq!(usage["usage"]["prompt_tokens"], 5);
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[2], "[DONE]");
        // Python reports a zero cache hit rather than omitting it, but not
        // in per-chunk usage.
        let details = &usage["usage"]["prompt_tokens_details"];
        match cache_report {
            true => {
                assert_eq!(details["cached_tokens"], 0, "{usage}");
                assert_eq!(first["usage"]["prompt_tokens"], 5);
                assert!(first["usage"]["prompt_tokens_details"].is_null());
            }
            false => assert!(details.is_null(), "{usage}"),
        }
    }
}
