//! Control operations across the listeners of a DP deployment: each rank's
//! scheduler is asked, and the outcomes are merged into one answer.

use std::time::Duration;

use serde_json::json;
use sglang_server_test::{ApiKeys, OutputDetails};

use crate::common::{Sse, dp_deployment, http_bytes, next_scheduler_request, next_scheduler_work};

/// `/flush_cache` on one listener asks its own scheduler and every other
/// rank's, and one rank's failure is the answer.
#[test]
fn flush_cache_merges_the_local_rank_with_every_target() {
    let [(rt0, addr0), (rt1, _)] = dp_deployment(|_, _| {});
    let client = std::thread::spawn(move || http_bytes(addr0, "POST", "/flush_cache", &[], b""));
    let local = next_scheduler_work(&rt0);
    assert_eq!(local.kind().as_deref(), Some("FlushCacheReqInput"));
    rt0.push_control_result(
        &local.rid().expect("rid"),
        &json!({"success": true, "message": ""}),
    )
    .unwrap();
    let relayed = next_scheduler_work(&rt1);
    assert_eq!(relayed.kind().as_deref(), Some("FlushCacheReqInput"));
    let busy = json!({"success": false, "message": "peer busy"});
    rt1.push_control_result(&relayed.rid().expect("rid"), &busy)
        .unwrap();

    let response = client.join().unwrap();
    assert_eq!(response.status, 400);
    assert_eq!(response.body, "peer busy");
    assert!(
        rt0.next_request(Duration::from_millis(200)).is_none(),
        "one fan-out submits one local request"
    );
}

/// The listener an abort reaches cancels the named generations locally,
/// answers at once, and relays the abort to every other rank's listener,
/// which cancels its own. A relayed abort is served on the receiving
/// listener alone, and a no-op abort is neither served nor relayed.
#[test]
fn abort_request_reaches_every_dp_listener() {
    let [(rt0, addr0), (rt1, addr1)] = dp_deployment(|_, _| {});
    let generate = json!({"rid": "client-rid", "input_ids": [1], "stream": true});
    let relayed_abort = rmp_serde::to_vec(&("AbortReq", "client-rid", (), false, (), ())).unwrap();
    for (path, body, local, forwarded) in [
        (
            "/abort_request",
            br#"{"rid": "client-rid"}"#.to_vec(),
            true,
            true,
        ),
        (
            "/abort_request",
            br#"{"abort_all": true}"#.to_vec(),
            true,
            true,
        ),
        ("/internal/peer", relayed_abort, true, false),
        ("/abort_request", br#"{"rid": ""}"#.to_vec(), false, false),
    ] {
        let case = format!("{path} {}", String::from_utf8_lossy(&body));
        // A live generation on each rank, under the rid the aborts name.
        let live0 = Sse::post(addr0, "/generate", &generate);
        let rid0 = next_scheduler_request(&rt0).rid().expect("rid");
        let live1 = Sse::post(addr1, "/generate", &generate);
        let rid1 = next_scheduler_request(&rt1).rid().expect("rid");

        let headers: &[&str] = if path == "/abort_request" {
            &["Content-Type: application/json"]
        } else {
            &[]
        };
        let response = http_bytes(addr0, "POST", path, headers, &body);
        assert_eq!(response.status, 200, "{case}");
        for (rt, rid, aborted, rank) in [(&rt0, &rid0, local, 0), (&rt1, &rid1, forwarded, 1)] {
            if aborted {
                let abort = next_scheduler_request(rt);
                assert_eq!(
                    abort.kind().as_deref(),
                    Some("AbortReq"),
                    "{case}: rank {rank}"
                );
                assert_eq!(abort.rid().as_ref(), Some(rid), "{case}: rank {rank}");
            } else {
                assert!(
                    rt.next_request(Duration::from_millis(300)).is_none(),
                    "{case}: rank {rank} was told to abort"
                );
            }
            // End the generation, so the next case starts clean.
            let details = OutputDetails {
                abort: aborted,
                finish_reason: Some(json!({"type": "abort", "message": ""})),
                ..Default::default()
            };
            rt.push_output_with(rid, &[], details).unwrap();
        }
        for mut events in [live0, live1] {
            while events.next().is_some() {}
        }
    }
}

/// A listener presents its admin key to the ranks it relays to, whose peer
/// route takes nothing less.
#[test]
fn the_control_relay_presents_the_admin_key_to_its_peers() {
    for (case, relay_keys, key, accepted) in [
        (
            "admin key",
            ApiKeys::new(Some("sk".into()), Some("sk-admin".into())),
            "Authorization: Bearer sk-admin",
            true,
        ),
        (
            "API key only",
            ApiKeys::new(Some("sk".into()), None),
            "Authorization: Bearer sk",
            false,
        ),
    ] {
        // Rank 1 always asks for both keys; rank 0 holds `relay_keys`.
        let [(rt0, addr0), (rt1, _)] = dp_deployment(|rank, args| {
            args.api_keys = if rank == 0 {
                relay_keys.clone()
            } else {
                ApiKeys::new(Some("sk".into()), Some("sk-admin".into()))
            };
        });
        let client =
            std::thread::spawn(move || http_bytes(addr0, "POST", "/flush_cache", &[key], b""));
        let ok = json!({"success": true, "message": ""});
        let local = next_scheduler_work(&rt0);
        rt0.push_control_result(&local.rid().expect("rid"), &ok)
            .unwrap();
        if accepted {
            let relayed = next_scheduler_work(&rt1);
            rt1.push_control_result(&relayed.rid().expect("rid"), &ok)
                .unwrap();
        }
        let response = client.join().unwrap();
        assert_eq!(
            response.status == 200,
            accepted,
            "{case}: {}",
            response.body
        );
        if !accepted {
            assert!(
                rt1.next_request(Duration::from_millis(200)).is_none(),
                "{case}: the peer's scheduler was asked"
            );
        }
    }
}
