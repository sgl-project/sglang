//! `/server_info` across the listeners of a DP deployment.

use serde_json::{Value, json};
use sglang_server_test::SchedulerlessServer;

use crate::common::{HttpResponse, dp_deployment, http_bytes, next_scheduler_work};

fn answer_state(rt: &SchedulerlessServer, state: Value) {
    let request = next_scheduler_work(rt);
    assert_eq!(request.kind().as_deref(), Some("GetInternalStateReq"));
    rt.push_control_result(
        &request.rid().expect("rid"),
        &json!({"internal_state": state}),
    )
    .unwrap();
}

/// `/server_info` on the listener of DP rank 1, whose state reports a
/// throughput of 1.5, while its peer, DP rank 0, reports `peer_throughput`.
fn rank_one_server_info(peer_throughput: Value) -> HttpResponse {
    let [(rt0, _), (rt1, addr1)] = dp_deployment(|_, _| {});
    let client = std::thread::spawn(move || http_bytes(addr1, "GET", "/server_info", &[], b""));
    answer_state(&rt1, json!({"last_gen_throughput": 1.5}));
    answer_state(&rt0, json!({"last_gen_throughput": peer_throughput}));
    client.join().unwrap()
}

/// `/server_info` lists one internal state per DP rank, in rank order:
/// a rank lower than this listener's comes first.
#[test]
fn server_info_lists_internal_states_in_rank_order() {
    let response = rank_one_server_info(json!(2.5));
    assert_eq!(response.status, 200, "{}", response.body);
    let throughputs: Vec<Value> = response.json()["internal_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|state| state["last_gen_throughput"].clone())
        .collect();
    assert_eq!(throughputs, [json!(2.5), json!(1.5)]);

    // A rank whose state breaks the typed contract fails the read.
    let response = rank_one_server_info(json!("fast"));
    assert_eq!(response.status, 500, "{}", response.body);
}

/// The listener of DP rank 0 adds every rank's listener as `dp_listeners`,
/// by rank; the other listeners do not.
#[test]
fn server_info_lists_dp_listeners_on_rank_zero_only() {
    for asked in [0, 1] {
        let ranks = dp_deployment::<2>(|_, _| {});
        let addr = ranks[asked].1;
        let client = std::thread::spawn(move || http_bytes(addr, "GET", "/server_info", &[], b""));
        // The asked listener's own scheduler first, then its peer's.
        answer_state(&ranks[asked].0, json!({}));
        answer_state(&ranks[1 - asked].0, json!({}));
        let response = client.join().unwrap();
        assert_eq!(response.status, 200, "{}", response.body);
        let body = response.json();
        let expected = (asked == 0).then(|| {
            json!([
                {"dp_rank": 0, "url": format!("http://{}", ranks[0].1)},
                {"dp_rank": 1, "url": format!("http://{}", ranks[1].1)},
            ])
        });
        assert_eq!(
            body.get("dp_listeners"),
            expected.as_ref(),
            "asked rank {asked}"
        );
    }
}
