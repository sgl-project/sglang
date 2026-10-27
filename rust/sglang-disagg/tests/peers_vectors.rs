//! Peer bookkeeping replay: every scenario in `wire.json` "peers" was
//! produced by driving the REAL Python `NixlKVManager._add_remote_peer` /
//! `_reload_invalidated_peers` against a recording stub agent. The Rust
//! `PeerTable` must produce the identical agent-call trace and final state.

use sglang_disagg::json::{JVal, loads};
use sglang_disagg::peers::{AddOutcome, PeerOps, PeerTable};
use sglang_disagg::{Error, Result};

fn vectors() -> JVal {
    let path = format!("{}/tests/vectors/wire.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read(&path).expect("wire.json missing — run scripts/gen_wire_vectors.py");
    loads(&text).expect("wire.json is valid json")
}

#[derive(Clone, Copy, PartialEq)]
enum CheckMode {
    Ok,
    Invalid,
    Raise,
}

/// Recording mock with the exact call surface of the Python stub agent.
struct MockAgent {
    trace: Vec<Vec<String>>,
    check_mode: CheckMode,
}

impl PeerOps for MockAgent {
    fn add_remote_agent(&mut self, metadata: &[u8]) -> Result<()> {
        self.trace.push(vec![
            "add_remote_agent".into(),
            String::from_utf8(metadata.to_vec()).unwrap(),
        ]);
        Ok(())
    }

    fn check_remote_metadata(&mut self, name: &str) -> Result<bool> {
        self.trace
            .push(vec!["check_remote_metadata".into(), name.into()]);
        match self.check_mode {
            CheckMode::Raise => Err(Error::Invalid("check blew up".into())),
            CheckMode::Ok => Ok(true),
            CheckMode::Invalid => Ok(false),
        }
    }

    fn remove_remote_agent(&mut self, name: &str) -> Result<()> {
        self.trace
            .push(vec!["remove_remote_agent".into(), name.into()]);
        Ok(())
    }
}

/// The prefill-side add hook: aux check + payload plan + state plan, exactly
/// the stubbed sequence the Python `_add_remote_peer` drives.
fn add_prepare(agent: &mut MockAgent, name: &str) -> Result<()> {
    agent
        .trace
        .push(vec!["aux_check".into(), name.into(), "2".into()]);
    agent
        .trace
        .push(vec!["prepare_payload".into(), name.into()]);
    agent.trace.push(vec!["prepare_state".into(), name.into()]);
    Ok(())
}

fn reload_prepare(agent: &mut MockAgent, name: &str) -> Result<()> {
    agent
        .trace
        .push(vec!["prepare_payload".into(), name.into()]);
    agent.trace.push(vec!["prepare_state".into(), name.into()]);
    Ok(())
}

fn want_trace(case: &JVal) -> Vec<Vec<String>> {
    case.get("agent_trace")
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|e| {
            e.as_list()
                .unwrap()
                .iter()
                .map(|x| match x {
                    JVal::Str(s) => s.clone(),
                    JVal::Int(i) => i.to_string(),
                    _ => panic!("trace entry"),
                })
                .collect()
        })
        .collect()
}

#[test]
fn peer_vectors() {
    let v = vectors();
    for case in v.get("peers").unwrap().as_list().unwrap() {
        let name = case.get("name").unwrap().as_str().unwrap();
        let is_prefill = case.get("is_prefill").unwrap() == &JVal::Bool(true);
        let mut agent = MockAgent {
            trace: Vec::new(),
            check_mode: CheckMode::Ok,
        };
        let mut table = PeerTable::default();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for op in case.get("ops").unwrap().as_list().unwrap() {
            let o = op.as_list().unwrap();
            match o[0].as_str().unwrap() {
                "add" => {
                    let (peer, meta) = (o[1].as_str().unwrap(), o[2].as_str().unwrap());
                    let outcome = table
                        .add_remote_peer(
                            &mut agent,
                            peer,
                            meta.as_bytes().to_vec(),
                            is_prefill,
                            |a, n| {
                                if is_prefill {
                                    add_prepare(a, n)
                                } else {
                                    Ok(())
                                }
                            },
                        )
                        .unwrap();
                    let want = if seen.insert(peer.to_string()) {
                        AddOutcome::Added
                    } else {
                        AddOutcome::AlreadyRegistered
                    };
                    assert_eq!(outcome, want, "{name} add {peer}");
                }
                "reload" | "reload_fail_prepare" => {
                    let fail_prepare = o[0].as_str().unwrap() == "reload_fail_prepare";
                    let agents: Vec<String> = o[1]
                        .as_list()
                        .unwrap()
                        .iter()
                        .map(|a| a.as_str().unwrap().to_string())
                        .collect();
                    let now = o[2].as_int().unwrap() as f64 / 1000.0;
                    agent.check_mode = if fail_prepare {
                        CheckMode::Invalid
                    } else {
                        match o[3].as_str().unwrap() {
                            "ok" => CheckMode::Ok,
                            "invalid" => CheckMode::Invalid,
                            "raise" => CheckMode::Raise,
                            m => panic!("check mode {m}"),
                        }
                    };
                    let failures = table.reload_invalidated(&mut agent, &agents, now, |a, n| {
                        if fail_prepare {
                            Err(Error::Invalid("prepare blew up".into()))
                        } else {
                            reload_prepare(a, n)
                        }
                    });
                    // Contained per-agent failures: every room agent whose
                    // check raised or whose prepare failed — all registered.
                    let want_failures: usize =
                        if agent.check_mode == CheckMode::Raise || fail_prepare {
                            agents.len()
                        } else {
                            0
                        };
                    assert_eq!(failures.len(), want_failures, "{name} failures");
                    agent.check_mode = CheckMode::Ok;
                }
                other => panic!("unknown peer op {other}"),
            }
        }

        assert_eq!(agent.trace, want_trace(case), "{name} trace");

        let peers = case.get("final_peers").unwrap().as_list().unwrap();
        for p in peers {
            assert!(table.contains(p.as_str().unwrap()), "{name} peer {p:?}");
        }
        for (peer, prepared) in case.get("final_prepared").unwrap().as_obj().unwrap() {
            assert_eq!(
                table.is_prepared(peer),
                prepared == &JVal::Bool(true),
                "{name} prepared {peer}"
            );
        }
        for (peer, t) in case.get("final_reload_times").unwrap().as_obj().unwrap() {
            assert_eq!(
                table.last_reload(peer),
                Some(t.as_int().unwrap() as f64 / 1000.0),
                "{name} time {peer}"
            );
        }
    }
}
