//! Room state machine replay: every scenario in `wire.json` "room" was
//! produced by driving the REAL Python `CommonKVManager` methods
//! (`update_status`, `apply_prefill_status`, the deferred-abort ACK
//! methods) plus the inline nixl bootstrap-thread block. The Rust machines
//! must land on identical observable state.

use std::collections::{BTreeMap, BTreeSet};

use disagg_wire::json::{JVal, loads};
use disagg_wire::room::{
    ApplyOutcome, DecodeRooms, DeferredAbortTracker, PrefillRooms, StatusTable, TransferInfoEntry,
};
use disagg_wire::{KVPoll, Room};

fn vectors() -> JVal {
    let path = format!("{}/tests/vectors/wire.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read(&path).expect("wire.json missing — run scripts/gen_wire_vectors.py");
    loads(&text).expect("wire.json is valid json")
}

fn poll(v: &JVal) -> KVPoll {
    KVPoll::from_i32(v.as_int().unwrap() as i32).unwrap()
}

/// Expected room -> status map from the vector ("final" / "final_status").
fn want_status(v: &JVal, key: &str) -> BTreeMap<Room, KVPoll> {
    v.get(key)
        .unwrap()
        .as_obj()
        .unwrap()
        .iter()
        .map(|(k, s)| (k.parse().unwrap(), poll(s)))
        .collect()
}

fn want_failures(v: &JVal) -> BTreeMap<Room, String> {
    v.get("final_failures")
        .unwrap()
        .as_obj()
        .unwrap()
        .iter()
        .map(|(k, s)| (k.parse().unwrap(), s.as_str().unwrap().to_string()))
        .collect()
}

fn want_tracker(v: &JVal) -> BTreeMap<Room, BTreeSet<u32>> {
    v.get("final_tracker")
        .unwrap()
        .as_obj()
        .unwrap()
        .iter()
        .map(|(k, ranks)| {
            (
                k.parse().unwrap(),
                ranks
                    .as_list()
                    .unwrap()
                    .iter()
                    .map(|r| r.as_int().unwrap() as u32)
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn update_status_vectors() {
    let v = vectors();
    let room = v.get("room").unwrap();
    for case in room.get("update_status").unwrap().as_list().unwrap() {
        let name = case.get("name").unwrap().as_str().unwrap();
        let mut table = StatusTable::default();
        for step in case.get("steps").unwrap().as_list().unwrap() {
            let s = step.as_list().unwrap();
            table.update_status(s[0].as_int().unwrap(), poll(&s[1]));
        }
        assert_eq!(table.status, want_status(case, "final"), "{name}");
    }
}

#[test]
fn decode_room_vectors() {
    let v = vectors();
    let room = v.get("room").unwrap();
    for case in room.get("decode").unwrap().as_list().unwrap() {
        let name = case.get("name").unwrap().as_str().unwrap();
        let mut rooms = DecodeRooms {
            defer_decode_allocation: case.get("defer").unwrap() == &JVal::Bool(true),
            ..Default::default()
        };
        for op in case.get("ops").unwrap().as_list().unwrap() {
            let o = op.as_list().unwrap();
            let tag = o[0].as_str().unwrap();
            match tag {
                "open" => rooms.open(o[1].as_int().unwrap(), o[2].as_int().unwrap() as u32),
                "open_tracker_only" => {
                    let room = o[1].as_int().unwrap();
                    rooms.response_tracker.insert(room, BTreeSet::new());
                    rooms.table.update_status(room, KVPoll::Bootstrapping);
                }
                "apply" => {
                    let (r, st, rank, reason) = (
                        o[1].as_int().unwrap(),
                        poll(&o[2]),
                        o[3].as_int().unwrap() as u32,
                        o[4].as_str(),
                    );
                    // Python runs the (absent) staging hook then publishes
                    // Success inside one call; mirror with publish_success.
                    if let ApplyOutcome::SuccessQuorum { room } =
                        rooms.apply_prefill_status(r, st, rank, reason)
                    {
                        rooms.publish_success(room);
                    }
                }
                "abort" => rooms.abort(o[1].as_int().unwrap()),
                "abort_msg" => {
                    rooms.table.apply_abort_message(o[1].as_int().unwrap());
                }
                "clear" => rooms.clear(o[1].as_int().unwrap()),
                _ => panic!("unknown decode op {tag}"),
            }
        }
        assert_eq!(
            rooms.table.status,
            want_status(case, "final_status"),
            "{name} status"
        );
        assert_eq!(rooms.table.failures, want_failures(case), "{name} failures");
        assert_eq!(rooms.response_tracker, want_tracker(case), "{name} tracker");
    }
}

#[test]
fn deferred_abort_vectors() {
    let v = vectors();
    let room = v.get("room").unwrap();
    let mut tracker = DeferredAbortTracker::default();
    for op in room.get("deferred_abort").unwrap().as_list().unwrap() {
        let o = op.as_list().unwrap();
        match o[0].as_str().unwrap() {
            "register" => {
                let generation = tracker.register_room(o[1].as_int().unwrap());
                assert_eq!(generation, o[2].as_int().unwrap(), "register generation");
            }
            "note_ack" => tracker.note_ack(
                o[1].as_int().unwrap(),
                o[2].as_int().unwrap() as u32,
                o[3].as_int().unwrap(),
            ),
            "is_safe" => {
                let safe = tracker
                    .is_release_safe(o[1].as_int().unwrap(), o[2].as_int().unwrap() as usize);
                assert_eq!(safe, o[3] == JVal::Bool(true), "is_safe");
            }
            "clear" => tracker.clear(o[1].as_int().unwrap()),
            other => panic!("unknown deferred-abort op {other}"),
        }
    }
}

#[test]
fn prefill_room_vectors() {
    let v = vectors();
    let room = v.get("room").unwrap();
    for case in room.get("prefill").unwrap().as_list().unwrap() {
        let name = case.get("name").unwrap().as_str().unwrap();
        let mut rooms = PrefillRooms::default();
        let results = case.get("results").unwrap().as_list().unwrap();
        for (i, add) in case
            .get("adds")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .enumerate()
        {
            let a = add.as_list().unwrap();
            let got = rooms.add_transfer_info(
                a[0].as_int().unwrap(),
                a[1].as_str().unwrap(),
                TransferInfoEntry {
                    required_dst_info_num: a[2].as_int().unwrap(),
                    decode_prefix_len: a[3].as_int(),
                    is_dummy: a[4] == JVal::Bool(true),
                },
            );
            assert_eq!(got, results[i].as_int(), "{name} add {i}");
        }
        assert_eq!(
            rooms.table.status,
            want_status(case, "final_status"),
            "{name} status"
        );
        let want_prefixes: BTreeMap<Room, i64> = case
            .get("final_prefixes")
            .unwrap()
            .as_obj()
            .unwrap()
            .iter()
            .map(|(k, p)| (k.parse().unwrap(), p.as_int().unwrap()))
            .collect();
        assert_eq!(rooms.decode_prefix_lens, want_prefixes, "{name} prefixes");
    }
}
