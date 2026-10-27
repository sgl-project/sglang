//! Bit-identity test: every vector in `tests/vectors/wire.json` was produced
//! by the REAL Python implementation (`scripts/gen_wire_vectors.py`, which
//! loads the actual `python/sglang/srt/disaggregation/` modules). The Rust
//! codecs must produce byte-identical frames and round-trip every parse.

use sglang_disagg::KVPoll;
use sglang_disagg::json::{JVal, dumps_default, loads};
use sglang_disagg::nixl::{self, MemKind, MetadataParams, RegistrationParams};
use sglang_disagg::pack;
use sglang_disagg::status::{self, AbortAck, AbortNotification};
use sglang_disagg::topology::{self, PrefillRankInfo, PrefillServerInfo, RegisterParams};

fn vectors() -> JVal {
    let path = format!("{}/tests/vectors/wire.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read(&path).expect("wire.json missing — run scripts/gen_wire_vectors.py");
    loads(&text).expect("wire.json is valid json")
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn frames(case: &JVal) -> Vec<Vec<u8>> {
    case.get("frames")
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|f| unhex(f.as_str().unwrap()))
        .collect()
}

fn hex_of(frames: &[Vec<u8>]) -> Vec<String> {
    frames
        .iter()
        .map(|f| f.iter().map(|b| format!("{:02x}", b)).collect())
        .collect()
}

fn assert_frames_eq(got: &[Vec<u8>], want_case: &JVal, ctx: &str) {
    let want: Vec<Vec<u8>> = frames(want_case);
    assert_eq!(
        hex_of(got),
        hex_of(&want),
        "{ctx}: built frames differ from Python"
    );
}

fn jstr<'a>(v: &'a JVal, key: &str) -> &'a str {
    v.get(key).unwrap().as_str().unwrap()
}

fn jint(v: &JVal, key: &str) -> i64 {
    v.get(key).unwrap().as_int().unwrap()
}

fn jopt_int(v: &JVal, key: &str) -> Option<i64> {
    v.get(key).unwrap().as_int()
}

fn ju64s(v: &JVal, key: &str) -> Vec<u64> {
    v.get(key)
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().parse().unwrap())
        .collect()
}

fn ju64_lists(v: &JVal, key: &str) -> Vec<Vec<u64>> {
    v.get(key)
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|l| {
            l.as_list()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().parse().unwrap())
                .collect()
        })
        .collect()
}

fn ju32s(v: &JVal, key: &str) -> Vec<u32> {
    v.get(key)
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|x| x.as_int().unwrap() as u32)
        .collect()
}

fn ju32_lists(v: &JVal, key: &str) -> Vec<Vec<u32>> {
    v.get(key)
        .unwrap()
        .as_list()
        .unwrap()
        .iter()
        .map(|l| {
            l.as_list()
                .unwrap()
                .iter()
                .map(|x| x.as_int().unwrap() as u32)
                .collect()
        })
        .collect()
}

// --------------------------------------------------------------------------- //
// pack helpers
// --------------------------------------------------------------------------- //

#[test]
fn pack_vectors() {
    let v = vectors();
    for case in v.get("pack").unwrap().as_list().unwrap() {
        let fmt = jstr(case, "fmt");
        let want = unhex(jstr(case, "packed"));
        let lists: Vec<Vec<i64>> = case
            .get("lists")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .map(|l| {
                l.as_list()
                    .unwrap()
                    .iter()
                    // i128 covers both 2^64-1 (Q lanes) and -2^31 (i lanes).
                    .map(|x| x.as_str().unwrap().parse::<i128>().unwrap() as i64)
                    .collect()
            })
            .collect();
        match fmt {
            "Q" => {
                let us: Vec<Vec<u64>> = lists
                    .iter()
                    .map(|l| l.iter().map(|&x| x as u64).collect())
                    .collect();
                assert_eq!(pack::pack_u64_lists(&us), want, "pack_u64_lists");
                assert_eq!(
                    pack::unpack_u64_lists(&want).unwrap(),
                    us,
                    "unpack_u64_lists"
                );
            }
            "I" => {
                let us: Vec<Vec<u32>> = lists
                    .iter()
                    .map(|l| l.iter().map(|&x| x as u32).collect())
                    .collect();
                assert_eq!(pack::pack_u32_lists(&us), want, "pack_u32_lists");
                assert_eq!(
                    pack::unpack_u32_lists(&want).unwrap(),
                    us,
                    "unpack_u32_lists"
                );
            }
            "i" => {
                let is: Vec<Vec<i32>> = lists
                    .iter()
                    .map(|l| l.iter().map(|&x| x as i32).collect())
                    .collect();
                assert_eq!(pack::pack_i32_lists(&is), want, "pack_i32_lists");
                assert_eq!(
                    pack::unpack_i32_lists(&want).unwrap(),
                    is,
                    "unpack_i32_lists"
                );
            }
            _ => panic!("unknown fmt {fmt}"),
        }
    }
}

#[test]
fn pack_buffers_vectors() {
    let v = vectors();
    for case in v.get("pack_buffers").unwrap().as_list().unwrap() {
        let buffers: Vec<Vec<u8>> = case
            .get("buffers")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .map(|b| unhex(b.as_str().unwrap()))
            .collect();
        let want = unhex(jstr(case, "packed"));
        let refs: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();
        assert_eq!(
            pack::pack_list_of_buffers(&refs),
            want,
            "pack_list_of_buffers"
        );
        assert_eq!(
            pack::unpack_list_of_buffers(&want).unwrap(),
            buffers,
            "unpack"
        );
    }
}

// --------------------------------------------------------------------------- //
// KV_STATUS
// --------------------------------------------------------------------------- //

#[test]
fn kv_status_vectors() {
    let v = vectors();
    for case in v.get("kv_status").unwrap().as_list().unwrap() {
        let name = jstr(case, "name");
        let tag: Option<Vec<u8>> = case
            .get("tag")
            .unwrap()
            .as_str()
            .map(|s| s.as_bytes().to_vec());
        let carries = case.get("carries_reason").unwrap() == &JVal::Bool(true);
        let parsed = case.get("parsed").unwrap().as_list().unwrap();
        let room = parsed[0].as_int().unwrap();
        let status_i = parsed[1].as_int().unwrap() as i32;
        let prefill_rank = parsed[2].as_int().unwrap();
        let reason = parsed[3].as_str();

        let got = status::encode_kv_status(
            room,
            KVPoll::from_i32(status_i).unwrap(),
            prefill_rank,
            reason,
            tag.as_deref(),
            carries,
        );
        assert_frames_eq(&got, case, name);

        let want_frames = frames(case);
        let (r, s, p, rs) = status::parse_kv_status(&want_frames, tag.as_deref()).expect(name);
        assert_eq!(
            (r, s, p, rs.as_deref()),
            (room, status_i, prefill_rank, reason),
            "{name}"
        );
    }
}

#[test]
fn kv_status_parse_drops() {
    let good = status::encode_kv_status(
        1,
        KVPoll::Success,
        2,
        None,
        Some(status::KV_STATUS_TAG_NIXL),
        true,
    );
    assert!(status::parse_kv_status(&good, Some(b"WRONG")).is_none());
    assert!(status::parse_kv_status(&good[..2], Some(status::KV_STATUS_TAG_NIXL)).is_none());
    let mut bad = good.clone();
    bad[1] = b"notanum".to_vec();
    assert!(status::parse_kv_status(&bad, Some(status::KV_STATUS_TAG_NIXL)).is_none());
}

// --------------------------------------------------------------------------- //
// ABORT / ABORT_ACK
// --------------------------------------------------------------------------- //

#[test]
fn abort_vectors() {
    let v = vectors();
    for case in v.get("abort").unwrap().as_list().unwrap() {
        let name = jstr(case, "name");
        let p = case.get("parsed").unwrap();
        let n = AbortNotification {
            room: jint(p, "room"),
            decode_ip: p.get("decode_ip").unwrap().as_str().map(str::to_string),
            decode_port: jopt_int(p, "decode_port"),
            generation: jopt_int(p, "generation"),
        };
        // The degrade vector's frames contain a bad address the builder
        // would never emit — parse-only.
        if name != "bad_addr_degrades" {
            assert_frames_eq(&status::build_abort(&n), case, name);
        }
        let want = frames(case);
        assert_eq!(status::parse_abort(&want), Some(n), "{name}");
    }
}

#[test]
fn abort_ack_vectors() {
    let v = vectors();
    for case in v.get("abort_ack").unwrap().as_list().unwrap() {
        let name = jstr(case, "name");
        let p = case.get("parsed").unwrap();
        let a = AbortAck {
            room: jint(p, "room"),
            prefill_rank: jint(p, "prefill_rank"),
            generation: jopt_int(p, "generation"),
        };
        assert_frames_eq(&status::build_abort_ack(&a), case, name);
        let want = frames(case);
        assert_eq!(status::parse_abort_ack(&want), Some(a), "{name}");
    }
}

// --------------------------------------------------------------------------- //
// BOOTSTRAP / CHUNK_READY
// --------------------------------------------------------------------------- //

#[test]
fn bootstrap_vectors() {
    let v = vectors();
    for case in v.get("bootstrap").unwrap().as_list().unwrap() {
        let p = case.get("parsed").unwrap();
        let (room, ip, port) = (jint(p, "room"), jstr(p, "ip"), jint(p, "port"));
        assert_frames_eq(&status::build_bootstrap(room, ip, port), case, "bootstrap");
        let want = frames(case);
        assert_eq!(
            status::parse_bootstrap(&want),
            Some((room, ip.to_string(), port))
        );
    }
}

#[test]
fn chunk_ready_vectors() {
    let v = vectors();
    for case in v.get("chunk_ready").unwrap().as_list().unwrap() {
        let p = case.get("parsed").unwrap();
        let (room, chunk_idx, page_start, num_pages, writer_id) = (
            jint(p, "room"),
            jint(p, "chunk_idx"),
            jint(p, "page_start"),
            jint(p, "num_pages"),
            jstr(p, "writer_id"),
        );
        // The Python builder appends prefill_rank; the parser ignores it.
        assert_frames_eq(
            &status::build_chunk_ready(room, chunk_idx, page_start, num_pages, writer_id, 5),
            case,
            "chunk_ready",
        );
        let want = frames(case);
        assert_eq!(
            status::parse_chunk_ready(&want),
            Some((
                room,
                chunk_idx,
                page_start,
                num_pages,
                writer_id.to_string()
            ))
        );
    }
}

// --------------------------------------------------------------------------- //
// NIXL registration
// --------------------------------------------------------------------------- //

/// Mirrors `reg_ns` in the generator.
fn reg_params(state_entries: usize, staging: bool) -> RegistrationParams {
    RegistrationParams {
        local_ip: "192.168.1.37".into(),
        rank_port: 30001,
        agent_name: "agent-X".into(),
        agent_metadata: unhex("01026167656e742d6d657461ff"),
        kv_data_ptrs: vec![0x7F00_0000_1000, 0x7F00_0001_2000],
        kv_data_lens: vec![1 << 20, 2 << 20],
        kv_data_mem_kinds: vec![MemKind::Vram, MemKind::Dram],
        kv_item_lens: vec![4096, 4096],
        kv_layer_ids: vec![3, 7, 11, 59],
        aux_data_ptrs: vec![0x7F00_0002_0000],
        state_data_ptrs: (0..state_entries)
            .map(|i| vec![0x7F00_0010_0000 + 0x1000 * i as u64])
            .collect(),
        state_item_lens: (0..state_entries).map(|_| vec![4096 * 4]).collect(),
        state_data_lens: (0..state_entries).map(|_| vec![1 << 16]).collect(),
        state_dim_per_tensor: (0..state_entries).map(|_| vec![2, 128]).collect(),
        state_layer_ids: (0..state_entries).map(|i| vec![i as u32]).collect(),
        gpu_id: 5,
        attn_tp_size: 16,
        engine_rank: 3,
        dcp_size: 1,
        dcp_rank: 0,
        staging: staging.then_some((0x7F00_8000_0000, 1 << 24)),
    }
}

#[test]
fn registration_vectors() {
    let v = vectors();
    for case in v.get("registration").unwrap().as_list().unwrap() {
        let name = jstr(case, "name");
        let (state_entries, staging) = match name {
            "basic" => (4, false),
            "staging" => (4, true),
            "empty_state" => (0, false),
            _ => panic!("unknown registration case {name}"),
        };
        let want = frames(case);
        assert_frames_eq(
            &nixl::build_registration(&reg_params(state_entries, staging)),
            case,
            name,
        );

        let p = case.get("parsed").unwrap();
        let info = nixl::parse_registration(&want[1..]).expect(name);
        assert_eq!(info.room, jstr(p, "room"), "{name} room");
        assert_eq!(info.endpoint, jstr(p, "endpoint"), "{name} endpoint");
        assert_eq!(info.dst_port, jint(p, "dst_port"), "{name} dst_port");
        assert_eq!(info.agent_name, jstr(p, "agent_name"), "{name} agent_name");
        assert_eq!(
            info.agent_metadata,
            unhex(jstr(p, "agent_metadata")),
            "{name} agent_metadata"
        );
        assert_eq!(info.dst_kv_ptrs, ju64s(p, "dst_kv_ptrs"), "{name} kv_ptrs");
        let kinds: Vec<MemKind> = p
            .get("dst_kv_mem_kinds")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .map(|k| match k.as_str().unwrap() {
                "VRAM" => MemKind::Vram,
                "DRAM" => MemKind::Dram,
                _ => panic!(),
            })
            .collect();
        assert_eq!(info.dst_kv_mem_kinds, kinds, "{name} mem_kinds");
        assert_eq!(
            info.dst_aux_ptrs,
            ju64s(p, "dst_aux_ptrs"),
            "{name} aux_ptrs"
        );
        assert_eq!(
            info.dst_state_data_ptrs,
            ju64_lists(p, "dst_state_data_ptrs"),
            "{name} state_ptrs"
        );
        assert_eq!(info.gpu_id, jint(p, "gpu_id"), "{name} gpu_id");
        assert_eq!(
            info.decode_tp_size,
            jint(p, "decode_tp_size"),
            "{name} tp_size"
        );
        assert_eq!(
            info.decode_tp_rank,
            jint(p, "decode_tp_rank"),
            "{name} tp_rank"
        );
        assert_eq!(
            info.dst_kv_item_len,
            jstr(p, "dst_kv_item_len").parse::<u64>().unwrap(),
            "{name} item_len"
        );
        assert_eq!(
            info.dst_kv_item_lens,
            ju64s(p, "dst_kv_item_lens"),
            "{name} item_lens"
        );
        assert_eq!(
            info.dst_kv_layer_ids,
            ju32s(p, "dst_kv_layer_ids"),
            "{name} kv_layer_ids"
        );
        assert_eq!(
            info.dst_dcp_size,
            jint(p, "dst_dcp_size"),
            "{name} dcp_size"
        );
        assert_eq!(
            info.dst_dcp_rank,
            jint(p, "dst_dcp_rank"),
            "{name} dcp_rank"
        );
        let want_slots = match p.get("dst_num_slots").unwrap() {
            JVal::Null => None,
            JVal::Str(s) => Some(s.parse::<u64>().unwrap()),
            _ => panic!(),
        };
        assert_eq!(info.dst_num_slots, want_slots, "{name} num_slots");
        assert_eq!(
            info.dst_state_item_lens,
            ju32_lists(p, "dst_state_item_lens"),
            "{name} state_item_lens"
        );
        assert_eq!(
            info.dst_state_dim_per_tensor,
            ju32_lists(p, "dst_state_dim_per_tensor"),
            "{name} dim_per_tensor"
        );
        assert_eq!(
            info.dst_state_layer_ids,
            ju32_lists(p, "dst_state_layer_ids"),
            "{name} state_layer_ids"
        );
        assert_eq!(
            info.staging_base_ptr,
            jstr(p, "staging_base_ptr").parse::<u64>().unwrap(),
            "{name} staging_base"
        );
        assert_eq!(
            info.staging_total_size,
            jstr(p, "staging_total_size").parse::<u64>().unwrap(),
            "{name} staging_total"
        );
        assert_eq!(
            info.dst_state_data_lens,
            ju64_lists(p, "dst_state_data_lens"),
            "{name} state_data_lens"
        );
    }
}

// --------------------------------------------------------------------------- //
// NIXL metadata
// --------------------------------------------------------------------------- //

/// Mirrors the generator's per-case `call_meta` inputs.
fn meta_params(name: &str) -> MetadataParams {
    let (room, kv, states, prefix, dummy): (
        i64,
        Vec<i32>,
        Option<Vec<Option<Vec<i32>>>>,
        Option<i64>,
        bool,
    ) = match name {
        "kv_run" => (777, (100..106).collect(), None, None, false),
        "state_only" => (
            778,
            vec![],
            Some(vec![Some(vec![1, 2]), Some(vec![3])]),
            None,
            false,
        ),
        "prefix_hit" => (
            779,
            vec![],
            Some(vec![Some(vec![9]), Some(vec![10])]),
            Some(7),
            false,
        ),
        "aux_only" => (780, vec![], None, None, false),
        "kv_and_state" => (
            781,
            (5..9).collect(),
            Some(vec![Some(vec![9]), None]),
            None,
            false,
        ),
        "dummy" => (782, vec![], None, None, true),
        _ => panic!("unknown metadata case {name}"),
    };
    MetadataParams {
        room,
        local_ip: "10.0.0.9".into(),
        rank_port: 30002,
        agent_name: "agent-Y".into(),
        kv_indices: kv,
        aux_index: Some(2),
        required_dst_info_num: 8,
        state_indices: states,
        decode_prefix_len: prefix,
        is_dummy: dummy,
    }
}

#[test]
fn metadata_vectors() {
    let v = vectors();
    for case in v.get("metadata").unwrap().as_list().unwrap() {
        let name = jstr(case, "name");
        let want = frames(case);
        assert_frames_eq(&nixl::build_metadata(&meta_params(name)), case, name);

        let p = case.get("parsed").unwrap();
        let info = nixl::parse_metadata(&want[1..]).expect(name);
        assert_eq!(info.room, jint(p, "room"), "{name} room");
        assert_eq!(info.endpoint, jstr(p, "endpoint"), "{name} endpoint");
        assert_eq!(info.dst_port, jint(p, "dst_port"), "{name} dst_port");
        assert_eq!(info.agent_name, jstr(p, "agent_name"), "{name} agent_name");
        let want_kv: Vec<i32> = p
            .get("dst_kv_indices")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .map(|x| x.as_int().unwrap() as i32)
            .collect();
        assert_eq!(info.dst_kv_indices, want_kv, "{name} kv_indices");
        assert_eq!(
            info.dst_aux_index,
            jint(p, "dst_aux_index"),
            "{name} aux_index"
        );
        assert_eq!(
            info.required_dst_info_num,
            jint(p, "required_dst_info_num"),
            "{name} required"
        );
        let want_states: Vec<Vec<i32>> = p
            .get("dst_state_indices")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .map(|l| {
                l.as_list()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_int().unwrap() as i32)
                    .collect()
            })
            .collect();
        assert_eq!(info.dst_state_indices, want_states, "{name} state_indices");
        assert_eq!(
            info.decode_prefix_len,
            jopt_int(p, "decode_prefix_len"),
            "{name} prefix_len"
        );
        assert_eq!(
            info.is_dummy,
            p.get("is_dummy").unwrap() == &JVal::Bool(true),
            "{name} is_dummy"
        );
    }
}

// --------------------------------------------------------------------------- //
// /route HTTP bodies
// --------------------------------------------------------------------------- //

#[test]
fn http_put_row() {
    let v = vectors();
    let body = jstr(v.get("http").unwrap(), "put_row");
    let parsed = topology::parse_register_body(&loads(body.as_bytes()).unwrap()).unwrap();
    assert_eq!(
        parsed,
        RegisterParams {
            attn_tp_size: 16,
            attn_tp_rank: 3,
            attn_cp_size: 1,
            attn_cp_rank: 0,
            attn_dp_size: 2,
            attn_dp_rank: 1,
            pp_size: 1,
            pp_rank: 0,
            system_dp_size: 1,
            system_dp_rank: 0,
            rank_ip: "10.0.0.5".into(),
            rank_port: 30011,
            page_size: Some(1),
            kv_cache_dtype: Some("auto".into()),
            kv_layout: None,
            dsv41_spec_layout: None,
            load_balance_method: Some("round_robin".into()),
            enable_dsa_cache_layer_split: false,
            enable_kv_cache_sharding: false,
            prefill_http_port: Some(8100),
            decode_allocation_policy: "early".into(),
        }
    );
    // Byte-exact reserialization (key order + separators + nulls).
    assert_eq!(dumps_default(&topology::register_body(&parsed)), body);
}

#[test]
fn http_rank_row() {
    let v = vectors();
    let body = jstr(v.get("http").unwrap(), "rank_row");
    let parsed = topology::parse_rank_info(&loads(body.as_bytes()).unwrap()).unwrap();
    assert_eq!(
        parsed,
        PrefillRankInfo {
            rank_ip: "10.0.0.5".into(),
            rank_port: 30011,
        }
    );
    assert_eq!(dumps_default(&topology::rank_info_body(&parsed)), body);
}

#[test]
fn http_get_rows() {
    let v = vectors();
    let http = v.get("http").unwrap();
    for key in ["get_row", "get_row_kv_layout", "get_row_dsv41"] {
        let body = jstr(http, key);
        let parsed = topology::parse_server_info(&loads(body.as_bytes()).unwrap()).unwrap();
        let expected = PrefillServerInfo {
            attn_tp_size: 8,
            attn_cp_size: 1,
            dp_size: 16,
            pp_size: 1,
            page_size: Some(1),
            kv_cache_dtype: Some("auto".into()),
            follow_bootstrap_room: true,
            enable_dsa_cache_layer_split: false,
            dsv41_spec_layout: if key == "get_row_dsv41" {
                Some(JVal::Obj(vec![
                    ("a".to_string(), JVal::Int(1)),
                    ("b".to_string(), JVal::Str("x".into())),
                ]))
            } else {
                None
            },
            kv_layout: if key == "get_row" {
                None
            } else {
                Some("layer_first".into())
            },
            decode_allocation_policy: "early".into(),
            enable_kv_cache_sharding: false,
            prefill_http_port: Some(8100),
        };
        assert_eq!(parsed, expected, "{key}");
        assert_eq!(
            dumps_default(&topology::server_info_body(&parsed)),
            body,
            "{key}"
        );
    }
}
