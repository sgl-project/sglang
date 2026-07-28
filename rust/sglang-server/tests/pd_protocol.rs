use std::fs::{self, OpenOptions};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt, symlink};
use std::path::PathBuf;

use hmac::{Hmac, Mac};
use rmpv::Value as MessagePackValue;
use serde::Deserialize;
use serde_json::Value;
use sglang_server::pd::protocol::{
    AuthKind, ControlPayload, Direction, FixedBytes, FrameCodec, FrameError, KvBlock, MessageKind,
    PrepareAccepted, Psk, RoomFields, SessionError, derive_session_keys, frame_hash,
    read_raw_frame, transcript_hash,
};
use sha2::Sha256;

const GOLDEN: &[u8] = include_bytes!("../contracts/control-v1-golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    clock: GoldenClock,
    crypto: GoldenCrypto,
    frames: Vec<GoldenFrame>,
}

#[derive(Debug, Deserialize)]
struct GoldenClock {
    now_unix_ms: u64,
}

#[derive(Debug, Deserialize)]
struct GoldenCrypto {
    psk_hex: String,
    decode_nonce_hex: String,
    prefill_nonce_hex: String,
    expected_psk_id_hex: String,
    expected_client_hello_hash_hex: String,
    expected_transcript_hash_hex: String,
    expected_decode_to_prefill_key_hex: String,
    expected_prefill_to_decode_key_hex: String,
}

#[derive(Debug, Deserialize)]
struct GoldenFrame {
    kind: u16,
    direction: Direction,
    auth: AuthKind,
    sequence: u64,
    deadline_unix_ms: u64,
    payload: Value,
    expected_frame_hex: String,
}

fn golden() -> Golden {
    serde_json::from_slice(GOLDEN).expect("checked-in control golden must parse")
}

fn fixed<const N: usize>(hex_value: &str) -> FixedBytes<N> {
    FixedBytes::from_hex(hex_value).expect("golden fixed bytes")
}

fn frame_key<'a>(fixture: &GoldenFrame, psk: &'a Psk, keys: &'a SessionTestKeys) -> &'a [u8; 32] {
    match (fixture.auth, fixture.direction) {
        (AuthKind::Psk, _) => psk.as_bytes(),
        (AuthKind::Session, Direction::DecodeToPrefill) => &keys.decode_to_prefill,
        (AuthKind::Session, Direction::PrefillToDecode) => &keys.prefill_to_decode,
    }
}

struct SessionTestKeys {
    decode_to_prefill: [u8; 32],
    prefill_to_decode: [u8; 32],
}

fn golden_material() -> (Golden, Psk, SessionTestKeys) {
    let golden = golden();
    let psk_path = temp_path();
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o400);
    let mut file = options.open(&psk_path).expect("create golden PSK");
    std::io::Write::write_all(&mut file, &fixed::<32>(&golden.crypto.psk_hex).into_array())
        .expect("write golden PSK");
    drop(file);
    let psk = Psk::load(&psk_path).expect("load golden PSK");
    fs::remove_file(psk_path).expect("remove golden PSK");
    let keys = SessionTestKeys {
        decode_to_prefill: fixed::<32>(&golden.crypto.expected_decode_to_prefill_key_hex)
            .into_array(),
        prefill_to_decode: fixed::<32>(&golden.crypto.expected_prefill_to_decode_key_hex)
            .into_array(),
    };
    (golden, psk, keys)
}

#[test]
fn all_21_golden_frames_encode_decode_and_reencode_exactly() {
    let (golden, psk, keys) = golden_material();

    assert_eq!(golden.frames.len(), 21);
    for fixture in &golden.frames {
        let kind = MessageKind::try_from(fixture.kind).expect("known golden kind");
        let payload =
            ControlPayload::from_json(kind, fixture.payload.clone()).expect("typed fixture");
        let key = frame_key(fixture, &psk, &keys);
        let expected = hex::decode(&fixture.expected_frame_hex).expect("golden frame hex");

        let encoded = FrameCodec::encode(
            kind,
            fixture.direction,
            fixture.sequence,
            fixture.deadline_unix_ms,
            &payload,
            key,
        )
        .expect("golden frame must encode");
        assert_eq!(encoded, expected, "kind {} encode drifted", fixture.kind);

        let decoded = FrameCodec::decode(
            &expected,
            fixture.direction,
            fixture.sequence,
            golden.clock.now_unix_ms,
            key,
        )
        .expect("golden frame must decode");
        assert_eq!(decoded.payload, payload, "kind {} payload", fixture.kind);

        let reencoded = FrameCodec::encode(
            decoded.header.kind,
            fixture.direction,
            decoded.header.sequence,
            decoded.header.deadline_unix_ms,
            &decoded.payload,
            key,
        )
        .expect("decoded frame must re-encode");
        assert_eq!(reencoded, expected, "kind {} re-encode", fixture.kind);
    }
}

#[test]
fn crypto_known_answers_match_the_external_oracle() {
    let (golden, psk, _) = golden_material();
    let client = hex::decode(&golden.frames[0].expected_frame_hex).expect("client frame");
    let server = hex::decode(&golden.frames[1].expected_frame_hex).expect("server frame");

    assert_eq!(hex::encode(psk.id()), golden.crypto.expected_psk_id_hex);
    assert_eq!(
        hex::encode(frame_hash(&client)),
        golden.crypto.expected_client_hello_hash_hex
    );
    let transcript = transcript_hash(&client, &server);
    assert_eq!(
        hex::encode(transcript),
        golden.crypto.expected_transcript_hash_hex
    );

    let session_keys = derive_session_keys(
        &psk,
        fixed::<32>(&golden.crypto.decode_nonce_hex),
        fixed::<32>(&golden.crypto.prefill_nonce_hex),
        FixedBytes::new(transcript),
        fixed::<16>("11111111111141118111111111111111"),
        fixed::<16>("22222222222242228222222222222222"),
    )
    .expect("HKDF must expand");
    assert_eq!(
        hex::encode(session_keys.decode_to_prefill),
        golden.crypto.expected_decode_to_prefill_key_hex
    );
    assert_eq!(
        hex::encode(session_keys.prefill_to_decode),
        golden.crypto.expected_prefill_to_decode_key_hex
    );
}

#[test]
fn authenticated_decoder_rejects_header_auth_replay_deadline_and_truncation_mutations() {
    let (golden, psk, keys) = golden_material();
    let fixture = &golden.frames[17];
    let original = hex::decode(&fixture.expected_frame_hex).expect("ping frame");
    let key = frame_key(fixture, &psk, &keys);

    let mut cases = Vec::new();
    for (name, offset) in [
        ("magic", 0_usize),
        ("major", 5),
        ("kind", 9),
        ("flags", 11),
        ("payload_length", 15),
        ("sequence", 23),
        ("deadline", 31),
        ("tag", original.len() - 1),
    ] {
        let mut mutated = original.clone();
        mutated[offset] ^= 1;
        cases.push((name, mutated));
    }
    cases.push(("truncated", original[..original.len() - 1].to_vec()));

    for (name, mutated) in cases {
        assert!(
            FrameCodec::decode(
                &mutated,
                fixture.direction,
                fixture.sequence,
                golden.clock.now_unix_ms,
                key,
            )
            .is_err(),
            "{name} mutation unexpectedly decoded"
        );
    }

    assert!(
        FrameCodec::decode(
            &original,
            fixture.direction,
            fixture.sequence + 1,
            golden.clock.now_unix_ms,
            key,
        )
        .is_err(),
        "replay/jump sequence unexpectedly decoded"
    );
    assert!(
        FrameCodec::decode(
            &original,
            fixture.direction,
            fixture.sequence,
            fixture.deadline_unix_ms + 1,
            key,
        )
        .is_err(),
        "expired deadline unexpectedly decoded"
    );
    let directional_fixture = &golden.frames[0];
    let directional_frame =
        hex::decode(&directional_fixture.expected_frame_hex).expect("ClientHello frame");
    assert!(
        FrameCodec::decode(
            &directional_frame,
            Direction::PrefillToDecode,
            directional_fixture.sequence,
            golden.clock.now_unix_ms,
            psk.as_bytes(),
        )
        .is_err(),
        "wrong direction unexpectedly decoded"
    );
}

#[test]
fn decoder_rejects_validly_authenticated_noncanonical_and_wrong_type_payloads() {
    let (golden, _psk, keys) = golden_material();
    let fixture = &golden.frames[17];
    let key = &keys.decode_to_prefill;

    let non_minimal_integer = [
        0x81, 0xa7, b'p', b'i', b'n', b'g', b'_', b'i', b'd', 0xcc, 0x01,
    ];
    let duplicate_key = [
        0x82, 0xa7, b'p', b'i', b'n', b'g', b'_', b'i', b'd', 0x01, 0xa7, b'p', b'i', b'n', b'g',
        b'_', b'i', b'd', 0x01,
    ];
    let non_string_key = [0x81, 0x01, 0x01];
    let wrong_type = [
        0x81, 0xa7, b'p', b'i', b'n', b'g', b'_', b'i', b'd', 0xa1, b'1',
    ];
    let unknown_key = [0x81, 0xa7, b'u', b'n', b'k', b'n', b'o', b'w', b'n', 0x01];
    let float_value = [
        0x81, 0xa7, b'p', b'i', b'n', b'g', b'_', b'i', b'd', 0xcb, 0x3f, 0xf0, 0, 0, 0, 0, 0, 0,
    ];

    for (name, payload) in [
        ("non_minimal_integer", non_minimal_integer.as_slice()),
        ("duplicate_key", duplicate_key.as_slice()),
        ("non_string_key", non_string_key.as_slice()),
        ("wrong_type", wrong_type.as_slice()),
        ("unknown_key", unknown_key.as_slice()),
        ("float_value", float_value.as_slice()),
    ] {
        let frame = raw_authenticated_frame(
            fixture.kind,
            fixture.sequence,
            fixture.deadline_unix_ms,
            payload,
            key,
        );
        assert!(
            FrameCodec::decode(
                &frame,
                fixture.direction,
                fixture.sequence,
                golden.clock.now_unix_ms,
                key,
            )
            .is_err(),
            "{name} payload unexpectedly decoded"
        );
    }
}

#[test]
fn decoder_rejects_order_length_and_hello_semantic_mutations() {
    let (golden, psk, keys) = golden_material();

    let client = &golden.frames[0];
    let client_frame = hex::decode(&client.expected_frame_hex).expect("ClientHello");
    let mut client_value = payload_value(&client_frame);
    let MessagePackValue::Map(entries) = &mut client_value else {
        panic!("ClientHello payload must be a map");
    };
    entries.swap(0, 1);
    let out_of_order = encode_value(&client_value);
    let frame = raw_authenticated_frame(
        client.kind,
        client.sequence,
        client.deadline_unix_ms,
        &out_of_order,
        psk.as_bytes(),
    );
    assert!(
        FrameCodec::decode(
            &frame,
            client.direction,
            client.sequence,
            golden.clock.now_unix_ms,
            psk.as_bytes(),
        )
        .is_err()
    );

    for (field, replacement) in [
        ("role", MessagePackValue::String("prefill".into())),
        ("profile_digest", MessagePackValue::Binary(vec![0x99; 32])),
        ("process_epoch", MessagePackValue::Binary(vec![0; 16])),
        ("nonce", MessagePackValue::Binary(vec![0; 32])),
        ("rank", MessagePackValue::from(1_u64)),
    ] {
        let mut value = payload_value(&client_frame);
        let MessagePackValue::Map(entries) = &mut value else {
            panic!("ClientHello payload must be a map");
        };
        let (_, field_value) = entries
            .iter_mut()
            .find(|(key, _)| key.as_str() == Some(field))
            .expect("Hello field");
        *field_value = replacement;
        let payload = encode_value(&value);
        let frame = raw_authenticated_frame(
            client.kind,
            client.sequence,
            client.deadline_unix_ms,
            &payload,
            psk.as_bytes(),
        );
        assert!(
            FrameCodec::decode(
                &frame,
                client.direction,
                client.sequence,
                golden.clock.now_unix_ms,
                psk.as_bytes(),
            )
            .is_err(),
            "{field} semantic mutation unexpectedly decoded"
        );
    }

    let register = &golden.frames[4];
    let register_frame = hex::decode(&register.expected_frame_hex).expect("RegisterRegions");
    let mut register_value = payload_value(&register_frame);
    let MessagePackValue::Map(entries) = &mut register_value else {
        panic!("RegisterRegions payload must be a map");
    };
    let regions = entries
        .iter_mut()
        .find(|(key, _)| key.as_str() == Some("regions"))
        .expect("regions field");
    let MessagePackValue::Array(regions) = &mut regions.1 else {
        panic!("regions must be an array");
    };
    regions.reverse();
    let reversed_regions = encode_value(&register_value);
    let frame = raw_authenticated_frame(
        register.kind,
        register.sequence,
        register.deadline_unix_ms,
        &reversed_regions,
        &keys.decode_to_prefill,
    );
    assert!(
        FrameCodec::decode(
            &frame,
            register.direction,
            register.sequence,
            golden.clock.now_unix_ms,
            &keys.decode_to_prefill,
        )
        .is_err()
    );

    let mut invalid_registration = payload_value(&register_frame);
    let MessagePackValue::Map(entries) = &mut invalid_registration else {
        panic!("RegisterRegions payload must be a map");
    };
    let (_, registration_epoch) = entries
        .iter_mut()
        .find(|(key, _)| key.as_str() == Some("registration_epoch"))
        .expect("registration_epoch field");
    *registration_epoch = MessagePackValue::Binary(vec![0; 16]);
    let payload = encode_value(&invalid_registration);
    let frame = raw_authenticated_frame(
        register.kind,
        register.sequence,
        register.deadline_unix_ms,
        &payload,
        &keys.decode_to_prefill,
    );
    assert!(
        FrameCodec::decode(
            &frame,
            register.direction,
            register.sequence,
            golden.clock.now_unix_ms,
            &keys.decode_to_prefill,
        )
        .is_err(),
        "non-UUID registration epoch unexpectedly decoded"
    );

    for field in ["process_epoch", "profile_digest"] {
        let mut value = payload_value(&client_frame);
        let MessagePackValue::Map(entries) = &mut value else {
            panic!("ClientHello payload must be a map");
        };
        let (_, field_value) = entries
            .iter_mut()
            .find(|(key, _)| key.as_str() == Some(field))
            .expect("fixed-length field");
        *field_value =
            MessagePackValue::Binary(vec![0; if field == "process_epoch" { 15 } else { 31 }]);
        let payload = encode_value(&value);
        let frame = raw_authenticated_frame(
            client.kind,
            client.sequence,
            client.deadline_unix_ms,
            &payload,
            psk.as_bytes(),
        );
        assert!(
            FrameCodec::decode(
                &frame,
                client.direction,
                client.sequence,
                golden.clock.now_unix_ms,
                psk.as_bytes(),
            )
            .is_err(),
            "{field} wrong length unexpectedly decoded"
        );
    }
}

#[test]
fn direction_matrix_accepts_only_frozen_directions() {
    let (golden, psk, keys) = golden_material();

    for fixture in &golden.frames {
        let kind = MessageKind::try_from(fixture.kind).expect("known kind");
        let opposite = match fixture.direction {
            Direction::DecodeToPrefill => Direction::PrefillToDecode,
            Direction::PrefillToDecode => Direction::DecodeToPrefill,
        };
        let payload =
            ControlPayload::from_json(kind, fixture.payload.clone()).expect("typed fixture");
        let opposite_key = match opposite {
            Direction::DecodeToPrefill => &keys.decode_to_prefill,
            Direction::PrefillToDecode => &keys.prefill_to_decode,
        };
        let encoded = FrameCodec::encode(
            kind,
            opposite,
            fixture.sequence,
            fixture.deadline_unix_ms,
            &payload,
            if fixture.auth == AuthKind::Psk {
                psk.as_bytes()
            } else {
                opposite_key
            },
        );
        if kind.allows(opposite) {
            let frame = encoded.expect("bidirectional kind must encode");
            FrameCodec::decode(
                &frame,
                opposite,
                fixture.sequence,
                golden.clock.now_unix_ms,
                opposite_key,
            )
            .expect("bidirectional kind must decode in the other direction");
        } else {
            assert!(
                encoded.is_err(),
                "kind {} allowed wrong direction",
                fixture.kind
            );
        }
    }
}

#[test]
fn maximum_fragmented_transfer_plan_fits_the_frozen_control_payload() {
    const KV_REGION_COUNT: u16 = 56;
    const KV_PAGES_PER_ROOM: u32 = 64;
    const KV_PAGE_BYTES: u64 = 131_072;
    const CONTROL_FRAME_OVERHEAD: usize = 64;
    const MAX_CONTROL_PAYLOAD_BYTES: usize = 524_288;

    let kv_blocks = (0..KV_REGION_COUNT)
        .flat_map(|region_id| {
            (0..KV_PAGES_PER_ROOM).map(move |page| KvBlock {
                region_id,
                source_page: page,
                destination_page: (page * 37) % KV_PAGES_PER_ROOM,
                byte_offset: 0,
                byte_length: KV_PAGE_BYTES,
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(kv_blocks.len(), 3_584);

    let payload = ControlPayload::PrepareAccepted(PrepareAccepted {
        room: RoomFields {
            decode_process_epoch: fixed("11111111111141118111111111111111"),
            bootstrap_room: 0,
            attempt_id: fixed("33333333333343338333333333333333"),
            generation: 1,
            request_contract_digest: FixedBytes::new([0x55; 32]),
        },
        source_registration_epoch: fixed("aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa"),
        destination_registration_epoch: fixed("bbbbbbbbbbbb4bbb8bbbbbbbbbbbbbbb"),
        kv_blocks,
        source_aux_slot: 1,
        destination_aux_slot: 1,
        source_completion_slot: 1,
        destination_completion_slot: 1,
        valid_token_count: 4_096,
        chunk_sequence: 0,
        chunk_count: 1,
        is_last_chunk: true,
        transfer_plan_digest: FixedBytes::new([0x66; 32]),
    });

    let frame = FrameCodec::encode(
        MessageKind::PrepareAccepted,
        Direction::PrefillToDecode,
        1,
        1_700_000_030_000,
        &payload,
        &[0x77; 32],
    )
    .expect("the frozen 4096-token fragmented plan must encode");
    let payload_bytes = frame.len() - CONTROL_FRAME_OVERHEAD;
    assert_eq!(payload_bytes, 262_107);
    assert!(payload_bytes <= MAX_CONTROL_PAYLOAD_BYTES);
}

#[test]
fn decoder_rejects_wrong_key_and_oversized_authenticated_payload() {
    let (golden, _psk, keys) = golden_material();
    let fixture = &golden.frames[17];
    let original = hex::decode(&fixture.expected_frame_hex).expect("ping frame");
    assert!(
        FrameCodec::decode(
            &original,
            fixture.direction,
            fixture.sequence,
            golden.clock.now_unix_ms,
            &[0x55; 32],
        )
        .is_err()
    );

    let oversized = vec![0_u8; 524_289];
    let frame = raw_authenticated_frame(
        fixture.kind,
        fixture.sequence,
        fixture.deadline_unix_ms,
        &oversized,
        &keys.decode_to_prefill,
    );
    assert!(matches!(
        FrameCodec::decode(
            &frame,
            fixture.direction,
            fixture.sequence,
            golden.clock.now_unix_ms,
            &keys.decode_to_prefill,
        ),
        Err(FrameError::PayloadTooLarge)
    ));
}

#[tokio::test]
async fn bounded_reader_rejects_524289_byte_header_before_reading_the_payload() {
    let (mut writer, mut reader) = tokio::io::duplex(64);
    let mut header = [0_u8; 32];
    header[..4].copy_from_slice(b"SGPD");
    header[4..6].copy_from_slice(&1_u16.to_be_bytes());
    header[8..10].copy_from_slice(&(MessageKind::Ping as u16).to_be_bytes());
    header[12..16].copy_from_slice(&524_289_u32.to_be_bytes());
    header[16..24].copy_from_slice(&1_u64.to_be_bytes());
    header[24..32].copy_from_slice(&1_700_000_030_000_u64.to_be_bytes());
    tokio::io::AsyncWriteExt::write_all(&mut writer, &header)
        .await
        .expect("write oversized header");

    assert!(matches!(
        read_raw_frame(&mut reader).await,
        Err(SessionError::PayloadTooLarge)
    ));
}

fn raw_authenticated_frame(
    kind: u16,
    sequence: u64,
    deadline_unix_ms: u64,
    payload: &[u8],
    key: &[u8; 32],
) -> Vec<u8> {
    let mut frame = Vec::with_capacity(32 + payload.len() + 32);
    frame.extend_from_slice(b"SGPD");
    frame.extend_from_slice(&1_u16.to_be_bytes());
    frame.extend_from_slice(&0_u16.to_be_bytes());
    frame.extend_from_slice(&kind.to_be_bytes());
    frame.extend_from_slice(&0_u16.to_be_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&sequence.to_be_bytes());
    frame.extend_from_slice(&deadline_unix_ms.to_be_bytes());
    frame.extend_from_slice(payload);
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC key");
    mac.update(&frame);
    frame.extend_from_slice(&mac.finalize().into_bytes());
    frame
}

fn payload_value(frame: &[u8]) -> MessagePackValue {
    let payload_len =
        u32::from_be_bytes(frame[12..16].try_into().expect("payload length")) as usize;
    let mut cursor = std::io::Cursor::new(&frame[32..32 + payload_len]);
    rmpv::decode::read_value(&mut cursor).expect("golden payload")
}

fn encode_value(value: &MessagePackValue) -> Vec<u8> {
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, value).expect("test payload encoding");
    bytes
}

#[test]
fn psk_loader_accepts_only_exact_mode_regular_non_symlink_files() {
    let temp = temp_path();
    fs::create_dir(&temp).expect("temp directory");
    let good = temp.join("good.psk");
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o400);
    let mut file = options.open(&good).expect("create PSK");
    std::io::Write::write_all(&mut file, &[7; 32]).expect("write PSK");
    drop(file);
    fs::set_permissions(&good, fs::Permissions::from_mode(0o400)).expect("PSK mode");

    assert!(Psk::load(&good).is_ok());

    let wrong_mode = temp.join("wrong-mode.psk");
    fs::copy(&good, &wrong_mode).expect("copy PSK");
    fs::set_permissions(&wrong_mode, fs::Permissions::from_mode(0o600)).expect("wrong mode");
    assert!(Psk::load(&wrong_mode).is_err());

    let wrong_length = temp.join("wrong-length.psk");
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o400);
    let mut file = options.open(&wrong_length).expect("create short PSK");
    std::io::Write::write_all(&mut file, &[7; 31]).expect("write short PSK");
    drop(file);
    assert!(Psk::load(&wrong_length).is_err());

    let link = temp.join("link.psk");
    symlink(&good, &link).expect("create symlink");
    assert!(Psk::load(&link).is_err());
    assert!(Psk::load(&temp).is_err());

    fs::remove_dir_all(&temp).expect("remove temp directory");
}

fn temp_path() -> PathBuf {
    std::env::temp_dir().join(format!("sglang-pd-protocol-{}", uuid::Uuid::new_v4()))
}
