use serde::Deserialize;
use sglang_server::pd::buffer::{
    AUX_BYTES, AuxRecord, AuxRecordInput, BufferError, COMPLETION_BYTES, CompletionRecordInput,
    CompletionWrites, TransferPlanDigest, clear_partial_page_tail, crc32c, validate_completion,
};
use sglang_server::pd::protocol::FixedBytes;
use sglang_server::pd::room::{AttemptId, ProcessEpoch, RegistrationEpoch};

const DATA_GOLDEN: &[u8] = include_bytes!("../contracts/data-v1-golden.json");

#[derive(Debug, Deserialize)]
struct DataGolden {
    aux_v1: AuxGolden,
    completion_v1: CompletionGolden,
    page_boundaries: Vec<PageBoundary>,
}

#[derive(Debug, Deserialize)]
struct AuxGolden {
    input: AuxInput,
    expected_bytes_hex: String,
    expected_crc32c: u32,
}

#[derive(Debug, Deserialize)]
struct AuxInput {
    first_token_valid: bool,
    first_token_id: i32,
    prompt_token_count: u32,
    prefill_output_count: u32,
    request_digest: String,
}

#[derive(Debug, Deserialize)]
struct CompletionGolden {
    input: CompletionInput,
    expected_record_crc32c: u32,
    expected_body_and_crc_hex: String,
    expected_bytes_hex: String,
    commit_marker_hex: String,
}

#[derive(Debug, Deserialize)]
struct CompletionInput {
    decode_process_epoch: String,
    attempt_id: String,
    source_registration_epoch: String,
    destination_registration_epoch: String,
    bootstrap_room: u64,
    transfer_generation: u64,
    chunk_sequence: u32,
    chunk_count: u32,
    page_count: u32,
    valid_token_count: u32,
    request_digest: String,
    transfer_plan_digest: String,
}

#[derive(Debug, Deserialize)]
struct PageBoundary {
    valid_token_count: u32,
    final_page_valid_rows: usize,
    final_page_invalid_tail_rows: usize,
}

fn golden() -> DataGolden {
    serde_json::from_slice(DATA_GOLDEN).expect("data golden")
}

fn aux_input(input: &AuxInput) -> AuxRecordInput {
    AuxRecordInput {
        first_token_valid: input.first_token_valid,
        first_token_id: input.first_token_id,
        prompt_token_count: input.prompt_token_count,
        prefill_output_count: input.prefill_output_count,
        request_digest: FixedBytes::from_hex(&input.request_digest).expect("request digest"),
    }
}

fn completion_input(input: &CompletionInput) -> CompletionRecordInput {
    CompletionRecordInput {
        decode_process_epoch: ProcessEpoch::parse(&input.decode_process_epoch)
            .expect("process epoch"),
        attempt_id: AttemptId::parse(&input.attempt_id).expect("attempt id"),
        source_registration_epoch: RegistrationEpoch::parse(&input.source_registration_epoch)
            .expect("source epoch"),
        destination_registration_epoch: RegistrationEpoch::parse(
            &input.destination_registration_epoch,
        )
        .expect("destination epoch"),
        bootstrap_room: input.bootstrap_room,
        transfer_generation: input.transfer_generation,
        chunk_sequence: input.chunk_sequence,
        chunk_count: input.chunk_count,
        page_count: input.page_count,
        valid_token_count: input.valid_token_count,
        request_digest: FixedBytes::from_hex(&input.request_digest).expect("request digest"),
        transfer_plan_digest: TransferPlanDigest::from_hex(&input.transfer_plan_digest)
            .expect("plan digest"),
    }
}

#[test]
fn aux_and_completion_match_the_external_exact_byte_crc32c_oracle() {
    let golden = golden();
    assert_eq!(crc32c(b"123456789"), 0xe306_9283);

    let aux = AuxRecord::encode(aux_input(&golden.aux_v1.input)).expect("encode golden aux");
    assert_eq!(
        aux.as_slice(),
        hex::decode(&golden.aux_v1.expected_bytes_hex)
            .expect("aux hex")
            .as_slice()
    );
    assert_eq!(crc32c(&aux), golden.aux_v1.expected_crc32c);

    let expected = completion_input(&golden.completion_v1.input);
    let writes = CompletionWrites::encode(&expected, &aux).expect("completion writes");
    assert_eq!(
        writes.body_and_crc().as_slice(),
        hex::decode(&golden.completion_v1.expected_body_and_crc_hex)
            .expect("body hex")
            .as_slice()
    );
    assert_eq!(
        writes.commit_marker().as_slice(),
        hex::decode(&golden.completion_v1.commit_marker_hex)
            .expect("marker hex")
            .as_slice()
    );
    let completion = writes.committed_bytes();
    assert_eq!(
        completion.as_slice(),
        hex::decode(&golden.completion_v1.expected_bytes_hex)
            .expect("completion hex")
            .as_slice()
    );
    assert_eq!(
        u32::from_be_bytes(completion[184..188].try_into().expect("record CRC")),
        golden.completion_v1.expected_record_crc32c
    );
    let validated = validate_completion(&completion, &aux, &expected).expect("validate completion");
    assert_eq!(validated.valid_token_count, 65);
    assert_eq!(validated.aux.first_token_id, 42);
}

#[test]
fn completion_validation_is_marker_then_record_crc_then_aux_crc_then_identity() {
    let golden = golden();
    let aux = AuxRecord::encode(aux_input(&golden.aux_v1.input)).expect("encode golden aux");
    let expected = completion_input(&golden.completion_v1.input);
    let completion = CompletionWrites::encode(&expected, &aux)
        .expect("completion writes")
        .committed_bytes();

    let mut bad_marker = completion;
    bad_marker[191] ^= 1;
    assert!(matches!(
        validate_completion(&bad_marker, &aux, &expected),
        Err(BufferError::DataRecord {
            check: "completion_marker"
        })
    ));

    let mut bad_record_crc = completion;
    bad_record_crc[100] ^= 1;
    assert!(matches!(
        validate_completion(&bad_record_crc, &aux, &expected),
        Err(BufferError::DataRecord {
            check: "completion_crc"
        })
    ));

    let mut bad_aux = aux;
    bad_aux[12] ^= 1;
    assert!(matches!(
        validate_completion(&completion, &bad_aux, &expected),
        Err(BufferError::DataRecord { check: "aux_crc" })
    ));

    let mut bad_identity = completion;
    bad_identity[80] ^= 1;
    let crc = crc32c(&bad_identity[..184]);
    bad_identity[184..188].copy_from_slice(&crc.to_be_bytes());
    assert!(matches!(
        validate_completion(&bad_identity, &aux, &expected),
        Err(BufferError::DataRecord {
            check: "completion_identity"
        })
    ));
}

#[test]
fn partial_page_tail_is_zero_for_every_frozen_token_boundary() {
    let golden = golden();
    for boundary in golden.page_boundaries {
        let mut final_page = [0x5a; 131_072];
        clear_partial_page_tail(&mut final_page, boundary.valid_token_count)
            .expect("clear partial tail");
        let valid_bytes = boundary.final_page_valid_rows * 2_048;
        assert!(final_page[..valid_bytes].iter().all(|byte| *byte == 0x5a));
        assert!(final_page[valid_bytes..].iter().all(|byte| *byte == 0));
        assert_eq!(
            final_page[valid_bytes..].len(),
            boundary.final_page_invalid_tail_rows * 2_048
        );
    }
}

#[test]
fn zero_first_token_contract_is_explicit_and_invalid_combinations_fail_closed() {
    let request_digest = FixedBytes::new([0x44; 32]);
    let no_token = AuxRecord::encode(AuxRecordInput {
        first_token_valid: false,
        first_token_id: 0,
        prompt_token_count: 1,
        prefill_output_count: 0,
        request_digest,
    })
    .expect("max_new_tokens zero aux");
    assert_eq!(no_token.len(), AUX_BYTES);
    assert!(
        !AuxRecord::decode(&no_token)
            .expect("decode no-token aux")
            .first_token_valid
    );

    for input in [
        AuxRecordInput {
            first_token_valid: false,
            first_token_id: 1,
            prompt_token_count: 1,
            prefill_output_count: 0,
            request_digest,
        },
        AuxRecordInput {
            first_token_valid: true,
            first_token_id: 1,
            prompt_token_count: 1,
            prefill_output_count: 0,
            request_digest,
        },
    ] {
        assert!(AuxRecord::encode(input).is_err());
    }
    let completion_input = CompletionRecordInput {
        decode_process_epoch: ProcessEpoch::random(),
        attempt_id: AttemptId::random(),
        source_registration_epoch: RegistrationEpoch::random(),
        destination_registration_epoch: RegistrationEpoch::random(),
        bootstrap_room: 0,
        transfer_generation: 1,
        chunk_sequence: 0,
        chunk_count: 1,
        page_count: 1,
        valid_token_count: 1,
        request_digest,
        transfer_plan_digest: TransferPlanDigest::from_hex(&"55".repeat(32)).expect("digest"),
    };
    assert_eq!(
        CompletionWrites::encode(&completion_input, &no_token)
            .expect("completion with no first token")
            .committed_bytes()
            .len(),
        COMPLETION_BYTES
    );
    let mismatched_aux = AuxRecord::encode(AuxRecordInput {
        first_token_valid: false,
        first_token_id: 0,
        prompt_token_count: 2,
        prefill_output_count: 0,
        request_digest,
    })
    .expect("mismatched aux");
    assert!(matches!(
        CompletionWrites::encode(&completion_input, &mismatched_aux),
        Err(BufferError::DataRecord {
            check: "aux_identity"
        })
    ));
}
