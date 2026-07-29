use serde::Deserialize;
use sglang_server::pd::protocol::FixedBytes;
use sglang_server::pd::request::{
    AuxSchema, KVPoll, PdPublicError, RequestContractDigest, RequestContractInput, RequestSampling,
};
use sglang_server::pd::room::PdReason;

const GOLDEN: &str = include_str!("../contracts/request-v1-golden.json");

#[derive(Debug, Deserialize)]
struct RequestGolden {
    digest_inputs: DigestInputs,
    request_cases: Vec<RequestCase>,
    mutation_checks: Vec<MutationCheck>,
    public_errors: Vec<PublicErrorFixture>,
    kv_poll: Vec<KvPollFixture>,
}

#[derive(Debug, Deserialize)]
struct DigestInputs {
    model_manifest_digest: String,
    tokenizer_manifest_digest: String,
    layout_fingerprint: String,
    profile_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RequestCase {
    request: RequestFixture,
    expected_canonical_bytes: usize,
    expected_canonical_hex: String,
    expected_digest_hex: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RequestFixture {
    batch_index: u32,
    normalized_input_ids: Vec<u32>,
    sampling: SamplingFixture,
    aux_schema: AuxFixture,
}

#[derive(Debug, Clone, Deserialize)]
struct SamplingFixture {
    temperature: f64,
    top_k: u32,
    top_p: f64,
    min_p: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    repetition_penalty: f64,
    min_new_tokens: u32,
    max_new_tokens: u32,
    stop: Vec<String>,
    stop_regex: Vec<String>,
    stop_token_ids: Vec<u32>,
    ignore_eos: bool,
    n: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct AuxFixture {
    version: u16,
    bytes: u16,
}

#[derive(Debug, Deserialize)]
struct MutationCheck {
    name: String,
    expected_digest_hex: String,
}

#[derive(Debug, Deserialize)]
struct PublicErrorFixture {
    reason: String,
    status: u16,
    retryable: bool,
    headers: PublicHeaders,
    body: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct PublicHeaders {
    #[serde(rename = "x-sglang-pd-reason")]
    reason: String,
    #[serde(rename = "x-sglang-retryable")]
    retryable: String,
}

#[derive(Debug, Deserialize)]
struct KvPollFixture {
    value: u8,
    name: String,
}

fn golden() -> RequestGolden {
    serde_json::from_str(GOLDEN).expect("request golden")
}

fn input(golden: &RequestGolden, case: &RequestCase) -> RequestContractInput {
    let request = &case.request;
    let sampling = &request.sampling;
    RequestContractInput {
        model_manifest_digest: FixedBytes::from_hex(&golden.digest_inputs.model_manifest_digest)
            .expect("model digest"),
        tokenizer_manifest_digest: FixedBytes::from_hex(
            &golden.digest_inputs.tokenizer_manifest_digest,
        )
        .expect("tokenizer digest"),
        layout_fingerprint: FixedBytes::from_hex(&golden.digest_inputs.layout_fingerprint)
            .expect("layout digest"),
        profile_digest: FixedBytes::from_hex(&golden.digest_inputs.profile_digest)
            .expect("profile digest"),
        batch_index: request.batch_index,
        normalized_input_ids: request.normalized_input_ids.clone(),
        sampling: RequestSampling {
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            min_p: sampling.min_p,
            frequency_penalty: sampling.frequency_penalty,
            presence_penalty: sampling.presence_penalty,
            repetition_penalty: sampling.repetition_penalty,
            min_new_tokens: sampling.min_new_tokens,
            max_new_tokens: sampling.max_new_tokens,
            stop: sampling.stop.clone(),
            stop_regex: sampling.stop_regex.clone(),
            stop_token_ids: sampling.stop_token_ids.clone(),
            ignore_eos: sampling.ignore_eos,
            n: sampling.n,
        },
        aux_schema: AuxSchema {
            version: request.aux_schema.version,
            bytes: request.aux_schema.bytes,
        },
    }
}

#[test]
fn request_contract_digest_matches_external_golden() {
    let golden = golden();
    for case in &golden.request_cases {
        let digest = RequestContractDigest::new(input(&golden, case)).expect("request digest");
        assert_eq!(
            digest.canonical_bytes().len(),
            case.expected_canonical_bytes
        );
        assert_eq!(
            hex::encode(digest.canonical_bytes()),
            case.expected_canonical_hex
        );
        assert_eq!(digest.to_hex(), case.expected_digest_hex);
    }
}

#[test]
fn request_contract_canonicalization_and_mutations_match_external_golden() {
    let golden = golden();
    let base = input(&golden, &golden.request_cases[0]);

    let mut positive_zero = base.clone();
    positive_zero.sampling.frequency_penalty = 0.0;

    let mut reordered_sets = base.clone();
    reordered_sets.sampling.stop = vec!["END".into(), "停止".into()];
    reordered_sets.sampling.stop_regex = vec!["a{1,3}".into(), "z+$".into(), "z+$".into()];
    reordered_sets.sampling.stop_token_ids = vec![151643, 151645];

    let mut changed_input = base.clone();
    changed_input.normalized_input_ids[0] = 2;

    let mut changed_batch = base.clone();
    changed_batch.batch_index = 1;

    let mut changed_sampling = base.clone();
    changed_sampling.sampling.max_new_tokens = 2;

    let cases = [
        (
            "bootstrap_identity_is_excluded",
            RequestContractDigest::new(base.clone()).expect("base"),
        ),
        (
            "negative_zero_is_positive_zero",
            RequestContractDigest::new(positive_zero).expect("positive zero"),
        ),
        (
            "unordered_sets_are_sorted_and_deduplicated",
            RequestContractDigest::new(reordered_sets).expect("sets"),
        ),
        (
            "input_id_mutation_changes_digest",
            RequestContractDigest::new(changed_input).expect("changed input"),
        ),
        (
            "batch_index_mutation_changes_digest",
            RequestContractDigest::new(changed_batch).expect("changed batch"),
        ),
        (
            "sampling_mutation_changes_digest",
            RequestContractDigest::new(changed_sampling).expect("changed sampling"),
        ),
    ];
    for (name, actual) in cases {
        let expected = golden
            .mutation_checks
            .iter()
            .find(|fixture| fixture.name == name)
            .expect("mutation fixture");
        assert_eq!(actual.to_hex(), expected.expected_digest_hex, "{name}");
    }
}

#[test]
fn public_errors_and_kv_poll_match_external_golden() {
    let golden = golden();
    let reasons = [
        PdReason::RequestInvalid,
        PdReason::Unsupported,
        PdReason::CapacityExhausted,
        PdReason::ProtocolMismatch,
        PdReason::PeerUnavailable,
        PdReason::RendezvousTimeout,
        PdReason::TransferTimeout,
        PdReason::TransferFailed,
        PdReason::AckTimeout,
        PdReason::Aborted,
        PdReason::StaleEpoch,
        PdReason::LocalFatal,
    ];
    assert_eq!(golden.public_errors.len(), reasons.len());
    for (fixture, reason) in golden.public_errors.iter().zip(reasons) {
        let actual = PdPublicError::new(reason).expect("public error");
        assert_eq!(actual.reason(), fixture.reason);
        assert_eq!(actual.status(), fixture.status);
        assert_eq!(actual.retryable(), fixture.retryable);
        assert_eq!(actual.reason(), fixture.headers.reason);
        assert_eq!(actual.retryable().to_string(), fixture.headers.retryable);
        assert_eq!(actual.body(), fixture.body);
    }

    let statuses = [
        KVPoll::Failed,
        KVPoll::Bootstrapping,
        KVPoll::WaitingForInput,
        KVPoll::Transferring,
        KVPoll::Success,
    ];
    for (fixture, status) in golden.kv_poll.iter().zip(statuses) {
        assert_eq!(status as u8, fixture.value);
        assert_eq!(status.name(), fixture.name);
    }
}
