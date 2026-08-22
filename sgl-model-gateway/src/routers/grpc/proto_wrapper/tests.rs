use super::*;
use crate::core::BasicWorkerBuilder;

fn worker(name: &str) -> Arc<dyn Worker> {
    Arc::new(BasicWorkerBuilder::new(format!("grpc://{name}")).build())
}

#[test]
fn detects_legacy_sglang_errors_without_interpreting_http_hints() {
    for http_status_code in ["499", "400", "500", "", "not-a-status"] {
        let response = ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
            response: Some(sglang::generate_response::Response::Error(
                sglang::GenerateError {
                    http_status_code: http_status_code.to_string(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        }));
        assert_eq!(
            response.legacy_in_band_attempt_outcome(),
            Some(AttemptOutcome::AttemptFailure)
        );
    }
    assert_eq!(
        ProtoGenerateResponse::Vllm(vllm::GenerateResponse::default())
            .legacy_in_band_attempt_outcome(),
        None
    );
}

#[test]
fn receipt_consumes_the_publication_right_exactly_once() {
    let selected = worker("selected");
    let mut failed = BreakerReceipt::Active(Arc::clone(&selected));
    failed.resolve(AttemptOutcome::AttemptFailure);
    failed.resolve(AttemptOutcome::Success);
    assert_eq!(selected.circuit_breaker().total_failures(), 1);
    assert_eq!(selected.circuit_breaker().total_successes(), 0);

    let abandoned_worker = worker("abandoned");
    let mut abandoned = BreakerReceipt::Active(Arc::clone(&abandoned_worker));
    abandoned.resolve(AttemptOutcome::Abandoned);
    abandoned.resolve(AttemptOutcome::AttemptFailure);
    assert_eq!(abandoned_worker.circuit_breaker().total_failures(), 0);
    assert_eq!(abandoned_worker.circuit_breaker().total_successes(), 0);
}
