use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tonic::Code;

#[derive(Default)]
struct TestRuntime {
    healthy: AtomicBool,
    calls: AtomicUsize,
}

fn test_service() -> (StandardHealthService, Arc<TestRuntime>, watch::Sender<bool>) {
    let runtime = Arc::new(TestRuntime::default());
    let state = runtime.clone();
    let (shutdown, stopping) = watch::channel(false);
    let service = StandardHealthService {
        check: Arc::new(move || {
            state.calls.fetch_add(1, Ordering::SeqCst);
            state.healthy.load(Ordering::SeqCst)
        }),
        stopping,
    };
    (service, runtime, shutdown)
}

fn request(service: &str) -> Request<HealthCheckRequest> {
    Request::new(HealthCheckRequest {
        service: service.to_owned(),
    })
}

#[tokio::test]
async fn check_tracks_runtime_readiness_for_overall_and_named_service() {
    let (service, runtime, _shutdown) = test_service();
    for (ready, expected) in [
        (false, ServingStatus::NotServing),
        (true, ServingStatus::Serving),
        (false, ServingStatus::NotServing),
    ] {
        runtime.healthy.store(ready, Ordering::SeqCst);
        for name in ["", SGLANG_SERVICE] {
            let response = service.check(request(name)).await.unwrap().into_inner();
            assert_eq!(response.status, expected as i32);
        }
    }
}

#[tokio::test]
async fn check_rejects_unknown_service_without_calling_runtime() {
    let (service, runtime, _shutdown) = test_service();
    let err = service.check(request("unknown")).await.unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert_eq!(runtime.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn check_reports_not_serving_when_runtime_task_panics() {
    let (mut service, _, _shutdown) = test_service();
    service.check = Arc::new(|| panic!("runtime failed"));
    let response = service.check(request("")).await.unwrap().into_inner();
    assert_eq!(response.status, ServingStatus::NotServing as i32);
}

#[tokio::test]
async fn shutdown_overrides_runtime_readiness_without_calling_runtime() {
    let (service, runtime, shutdown) = test_service();
    runtime.healthy.store(true, Ordering::SeqCst);
    shutdown.send(true).unwrap();
    for name in ["", SGLANG_SERVICE] {
        let response = service.check(request(name)).await.unwrap().into_inner();
        assert_eq!(response.status, ServingStatus::NotServing as i32);
    }
    assert_eq!(runtime.calls.load(Ordering::SeqCst), 0);
}
