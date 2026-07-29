//! Read-only PD readiness endpoint. It cannot mutate or stop the transport.

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Serialize;

use super::AppState;
use crate::pd::protocol::Role;
use crate::pd::runtime::RuntimeLifecycle;
use crate::pd::transport::TransportSnapshot;

pub(super) fn routes() -> Router<AppState> {
    Router::new().route("/readiness", get(readiness))
}

async fn readiness(State(state): State<AppState>) -> Response {
    let snapshot = state.pd_readiness.as_ref().map(|handle| handle.snapshot());
    readiness_response(snapshot.as_ref())
}

#[derive(Debug, Serialize)]
struct ReadinessBody {
    role: Option<Role>,
    ready: bool,
    local_process_epoch: Option<String>,
    local_registration_epoch: Option<String>,
    profile_digest: Option<String>,
}

fn readiness_response(snapshot: Option<&TransportSnapshot>) -> Response {
    let ready = snapshot.is_some_and(|snapshot| {
        snapshot.runtime.lifecycle == RuntimeLifecycle::PairReady && snapshot.runtime.pair_ready
    });
    let body = match snapshot {
        Some(snapshot) => ReadinessBody {
            role: Some(snapshot.runtime.role),
            ready,
            local_process_epoch: Some(
                uuid::Uuid::from_bytes(snapshot.runtime.process_epoch.as_bytes()).to_string(),
            ),
            local_registration_epoch: Some(
                uuid::Uuid::from_bytes(snapshot.runtime.registration_epoch.as_bytes()).to_string(),
            ),
            profile_digest: Some(hex::encode(snapshot.runtime.profile_digest.as_bytes())),
        },
        None => ReadinessBody {
            role: None,
            ready: false,
            local_process_epoch: None,
            local_registration_epoch: None,
            profile_digest: None,
        },
    };
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pd::protocol::FixedBytes;
    use crate::pd::room::{ProcessEpoch, RegistrationEpoch};
    use crate::pd::runtime::RuntimeSnapshot;

    fn transport_snapshot(lifecycle: RuntimeLifecycle) -> TransportSnapshot {
        let process_epoch = ProcessEpoch::parse("123e4567-e89b-42d3-a456-426614174000").unwrap();
        let registration_epoch =
            RegistrationEpoch::parse("123e4567-e89b-42d3-a456-426614174001").unwrap();
        let mut runtime = RuntimeSnapshot::starting(
            Role::Prefill,
            process_epoch,
            registration_epoch,
            FixedBytes::new([7; 32]),
        );
        runtime.lifecycle = lifecycle;
        runtime.local_ready = lifecycle != RuntimeLifecycle::Starting;
        runtime.pair_ready = lifecycle == RuntimeLifecycle::PairReady;
        TransportSnapshot {
            runtime,
            model_manifest_digest: FixedBytes::new([1; 32]),
            tokenizer_manifest_digest: FixedBytes::new([2; 32]),
            layout_fingerprint: FixedBytes::new([3; 32]),
            expected_bootstrap_host: "prefill.internal".to_string(),
            allowed_bootstrap_ports: [8998].into_iter().collect(),
            accepting_rooms: true,
            active_handles: 0,
            result_slots: 0,
            abort_generation: 0,
            last_abort_reason: None,
        }
    }

    #[test]
    fn only_pair_ready_is_http_200() {
        for lifecycle in [
            RuntimeLifecycle::Starting,
            RuntimeLifecycle::LocalReady,
            RuntimeLifecycle::Draining,
            RuntimeLifecycle::Fatal,
            RuntimeLifecycle::Stopped,
        ] {
            assert_eq!(
                readiness_response(Some(&transport_snapshot(lifecycle))).status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{lifecycle:?}"
            );
        }
        assert_eq!(
            readiness_response(Some(&transport_snapshot(RuntimeLifecycle::PairReady))).status(),
            StatusCode::OK
        );
        assert_eq!(
            readiness_response(None).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn temporary_capacity_exhaustion_does_not_lower_pair_readiness() {
        let mut snapshot = transport_snapshot(RuntimeLifecycle::PairReady);
        snapshot.accepting_rooms = false;
        assert_eq!(readiness_response(Some(&snapshot)).status(), StatusCode::OK);
    }
}
