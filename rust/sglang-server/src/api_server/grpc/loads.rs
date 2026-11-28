use std::path::Path;

use rustix::fs::FlockOperation;
use serde::Deserialize;
use sglang_api_types::api::v1::{GetLoadRequest, GetLoadResponse};
use tonic::Status;

use crate::api_server::load_snapshot::read_snapshots;
use crate::message::config::ServerArgs;

// The publisher omits fields at their default values.
#[derive(Default, Deserialize)]
#[serde(default)]
struct LoadSnapshot {
    timestamp: f64,
    dp_rank: u32,
    num_running_reqs: u64,
    num_waiting_reqs: u64,
    num_waiting_uncached_tokens: u64,
    num_used_tokens: u64,
    max_total_num_tokens: u64,
    max_running_requests: u64,
}

pub(super) async fn get_load(
    args: &ServerArgs,
    instance_id: &str,
    request: GetLoadRequest,
) -> Result<GetLoadResponse, Status> {
    let path = args.load_snapshot_path.clone().ok_or_else(|| {
        Status::unavailable("load snapshots are not published in shared memory for this listener")
    })?;
    let slot = args.load_snapshot_dp_rank;
    let dp_rank =
        u32::try_from(slot).map_err(|_| Status::internal("invalid load snapshot rank"))?;
    if request
        .dp_rank
        .is_some_and(|requested| requested != dp_rank)
    {
        return Err(Status::invalid_argument(
            "DP rank is not served by this listener",
        ));
    }

    let dp_size = args.load_snapshot_dp_size;
    // Avoid waiting behind a publisher that stops while holding the file lock.
    // The file read itself runs outside the async worker pool.
    let snapshot = tokio::task::spawn_blocking(move || {
        let raw = read_snapshots(
            Path::new(&path),
            dp_size,
            &[slot],
            FlockOperation::NonBlockingLockShared,
        )
        .pop()?;
        rmpv::ext::from_value::<LoadSnapshot>(raw).ok()
    })
    .await
    .map_err(|_| Status::internal("load snapshot reader stopped"))?
    .filter(|s| s.dp_rank == dp_rank && s.timestamp.is_finite() && s.timestamp > 0.0)
    .ok_or_else(|| Status::unavailable("no load snapshot is available for this rank"))?;

    Ok(GetLoadResponse {
        dp_rank,
        instance_id: instance_id.to_owned(),
        snapshot_unix_seconds: snapshot.timestamp,
        num_running_requests: snapshot.num_running_reqs,
        num_waiting_requests: snapshot.num_waiting_reqs,
        num_waiting_uncached_tokens: snapshot.num_waiting_uncached_tokens,
        num_used_tokens: snapshot.num_used_tokens,
        max_total_num_tokens: snapshot.max_total_num_tokens,
        max_running_requests: snapshot.max_running_requests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::load_snapshot::test_support::SnapshotFile;
    use rustix::fs::flock;
    use std::fs::File;
    use tonic::Code;

    #[tokio::test]
    async fn returns_the_addressed_ranks_snapshot_and_original_timestamp() {
        let file = SnapshotFile::from_payloads(&[
            rmp_serde::to_vec_named(&serde_json::json!({"timestamp": 2.5, "num_running_reqs": 9}))
                .unwrap(),
            rmp_serde::to_vec_named(&serde_json::json!({
                "timestamp": 1.25,
                "dp_rank": 1,
                "num_waiting_reqs": 3,
                "num_used_tokens": 128,
            }))
            .unwrap(),
        ]);
        let args = ServerArgs {
            load_snapshot_path: Some(file.0.display().to_string()),
            load_snapshot_dp_size: 2,
            load_snapshot_dp_rank: 1,
            ..Default::default()
        };
        let load = get_load(&args, "boot", GetLoadRequest { dp_rank: Some(1) })
            .await
            .unwrap();
        assert_eq!(load.dp_rank, 1);
        assert_eq!(load.instance_id, "boot");
        assert_eq!(load.snapshot_unix_seconds, 1.25);
        assert_eq!(load.num_running_requests, 0);
        assert_eq!(load.num_waiting_requests, 3);
        assert_eq!(load.num_used_tokens, 128);
        assert_eq!(
            get_load(&args, "boot", GetLoadRequest { dp_rank: Some(0) })
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }

    #[tokio::test]
    async fn rejects_unpublished_or_busy_snapshots() {
        let file = SnapshotFile::new(2);
        let mut args = ServerArgs {
            load_snapshot_path: Some(file.0.display().to_string()),
            load_snapshot_dp_size: 2,
            load_snapshot_dp_rank: 1,
            ..Default::default()
        };
        // The fixture's second slot has been initialized but never sampled.
        assert_eq!(
            get_load(&args, "boot", GetLoadRequest::default())
                .await
                .unwrap_err()
                .code(),
            Code::Unavailable
        );
        args.load_snapshot_dp_rank = 0;
        let writer = File::open(&file.0).unwrap();
        flock(&writer, FlockOperation::LockExclusive).unwrap();
        assert_eq!(
            get_load(&args, "boot", GetLoadRequest::default())
                .await
                .unwrap_err()
                .code(),
            Code::Unavailable
        );
        drop(writer);
        assert!(
            get_load(&args, "boot", GetLoadRequest::default())
                .await
                .is_ok()
        );
    }
}
