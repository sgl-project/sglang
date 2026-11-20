//! HiCache storage changes across the listeners of a DP deployment.

use std::time::Duration;

use serde_json::json;
use sglang_server_test::ApiKeys;

use crate::common::{dp_deployment, http, next_scheduler_work};

/// Without an admin key, attach and detach are refused before the
/// fan-out, so neither the local scheduler nor a peer rank receives them;
/// with one, both do.
#[test]
fn storage_changes_without_an_admin_key_reach_no_rank() {
    let attach = r#"{"hicache_storage_backend": "file"}"#;
    let [(rt0, addr0), (rt1, _)] = dp_deployment(|_, _| {});
    for method in ["PUT", "DELETE"] {
        let response = http(addr0, method, "/hicache/storage-backend", &[], attach);
        assert_eq!(response.status, 403, "{method}: {}", response.body);
    }
    for (rt, rank) in [(&rt0, 0), (&rt1, 1)] {
        assert!(
            rt.next_request(Duration::from_millis(200)).is_none(),
            "rank {rank}'s scheduler received a refused change"
        );
    }

    let [(rt0, addr0), (rt1, _)] = dp_deployment(|_, args| {
        args.api_keys = ApiKeys::new(None, Some("sk-admin".into()));
    });
    let client = std::thread::spawn(move || {
        let admin = "Authorization: Bearer sk-admin";
        http(addr0, "PUT", "/hicache/storage-backend", &[admin], attach)
    });
    for rt in [&rt0, &rt1] {
        let request = next_scheduler_work(rt);
        assert_eq!(
            request.kind().as_deref(),
            Some("AttachHiCacheStorageReqInput")
        );
        rt.push_control_result(&request.rid().expect("rid"), &json!({"success": true}))
            .unwrap();
    }
    let response = client.join().unwrap();
    assert_eq!(response.status, 200, "{}", response.body);
    assert_eq!(response.json()["success"], true);
}
