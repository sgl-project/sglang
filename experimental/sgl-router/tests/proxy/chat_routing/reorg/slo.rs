// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn slo_headers_order_complete_pd_buckets_and_validate_before_dispatch() {
    use sgl_router::buckets_reorg::SloPreference;
    let fast_p = MockWorker::start(vec![]).await;
    let slow_p = MockWorker::start(vec![]).await;
    let fast_d = MockWorker::start(vec![]).await;
    let slow_d = MockWorker::start(vec![]).await;
    let mut ctx = Arc::try_unwrap(context(
        &[
            ("fast-p", Stage::Prefill, &fast_p),
            ("slow-p", Stage::Prefill, &slow_p),
            ("fast-d", Stage::Decode, &fast_d),
            ("slow-d", Stage::Decode, &slow_d),
        ],
        vec![],
    ))
    .unwrap_or_else(|_| panic!("context is not shared yet"));
    let mut prefill = Bucket::new(
        "a-prefill",
        BucketGroups::Pd {
            prefill: group("fast-p", Arc::new(FirstPolicy::default())),
            decode: group("slow-d", Arc::new(FirstPolicy::default())),
        },
    );
    prefill.ttft_ms = Some(50);
    prefill.tokens_per_second = Some(10.0);
    let mut decode = Bucket::new(
        "b-decode",
        BucketGroups::Pd {
            prefill: group("slow-p", Arc::new(FirstPolicy::default())),
            decode: group("fast-d", Arc::new(FirstPolicy::default())),
        },
    );
    decode.ttft_ms = Some(100);
    decode.tokens_per_second = Some(100.0);
    let mut resolver = BucketResolver::new(vec![decode, prefill]).unwrap();
    resolver.ttft_slo = SloPreference::SloFirst;
    resolver.tps_slo = SloPreference::SloFirst;
    ctx.chat_routing = ChatRouting::Reorg([(ModelId("tiny".into()), resolver)].into());
    let app = build_router(Arc::new(ctx));
    for (ttft, tps) in [("0", "100"), ("50", "NaN"), ("50", "0"), ("abc", "100")] {
        let mut req = request(body("hi"));
        req.headers_mut()
            .insert("x-sgl-ttft-slo-ms", ttft.parse().unwrap());
        req.headers_mut()
            .insert("x-sgl-tps-slo", tps.parse().unwrap());
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }
    for worker in [&fast_p, &slow_p, &fast_d, &slow_d] {
        assert!(worker.captured.lock().unwrap().last_body.is_none());
    }
    // With both targets unmet once, the existing bucket order breaks the tie.
    // With TTFT unconstrained, throughput chooses the other complete bucket.
    for (ttft, expected_d, expected_p, unused_d, unused_p) in [
        ("50", &slow_d, &fast_p, &fast_d, &slow_p),
        ("100", &fast_d, &slow_p, &slow_d, &fast_p),
    ] {
        for worker in [&fast_p, &slow_p, &fast_d, &slow_d] {
            worker.captured.lock().unwrap().last_body = None;
        }
        let mut req = request(body("hi"));
        req.headers_mut()
            .insert("x-sgl-ttft-slo-ms", ttft.parse().unwrap());
        req.headers_mut()
            .insert("x-sgl-tps-slo", "100".parse().unwrap());
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-sgl-decode-url"], expected_d.url);
        response.into_body().collect().await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while expected_p.captured.lock().unwrap().last_body.is_none() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert!(unused_p.captured.lock().unwrap().last_body.is_none());
        assert!(unused_d.captured.lock().unwrap().last_body.is_none());
    }
}

#[tokio::test]
async fn preferred_bucket_rejection_falls_back_and_disabled_headers_are_ignored() {
    use sgl_router::buckets_reorg::SloPreference;
    for enabled in [false, true] {
        let rejected = MockWorker::start(vec![]).await;
        let accepted = MockWorker::start(vec![]).await;
        let mut ctx = Arc::try_unwrap(context(
            &[
                ("rejected", Stage::Plain, &rejected),
                ("accepted", Stage::Plain, &accepted),
            ],
            vec![],
        ))
        .unwrap_or_else(|_| panic!("context is not shared yet"));
        let mut fast = Bucket::new(
            "a-fast",
            BucketGroups::Plain(group("rejected", rejecting_policy())),
        );
        fast.ttft_ms = Some(10);
        let mut slow = Bucket::new(
            "b-slow",
            BucketGroups::Plain(group("accepted", Arc::new(FirstPolicy::default()))),
        );
        slow.ttft_ms = Some(100);
        let mut resolver = BucketResolver::new(vec![slow, fast]).unwrap();
        if enabled {
            resolver.ttft_slo = SloPreference::SloFirst;
        }
        ctx.chat_routing = ChatRouting::Reorg([(ModelId("tiny".into()), resolver)].into());
        let app = build_router(Arc::new(ctx));
        let mut req = request(body("hi"));
        req.headers_mut().insert(
            "x-sgl-ttft-slo-ms",
            if enabled { "50" } else { "invalid" }.parse().unwrap(),
        );
        // TPS is disabled even when TTFT is enabled.
        req.headers_mut()
            .insert("x-sgl-tps-slo", "NaN".parse().unwrap());
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response.into_body().collect().await.unwrap();
        assert!(rejected.captured.lock().unwrap().last_body.is_none());
        assert!(accepted.captured.lock().unwrap().last_body.is_some());
    }
}
