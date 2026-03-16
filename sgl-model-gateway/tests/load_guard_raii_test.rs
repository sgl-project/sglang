//! Tests for WorkerLoadGuard RAII pattern with response body attachment
//!
//! These tests verify that load guards properly decrement worker load when:
//! - Response body is fully consumed
//! - Response body is dropped (client disconnect simulation)
//! - Multiple guards are attached (dual prefill/decode workers)

use std::sync::Arc;

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::BodyExt;
use smg::core::{AttachedBody, BasicWorkerBuilder, Worker, WorkerLoadGuard};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Helper to create an SSE streaming response
fn create_sse_response(rx: mpsc::UnboundedReceiver<Bytes>) -> Response {
    let stream = UnboundedReceiverStream::new(rx).map(Ok::<_, std::io::Error>);
    let body = Body::from_stream(stream);
    Response::new(body)
}

/// Helper to create a test worker
fn create_test_worker() -> Arc<dyn Worker> {
    Arc::new(BasicWorkerBuilder::new("http://localhost:8000").build())
}

#[tokio::test]
async fn test_guard_dropped_when_response_body_consumed() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    // Create a simple response with some data
    let body = Body::from("Hello, World!");
    let response = Response::new(body);

    // Attach guard
    let guard = WorkerLoadGuard::new(worker.clone(), None);
    assert_eq!(worker.load(), 1);

    let guarded_response = AttachedBody::wrap_response(response, guard);

    // Load should still be 1 (guard is in the body)
    assert_eq!(worker.load(), 1);

    // Consume the response body
    let body = guarded_response.into_body();
    let _bytes = body.collect().await.unwrap().to_bytes();

    // After consuming, guard should be dropped, load should be 0
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn test_guard_dropped_when_response_dropped_without_consumption() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    {
        let body = Body::from("Hello, World!");
        let response = Response::new(body);

        let guard = WorkerLoadGuard::new(worker.clone(), None);
        assert_eq!(worker.load(), 1);

        let _guarded_response = AttachedBody::wrap_response(response, guard);

        // Load is still 1
        assert_eq!(worker.load(), 1);

        // Response goes out of scope here
    }

    // After response is dropped, guard should be dropped, load should be 0
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn test_streaming_guard_dropped_when_stream_ends() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    // Create a channel for SSE streaming
    let (tx, rx) = mpsc::unbounded_channel::<Bytes>();

    let response = create_sse_response(rx);
    let guard = WorkerLoadGuard::new(worker.clone(), None);
    assert_eq!(worker.load(), 1);

    let guarded_response = AttachedBody::wrap_response(response, guard);

    // Spawn a task to consume the response
    let worker_clone = worker.clone();
    let consume_task = tokio::spawn(async move {
        {
            let mut body = guarded_response.into_body();
            while let Some(result) = body.frame().await {
                if result.is_err() {
                    break;
                }
            }
            // Body is still in scope here, guard not dropped yet
        }
        // Body dropped here, guard should be dropped
        assert_eq!(worker_clone.load(), 0);
    });

    // Send some data
    tx.send(Bytes::from("data: chunk1\n\n")).unwrap();
    tx.send(Bytes::from("data: chunk2\n\n")).unwrap();

    // Load should still be 1 while streaming
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    assert_eq!(worker.load(), 1);

    // Close the sender to end the stream
    drop(tx);

    // Wait for consumer to finish
    consume_task.await.unwrap();

    // Load should now be 0
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn test_streaming_guard_dropped_on_client_disconnect() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    let (tx, rx) = mpsc::unbounded_channel::<Bytes>();

    let response = create_sse_response(rx);
    let guard = WorkerLoadGuard::new(worker.clone(), None);
    assert_eq!(worker.load(), 1);

    let guarded_response = AttachedBody::wrap_response(response, guard);

    // Start consuming but drop early (simulate client disconnect)
    {
        let mut body = guarded_response.into_body();

        // Read one frame
        tx.send(Bytes::from("data: chunk1\n\n")).unwrap();
        let _ = body.frame().await;

        // Load still 1
        assert_eq!(worker.load(), 1);

        // Body dropped here (simulating client disconnect)
    }

    // Guard should be dropped when body is dropped
    assert_eq!(worker.load(), 0);

    // tx is still open but no one is listening
    drop(tx);
}

#[tokio::test]
async fn test_multiple_guards_all_dropped() {
    let worker1 = create_test_worker();
    let worker2 = create_test_worker();
    assert_eq!(worker1.load(), 0);
    assert_eq!(worker2.load(), 0);

    {
        let body = Body::from("Hello");
        let response = Response::new(body);

        // Create guards for both workers (simulates dual prefill/decode)
        let guard1 = WorkerLoadGuard::new(worker1.clone(), None);
        let guard2 = WorkerLoadGuard::new(worker2.clone(), None);
        assert_eq!(worker1.load(), 1);
        assert_eq!(worker2.load(), 1);

        let _response = AttachedBody::wrap_response(response, vec![guard1, guard2]);

        // Both loads are 1
        assert_eq!(worker1.load(), 1);
        assert_eq!(worker2.load(), 1);
    }

    // Both guards dropped when response goes out of scope
    assert_eq!(worker1.load(), 0);
    assert_eq!(worker2.load(), 0);
}

#[tokio::test]
async fn test_guard_with_empty_body() {
    let worker = create_test_worker();
    assert_eq!(worker.load(), 0);

    {
        let body = Body::empty();
        let response = Response::new(body);

        let guard = WorkerLoadGuard::new(worker.clone(), None);
        assert_eq!(worker.load(), 1);

        let guarded_response = AttachedBody::wrap_response(response, guard);

        // Consume empty body
        let body = guarded_response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        assert!(bytes.is_empty());
    }

    assert_eq!(worker.load(), 0);
}
