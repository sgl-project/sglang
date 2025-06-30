//! Common test utilities and mock servers for testing health check functionality
//!
//! This module provides shared test infrastructure that can be used across
//! different test modules to create realistic test scenarios.

#[cfg(test)]
pub mod mock_servers {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Enhanced mock server that can simulate various health check scenarios
    ///
    /// This mock server can be configured to return different HTTP responses,
    /// introduce delays, and limit the number of calls it accepts. It's designed
    /// to test worker health check functionality under various conditions.
    ///
    /// # Arguments
    /// * `health_responses` - Vector of (status_code, response_body) tuples
    /// * `response_delays` - Vector of delays to apply to each response
    /// * `call_limit` - Optional maximum number of calls to accept
    ///
    /// # Returns
    /// * `(String, Arc<AtomicUsize>)` - Server URL and call counter
    pub async fn create_enhanced_mock_health_server(
        health_responses: Vec<(u16, String)>, // (status_code, response_body)
        response_delays: Vec<Duration>,       // Delays for each response
        call_limit: Option<usize>,            // Maximum number of calls to accept
    ) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let current_call = call_count_clone.fetch_add(1, Ordering::SeqCst);

                    // Check call limit
                    if let Some(limit) = call_limit {
                        if current_call >= limit {
                            // Close connection immediately if limit exceeded
                            let _ = stream.shutdown().await;
                            continue;
                        }
                    }

                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    // Get response for this call (apply to all endpoints)
                    let default_response = (200, r#"{"status": "healthy"}"#.to_string());
                    let response_index = if health_responses.is_empty() {
                        0
                    } else {
                        current_call % health_responses.len()
                    };
                    let (status_code, response_body) = health_responses
                        .get(response_index)
                        .unwrap_or(&default_response);

                    // Apply delay if specified
                    let delay_index = if response_delays.is_empty() {
                        0
                    } else {
                        current_call % response_delays.len()
                    };
                    if let Some(delay) = response_delays.get(delay_index) {
                        if !delay.is_zero() {
                            tokio::time::sleep(*delay).await;
                        }
                    }

                    let response = format!(
                        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        status_code,
                        match *status_code {
                            200 => "OK",
                            401 => "Unauthorized",
                            429 => "Too Many Requests",
                            500 => "Internal Server Error",
                            503 => "Service Unavailable",
                            _ => "Unknown"
                        },
                        response_body.len(),
                        response_body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;

                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            }
        });

        // Give server time to start
        tokio::time::sleep(Duration::from_millis(50)).await;
        (format!("http://127.0.0.1:{}", addr.port()), call_count)
    }

    /// Create a simple mock HTTP server for basic testing
    ///
    /// This is a simpler version of the mock server for basic HTTP testing scenarios.
    ///
    /// # Arguments
    /// * `response_body` - The response body to return
    /// * `status_code` - The HTTP status code to return
    ///
    /// # Returns
    /// * `String` - Server URL
    pub async fn create_mock_http_server(response_body: &str, status_code: u16) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let response_body = response_body.to_string();
        tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut buffer = [0; 1024];
                let _ = stream.read(&mut buffer).await;

                let response = format!(
                    "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status_code,
                    response_body.len(),
                    response_body
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
                let _ = stream.shutdown().await;
            }
        });

        // Give server time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        format!("http://127.0.0.1:{}", addr.port())
    }
}
