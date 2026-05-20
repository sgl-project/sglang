use std::{
    process::{Child, Command},
    sync::OnceLock,
    time::Duration,
};

use redis::RedisError;
use tracing::{info, warn};

static SHARED_SERVER: OnceLock<RedisTestServer> = OnceLock::new();

pub fn get_shared_server() -> &'static RedisTestServer {
    let server = SHARED_SERVER
        .get_or_init(|| RedisTestServer::start().expect("Failed to start shared Redis server"));
    server.wait_ready();
    server
}

pub struct RedisTestServer {
    process: Option<Child>,
    port: u16,
    url: String,
}

impl RedisTestServer {
    pub fn start() -> Result<Self, String> {
        let port = portpicker::pick_unused_port()
            .ok_or_else(|| "Failed to find available port".to_string())?;
        let url = format!("redis://127.0.0.1:{}", port);

        let mut cmd = Command::new("redis-server");
        cmd.args([
            "--port",
            &port.to_string(),
            "--save",
            "",
            "--appendonly",
            "no",
            "--daemonize",
            "no",
        ]);

        info!("Starting redis server... cmd={cmd:?}");
        let process = cmd.spawn().map_err(|e| {
            format!(
                "Failed to start redis-server: {}. Is redis-server installed?",
                e
            )
        })?;

        Ok(Self {
            process: Some(process),
            port,
            url,
        })
    }

    pub fn wait_ready(&self) {
        for _ in 0..200 {
            match self.is_ready() {
                Ok(()) => return,
                Err(e) => info!("wait_ready failed, will retry (e={e})"),
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("Timeout waiting Redis server ready on port {}", self.port);
    }

    pub fn is_ready(&self) -> Result<(), RedisError> {
        let client = redis::Client::open(self.url.as_str())?;
        let mut conn = client.get_connection()?;
        redis::cmd("PING").query::<String>(&mut conn)?;
        info!("Redis is_ready=true url={}", self.url);
        Ok(())
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for RedisTestServer {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            info!("Killing redis server... process={process:?}");
            if let Err(e) = process.kill() {
                warn!("Failed to kill redis-server process: {}", e);
            }
            if let Err(e) = process.wait() {
                warn!("Failed to wait for redis-server process: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redis_server_start_stop() {
        let server = RedisTestServer::start().unwrap();
        server.wait_ready();
        assert!(server.url().starts_with("redis://"));

        let client = redis::Client::open(server.url()).unwrap();
        let mut conn = client.get_connection().unwrap();

        let _: () = redis::cmd("SET")
            .arg("test_key")
            .arg("test_value")
            .query(&mut conn)
            .unwrap();

        let value: String = redis::cmd("GET").arg("test_key").query(&mut conn).unwrap();
        assert_eq!(value, "test_value");
    }
}
