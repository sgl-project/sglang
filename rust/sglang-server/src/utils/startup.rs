//! Helpers for the Python-facing server startup boundary.

use std::net::SocketAddr;

use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

use crate::message::config::ServerArgs;

/// A `ValueError` for a boot-time failure, as `"{context}: {err}"`.
pub(crate) fn value_error(context: &str, err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {err}"))
}

pub(crate) fn listen_addr(
    server_args: &ServerArgs,
    port_offset: Option<u16>,
    http_port: Option<u16>,
) -> Result<SocketAddr, String> {
    if port_offset.is_some() && http_port.is_some() {
        return Err("http_port and port_offset cannot both be specified".into());
    }
    let offset = port_offset.unwrap_or_default();
    let port = server_args
        .port
        .checked_add(offset)
        .ok_or_else(|| format!("port {} + offset {offset} exceeds 65535", server_args.port))?;
    let mut addr: SocketAddr = server_args
        .bind()
        .parse()
        .map_err(|err| format!("invalid host {:?}: {err}", server_args.host))?;
    addr.set_port(http_port.unwrap_or(port));
    Ok(addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listen_addr_uses_server_host_and_port_offset() {
        let args = ServerArgs {
            host: "::".into(),
            port: 30_000,
            ..Default::default()
        };
        assert_eq!(
            listen_addr(&args, None, None).unwrap(),
            "[::]:30000".parse().unwrap()
        );
        assert_eq!(
            listen_addr(&args, Some(7), None).unwrap(),
            "[::]:30007".parse().unwrap()
        );
        assert_eq!(
            listen_addr(&args, None, Some(0)).unwrap(),
            "[::]:0".parse().unwrap()
        );
        assert!(listen_addr(&args, Some(7), Some(0)).is_err());
    }

    #[test]
    fn listen_addr_rejects_port_overflow() {
        let args = ServerArgs {
            port: u16::MAX,
            ..Default::default()
        };
        assert!(
            listen_addr(&args, Some(1), None)
                .unwrap_err()
                .contains("exceeds")
        );
    }
}
