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
) -> Result<SocketAddr, String> {
    listen_addr_for_port(server_args, server_args.port, port_offset)
}

pub(crate) fn grpc_listen_addr(
    server_args: &ServerArgs,
    port_offset: Option<u16>,
) -> Result<Option<SocketAddr>, String> {
    server_args
        .grpc_port
        .map(|port| listen_addr_for_port(server_args, port, port_offset))
        .transpose()
}

fn listen_addr_for_port(
    server_args: &ServerArgs,
    base_port: u16,
    port_offset: Option<u16>,
) -> Result<SocketAddr, String> {
    let offset = port_offset.unwrap_or_default();
    let port = base_port
        .checked_add(offset)
        .ok_or_else(|| format!("port {base_port} + offset {offset} exceeds 65535"))?;
    let mut addr: SocketAddr = server_args
        .bind()
        .parse()
        .map_err(|err| format!("invalid host {:?}: {err}", server_args.host))?;
    addr.set_port(port);
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
            listen_addr(&args, None).unwrap(),
            "[::]:30000".parse().unwrap()
        );
        assert_eq!(
            listen_addr(&args, Some(7)).unwrap(),
            "[::]:30007".parse().unwrap()
        );
    }

    #[test]
    fn listen_addr_rejects_port_overflow() {
        let args = ServerArgs {
            port: u16::MAX,
            ..Default::default()
        };
        assert!(listen_addr(&args, Some(1)).unwrap_err().contains("exceeds"));
    }

    #[test]
    fn grpc_listen_addr_is_optional_and_uses_the_same_offset() {
        let mut args = ServerArgs {
            host: "::".into(),
            ..Default::default()
        };
        assert_eq!(grpc_listen_addr(&args, Some(7)).unwrap(), None);

        args.grpc_port = Some(50_051);
        assert_eq!(
            grpc_listen_addr(&args, Some(7)).unwrap(),
            Some("[::]:50058".parse().unwrap())
        );
    }

    #[test]
    fn grpc_listen_addr_rejects_port_overflow() {
        let args = ServerArgs {
            grpc_port: Some(u16::MAX),
            ..Default::default()
        };
        assert!(
            grpc_listen_addr(&args, Some(1))
                .unwrap_err()
                .contains("exceeds")
        );
    }
}
