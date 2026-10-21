//! Helpers for the Python-facing server startup boundary.

use std::net::SocketAddr;

use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

use crate::message::config::{DisaggregationMode, ServerArgs};

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

/// Rewrite the advertised ports to the ones this listener binds, so a DP
/// listener's server info names its own endpoints rather than the base ports.
pub(crate) fn advertise_listener_ports(
    mut server_args: ServerArgs,
    http_addr: SocketAddr,
    grpc_addr: Option<SocketAddr>,
) -> ServerArgs {
    server_args.port = http_addr.port();
    server_args.grpc_port = grpc_addr.map(|addr| addr.port());
    // Rust-mode prefill serves the PD bootstrap registry on its API port; on
    // decode the field names the prefill peer's port and must stay unchanged.
    if server_args.disaggregation_mode == DisaggregationMode::Prefill {
        server_args.deployment_info.disaggregation_bootstrap_port = Some(http_addr.port());
    }
    server_args
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

    /// A DP listener at a nonzero offset must not advertise rank 0's ports:
    /// decode would rendezvous with the wrong prefill bootstrap registry.
    #[test]
    fn dp_listener_advertises_its_own_ports() {
        let base = ServerArgs {
            host: "::".into(),
            port: 30_000,
            grpc_port: Some(50_051),
            ..Default::default()
        };
        let http = listen_addr(&base, Some(2)).unwrap();
        let grpc = grpc_listen_addr(&base, Some(2)).unwrap();

        let mut prefill = base.clone();
        prefill.disaggregation_mode = DisaggregationMode::Prefill;
        prefill.deployment_info.disaggregation_bootstrap_port = Some(30_000);
        let prefill = advertise_listener_ports(prefill, http, grpc);
        assert_eq!(prefill.port, 30_002);
        assert_eq!(prefill.grpc_port, Some(50_053));
        assert_eq!(
            prefill.deployment_info.disaggregation_bootstrap_port,
            Some(30_002)
        );

        let mut decode = base;
        decode.disaggregation_mode = DisaggregationMode::Decode;
        decode.deployment_info.disaggregation_bootstrap_port = Some(8_998);
        let decode = advertise_listener_ports(decode, http, grpc);
        assert_eq!(decode.port, 30_002);
        assert_eq!(
            decode.deployment_info.disaggregation_bootstrap_port,
            Some(8_998)
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
