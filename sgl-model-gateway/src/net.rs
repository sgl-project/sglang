//! Network address formatting helpers.

use std::net::IpAddr;

/// Format `host:port` as a URL authority, bracketing IPv6 literals.
///
/// - IPv4 addresses and DNS hostnames pass through unchanged.
/// - IPv6 literals are wrapped in `[]` (RFC 3986 §3.2.2).
/// - Already-bracketed input (`"[2001:db8::1]"`) is **not** double-bracketed.
pub fn format_authority(host: &str, port: u16) -> String {
    if host.starts_with('[') && host.ends_with(']') {
        return format!("{}:{}", host, port);
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V6(_)) => format!("[{}]:{}", host, port),
        Ok(IpAddr::V4(_)) | Err(_) => format!("{}:{}", host, port),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    #[test]
    fn ipv4_passes_through() {
        assert_eq!(format_authority("10.0.0.1", 8080), "10.0.0.1:8080");
    }

    #[test]
    fn ipv6_full_form_gets_brackets() {
        assert_eq!(
            format_authority("2001:0db8:0000:0000:0000:0000:0000:0001", 8080),
            "[2001:0db8:0000:0000:0000:0000:0000:0001]:8080"
        );
    }

    #[test]
    fn ipv6_compressed_form_gets_brackets() {
        assert_eq!(format_authority("2001:db8::1", 8080), "[2001:db8::1]:8080");
        assert_eq!(format_authority("::1", 8080), "[::1]:8080");
        assert_eq!(format_authority("::", 8080), "[::]:8080");
    }

    #[test]
    fn ipv4_mapped_ipv6_gets_brackets() {
        // ::ffff:10.0.0.1 parses as IpAddr::V6, so it's bracketed.
        assert_eq!(
            format_authority("::ffff:10.0.0.1", 8080),
            "[::ffff:10.0.0.1]:8080"
        );
    }

    #[test]
    fn already_bracketed_is_not_double_bracketed() {
        assert_eq!(
            format_authority("[2001:db8::1]", 8080),
            "[2001:db8::1]:8080"
        );
        assert_eq!(format_authority("[::1]", 8080), "[::1]:8080");
    }

    #[test]
    fn hostname_passes_through() {
        assert_eq!(
            format_authority("worker.svc.cluster.local", 8080),
            "worker.svc.cluster.local:8080"
        );
        assert_eq!(format_authority("localhost", 8080), "localhost:8080");
    }

    #[test]
    fn ipv6_unspecified_bind() {
        let s = format_authority("::", 8000);
        let addr: SocketAddr = s.parse().expect("must parse as SocketAddr");
        assert!(addr.is_ipv6());
        assert_eq!(addr.port(), 8000);
    }

    #[test]
    fn ipv4_unspecified_bind() {
        let s = format_authority("0.0.0.0", 8000);
        let addr: SocketAddr = s.parse().expect("must parse as SocketAddr");
        assert!(addr.is_ipv4());
        assert_eq!(addr.port(), 8000);
    }

    #[test]
    fn socket_addr_round_trip() {
        for (host, port) in [
            ("10.0.0.1", 8080u16),
            ("2001:db8::1", 8080),
            ("::1", 9000),
            ("127.0.0.1", 65535),
        ] {
            let s = format_authority(host, port);
            let addr: SocketAddr = s
                .parse()
                .unwrap_or_else(|e| panic!("parse {s:?} failed: {e}"));
            assert_eq!(addr.port(), port);
        }
    }

    #[test]
    fn url_round_trip_with_http_prefix() {
        for (host, port) in [
            ("10.0.0.1", 8080u16),
            ("2001:db8::1", 8080),
            ("::1", 9000),
            ("worker.svc", 8080),
        ] {
            let s = format!("http://{}", format_authority(host, port));
            let url = url::Url::parse(&s).unwrap_or_else(|e| panic!("parse {s:?}: {e}"));
            assert_eq!(url.port(), Some(port));
        }
    }

    #[test]
    fn empty_host_no_panic() {
        assert_eq!(format_authority("", 8080), ":8080");
    }
}
