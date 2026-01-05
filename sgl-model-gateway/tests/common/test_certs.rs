// TLS certificate generation for integration tests
#![allow(dead_code)]

use std::path::PathBuf;

use openssl::{
    asn1::Asn1Time,
    bn::{BigNum, MsbOption},
    hash::MessageDigest,
    pkey::{PKey, Private},
    rsa::Rsa,
    x509::{
        extension::{BasicConstraints, ExtendedKeyUsage, KeyUsage, SubjectAlternativeName},
        X509Builder, X509NameBuilder, X509,
    },
};
use tempfile::TempDir;

/// Container for generated test certificates
pub struct TestCertificates {
    /// Temporary directory containing the certificate files
    pub temp_dir: TempDir,
    /// Path to CA certificate
    pub ca_cert_path: PathBuf,
    /// Path to CA private key
    pub ca_key_path: PathBuf,
    /// Path to server certificate
    pub server_cert_path: PathBuf,
    /// Path to server private key
    pub server_key_path: PathBuf,
    /// Path to client certificate
    pub client_cert_path: PathBuf,
    /// Path to client private key
    pub client_key_path: PathBuf,
}

impl TestCertificates {
    /// Generate a complete set of test certificates for mTLS testing.
    ///
    /// Creates:
    /// - CA certificate and key (self-signed root CA)
    /// - Server certificate and key (signed by CA, for localhost)
    /// - Client certificate and key (signed by CA, for client authentication)
    pub fn generate() -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path();

        // Generate CA key pair
        let ca_key = generate_rsa_key()?;
        let ca_cert = generate_ca_certificate(&ca_key)?;

        // Generate server key pair and certificate
        let server_key = generate_rsa_key()?;
        let server_cert = generate_server_certificate(&server_key, &ca_cert, &ca_key)?;

        // Generate client key pair and certificate
        let client_key = generate_rsa_key()?;
        let client_cert = generate_client_certificate(&client_key, &ca_cert, &ca_key)?;

        // Write all files
        let ca_cert_path = base_path.join("ca_cert.pem");
        let ca_key_path = base_path.join("ca_key.pem");
        let server_cert_path = base_path.join("server_cert.pem");
        let server_key_path = base_path.join("server_key.pem");
        let client_cert_path = base_path.join("client_cert.pem");
        let client_key_path = base_path.join("client_key.pem");

        std::fs::write(&ca_cert_path, ca_cert.to_pem()?)?;
        std::fs::write(&ca_key_path, ca_key.private_key_to_pem_pkcs8()?)?;
        std::fs::write(&server_cert_path, server_cert.to_pem()?)?;
        std::fs::write(&server_key_path, server_key.private_key_to_pem_pkcs8()?)?;
        std::fs::write(&client_cert_path, client_cert.to_pem()?)?;
        std::fs::write(&client_key_path, client_key.private_key_to_pem_pkcs8()?)?;

        Ok(Self {
            temp_dir,
            ca_cert_path,
            ca_key_path,
            server_cert_path,
            server_key_path,
            client_cert_path,
            client_key_path,
        })
    }

    /// Get paths as string references for use with RouterConfig builder
    pub fn ca_cert_str(&self) -> &str {
        self.ca_cert_path.to_str().unwrap()
    }

    pub fn server_cert_str(&self) -> &str {
        self.server_cert_path.to_str().unwrap()
    }

    pub fn server_key_str(&self) -> &str {
        self.server_key_path.to_str().unwrap()
    }

    pub fn client_cert_str(&self) -> &str {
        self.client_cert_path.to_str().unwrap()
    }

    pub fn client_key_str(&self) -> &str {
        self.client_key_path.to_str().unwrap()
    }
}

/// Generate a 2048-bit RSA key pair
fn generate_rsa_key() -> Result<PKey<Private>, Box<dyn std::error::Error>> {
    let rsa = Rsa::generate(2048)?;
    Ok(PKey::from_rsa(rsa)?)
}

/// Generate a self-signed CA certificate
fn generate_ca_certificate(key: &PKey<Private>) -> Result<X509, Box<dyn std::error::Error>> {
    let mut name_builder = X509NameBuilder::new()?;
    name_builder.append_entry_by_text("C", "US")?;
    name_builder.append_entry_by_text("ST", "California")?;
    name_builder.append_entry_by_text("L", "Test City")?;
    name_builder.append_entry_by_text("O", "Test CA Organization")?;
    name_builder.append_entry_by_text("CN", "Test CA")?;
    let name = name_builder.build();

    let mut cert_builder = X509Builder::new()?;
    cert_builder.set_version(2)?; // X509 v3

    // Serial number
    let serial = {
        let mut bn = BigNum::new()?;
        bn.rand(128, MsbOption::MAYBE_ZERO, false)?;
        bn.to_asn1_integer()?
    };
    cert_builder.set_serial_number(&serial)?;

    cert_builder.set_subject_name(&name)?;
    cert_builder.set_issuer_name(&name)?; // Self-signed
    cert_builder.set_pubkey(key)?;

    // Validity: 1 year from now
    let not_before = Asn1Time::days_from_now(0)?;
    let not_after = Asn1Time::days_from_now(365)?;
    cert_builder.set_not_before(&not_before)?;
    cert_builder.set_not_after(&not_after)?;

    // Extensions for CA
    let basic_constraints = BasicConstraints::new().critical().ca().build()?;
    cert_builder.append_extension(basic_constraints)?;

    let key_usage = KeyUsage::new()
        .critical()
        .key_cert_sign()
        .crl_sign()
        .build()?;
    cert_builder.append_extension(key_usage)?;

    cert_builder.sign(key, MessageDigest::sha256())?;

    Ok(cert_builder.build())
}

/// Generate a server certificate signed by the CA
fn generate_server_certificate(
    key: &PKey<Private>,
    ca_cert: &X509,
    ca_key: &PKey<Private>,
) -> Result<X509, Box<dyn std::error::Error>> {
    let mut name_builder = X509NameBuilder::new()?;
    name_builder.append_entry_by_text("C", "US")?;
    name_builder.append_entry_by_text("ST", "California")?;
    name_builder.append_entry_by_text("L", "Test City")?;
    name_builder.append_entry_by_text("O", "Test Server Organization")?;
    name_builder.append_entry_by_text("CN", "localhost")?;
    let name = name_builder.build();

    let mut cert_builder = X509Builder::new()?;
    cert_builder.set_version(2)?;

    let serial = {
        let mut bn = BigNum::new()?;
        bn.rand(128, MsbOption::MAYBE_ZERO, false)?;
        bn.to_asn1_integer()?
    };
    cert_builder.set_serial_number(&serial)?;

    cert_builder.set_subject_name(&name)?;
    cert_builder.set_issuer_name(ca_cert.subject_name())?;
    cert_builder.set_pubkey(key)?;

    let not_before = Asn1Time::days_from_now(0)?;
    let not_after = Asn1Time::days_from_now(365)?;
    cert_builder.set_not_before(&not_before)?;
    cert_builder.set_not_after(&not_after)?;

    // Extensions for server certificate
    let basic_constraints = BasicConstraints::new().build()?;
    cert_builder.append_extension(basic_constraints)?;

    let key_usage = KeyUsage::new()
        .critical()
        .digital_signature()
        .key_encipherment()
        .build()?;
    cert_builder.append_extension(key_usage)?;

    let ext_key_usage = ExtendedKeyUsage::new().server_auth().build()?;
    cert_builder.append_extension(ext_key_usage)?;

    // Subject Alternative Names for localhost
    let san = SubjectAlternativeName::new()
        .dns("localhost")
        .ip("127.0.0.1")
        .ip("::1")
        .build(&cert_builder.x509v3_context(Some(ca_cert), None))?;
    cert_builder.append_extension(san)?;

    cert_builder.sign(ca_key, MessageDigest::sha256())?;

    Ok(cert_builder.build())
}

/// Generate a client certificate signed by the CA
fn generate_client_certificate(
    key: &PKey<Private>,
    ca_cert: &X509,
    ca_key: &PKey<Private>,
) -> Result<X509, Box<dyn std::error::Error>> {
    let mut name_builder = X509NameBuilder::new()?;
    name_builder.append_entry_by_text("C", "US")?;
    name_builder.append_entry_by_text("ST", "California")?;
    name_builder.append_entry_by_text("L", "Test City")?;
    name_builder.append_entry_by_text("O", "Test Client Organization")?;
    name_builder.append_entry_by_text("CN", "Test Client")?;
    let name = name_builder.build();

    let mut cert_builder = X509Builder::new()?;
    cert_builder.set_version(2)?;

    let serial = {
        let mut bn = BigNum::new()?;
        bn.rand(128, MsbOption::MAYBE_ZERO, false)?;
        bn.to_asn1_integer()?
    };
    cert_builder.set_serial_number(&serial)?;

    cert_builder.set_subject_name(&name)?;
    cert_builder.set_issuer_name(ca_cert.subject_name())?;
    cert_builder.set_pubkey(key)?;

    let not_before = Asn1Time::days_from_now(0)?;
    let not_after = Asn1Time::days_from_now(365)?;
    cert_builder.set_not_before(&not_before)?;
    cert_builder.set_not_after(&not_after)?;

    // Extensions for client certificate
    let basic_constraints = BasicConstraints::new().build()?;
    cert_builder.append_extension(basic_constraints)?;

    let key_usage = KeyUsage::new().critical().digital_signature().build()?;
    cert_builder.append_extension(key_usage)?;

    let ext_key_usage = ExtendedKeyUsage::new().client_auth().build()?;
    cert_builder.append_extension(ext_key_usage)?;

    cert_builder.sign(ca_key, MessageDigest::sha256())?;

    Ok(cert_builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certificate_generation() {
        let certs = TestCertificates::generate().expect("Failed to generate certificates");

        // Verify all files exist
        assert!(certs.ca_cert_path.exists(), "CA cert should exist");
        assert!(certs.ca_key_path.exists(), "CA key should exist");
        assert!(certs.server_cert_path.exists(), "Server cert should exist");
        assert!(certs.server_key_path.exists(), "Server key should exist");
        assert!(certs.client_cert_path.exists(), "Client cert should exist");
        assert!(certs.client_key_path.exists(), "Client key should exist");

        // Verify files are not empty
        assert!(
            std::fs::metadata(&certs.ca_cert_path).unwrap().len() > 0,
            "CA cert should not be empty"
        );
        assert!(
            std::fs::metadata(&certs.server_cert_path).unwrap().len() > 0,
            "Server cert should not be empty"
        );
        assert!(
            std::fs::metadata(&certs.client_cert_path).unwrap().len() > 0,
            "Client cert should not be empty"
        );
    }
}
