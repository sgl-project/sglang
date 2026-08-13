//! Prost messages, Tonic clients and servers, and descriptors for the SGLang runtime API.

tonic::include_proto!("sglang.runtime.v1");

/// Encoded `google.protobuf.FileDescriptorSet` for the SGLang runtime API.
pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("sglang_descriptor");
