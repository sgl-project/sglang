//! Memory management for FFI functions

use std::ffi::CString;
use std::os::raw::c_char;

/// Free a C string allocated by Rust
///
/// # Safety
/// This function must only be called with pointers returned by other FFI functions.
/// Calling with arbitrary pointers or multiple times on the same pointer is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn sgl_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// Free token IDs array allocated by Rust
///
/// # Safety
/// This function must only be called with pointers returned by `sgl_tokenizer_encode`.
/// The `count` parameter must match the length of the array.
#[no_mangle]
pub unsafe extern "C" fn sgl_free_token_ids(ptr: *mut u32, count: usize) {
    if !ptr.is_null() && count > 0 {
        let _ = Vec::from_raw_parts(ptr, count, count);
    }
}
