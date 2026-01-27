//! Error handling for FFI functions

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

/// Error codes returned by FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SglErrorCode {
    Success = 0,
    InvalidArgument = 1,
    TokenizationError = 2,
    ParsingError = 3,
    MemoryError = 4,
    UnknownError = 99,
}

/// Helper to set error message in FFI output parameter
pub fn set_error_message(error_out: *mut *mut c_char, message: &str) {
    unsafe {
        if !error_out.is_null() {
            if let Ok(cstr) = CString::new(message) {
                *error_out = cstr.into_raw();
            } else {
                *error_out = ptr::null_mut();
            }
        }
    }
}

/// Helper to set error message from format string
pub fn set_error_message_fmt(error_out: *mut *mut c_char, fmt: std::fmt::Arguments) {
    if !error_out.is_null() {
        let msg = format!("{}", fmt);
        set_error_message(error_out, &msg);
    }
}

/// Helper to clear error message
pub fn clear_error_message(error_out: *mut *mut c_char) {
    unsafe {
        if !error_out.is_null() {
            *error_out = ptr::null_mut();
        }
    }
}

// Helper functions for error handling
// Note: Some helper functions are kept for potential future use
