//! Minimal C ABI over the Hugging Face `tokenizers` crate.

use std::ffi::{c_char, CStr, CString};
use std::ptr;
use tokenizers::Tokenizer;

fn set_err(err: *mut *mut c_char, msg: String) {
    if !err.is_null() {
        unsafe { *err = CString::new(msg).unwrap_or_default().into_raw() };
    }
}

#[no_mangle]
pub extern "C" fn wemm_tok_from_file(path: *const c_char, err: *mut *mut c_char) -> *mut Tokenizer {
    let path = unsafe { CStr::from_ptr(path) }.to_string_lossy().into_owned();
    match Tokenizer::from_file(&path) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(e) => {
            set_err(err, format!("failed to load tokenizer {path}: {e}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn wemm_tok_free(tok: *mut Tokenizer) {
    if !tok.is_null() {
        unsafe { drop(Box::from_raw(tok)) };
    }
}

#[no_mangle]
pub extern "C" fn wemm_tok_free_str(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// Encodes `n` UTF-8 strings in parallel. On success, `out_ids[i]`/`out_lens[i]` hold a
/// buffer that must be released with `wemm_tok_free_ids`.
#[no_mangle]
pub extern "C" fn wemm_tok_encode_batch(
    tok: *const Tokenizer,
    texts: *const *const c_char,
    text_lens: *const usize,
    n: usize,
    add_special_tokens: bool,
    out_ids: *mut *mut u32,
    out_lens: *mut usize,
    err: *mut *mut c_char,
) -> i32 {
    let tok = unsafe { &*tok };
    let mut inputs = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = unsafe { std::slice::from_raw_parts(*texts.add(i) as *const u8, *text_lens.add(i)) };
        match std::str::from_utf8(bytes) {
            Ok(s) => inputs.push(s.to_owned()),
            Err(e) => {
                set_err(err, format!("input {i} is not valid UTF-8: {e}"));
                return -1;
            }
        }
    }
    let encodings = match tok.encode_batch(inputs, add_special_tokens) {
        Ok(e) => e,
        Err(e) => {
            set_err(err, format!("encode failed: {e}"));
            return -1;
        }
    };
    for (i, enc) in encodings.into_iter().enumerate() {
        let ids: Box<[u32]> = enc.get_ids().to_vec().into_boxed_slice();
        unsafe {
            *out_lens.add(i) = ids.len();
            *out_ids.add(i) = Box::into_raw(ids) as *mut u32;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn wemm_tok_free_ids(ids: *mut u32, len: usize) {
    if !ids.is_null() {
        unsafe { drop(Box::from_raw(ptr::slice_from_raw_parts_mut(ids, len))) };
    }
}

/// Returns -1 when the token is unknown.
#[no_mangle]
pub extern "C" fn wemm_tok_token_to_id(tok: *const Tokenizer, token: *const c_char) -> i64 {
    let tok = unsafe { &*tok };
    let token = unsafe { CStr::from_ptr(token) }.to_string_lossy();
    tok.token_to_id(&token).map(|v| v as i64).unwrap_or(-1)
}
