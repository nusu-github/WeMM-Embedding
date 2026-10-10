#include "wemm/tokenizer.h"

#include "wemm/common.h"

extern "C" {
void* wemm_tok_from_file(const char* path, char** err);
void wemm_tok_free(void* tok);
void wemm_tok_free_str(char* s);
int wemm_tok_encode_batch(const void* tok, const char* const* texts, const size_t* lens, size_t n,
                          bool add_special_tokens, uint32_t** out_ids, size_t* out_lens, char** err);
void wemm_tok_free_ids(uint32_t* ids, size_t len);
int64_t wemm_tok_token_to_id(const void* tok, const char* token);
}

namespace wemm {

namespace {
[[noreturn]] void raise(char* err) {
  std::string msg = err ? err : "unknown tokenizer error";
  wemm_tok_free_str(err);
  throw Error(msg);
}
}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
  char* err = nullptr;
  handle_ = wemm_tok_from_file(path.c_str(), &err);
  if (!handle_) raise(err);
}

Tokenizer::~Tokenizer() { wemm_tok_free(handle_); }

std::vector<std::vector<int32_t>> Tokenizer::encode_batch(const std::vector<std::string>& texts,
                                                          bool add_special_tokens) const {
  const size_t n = texts.size();
  std::vector<const char*> ptrs(n);
  std::vector<size_t> lens(n), out_lens(n);
  std::vector<uint32_t*> out_ids(n, nullptr);
  for (size_t i = 0; i < n; ++i) {
    ptrs[i] = texts[i].data();
    lens[i] = texts[i].size();
  }
  char* err = nullptr;
  if (wemm_tok_encode_batch(handle_, ptrs.data(), lens.data(), n, add_special_tokens, out_ids.data(),
                            out_lens.data(), &err) != 0)
    raise(err);
  std::vector<std::vector<int32_t>> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i].assign(out_ids[i], out_ids[i] + out_lens[i]);
    wemm_tok_free_ids(out_ids[i], out_lens[i]);
  }
  return out;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_special_tokens) const {
  return encode_batch({text}, add_special_tokens)[0];
}

int64_t Tokenizer::token_to_id(const std::string& token) const {
  return wemm_tok_token_to_id(handle_, token.c_str());
}

}  // namespace wemm
