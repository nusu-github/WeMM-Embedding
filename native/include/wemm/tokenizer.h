#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wemm {

// Thin RAII wrapper over the Rust `tokenizers` crate (see tokenizers-ffi/).
class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json);
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = true) const;
  std::vector<std::vector<int32_t>> encode_batch(const std::vector<std::string>& texts,
                                                 bool add_special_tokens = true) const;
  int64_t token_to_id(const std::string& token) const;

 private:
  void* handle_ = nullptr;
};

}  // namespace wemm
