#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace wemm {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace detail {
template <typename... Args>
std::string concat(Args&&... args) {
  std::ostringstream os;
  (os << ... << args);
  return os.str();
}
}  // namespace detail

}  // namespace wemm

#define WEMM_CHECK(cond, ...)                                                              \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      throw ::wemm::Error(::wemm::detail::concat(__FILE__, ":", __LINE__, ": ", __VA_ARGS__)); \
    }                                                                                      \
  } while (0)
