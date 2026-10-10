#include "wemm/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>

#include "wemm/common.h"

namespace fs = std::filesystem;

namespace wemm {

struct SafeTensors::Mapping {
  void* data = nullptr;
  size_t size = 0;
  explicit Mapping(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    WEMM_CHECK(fd >= 0, "cannot open ", path);
    struct stat st{};
    ::fstat(fd, &st);
    size = static_cast<size_t>(st.st_size);
    data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    WEMM_CHECK(data != MAP_FAILED, "mmap failed for ", path);
    ::madvise(data, size, MADV_SEQUENTIAL);
  }
  ~Mapping() {
    if (data && data != MAP_FAILED) ::munmap(data, size);
  }
};

static at::ScalarType parse_dtype(const std::string& s) {
  if (s == "BF16") return at::kBFloat16;
  if (s == "F16") return at::kHalf;
  if (s == "F32") return at::kFloat;
  if (s == "F64") return at::kDouble;
  if (s == "I64") return at::kLong;
  if (s == "I32") return at::kInt;
  if (s == "I16") return at::kShort;
  if (s == "I8") return at::kChar;
  if (s == "U8") return at::kByte;
  if (s == "BOOL") return at::kBool;
  throw Error("unsupported safetensors dtype " + s);
}

void SafeTensors::add_file(const std::string& file) {
  auto map = std::make_shared<Mapping>(file);
  WEMM_CHECK(map->size >= 8, "truncated safetensors file ", file);
  uint64_t header_len = 0;
  std::memcpy(&header_len, map->data, 8);
  WEMM_CHECK(8 + header_len <= map->size, "bad safetensors header in ", file);
  auto header = nlohmann::json::parse(static_cast<const char*>(map->data) + 8,
                                      static_cast<const char*>(map->data) + 8 + header_len);
  const size_t base = 8 + header_len;
  for (auto& [name, info] : header.items()) {
    if (name == "__metadata__") continue;
    Entry e;
    e.file = map;
    e.dtype = parse_dtype(info.at("dtype").get<std::string>());
    e.shape = info.at("shape").get<std::vector<int64_t>>();
    auto off = info.at("data_offsets").get<std::vector<size_t>>();
    e.begin = base + off.at(0);
    e.end = base + off.at(1);
    WEMM_CHECK(e.end <= map->size, "tensor ", name, " out of bounds in ", file);
    index_[name] = std::move(e);
  }
}

SafeTensors SafeTensors::open(const std::string& path) {
  SafeTensors st;
  if (fs::is_directory(path)) {
    auto index = fs::path(path) / "model.safetensors.index.json";
    if (fs::exists(index)) {
      std::ifstream in(index);
      auto j = nlohmann::json::parse(in);
      std::set<std::string> files;
      for (auto& [_, f] : j.at("weight_map").items()) files.insert(f.get<std::string>());
      for (auto& f : files) st.add_file((fs::path(path) / f).string());
    } else {
      st.add_file((fs::path(path) / "model.safetensors").string());
    }
  } else {
    st.add_file(path);
  }
  return st;
}

at::Tensor SafeTensors::get(const std::string& name) const {
  auto it = index_.find(name);
  WEMM_CHECK(it != index_.end(), "missing tensor ", name);
  const Entry& e = it->second;
  auto* ptr = static_cast<char*>(e.file->data) + e.begin;
  auto keep = e.file;  // keep the mapping alive as long as the tensor lives
  // MAP_PRIVATE pages are copy-on-write, so exposing them as writable is safe.
  return at::from_blob(ptr, e.shape, [keep](void*) {}, at::TensorOptions().dtype(e.dtype));
}

std::vector<std::string> SafeTensors::names() const {
  std::vector<std::string> out;
  out.reserve(index_.size());
  for (auto& kv : index_) out.push_back(kv.first);
  return out;
}

}  // namespace wemm
