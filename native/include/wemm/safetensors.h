#pragma once

#include <ATen/ATen.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace wemm {

// Memory-mapped reader for one or more .safetensors files. Tensors returned by
// `get` are zero-copy CPU views into the mapping and stay valid while the
// reader is alive.
class SafeTensors {
 public:
  // `path` is either a single .safetensors file or a model directory holding
  // model.safetensors / model.safetensors.index.json.
  static SafeTensors open(const std::string& path);

  bool contains(const std::string& name) const { return index_.count(name) != 0; }
  at::Tensor get(const std::string& name) const;
  std::vector<std::string> names() const;

 private:
  struct Mapping;
  struct Entry {
    std::shared_ptr<Mapping> file;
    at::ScalarType dtype;
    std::vector<int64_t> shape;
    size_t begin, end;
  };
  void add_file(const std::string& file);
  std::unordered_map<std::string, Entry> index_;
};

}  // namespace wemm
