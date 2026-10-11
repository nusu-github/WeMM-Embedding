// Command-line embedding tool.
//
//   wemm_embed --model DIR [--dimension D] [--output out.npy|out.jsonl] [--batch-tokens N]
//              (--text "..." | --image PATH | --video PATH | --input inputs.jsonl)...
//
// inputs.jsonl: one JSON per line, either HF-style messages
//   [{"role":"user","content":[{"type":"image","image":"a.jpg"},{"type":"text","text":"..."}]}]
// or a shorthand {"text": "..."} / {"image": "a.jpg"} / {"video": "v.mp4", ...}.

#include <c10/core/InferenceMode.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "wemm/embedder.h"
#include "wemm/json_input.h"

using namespace wemm;

static void write_npy(const std::string& path, const at::Tensor& t) {
  std::ofstream f(path, std::ios::binary);
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(t.size(0)) + ", " +
                       std::to_string(t.size(1)) + "), }";
  while ((10 + header.size() + 1) % 64) header += ' ';
  header += '\n';
  const char magic[] = "\x93NUMPY\x01\x00";
  f.write(magic, 8);
  uint16_t len = uint16_t(header.size());
  f.write(reinterpret_cast<const char*>(&len), 2);
  f.write(header.data(), std::streamsize(header.size()));
  at::Tensor c = t.contiguous();
  f.write(reinterpret_cast<const char*>(c.data_ptr<float>()), std::streamsize(c.numel() * sizeof(float)));
}

int main(int argc, char** argv) {
  std::string model, output;
  std::optional<int64_t> dim;
  int repeat = 1;
  std::vector<Conversation> inputs;
  EmbedderOptions opts;
  try {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        WEMM_CHECK(i + 1 < argc, "missing value for ", a);
        return argv[++i];
      };
      if (a == "--model") model = next();
      else if (a == "--dimension") dim = std::stoll(next());
      else if (a == "--output") output = next();
      else if (a == "--batch-tokens") opts.max_batch_tokens = std::stoll(next());
      else if (a == "--threads") opts.preprocess_threads = std::stoi(next());
      else if (a == "--repeat") repeat = std::max(1, std::stoi(next()));  // timing: report the last run
      else if (a == "--device") opts.model.device_index = std::stoi(next());
      else if (a == "--compress-weights") opts.model.compress_weights = true;
      else if (a == "--text") inputs.push_back(text_conversation(next()));
      else if (a == "--image") inputs.push_back(image_conversation(next()));
      else if (a == "--video") inputs.push_back(video_conversation(next()));
      else if (a == "--input") {
        std::ifstream in(next());
        WEMM_CHECK(in.good(), "cannot open input file");
        for (std::string line; std::getline(in, line);)
          if (!line.empty()) inputs.push_back(conversation_from_json(nlohmann::json::parse(line)));
      } else if (a == "-h" || a == "--help") {
        std::cout << "usage: wemm_embed --model DIR [--dimension D] [--output out.npy|out.jsonl]\n"
                     "                  (--text T | --image P | --video P | --input in.jsonl)...\n";
        return 0;
      } else {
        throw Error("unknown argument " + a);
      }
    }
    WEMM_CHECK(!model.empty(), "--model is required");
    WEMM_CHECK(!inputs.empty(), "no inputs given");

    c10::InferenceMode guard;
    auto t0 = std::chrono::steady_clock::now();
    Embedder emb(model, opts);
    auto t1 = std::chrono::steady_clock::now();
    at::Tensor out;
    for (int r = 0; r < repeat; ++r) {
      t1 = std::chrono::steady_clock::now();
      out = emb.embed(inputs, dim);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t2 - t1).count();
    std::fprintf(stderr, "loaded in %.2fs, embedded %zu inputs in %.3fs (%.1f inputs/s)\n",
                 std::chrono::duration<double>(t1 - t0).count(), inputs.size(), secs, inputs.size() / secs);

    if (output.size() > 4 && output.substr(output.size() - 4) == ".npy") {
      write_npy(output, out);
    } else {
      std::ofstream file;
      if (!output.empty()) file.open(output);
      std::ostream& os = output.empty() ? std::cout : file;
      for (int64_t i = 0; i < out.size(0); ++i) {
        nlohmann::json j = std::vector<float>(out[i].data_ptr<float>(), out[i].data_ptr<float>() + out.size(1));
        os << j.dump() << "\n";
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
