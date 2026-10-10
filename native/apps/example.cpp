// Minimal C++ API example (mirrors the README).
//   wemm_example <model_dir> <image> <video>
#include <c10/core/InferenceMode.h>

#include <iostream>

#include "wemm/embedder.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " <model_dir> <image> <video>\n";
    return 2;
  }
  c10::InferenceMode guard;
  wemm::Embedder emb(argv[1]);

  at::Tensor t = emb.embed_texts({"A dog is running on a beach."});  // [1, hidden], L2-normalized
  at::Tensor i = emb.embed_images({argv[2]}, /*dimension=*/256);     // Matryoshka prefix

  wemm::VideoOptions vo;
  vo.nframes = 64;
  std::vector<wemm::Conversation> convs = {
      {{"user", {wemm::ContentItem::Image(argv[2]), wemm::ContentItem::Text("Find a similar scene.")}}},
      {{"user", {wemm::ContentItem::Video(argv[3], vo)}}},
  };
  at::Tensor e = emb.embed(convs);

  std::cout << "text  " << t.sizes() << "\n"
            << "image " << i.sizes() << "\n"
            << "conv  " << e.sizes() << "\n"
            << "cos(text, image+text) = " << (t[0] * e[0]).sum().item<float>() << "\n";
  return 0;
}
