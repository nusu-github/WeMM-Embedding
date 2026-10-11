/* Smoke test of the C ABI: wemm_test_c_api <model_dir> <image> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "wemm/c_api.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model_dir> <image>\n", argv[0]);
    return 2;
  }
  wemm_embedder* e = wemm_create(argv[1], 0, "{\"max_batch_tokens\": 16384}");
  if (!e) {
    fprintf(stderr, "create failed: %s\n", wemm_last_error());
    return 1;
  }
  char img[1024];
  snprintf(img, sizeof img, "{\"image\": \"%s\"}", argv[2]);
  const char* inputs[] = {"{\"text\": \"A dog is running on a beach.\"}", img};
  const int dim = wemm_dim(e, 256);
  float* out = malloc(sizeof(float) * 2 * dim);
  if (wemm_embed_json(e, inputs, 2, 256, out) != 0) {
    fprintf(stderr, "embed failed: %s\n", wemm_last_error());
    return 1;
  }
  double n0 = 0, n1 = 0, dot = 0;
  for (int i = 0; i < dim; ++i) {
    n0 += out[i] * out[i];
    n1 += out[dim + i] * out[dim + i];
    dot += out[i] * out[dim + i];
  }
  printf("dim=%d |e0|=%.4f |e1|=%.4f cos=%.4f\n", dim, sqrt(n0), sqrt(n1), dot);
  /* error path */
  const char* bad[] = {"{\"image\": \"/nonexistent.jpg\"}"};
  int rc = wemm_embed_json(e, bad, 1, 0, out);
  printf("bad input rc=%d error=\"%.60s...\"\n", rc, wemm_last_error());
  free(out);
  wemm_destroy(e);
  return 0;
}
