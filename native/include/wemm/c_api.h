/* C ABI for embedding WeMM-Embedding into other languages (Rust, Go, C#, ...).
 *
 *   wemm_embedder* e = wemm_create("/models/WeMM-Embedding-2B", -1, NULL);
 *   const char* inputs[] = {"{\"text\": \"a dog\"}", "{\"image\": \"cat.jpg\"}"};
 *   int dim = wemm_dim(e, 0);
 *   float* out = malloc(sizeof(float) * 2 * dim);
 *   if (wemm_embed_json(e, inputs, 2, 0, out) != 0) puts(wemm_last_error());
 *   wemm_destroy(e);
 *
 * Inputs use the same JSON forms as the wemm_embed CLI: HF-style message lists
 * or the {"text"|"image"|"video": ...} shorthand. Not thread-safe per handle. */
#ifndef WEMM_C_API_H
#define WEMM_C_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wemm_embedder wemm_embedder;

/* device: CUDA device index (-1 = 0). options_json may be NULL or e.g.
 * {"max_batch_tokens": 32768, "preprocess_threads": 8, "compute_dtype": "fp16",
 *  "fp32_residual": true, "embeddings_on_gpu": false, "compress_weights": false}.
 * Returns NULL on failure (see wemm_last_error). */
wemm_embedder* wemm_create(const char* model_dir, int device, const char* options_json);
void wemm_destroy(wemm_embedder* e);

/* Embedding width for a Matryoshka dimension (0 = full); 0 if invalid. */
int wemm_dim(const wemm_embedder* e, int dimension);

/* Embeds n inputs into out (n * wemm_dim(e, dimension) floats, row-major,
 * L2-normalized). Returns 0 on success. */
int wemm_embed_json(wemm_embedder* e, const char* const* inputs_json, size_t n, int dimension, float* out);

/* Last error message of the calling thread ("" if none). */
const char* wemm_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* WEMM_C_API_H */
