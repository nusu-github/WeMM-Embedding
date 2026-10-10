// Plain-ATen reference implementation (ModelOptions::fused_kernels = false).
// Mirrors the Transformers modeling code op by op (fp32 where it computes in
// fp32) and serves as the oracle for the fused kernels. Slower, more memory.

#include <cmath>

#include "model_internal.h"
#include "wemm/common.h"
#include "wemm/model.h"

namespace wemm {

using namespace detail;

namespace {

at::Tensor ref_rmsnorm(const at::Tensor& x, const at::Tensor& w1p, double eps, at::ScalarType out) {
  at::Tensor xf = x.to(at::kFloat);
  return (xf * at::rsqrt(xf.pow(2).mean(-1, true) + eps) * w1p).to(out);
}

at::Tensor ref_rotate_half(const at::Tensor& x) {
  const int64_t h = x.size(-1) / 2;
  return at::cat({-x.narrow(-1, h, h), x.narrow(-1, 0, h)}, -1);
}

at::Tensor ref_layernorm(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, at::ScalarType out) {
  return at::layer_norm(x.to(at::kFloat), {x.size(-1)}, w, b, 1e-6).to(out);
}

}  // namespace

// ==================================================================== ViT ====
at::Tensor WeMMModel::vision_reference(const at::Tensor& pixel_values, const VisionPrelude& vp) {
  const auto& vc = cfg_.vision;
  const int64_t m = vc.spatial_merge_size, D = vc.hidden_size, nh = vc.num_heads, hd = D / nh;
  const int64_t P = pixel_values.size(0);
  at::Tensor x = at::linear(pixel_values.to(act_), patch_w, patch_b.to(act_));  // Conv3d == linear
  x = (x.to(at::kFloat) + vp.pos_embeds).to(act_);
  at::Tensor cos = vp.cos.unsqueeze(1), sin = vp.sin.unsqueeze(1);
  for (const auto& b : vblocks_) {
    at::Tensor y = ref_layernorm(x, b.norm1_w, b.norm1_b, act_);
    at::Tensor qkv = at::linear(y, b.qkv_w, b.qkv_b).view({P, 3, nh, hd});
    at::Tensor q = qkv.select(1, 0).to(at::kFloat), k = qkv.select(1, 1).to(at::kFloat);
    q = (q * cos + ref_rotate_half(q) * sin).to(act_);
    k = (k * cos + ref_rotate_half(k) * sin).to(act_);
    at::Tensor v = qkv.select(1, 2).contiguous();
    at::Tensor o = attention(q, k, v, vp.cu_seqlens, vp.cu_seqlens, vp.max_len, vp.max_len, false);
    x = x + at::linear(o.reshape({P, D}), b.proj_w, b.proj_b.to(act_));
    y = ref_layernorm(x, b.norm2_w, b.norm2_b, act_);
    x = x + at::linear(at::gelu(at::linear(y, b.fc1_w, b.fc1_b), "tanh"), b.fc2_w, b.fc2_b.to(act_));
  }
  at::Tensor y = ref_layernorm(x, merger_norm_w, merger_norm_b, act_).view({P / (m * m), D * m * m});
  return at::linear(at::gelu(at::linear(y, merger_fc1_w, merger_fc1_b)), merger_fc2_w, merger_fc2_b);
}

// ===================================================================== LM ====
at::Tensor WeMMModel::text_reference(at::Tensor h, const Packing& pk) {
  const auto& tc = cfg_.text;
  for (size_t i = 0; i < layers_.size(); ++i) {
    const auto& l = layers_[i];
    const bool last = i + 1 == layers_.size();
    at::Tensor x = ref_rmsnorm(h, l.in_norm, tc.rms_norm_eps, act_);
    at::Tensor mix;
    if (l.linear) {
      mix = linear_attention_reference(l, x, pk);
      if (last) mix = mix.index_select(0, pk.last_rows);
    } else {
      mix = full_attention_reference(l, x, pk, last);
    }
    if (last) h = h.index_select(0, pk.last_rows);  // only the <embedding> rows are needed from here on
    h = h + mix;
    at::Tensor gu = at::linear(ref_rmsnorm(h, l.post_norm, tc.rms_norm_eps, act_), l.gate_up);
    const int64_t I = gu.size(1) / 2;
    at::Tensor act = (at::silu(gu.narrow(1, 0, I).to(at::kFloat)) * gu.narrow(1, I, I).to(at::kFloat)).to(act_);
    h = h + at::linear(act, l.down);
  }
  return ref_rmsnorm(h, final_norm, tc.rms_norm_eps, at::kFloat);
}

at::Tensor WeMMModel::linear_attention_reference(const TextLayer& l, const at::Tensor& x, const Packing& pk) {
  const auto& tc = cfg_.text;
  const int64_t L = x.size(0);
  const int64_t Hk = tc.linear_num_key_heads, Hv = tc.linear_num_value_heads;
  const int64_t dk = tc.linear_key_head_dim, dv = tc.linear_value_head_dim;
  const int64_t key_dim = Hk * dk, value_dim = Hv * dv, conv_dim = 2 * key_dim + value_dim;

  at::Tensor proj = at::linear(x, l.in_proj);
  // Per-sequence depthwise causal conv in fp32, single rounding.
  at::Tensor xs = proj.narrow(1, 0, conv_dim).to(at::kFloat);
  std::vector<at::Tensor> outs;
  int64_t off = 0;
  for (int64_t len : pk.seq_lens) {
    at::Tensor xi = xs.narrow(0, off, len).t().unsqueeze(0);  // [1, C, len]
    at::Tensor yi = at::conv1d(xi, l.conv_w.unsqueeze(1), {}, 1, l.conv_w.size(1) - 1, 1, conv_dim);
    outs.push_back(at::silu(yi.narrow(2, 0, len)).squeeze(0).t());
    off += len;
  }
  at::Tensor mixed = at::cat(outs, 0).to(act_);
  at::Tensor z = proj.narrow(1, conv_dim, value_dim);
  at::Tensor b = proj.narrow(1, conv_dim + value_dim, Hv);
  at::Tensor a = proj.narrow(1, conv_dim + value_dim + Hv, Hv);

  at::Tensor q = mixed.narrow(1, 0, key_dim).view({L, Hk, dk});
  at::Tensor k = mixed.narrow(1, key_dim, key_dim).view({L, Hk, dk});
  at::Tensor v = mixed.narrow(1, 2 * key_dim, value_dim).view({L, Hv, dv});
  at::Tensor beta = at::sigmoid(b.to(at::kFloat));
  at::Tensor g = l.neg_A * at::softplus(a.to(at::kFloat) + l.dt_bias);
  if (Hv / Hk > 1) {
    q = q.repeat_interleave(Hv / Hk, 1);
    k = k.repeat_interleave(Hv / Hk, 1);
  }
  at::Tensor o = gated_delta_rule_reference(q, k, v, g, beta, pk);  // [L, Hv, dv] fp32
  // Qwen3_5RMSNormGated: norm(o) * w * silu(z)
  at::Tensor zf = z.reshape({L, Hv, dv}).to(at::kFloat);
  o = o * at::rsqrt(o.pow(2).mean(-1, true) + tc.rms_norm_eps) * l.gnorm_w * at::silu(zf);
  return at::linear(o.reshape({L, value_dim}).to(act_), l.out_proj);
}

// Chunked algorithm of torch_chunk_gated_delta_rule (fp32), batched over all
// chunks; the triangular solve replaces the row-by-row substitution loop.
at::Tensor WeMMModel::gated_delta_rule_reference(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                 const at::Tensor& g, const at::Tensor& beta, const Packing& pk) {
  const int64_t L = q.size(0), H = q.size(1), dk = q.size(2), dv = v.size(2);
  const int64_t C = int64_t(pk.chunk_start_h.size());
  // Chunk-padded layout: row index of every chunk slot (L = an appended zero row).
  std::vector<int64_t> gather(size_t(C * kChunk), L);
  for (int64_t c = 0; c < C; ++c)
    for (int64_t j = 0; j < pk.chunk_valid_h[size_t(c)]; ++j) gather[size_t(c * kChunk + j)] = pk.chunk_start_h[size_t(c)] + j;
  at::Tensor gidx = to_device_i64(gather, device_);

  auto l2norm = [](const at::Tensor& x) {
    at::Tensor xf = x.to(at::kFloat);
    return xf * at::rsqrt((xf * xf).sum(-1, true) + 1e-6);
  };
  auto pad_rows = [](const at::Tensor& x) { return at::cat({x, at::zeros_like(x.narrow(0, 0, 1))}, 0); };
  auto to_chunks = [&](const at::Tensor& x) {  // [L, H, d] -> [H, C, 64, d]
    return pad_rows(x).index_select(0, gidx).view({C, kChunk, H, -1}).permute({2, 0, 1, 3}).contiguous();
  };
  auto to_chunks2 = [&](const at::Tensor& x) {  // [L, H] -> [H, C, 64]
    return pad_rows(x).index_select(0, gidx).view({C, kChunk, H}).permute({2, 0, 1}).contiguous();
  };

  at::Tensor Q = to_chunks(l2norm(q) * (1.0 / std::sqrt(double(dk))));
  at::Tensor K = to_chunks(l2norm(k));
  at::Tensor V = to_chunks(v.to(at::kFloat));
  at::Tensor gc = to_chunks2(g).cumsum(-1);
  at::Tensor B = to_chunks2(beta);

  at::Tensor eg = gc.exp();
  at::Tensor kb = K * B.unsqueeze(-1);
  at::Tensor vb = V * B.unsqueeze(-1);
  at::Tensor lower = at::ones({kChunk, kChunk}, at::TensorOptions().dtype(at::kBool).device(device_)).tril();
  at::Tensor decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(lower.logical_not(), 0).exp() * lower;
  at::Tensor A = at::matmul(kb, K.transpose(-1, -2)) * decay;
  A.diagonal(0, -2, -1).fill_(1);  // I + strictly-lower(A)
  at::Tensor rhs = at::cat({vb, kb * eg.unsqueeze(-1)}, -1);
  at::Tensor sol = at::linalg_solve_triangular(A, rhs, /*upper=*/false, /*left=*/true, /*unitriangular=*/true);
  at::Tensor u = sol.narrow(-1, 0, dv), w = sol.narrow(-1, dv, dk);
  at::Tensor aqk = at::matmul(Q, K.transpose(-1, -2)) * decay;
  at::Tensor glast = gc.select(-1, kChunk - 1);
  at::Tensor qg = Q * eg.unsqueeze(-1);
  at::Tensor kd = K * (glast.unsqueeze(-1) - gc).exp().unsqueeze(-1);
  at::Tensor dl = glast.exp();

  at::Tensor o = at::empty({H, C, kChunk, dv}, Q.options());
  for (size_t s = 0; s < pk.seq_lens.size(); ++s) {
    at::Tensor S = at::zeros({H, dk, dv}, Q.options());
    for (int64_t c = pk.chunk_off_h[s]; c < pk.chunk_off_h[s + 1]; ++c) {
      at::Tensor vnew = u.select(1, c) - at::matmul(w.select(1, c), S);
      o.select(1, c).copy_(at::matmul(qg.select(1, c), S) + at::matmul(aqk.select(1, c), vnew));
      S = S * dl.select(1, c).view({H, 1, 1}) + at::matmul(kd.select(1, c).transpose(-1, -2), vnew);
    }
  }
  at::Tensor flat = o.permute({1, 2, 0, 3}).reshape({C * kChunk, H, dv});
  return flat.index_select(0, pk.padded_row.to(at::kLong));  // [L, H, dv] fp32
}

at::Tensor WeMMModel::full_attention_reference(const TextLayer& l, const at::Tensor& x, const Packing& pk, bool last) {
  const auto& tc = cfg_.text;
  const int64_t L = x.size(0), H = tc.num_attention_heads, Hkv = tc.num_key_value_heads, hd = tc.head_dim;
  const int64_t rot = pk.cos.size(-1);
  at::Tensor proj = at::linear(x, l.qkv_proj);
  at::Tensor qg = proj.narrow(1, 0, H * hd * 2).view({L, H, 2 * hd});
  at::Tensor q = qg.narrow(2, 0, hd), gate = qg.narrow(2, hd, hd);
  at::Tensor k = proj.narrow(1, H * hd * 2, Hkv * hd).view({L, Hkv, hd});
  at::Tensor v = proj.narrow(1, H * hd * 2 + Hkv * hd, Hkv * hd).view({L, Hkv, hd}).contiguous();
  at::Tensor qcos = pk.cos, qsin = pk.sin;
  if (last) {
    q = q.index_select(0, pk.last_rows);
    gate = gate.index_select(0, pk.last_rows);
    qcos = qcos.index_select(0, pk.last_rows);
    qsin = qsin.index_select(0, pk.last_rows);
  }
  auto norm_rope = [&](const at::Tensor& t, const at::Tensor& w, const at::Tensor& c, const at::Tensor& s) {
    at::Tensor f = ref_rmsnorm(t, w, tc.rms_norm_eps, at::kFloat);
    at::Tensor r = f.narrow(-1, 0, rot);
    r = r * c.unsqueeze(1) + ref_rotate_half(r) * s.unsqueeze(1);
    return at::cat({r, f.narrow(-1, rot, hd - rot)}, -1).to(act_);
  };
  q = norm_rope(q, l.q_norm, qcos, qsin);
  k = norm_rope(k, l.k_norm, pk.cos, pk.sin);
  at::Tensor o;
  if (last) {
    const int64_t N = q.size(0);
    at::Tensor cu_q = at::arange(0, N + 1, at::TensorOptions().dtype(at::kInt).device(device_));
    o = attention(q, k, v, cu_q, pk.cu_seqlens, 1, pk.max_len, /*causal=*/false);
  } else {
    o = attention(q, k, v, pk.cu_seqlens, pk.cu_seqlens, pk.max_len, pk.max_len, /*causal=*/true);
  }
  o = (o.to(at::kFloat) * at::sigmoid(gate.to(at::kFloat))).to(act_);
  return at::linear(o.reshape({o.size(0), H * hd}), l.o_proj);
}

}  // namespace wemm
