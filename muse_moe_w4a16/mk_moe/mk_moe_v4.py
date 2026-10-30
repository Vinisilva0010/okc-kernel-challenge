"""Marlin-layout NVFP4 MoE for Muse Spark (hidden 4096, intermediate 4096, top-8, 32 local experts, M <= 256).

Single source of truth: tools/build_sub_mk3.py pastes this file into the phase-1 check file and copies it unchanged to
python/sglang/srt/layers/moe/muse_mk_moe.py. Weights are SGLang's Marlin tensors as they are (no second copy).
v4: experts with >= 2 token groups (M > 16 only) run on mkh_k, which dequantizes every weight tile ONCE per CTA for up to 4 token groups (4 producer warps, k-steps interleaved); results are bit-identical to v2 at equal KS.
Path: route (GPU) -> gather -> GEMM1 -> SiLU(gate) * up -> GEMM2 -> fused post-expert norm + weighted sum.
Numerics follow stock Marlin's write-out: bf16(bf16(acc) * global_scale_bf16) per GEMM (the raw fp32 accumulator is rounded to bf16, then multiplied by the bf16 global scale and rounded again); SiLU * up is one fp32 expression with fast-math exp and one bf16 rounding,
the post-norm runs per slot in fp32 in ascending slot order. Split-K is deterministic: slice s writes its own fp32 partial (no atomics, no zeroing)
and the consumer (activation / epilogue) adds the partials in slice order 0..KS-1.
Graph safe: buffers are allocated once in alloc(), group counts live in device memory, grids depend on M only.
The Marlin dequant bit tricks follow sglang kernels/jit/csrc/gemm/marlin/dequant.h (Apache-2.0).
"""

import hashlib

import torch

MK_TOP_K = 8
MK_HIDDEN = 4096
MK_INTER = 4096
MK_MAX_TOKENS = 256
MK_EXPERTS = 32

_MK_CPP = r"""#include <torch/extension.h>
void mkp_init();
std::vector<int64_t> mkp_attrs();
void mkp_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctl, torch::Tensor gtab, torch::Tensor htab, torch::Tensor slot_of_row, torch::Tensor row_of_slot, torch::Tensor exp_of_row, int64_t n_slots, int64_t lo, int64_t hi, int64_t hot_min);
void mkp_hot(torch::Tensor W, torch::Tensor Sc, torch::Tensor X, torch::Tensor htab, torch::Tensor ctl, torch::Tensor out, int64_t max_chunks, int64_t K, int64_t N, int64_t KS, int64_t ksw);
void mkp_reduce(torch::Tensor part, torch::Tensor y, torch::Tensor ctl, int64_t max_rows, int64_t ks, int64_t ksw, torch::Tensor exp_of_row, torch::Tensor gs);
void mkp_epi(torch::Tensor y, torch::Tensor row_of_slot, torch::Tensor topk_w, torch::Tensor norm_w, torch::Tensor out, int64_t m, double eps, int64_t wf);
void mkp_gather(torch::Tensor x, torch::Tensor xg, torch::Tensor slot_of_row, torch::Tensor ctl, int64_t max_rows);
void mkp_act(torch::Tensor part, torch::Tensor inter, torch::Tensor ctl, int64_t max_rows, int64_t ks, int64_t ksw, torch::Tensor exp_of_row, torch::Tensor gs);
void mkp_gemm(torch::Tensor W, torch::Tensor Sc, torch::Tensor X, torch::Tensor gtab, torch::Tensor ctl, torch::Tensor out, int64_t max_groups, int64_t K, int64_t N, int64_t KS, int64_t ksw);
"""

_MK_CUDA = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

__device__ __forceinline__ void mk_dq(uint32_t q, __nv_bfloat162 (&fb)[2]) {
    const uint32_t MASK = 0x70007000u;
    uint32_t o1 = (q & 0x80008000u) | ((q & MASK) >> 6);
    q <<= 4;
    uint32_t o2 = (q & 0x80008000u) | ((q & MASK) >> 6);
    fb[1] = *reinterpret_cast<const __nv_bfloat162*>(&o1);
    fb[0] = *reinterpret_cast<const __nv_bfloat162*>(&o2);
}

__device__ __forceinline__ void mk_dqs(uint32_t q, __nv_bfloat162 (&fb)[2]) {
    const uint32_t MASK = 0x7F007F00u;
    uint32_t o1 = ((q & 0x80008000u) >> 1) | ((q & MASK) >> 4);
    q <<= 8;
    uint32_t o2 = ((q & 0x80008000u) >> 1) | ((q & MASK) >> 4);
    fb[1] = *reinterpret_cast<const __nv_bfloat162*>(&o1);
    fb[0] = *reinterpret_cast<const __nv_bfloat162*>(&o2);
}

__device__ __forceinline__ uint32_t mk_bits(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }

__device__ __forceinline__ void mk_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t mk_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mk_cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void mk_cp16pf(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void mk_cp16z(uint32_t dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void mk_cp8(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void mk_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int NW>
__device__ __forceinline__ void mk_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(NW) : "memory"); }
__device__ __forceinline__ uint4 mk_lds128(uint32_t a) {
    uint4 r;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a) : "memory");
    return r;
}
__device__ __forceinline__ uint2 mk_lds64(uint32_t a) {
    uint2 r;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "r"(a) : "memory");
    return r;
}
__device__ __forceinline__ void mk_ldsm4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void mk_fold(float& f, uint32_t v) { f = __uint_as_float(__float_as_uint(f) ^ v); }

// mk6: NT 1, the A ring is shared by the 4 warps of a CTA (they share g, ks, e, t0, nk: ntiles % 4 == 0). Each warp copies P/4 of every
// group's A tiles; one __syncthreads per group of P k-steps.
template <int G, int P>
__global__ void __launch_bounds__(128, 1) mkg_k(const uint4* __restrict__ W, const uint2* __restrict__ Sc,
                                                const __nv_bfloat16* __restrict__ X, const int* __restrict__ gtab, float* __restrict__ out,
                                                const int* __restrict__ ctl, int K, int N, int KS, size_t ksw) {
    extern __shared__ __align__(16) uint8_t mk3_smem[];
    constexpr int S = G * P;
    constexpr int PA = P / 4;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int item = blockIdx.x * 4 + warp;
    const int ntiles = N >> 6;
    if (item >= ctl[1] * KS * ntiles) return;  // CTA-uniform: items per CTA never straddle (ntiles % 4 == 0)
    const int nt = item % ntiles;
    const int rr = item / ntiles;
    const int ks = rr % KS;
    const int g = rr / KS;
    const int e = gtab[3 * g], t0 = gtab[3 * g + 1], nk = gtab[3 * g + 2];
    const int ktt = K >> 4, kper = ktt / KS, kt0 = ks * kper;
    const int gq = lane >> 2, cq = lane & 3;
    const size_t wst = (size_t)(N >> 1), sst = (size_t)(N >> 3);
    const uint32_t sbase = mk_smem_u32(mk3_smem);
    const uint32_t sW = sbase + (uint32_t)warp * (uint32_t)(S * 576);
    const uint32_t sA = sbase + (uint32_t)(4 * S * 576);
    const uint4* wp = W + ((size_t)e * ktt + kt0) * wst + nt * 32 + lane;
    const uint2* sp = Sc + ((size_t)e * ktt + kt0) * sst + nt * 8 + (lane & 7);
    const int arow = lane >> 1, akh = lane & 1;
    const bool av = arow < nk;
    const __nv_bfloat16* ap = X + (size_t)(t0 + (av ? arow : 0)) * K + (size_t)kt0 * 16 + akh * 8;
    const uint32_t a_dst = (uint32_t)(arow * 32 + ((akh ^ ((arow >> 2) & 1)) * 16));
    const int lr = lane & 15, lk = lane >> 4;
    const uint32_t a_ld = (uint32_t)(lr * 32 + ((lk ^ ((lr >> 2) & 1)) * 16));

#define MKG_ISSUE(GX)                                                                                  \
    {                                                                                                  \
        _Pragma("unroll") for (int p = 0; p < P; ++p) {                                                \
            const int kx = (GX) * P + p;                                                               \
            const uint32_t base_ = sW + (uint32_t)(kx % S) * 576;                                      \
            mk_cp16(base_ + lane * 16, wp + (size_t)kx * wst);                                         \
            if (lane < 8) mk_cp8(base_ + 512 + lane * 8, sp + (size_t)kx * sst);                       \
        }                                                                                              \
        _Pragma("unroll") for (int pa = 0; pa < PA; ++pa) {                                            \
            const int kx = (GX) * P + warp * PA + pa;                                                  \
            mk_cp16z(sA + (uint32_t)(kx % S) * 512 + a_dst, ap + (size_t)kx * 16, av ? 16 : 0);        \
        }                                                                                              \
    }

    const int ngrp = kper / P;
#pragma unroll
    for (int gi = 0; gi < G - 1; ++gi) {
        if (gi < ngrp) MKG_ISSUE(gi)
        mk_commit();
    }
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[j][q] = 0.f;
    for (int it = 0; it < ngrp; ++it) {
        mk_wait<G - 2>();
        __syncthreads();
        {
            const int gx = it + G - 1;
            if (gx < ngrp) MKG_ISSUE(gx)
            mk_commit();
        }
        uint4 wv[P];
        uint2 scv[P];
        uint32_t a[P][4];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint32_t slot = (uint32_t)((it * P + p) % S);
            wv[p] = mk_lds128(sW + slot * 576 + lane * 16);
            scv[p] = mk_lds64(sW + slot * 576 + 512 + gq * 8);
            mk_ldsm4(a[p], sA + slot * 512 + a_ld);
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint4 w = wv[p];
            const uint2 sc = scv[p];
            const uint32_t wq[4] = {w.x, w.y, w.z, w.w};
            {
                __nv_bfloat162 s01[2], s23[2];
                mk_dqs(sc.x, s01);
                mk_dqs(sc.y, s23);
                const __nv_bfloat162 spair[4] = {s01[0], s01[1], s23[0], s23[1]};
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    const __nv_bfloat162 slo = __bfloat162bfloat162(__low2bfloat16(spair[wi]));
                    const __nv_bfloat162 shi = __bfloat162bfloat162(__high2bfloat16(spair[wi]));
                    __nv_bfloat162 f[2];
                    mk_dq(wq[wi] << 8, f);
                    const uint32_t blo0 = mk_bits(__hmul2(f[0], slo));
                    const uint32_t blo1 = mk_bits(__hmul2(f[1], slo));
                    mk_dq(wq[wi], f);
                    const uint32_t bhi0 = mk_bits(__hmul2(f[0], shi));
                    const uint32_t bhi1 = mk_bits(__hmul2(f[1], shi));
                    mk_mma(acc[2 * wi], a[p], blo0, blo1);
                    mk_mma(acc[2 * wi + 1], a[p], bhi0, bhi1);
                }
            }
        }
    }
    float* ob = out + (size_t)ks * ksw;  // this K slice's raw partial (global scale not applied): [rows, N] fp32, summed by the consumer in slice order
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = nt * 64 + j * 8 + 2 * cq;
        const int r0 = gq, r1 = gq + 8;
        if (r0 < nk) *reinterpret_cast<float2*>(ob + (size_t)(t0 + r0) * N + col) = make_float2(acc[j][0], acc[j][1]);
        if (r1 < nk) *reinterpret_cast<float2*>(ob + (size_t)(t0 + r1) * N + col) = make_float2(acc[j][2], acc[j][3]);
    }
}


// ---- MK path kernels. Row space: rows = local routes sorted by expert; slot = token * 8 + k.
// ctl[0] = n_rows, ctl[1] = n_plain_groups, ctl[2] = n_hot_chunks (device ints, written by mkp_route_k). gtab rows: (expert, first row, rows <= 16); htab rows: (expert, first row, rows <= 64).
__global__ void __launch_bounds__(1024) mkp_route_k(const int* __restrict__ ids, int* __restrict__ local_ids, int* __restrict__ ctl,
                                                    int* __restrict__ gtab, int* __restrict__ htab, int* __restrict__ slot_of_row, int* __restrict__ row_of_slot, int* __restrict__ exp_of_row, int n_slots, int lo, int hi, int hot_min) {
    __shared__ int cnt[32];
    __shared__ int off[33];
    __shared__ int gbase[33];
    __shared__ int hbase[33];
    __shared__ int rank_s[2048];
    const int tid = threadIdx.x;
    if (tid < 32) cnt[tid] = 0;
    __syncthreads();
    for (int s = tid; s < n_slots; s += blockDim.x) {
        const int e = ids[s];
        const int l = (e >= lo && e < hi) ? e - lo : -1;
        local_ids[s] = l;
        rank_s[s] = l >= 0 ? atomicAdd(&cnt[l], 1) : 0;
    }
    __syncthreads();
    if (tid == 0) {
        int o = 0, gb = 0, hb = 0;
        for (int e = 0; e < 32; ++e) {
            const int ng = (cnt[e] + 15) >> 4;
            off[e] = o;
            gbase[e] = gb;  // plain groups (single-group experts, or all experts when the hot path is off)
            hbase[e] = hb;  // hot chunks of up to 4 groups
            o += cnt[e];
            if (ng >= hot_min) hb += (ng + 3) >> 2;
            else gb += ng;
        }
        ctl[0] = o;
        ctl[1] = gb;
        ctl[2] = hb;
    }
    __syncthreads();
    if (tid < 32) {
        const int c = cnt[tid];
        if (((c + 15) >> 4) >= hot_min) {
            for (int ci = 0; ci * 64 < c; ++ci) {
                int* t = htab + 3 * (hbase[tid] + ci);
                t[0] = tid;
                t[1] = off[tid] + ci * 64;
                t[2] = min(64, c - ci * 64);
            }
        } else {
            for (int gi = 0; gi * 16 < c; ++gi) {
                int* t = gtab + 3 * (gbase[tid] + gi);
                t[0] = tid;
                t[1] = off[tid] + gi * 16;
                t[2] = min(16, c - gi * 16);
            }
        }
    }
    for (int s = tid; s < n_slots; s += blockDim.x) {
        const int l = local_ids[s];
        if (l >= 0) {
            slot_of_row[off[l] + rank_s[s]] = s;
            exp_of_row[off[l] + rank_s[s]] = l;
        }
        row_of_slot[s] = l >= 0 ? off[l] + rank_s[s] : -1;
    }
}

// Block r = row r: Xg[r] = x[slot / 8].
__global__ void __launch_bounds__(256) mkp_gather_k(const uint4* __restrict__ x, uint4* __restrict__ xg, const int* __restrict__ slot_of_row,
                                                    const int* __restrict__ ctl) {
    const int r = blockIdx.x;
    if (r >= ctl[0]) return;
    const int tok = slot_of_row[r] >> 3;
    for (int i = threadIdx.x; i < 512; i += 256) xg[(size_t)r * 512 + i] = x[(size_t)tok * 512 + i];
}

__device__ __forceinline__ float mkp_rb(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// Marlin's write-out: bf16 round of the fp32 sum, then bf16 multiply by the bf16 global scale (exact in fp32: 8 x 8 bit mantissas), second rounding.
__device__ __forceinline__ float4 mkp_scale4(float4 a, float gs) {
    return make_float4(mkp_rb(mkp_rb(a.x) * gs), mkp_rb(mkp_rb(a.y) * gs), mkp_rb(mkp_rb(a.z) * gs), mkp_rb(mkp_rb(a.w) * gs));
}

// Sum of the KS partials of 4 consecutive floats, in slice order (fp32).
// The plane loads are issued 8 at a time (independent) and added in order, so the sum is the same as a serial loop.
__device__ __forceinline__ float4 mkp_sum4(const float4* __restrict__ part, size_t idx, size_t ks_stride4, int ks) {
    float4 a = part[idx];
    for (int s0 = 1; s0 < ks; s0 += 8) {
        float4 v[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) v[t] = (s0 + t < ks) ? part[idx + (size_t)(s0 + t) * ks_stride4] : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (s0 + t < ks) {
                a.x += v[t].x;
                a.y += v[t].y;
                a.z += v[t].z;
                a.w += v[t].w;
            }
        }
    }
    return a;
}

__device__ __forceinline__ float4 mkp_ld4(const uint2* __restrict__ yb, size_t idx) {
    const uint2 raw = yb[idx];
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), b = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

// silu_and_mul as stock builds it for sm90 (fast math): bf16 gate and up in, fp32 silu(gate) * up, one bf16 rounding.
__device__ __forceinline__ float mkp_silu_mul(float g, float u) { return __fdividef(g, 1.f + __expf(-g)) * u; }

// inter[r] = bf16(silu(gate) * up), gate = bf16(sum of GEMM1 partials)[r][0:4096], up = [r][4096:8192].
__global__ void __launch_bounds__(256) mkp_act_k(const float4* __restrict__ part, __nv_bfloat162* __restrict__ inter, const int* __restrict__ ctl, int ks,
                                                 size_t ksw, const int* __restrict__ exp_of_row, const float* __restrict__ gs) {
    const int r = blockIdx.x;
    if (r >= ctl[0]) return;
    const size_t ks4 = ksw / 4;
    const int i = blockIdx.y * 256 + threadIdx.x;  // grid (rows, 4): 1024 float4 per row
    const float gse = gs[exp_of_row[r]];
    const float4 g = mkp_scale4(mkp_sum4(part, (size_t)r * 2048 + i, ks4, ks), gse);
    const float4 u = mkp_scale4(mkp_sum4(part, (size_t)r * 2048 + 1024 + i, ks4, ks), gse);
    inter[(size_t)r * 2048 + 2 * i] = __floats2bfloat162_rn(mkp_silu_mul(g.x, u.x), mkp_silu_mul(g.y, u.y));
    inter[(size_t)r * 2048 + 2 * i + 1] = __floats2bfloat162_rn(mkp_silu_mul(g.z, u.z), mkp_silu_mul(g.w, u.w));
}

// y[r] = bf16(sum of the GEMM2 partials), dense [rows, 4096]; grid (rows, 4).
__global__ void __launch_bounds__(256) mkp_red_k(const float4* __restrict__ part, uint2* __restrict__ y, const int* __restrict__ ctl, int ks, size_t ksw,
                                                 const int* __restrict__ exp_of_row, const float* __restrict__ gs) {
    const int r = blockIdx.x;
    if (r >= ctl[0]) return;
    const int i = blockIdx.y * 256 + threadIdx.x;
    const float4 v = mkp_scale4(mkp_sum4(part, (size_t)r * 1024 + i, ksw / 4, ks), gs[exp_of_row[r]]);
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&lo);
    packed.y = *reinterpret_cast<uint32_t*>(&hi);
    y[(size_t)r * 1024 + i] = packed;
}


// Fused scatter + post-expert norm + weighted sum, per slot in fp32 like stock's fused kernel:
//   weight_first: y = (x * w) * rsqrt(mean((x * w)^2) + eps);  otherwise y = (x * rsqrt(mean(x^2) + eps)) * w;  out = bf16(sum over live k ascending of y * s_k).
// x = the bf16 GEMM2 output row of the route (written by mkp_red_k).
__global__ void __launch_bounds__(256) mkp_epi_k(const uint2* __restrict__ yb, const int* __restrict__ row_of_slot, const float* __restrict__ topk_w,
                                                 const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ out, float eps, int wf) {
    __shared__ float red[8][8];
    const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int rows[8];
    float ss[8];
    float4 w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint2 raw = *reinterpret_cast<const uint2*>(norm_w + (tid + 256 * j) * 4);
        const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), b = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        w[j] = make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        rows[k] = row_of_slot[tok * 8 + k];
        float s = 0.f;
        if (rows[k] >= 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 y = mkp_ld4(yb, (size_t)rows[k] * 1024 + tid + 256 * j);
                const float a = wf ? y.x * w[j].x : y.x, b = wf ? y.y * w[j].y : y.y, c = wf ? y.z * w[j].z : y.z, d = wf ? y.w * w[j].w : y.w;
                s += a * a + b * b + c * c + d * d;
            }
        }
        ss[k] = s;
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
        if (lane == 0) red[wid][k] = ss[k];
    }
    __syncthreads();
    float rinv[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float t = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += red[i][k];
        rinv[k] = rsqrtf(t / 4096.f + eps);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (rows[k] >= 0) {
                const float4 y = mkp_ld4(yb, (size_t)rows[k] * 1024 + tid + 256 * j);
                const float sk = topk_w[tok * 8 + k];
                if (wf) {
                    acc.x += (y.x * w[j].x * rinv[k]) * sk;
                    acc.y += (y.y * w[j].y * rinv[k]) * sk;
                    acc.z += (y.z * w[j].z * rinv[k]) * sk;
                    acc.w += (y.w * w[j].w * rinv[k]) * sk;
                } else {
                    acc.x += (y.x * rinv[k] * w[j].x) * sk;
                    acc.y += (y.y * rinv[k] * w[j].y) * sk;
                    acc.z += (y.z * rinv[k] * w[j].z) * sk;
                    acc.w += (y.w * rinv[k] * w[j].w) * sk;
                }
            }
        }
        __nv_bfloat162 lo = __floats2bfloat162_rn(acc.x, acc.y), hi = __floats2bfloat162_rn(acc.z, acc.w);
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&lo);
        packed.y = *reinterpret_cast<uint32_t*>(&hi);
        *reinterpret_cast<uint2*>(out + (size_t)tok * 4096 + (tid + 256 * j) * 4) = packed;
    }
}

// ---- Hot-expert kernel (experts with >= 2 token groups): ONE producer warp dequantizes each Marlin tile once into a shared-memory fragment ring,
// four consumer warps (one 16-token group each) run the same mma sequence as mkg_k on those fragments. Bit-identical to mkg_k per output element:
// same B bits, same A tile, same k order, same split-K plane layout. 32 * (NP + 4) threads; named barriers 1..R (full) and R+1..2R (empty), each completes at 160 arrivals (1 producer warp + 4 consumer warps).
#define MKH_NP 4   // producer warps; producer p dequantizes k-steps i with i % MKH_NP == p
#define MKH_DP 4   // per-producer cp.async ring: own k-steps of Marlin tile + scales
#define MKH_R 4    // fragment ring slots (== MKH_NP: slot p belongs to producer p, so every barrier has one producer)
#define MKH_SA 6   // consumer A ring depth
#define MKH_NT (32 * (MKH_NP + 4))
#define MKH_SMEM (MKH_NP * MKH_DP * 576 + MKH_R * 2048 + 4 * MKH_SA * 512)

__device__ __forceinline__ void mkh_bar_sync(int id) { asm volatile("bar.sync %0, 160;\n" :: "r"(id) : "memory"); }
__device__ __forceinline__ void mkh_bar_arrive(int id) { asm volatile("bar.arrive %0, 160;\n" :: "r"(id) : "memory"); }
__device__ __forceinline__ void mkh_sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__global__ void __launch_bounds__(MKH_NT, 3) mkh_k(const uint4* __restrict__ W, const uint2* __restrict__ Sc, const __nv_bfloat16* __restrict__ X,
                                                const int* __restrict__ htab, float* __restrict__ out, const int* __restrict__ ctl, int K, int N, int KS,
                                                size_t ksw) {
    extern __shared__ __align__(16) uint8_t mk3_smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ntiles = N >> 6;
    const int nt = blockIdx.x % ntiles;
    const int rr = blockIdx.x / ntiles;
    const int ks = rr % KS;
    const int h = rr / KS;
    if (h >= ctl[2]) return;  // CTA-uniform
    const int e = htab[3 * h], t0 = htab[3 * h + 1], nrows = htab[3 * h + 2];
    const int ktt = K >> 4, kper = ktt / KS, kt0 = ks * kper;
    const int gq = lane >> 2, cq = lane & 3;
    const size_t wst = (size_t)(N >> 1), sst = (size_t)(N >> 3);
    const uint32_t sbase = mk_smem_u32(mk3_smem);
    const uint32_t sF = sbase + MKH_NP * MKH_DP * 576;
    const uint32_t sAc = sF + MKH_R * 2048;
    if (warp < MKH_NP) {
        // ------------------------------------------------ producer p: cp.async ring -> dequant -> fragment ring slot p (k-steps p, p + NP, ...)
        const int p = warp;
        const int nown = kper > p ? (kper - p + MKH_NP - 1) / MKH_NP : 0;
        const uint4* wp = W + ((size_t)e * ktt + kt0 + p) * wst + nt * 32 + lane;
        const uint2* sp = Sc + ((size_t)e * ktt + kt0 + p) * sst + nt * 8 + (lane & 7);
        const size_t wstp = wst * MKH_NP, sstp = sst * MKH_NP;
        const uint32_t sWp = sbase + (uint32_t)(p * MKH_DP * 576);
#pragma unroll
        for (int st = 0; st < MKH_DP - 1; ++st) {
            if (st < nown) {
                mk_cp16(sWp + st * 576 + lane * 16, wp + (size_t)st * wstp);
                if (lane < 8) mk_cp8(sWp + st * 576 + 512 + lane * 8, sp + (size_t)st * sstp);
            }
            mk_commit();
        }
        for (int j = 0; j < nown; ++j) {
            mk_wait<MKH_DP - 2>();
            __syncwarp();
            const int nx = j + MKH_DP - 1;
            if (nx < nown) {
                mk_cp16(sWp + (nx % MKH_DP) * 576 + lane * 16, wp + (size_t)nx * wstp);
                if (lane < 8) mk_cp8(sWp + (nx % MKH_DP) * 576 + 512 + lane * 8, sp + (size_t)nx * sstp);
            }
            mk_commit();
            const uint32_t sl = sWp + (uint32_t)(j % MKH_DP) * 576;
            const uint4 w = mk_lds128(sl + lane * 16);
            const uint2 sc = mk_lds64(sl + 512 + gq * 8);
            const uint32_t wq[4] = {w.x, w.y, w.z, w.w};
            __nv_bfloat162 s01[2], s23[2];
            mk_dqs(sc.x, s01);
            mk_dqs(sc.y, s23);
            const __nv_bfloat162 spair[4] = {s01[0], s01[1], s23[0], s23[1]};
            uint32_t fr[4][4];
#pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                const __nv_bfloat162 slo = __bfloat162bfloat162(__low2bfloat16(spair[wi]));
                const __nv_bfloat162 shi = __bfloat162bfloat162(__high2bfloat16(spair[wi]));
                __nv_bfloat162 f[2];
                mk_dq(wq[wi] << 8, f);
                fr[wi][0] = mk_bits(__hmul2(f[0], slo));
                fr[wi][1] = mk_bits(__hmul2(f[1], slo));
                mk_dq(wq[wi], f);
                fr[wi][2] = mk_bits(__hmul2(f[0], shi));
                fr[wi][3] = mk_bits(__hmul2(f[1], shi));
            }
            if (j >= 1) mkh_bar_sync(1 + MKH_R + p);  // consumers released slot p (they arrive once per step i = p mod NP)
#pragma unroll
            for (int wi = 0; wi < 4; ++wi) mkh_sts128(sF + (uint32_t)p * 2048 + wi * 512 + lane * 16, fr[wi][0], fr[wi][1], fr[wi][2], fr[wi][3]);
            mkh_bar_arrive(1 + p);
        }
        // the consumers arrive on the empty barrier of every step; drain the last arrival of this producer's slot
        if (nown > 0) mkh_bar_sync(1 + MKH_R + p);
    } else {
        // ------------------------------------------------ consumer c: A ring -> mma on the shared fragments
        const int c = warp - MKH_NP;
        const int nk = min(16, nrows - 16 * c);  // <= 0: idle (still takes part in every barrier)
        const int t0c = t0 + 16 * c;
        const int arow = lane >> 1, akh = lane & 1;
        const bool av = arow < nk;
        const __nv_bfloat16* ap = X + (size_t)(t0c + (av ? arow : 0)) * K + (size_t)kt0 * 16 + akh * 8;
        const uint32_t a_dst = (uint32_t)(arow * 32 + ((akh ^ ((arow >> 2) & 1)) * 16));
        const int lr = lane & 15, lk = lane >> 4;
        const uint32_t a_ld = (uint32_t)(lr * 32 + ((lk ^ ((lr >> 2) & 1)) * 16));
        const uint32_t sA = sAc + (uint32_t)c * (MKH_SA * 512);
#pragma unroll
        for (int st = 0; st < MKH_SA - 1; ++st) {
            if (st < kper) mk_cp16z(sA + st * 512 + a_dst, ap + (size_t)st * 16, av ? 16 : 0);
            mk_commit();
        }
        float acc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[j][q] = 0.f;
        for (int i = 0; i < kper; ++i) {
            mk_wait<MKH_SA - 2>();
            __syncwarp();
            const int nx = i + MKH_SA - 1;
            if (nx < kper) mk_cp16z(sA + (nx % MKH_SA) * 512 + a_dst, ap + (size_t)nx * 16, av ? 16 : 0);
            mk_commit();
            uint32_t a[4] = {0u, 0u, 0u, 0u};
            if (nk > 0) mk_ldsm4(a, sA + (uint32_t)(i % MKH_SA) * 512 + a_ld);
            const int fs = i % MKH_R;
            mkh_bar_sync(1 + fs);  // fragments of step i are in the ring
            if (nk > 0) {
                uint4 f[4];
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) f[wi] = mk_lds128(sF + (uint32_t)fs * 2048 + wi * 512 + lane * 16);
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    mk_mma(acc[2 * wi], a, f[wi].x, f[wi].y);
                    mk_mma(acc[2 * wi + 1], a, f[wi].z, f[wi].w);
                }
            }
            mkh_bar_arrive(1 + MKH_R + fs);
        }
        if (nk > 0) {
            float* ob = out + (size_t)ks * ksw;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = nt * 64 + j * 8 + 2 * cq;
                const int r0 = gq, r1 = gq + 8;
                if (r0 < nk) *reinterpret_cast<float2*>(ob + (size_t)(t0c + r0) * N + col) = make_float2(acc[j][0], acc[j][1]);
                if (r1 < nk) *reinterpret_cast<float2*>(ob + (size_t)(t0c + r1) * N + col) = make_float2(acc[j][2], acc[j][3]);
            }
        }
    }
}

static const int MKP_G = 2, MKP_P = 4;
static const int MKP_SMEM = 4 * MKP_G * MKP_P * 576 + MKP_G * MKP_P * 512;

std::vector<int64_t> mkp_attrs() {
    cudaFuncAttributes h, g;
    cudaFuncGetAttributes(&h, mkh_k);
    cudaFuncGetAttributes(&g, mkg_k<MKP_G, MKP_P>);
    return {(int64_t)h.numRegs, (int64_t)h.localSizeBytes, (int64_t)g.numRegs, (int64_t)g.localSizeBytes};
}

void mkp_init() {
    cudaFuncSetAttribute(mkh_k, cudaFuncAttributeMaxDynamicSharedMemorySize, MKH_SMEM);
    cudaFuncSetAttribute(mkg_k<MKP_G, MKP_P>, cudaFuncAttributeMaxDynamicSharedMemorySize, MKP_SMEM);
    cudaFuncSetAttribute(mkg_k<MKP_G, MKP_P>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);
}

void mkp_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctl, torch::Tensor gtab, torch::Tensor htab, torch::Tensor slot_of_row, torch::Tensor row_of_slot, torch::Tensor exp_of_row,
               int64_t n_slots, int64_t lo, int64_t hi, int64_t hot_min) {
    auto st = at::cuda::getCurrentCUDAStream();
    mkp_route_k<<<1, 1024, 0, st>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), ctl.data_ptr<int>(), gtab.data_ptr<int>(), htab.data_ptr<int>(),
                                    slot_of_row.data_ptr<int>(), row_of_slot.data_ptr<int>(), exp_of_row.data_ptr<int>(), (int)n_slots, (int)lo, (int)hi, (int)hot_min);
}

void mkp_gather(torch::Tensor x, torch::Tensor xg, torch::Tensor slot_of_row, torch::Tensor ctl, int64_t max_rows) {
    auto st = at::cuda::getCurrentCUDAStream();
    mkp_gather_k<<<(int)max_rows, 256, 0, st>>>(reinterpret_cast<const uint4*>(x.data_ptr<at::BFloat16>()), reinterpret_cast<uint4*>(xg.data_ptr<at::BFloat16>()),
                                                slot_of_row.data_ptr<int>(), ctl.data_ptr<int>());
}

void mkp_act(torch::Tensor part, torch::Tensor inter, torch::Tensor ctl, int64_t max_rows, int64_t ks, int64_t ksw, torch::Tensor exp_of_row, torch::Tensor gs) {
    auto st = at::cuda::getCurrentCUDAStream();
    mkp_act_k<<<dim3((unsigned)max_rows, 4), 256, 0, st>>>(reinterpret_cast<const float4*>(part.data_ptr<float>()),
                                             reinterpret_cast<__nv_bfloat162*>(inter.data_ptr<at::BFloat16>()), ctl.data_ptr<int>(), (int)ks, (size_t)ksw, exp_of_row.data_ptr<int>(), gs.data_ptr<float>());
}

void mkp_reduce(torch::Tensor part, torch::Tensor y, torch::Tensor ctl, int64_t max_rows, int64_t ks, int64_t ksw, torch::Tensor exp_of_row, torch::Tensor gs) {
    auto st = at::cuda::getCurrentCUDAStream();
    mkp_red_k<<<dim3((unsigned)max_rows, 4), 256, 0, st>>>(reinterpret_cast<const float4*>(part.data_ptr<float>()), reinterpret_cast<uint2*>(y.data_ptr<at::BFloat16>()),
                                                           ctl.data_ptr<int>(), (int)ks, (size_t)ksw, exp_of_row.data_ptr<int>(), gs.data_ptr<float>());
}

void mkp_epi(torch::Tensor y, torch::Tensor row_of_slot, torch::Tensor topk_w, torch::Tensor norm_w, torch::Tensor out, int64_t m, double eps, int64_t wf) {
    auto st = at::cuda::getCurrentCUDAStream();
    mkp_epi_k<<<(int)m, 256, 0, st>>>(reinterpret_cast<const uint2*>(y.data_ptr<at::BFloat16>()), row_of_slot.data_ptr<int>(), topk_w.data_ptr<float>(),
                                      reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                                      (float)eps, (int)wf);
}

// max_groups: worst-case group count for the call's M (shape only); warps beyond the device-side n_groups exit at once.
void mkp_gemm(torch::Tensor W, torch::Tensor Sc, torch::Tensor X, torch::Tensor gtab, torch::Tensor ctl, torch::Tensor out,
              int64_t max_groups, int64_t K, int64_t N, int64_t KS, int64_t ksw) {
    const int items = (int)(max_groups * KS * (N / 64));
    const int blocks = (items + 3) / 4;
    auto st = at::cuda::getCurrentCUDAStream();
    mkg_k<MKP_G, MKP_P><<<blocks, 128, MKP_SMEM, st>>>(reinterpret_cast<const uint4*>(W.data_ptr<int>()),
                                                       reinterpret_cast<const uint2*>(Sc.data_ptr<uint8_t>()),
                                                       reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), gtab.data_ptr<int>(),
                                                       out.data_ptr<float>(), ctl.data_ptr<int>(), (int)K, (int)N, (int)KS, (size_t)ksw);
}

// Hot experts (>= hot_min groups): max_chunks = worst-case chunk count of the call's M (shape only); CTAs past the device-side ctl[2] exit at once.
void mkp_hot(torch::Tensor W, torch::Tensor Sc, torch::Tensor X, torch::Tensor htab, torch::Tensor ctl, torch::Tensor out, int64_t max_chunks, int64_t K,
             int64_t N, int64_t KS, int64_t ksw) {
    const int blocks = (int)(max_chunks * KS * (N / 64));
    auto st = at::cuda::getCurrentCUDAStream();
    mkh_k<<<blocks, MKH_NT, MKH_SMEM, st>>>(reinterpret_cast<const uint4*>(W.data_ptr<int>()), reinterpret_cast<const uint2*>(Sc.data_ptr<uint8_t>()),
                                         reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), htab.data_ptr<int>(), out.data_ptr<float>(),
                                         ctl.data_ptr<int>(), (int)K, (int)N, (int)KS, (size_t)ksw);
}
"""

# (largest M, K split of GEMM1, K split of GEMM2): shape only; more splits for fewer groups.
_MK_KS = ((1, 16, 16), (16, 8, 8), (32, 4, 4), (None, 2, 2))
MK_HOT_MIN_GROUPS = 2  # experts with at least this many 16-token groups use the shared-dequant kernel (only M > 16 can have them)
_MK_EXT = {}


def _mk_ext():
    key = hashlib.sha1((_MK_CPP + _MK_CUDA).encode()).hexdigest()[:12]
    if key not in _MK_EXT:
        from torch.utils.cpp_extension import load_inline

        ext = load_inline(
            name=f"muse_mk_moe_{key}",
            cpp_sources=_MK_CPP,
            cuda_sources=_MK_CUDA,
            functions=["mkp_init", "mkp_route", "mkp_gather", "mkp_act", "mkp_reduce", "mkp_epi", "mkp_gemm", "mkp_hot", "mkp_attrs"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        ext.mkp_init()
        _MK_EXT[key] = ext
    return _MK_EXT[key]


def _mk_ks(m):
    for top, ks1, ks2 in _MK_KS:
        if top is None or m <= top:
            return ks1, ks2


def _mk_use_hot(m):
    return m > 16


def _mk_max_chunks(m):
    ng = _mk_max_groups(m)
    return max(1, min(ng // 2, ng // 4 + 32))


def _mk_max_groups(m):
    rows = MK_TOP_K * m
    return min(rows, 32 * ((m + 15) // 16), rows // 16 + 32)


class MkLayer:
    """One MoE layer's Marlin tensors plus the fp32 global scales (skip-flop: the bf16 scale already carries 2^119)."""

    __slots__ = ("w13", "w13_s", "w2", "w2_s", "gs13", "gs2")

    def __init__(self, w13, w13_s, w13_gs, w2, w2_s, w2_gs):
        assert w13.shape[1] * 16 == MK_HIDDEN and w13.shape[2] == 2 * 2 * MK_INTER, tuple(w13.shape)
        assert w2.shape[1] * 16 == MK_INTER and w2.shape[2] == 2 * MK_HIDDEN, tuple(w2.shape)
        assert w13.shape[0] == MK_EXPERTS and w2.shape[0] == MK_EXPERTS
        assert w13_s.element_size() == 1 and w2_s.element_size() == 1
        self.w13 = w13
        self.w13_s = w13_s.view(torch.uint8)
        self.w2 = w2
        self.w2_s = w2_s.view(torch.uint8)
        self.gs13 = w13_gs.reshape(MK_EXPERTS, -1)[:, 0].float().contiguous()
        self.gs2 = w2_gs.reshape(MK_EXPERTS, -1)[:, 0].float().contiguous()


class MkMoE:
    def __init__(self):
        self.ext = None
        self.buf = None
        self.hot = True  # shared-dequant kernel for experts with >= 2 groups; False = v2's single path

    def build(self, device):
        self.ext = _mk_ext()
        return self

    def alloc(self, max_tokens, top_k, hidden, inter, device):
        assert top_k == MK_TOP_K and hidden == MK_HIDDEN and inter == MK_INTER and max_tokens <= MK_MAX_TOKENS
        self.max_tokens = max_tokens
        rmax = top_k * max_tokens
        self.buf = {
            "ctl": torch.zeros(4, dtype=torch.int32, device=device),
            "local_ids": torch.zeros(rmax, dtype=torch.int32, device=device),
            "gtab": torch.zeros(3 * (rmax // 16 + 64), dtype=torch.int32, device=device),
            "htab": torch.zeros(3 * (rmax // 16 + 64), dtype=torch.int32, device=device),
            "slot_of_row": torch.zeros(rmax, dtype=torch.int32, device=device),
            "row_of_slot": torch.zeros(rmax, dtype=torch.int32, device=device),
            "exp_of_row": torch.zeros(rmax, dtype=torch.int32, device=device),
            "xg": torch.zeros(rmax, hidden, dtype=torch.bfloat16, device=device),
            "inter": torch.zeros(rmax, inter, dtype=torch.bfloat16, device=device),
            "y": torch.zeros(rmax, hidden, dtype=torch.bfloat16, device=device),
            # fp32 split-K partials [KS, 8 * M, N] of the call in flight; GEMM2 reuses GEMM1's after the activation has consumed them
            "part": torch.zeros(max(max(_mk_ks(m)[0] * 2 * inter, _mk_ks(m)[1] * hidden) * top_k * m for m in range(1, max_tokens + 1)), dtype=torch.float32, device=device),
        }
        return self

    def scratch_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.buf.values())

    def bind_layer(self, w13, w13_s, w13_gs, w2, w2_s, w2_gs):
        return MkLayer(w13, w13_s, w13_gs, w2, w2_s, w2_gs)

    def forward(self, hidden_states, topk_ids, topk_weights, local_offset, local_count, layer, norm_weight, norm_eps, weight_first, timer=None):
        """hidden_states bf16 [M, 4096]; topk_ids int [M, 8] global ids; topk_weights [M, 8]; norm_weight bf16 [4096] (effective weight).

        Returns bf16 [M, 4096] = sum_k s_k * post_norm(expert_k(x)), with the norm form selected by weight_first
        (True: rms_norm(w * y), False: rms_norm(y) * w); routes to experts outside [local_offset, local_offset + local_count) are dropped.
        """
        e, b = self.ext, self.buf
        x = hidden_states.contiguous()
        m = x.shape[0]
        assert m <= self.max_tokens and local_count <= MK_EXPERTS
        n_slots = m * MK_TOP_K
        ng = _mk_max_groups(m)
        ks1, ks2 = _mk_ks(m)
        hot = _mk_use_hot(m) and self.hot
        e.mkp_route(topk_ids.to(torch.int32).contiguous(), b["local_ids"], b["ctl"], b["gtab"], b["htab"], b["slot_of_row"], b["row_of_slot"], b["exp_of_row"], n_slots,
                    local_offset, local_offset + local_count, MK_HOT_MIN_GROUPS if hot else (1 << 20))
        ksw1 = n_slots * 2 * MK_INTER
        ksw2 = n_slots * MK_HIDDEN
        if timer:
            timer("route")
        e.mkp_gather(x, b["xg"], b["slot_of_row"], b["ctl"], n_slots)
        if timer:
            timer("gather")
        nch = _mk_max_chunks(m)
        if hot:
            e.mkp_hot(layer.w13, layer.w13_s, b["xg"], b["htab"], b["ctl"], b["part"], nch, MK_HIDDEN, 2 * MK_INTER, ks1, ksw1)
            if timer:
                timer("g1_hot")
        e.mkp_gemm(layer.w13, layer.w13_s, b["xg"], b["gtab"], b["ctl"], b["part"], ng, MK_HIDDEN, 2 * MK_INTER, ks1, ksw1)
        if timer:
            timer("g1_small")
        e.mkp_act(b["part"], b["inter"], b["ctl"], n_slots, ks1, ksw1, b["exp_of_row"], layer.gs13)
        if timer:
            timer("act")
        if hot:
            e.mkp_hot(layer.w2, layer.w2_s, b["inter"], b["htab"], b["ctl"], b["part"], nch, MK_INTER, MK_HIDDEN, ks2, ksw2)
            if timer:
                timer("g2_hot")
        e.mkp_gemm(layer.w2, layer.w2_s, b["inter"], b["gtab"], b["ctl"], b["part"], ng, MK_INTER, MK_HIDDEN, ks2, ksw2)
        if timer:
            timer("g2_small")
        e.mkp_reduce(b["part"], b["y"], b["ctl"], n_slots, ks2, ksw2, b["exp_of_row"], layer.gs2)
        if timer:
            timer("reduce")
        out = torch.empty((m, MK_HIDDEN), dtype=torch.bfloat16, device=x.device)
        e.mkp_epi(b["y"], b["row_of_slot"], topk_weights.to(torch.float32).contiguous(), norm_weight, out, m, float(norm_eps), 1 if weight_first else 0)
        if timer:
            timer("epi")
        return out


_MK_SHARED = {}


def mk_shared(device):
    """One MkMoE (compiled module and buffers) per device, shared by all layers: layers run one after the other on a stream."""
    key = (device.type, device.index)
    if key not in _MK_SHARED:
        _MK_SHARED[key] = MkMoE().build(device).alloc(MK_MAX_TOKENS, MK_TOP_K, MK_HIDDEN, MK_INTER, device)
    return _MK_SHARED[key]
