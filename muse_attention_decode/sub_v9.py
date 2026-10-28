# okc: problem=muse_attention_decode
# Based on infraax/okc-kernel-challenge (Apache-2.0): kernels/muse_attention_decode/sonnet_v5.py
# v7: verify-full at batch >= 16 (page 1, sm_90) runs a warp-specialized Gluon kernel: one producer warpgroup (cp.async ring,
#     page ids 2 tiles ahead, V FP8->FP16) and one consumer warpgroup per KV head with FA3-style overlap (q.K of tile i and
#     P.V of tile i-1 in flight while the softmax runs). The block's own keys still run sonnet_v5's program (S_OFFSET).
#     Measured (check 0231e0c08f654640): verify-full b16 163 -> 146 us, b32 322 -> 280 us; outputs within 3e-4 of v5.
# v9: the overlapped consumers mask only the tail tile of a split (check 5f20aaf62c1d4318: 151 -> 138 us at b16,
#     288 -> 269 us at b32; same output).
# sonnet_v5 (C27) = sonnet_v3 + ONE config change: verify-full (_full_kernel, q_len > 1) with batch >= FULL_BIG_MIN_BATCH uses BN 128 / 3 stages instead of BN 64 / 4.
#   Why: C26' judge check (job 9206dc7a1a3340bc, design/breakdown/c26-attention-verify-result.md): exp2 is 7 % and the FP8 convert 1-4 % of the verify-full kernel, the
#   rest is per-tile bookkeeping, so bigger tiles amortise it: BN128 (255 regs, 96 KB smem, 2 programs / SM) measured #9 189.1 -> 163.5 us (x1.16), #11 364.1 -> 322.2 (x1.13),
#   seeds within +-1 us, output err 7.6e-5 vs base; at batch 1 (#7) it is WORSE (20.9 -> 24.8: the plan has only ~157 programs of >= 4 x 128 keys for 264 slots).
#   Threshold: only batch 1 / 16 / 32 were measured; 16 is the smallest batch measured to win and 1 the one measured to lose, so the switch sits at 16 (batches 2-15 keep v3's
#   BN 64, i.e. unmeasured shapes run the measured-good code). Decode-full (_full_cat_kernel), sliding layers, the reducer and every other line are v3's, unchanged.
# sonnet_v3 -- v2 with new kernels for the FULL-attention layers (window 0, no RoPE); sliding layers run v2's kernels unchanged
#   (see sonnet_v2.py for the RoPE / own-keys / cold-code changes that made them 1.3-2.6x). Two Triton launches as before.
# WHAT THE JUDGE SHOWED for v2 (f6f326822293406f, 1.484x): sliding shapes are good; what is left is the full layers, 164 / 321 us at batch 16 /
#   32 decode and 196 / 358 us verify (floor 63 / 127: v2 moves ~1.8 TB/s), and batch 1 full (20 us, floor 4). By load weight (46 sliding : 16 full)
#   the full layers are 80% of a batch-16 load and 47% of a batch-1 load.
# WHAT AN sm_90a COMPILE OF v2 SHOWS (ptxas + nvdisasm; offline, no GPU needed): the loop IS cp.async pipelined (3 stages, vectorised 16-B copies),
#   but Triton also turns the page-id load into a cp.async and waits for it with an async_wait that drains every K/V copy still in flight, so there
#   is ONE tile (128 keys x 64 B, half a 128-B line per row) in flight per program and the page-id -> row-gather latency is paid every tile
#   (~4 us per tile per program measured: 159 us / 37 tiles at batch 16). It issues ~18 warp-instructions per key and head (hot loop, masks excluded).
# CHANGES (full layers)
#   1. Software pipeline: page ids are ordinary register loads carried one tile ahead, so Triton leaves them alone and pipelines only the K / V
#      copies: with num_stages S the SASS keeps S-1 tiles pending (DEPBAR.LE SB0, 4*(S-2)) and issues the copies of tile i+S-1 while tile i is computed.
#      The page-table lines of tiles 1-2 are touched in parallel with the seq_lens / q loads (a * 0.0 term keeps the load alive), so the prologue
#      does not chain three cold page loads; q prep runs before the loop (startup: seq / pages / q in parallel, then K / V: ~2 latencies, v2 ~4).
#   2. Both KV heads of a pair per program, rows are whole 128-B lines. Decode (q_len 1, 8 query heads per KV head): ONE attention problem, 16 q
#      rows (0-7 head 0, 8-15 head 1; q zero outside its head's 64 columns) against [BN, 128] K / V tiles: one logit dot, one softmax, one
#      reduction chain per tile (v2: 16 rows of which 8 padding, per head), P.V is computed for all 128 columns and each head keeps its own 64.
#      Verify (64 rows per head): two [64, 64] problems per tile, same pipeline; one dot over both heads would double the tensor work.
#   3. Programs: ~1056 for decode (4 resident per SM -> two waves), 256 for verify (2 resident per SM -> one wave: no partial last wave, 17-9
#      partials like v2 so the 15 us verify reducer does not grow), <= 255 splits and >= 4 tiles per split; from shapes only (page-table width).
#      The reducer merges up to 256 partials in one pass (8 warps above 128) so batch 1 can use ~224 programs.
#   4. Verify tile: BLOCK_M 64 with 4 warps (one warpgroup, wgmma M = 64, q.K in FP8 as v2), BN 64. Measured on the sm_90a SASS (hot loop, thread
#      instructions x warps / key-head): 4 warps 18.3, 8 warps 32.2 (cross-warpgroup reductions, P through shared memory), BN 128 15.4 but 255
#      registers (v3b). Two 32-row programs would read and convert every K / V tile twice; so 4 warps, 2 resident programs per SM.
#   5. Decode: 2 warps per program (205 regs, 4 programs per SM; 4 warps when the grid is <= 2 programs per SM, i.e. batch 1): 15.1 warp-instr / key-head against 20.4 for 4 warps (sep) and v2's 18.3.
#   6. Partials are stored [batch, kv head, row, split]: the reducer reads an output row's partials as one contiguous run (v2: 256-byte pieces 16 KB
#      apart). The judge's profile has the verify-full reducer at 14.7 / 15.9 us for 9-10 MB of partials (~0.6 TB/s), so it was not hitting L2; not
#      counted in the predictions below beyond a few us on the verify shapes. Sliding layers keep v2's layout.
# PREDICTED (no H200 here; model: HBM-bound at 2.6-3.0 TB/s on 128-B random rows, ~2.5 IPC/SM, + 3 us start, + reducer as in v2; honest range +-20%)
#   full-layer us (v2 judged -> v3): decode b1 20.1 -> 15.5, b16 164 -> 125 (105-150), b32 321 -> 245 (210-290); verify b1 20.5 -> 15.5,
#   b16 196 -> 150 (125-180), b32 358 -> 275 (240-330); sliding shapes as v2 (7.9 / 13.3 / 16.9 / 9.3 / 17.4 / 26.0).
#   per-load speedup vs SGLang (v2 judged -> v3): decode 1 2.12 -> 2.37, decode 16 1.37 -> 1.70, decode 32 1.28 -> 1.61, verify 1 2.17 -> 2.42,
#   verify 16 1.17 -> 1.44, verify 32 1.13 -> 1.40; geo-mean 1.484 -> ~1.78 (pessimistic 1.58, optimistic 1.98).
# UNVERIFIABLE HERE: every H200 timing; the HBM rate on 128-B random rows; residency (ptxas: decode 205 regs / 39 KB smem, verify 185 regs / 70 KB)
#   and how many tiles the memory system really keeps in flight; that Triton's wgmma / mma.sync FP8 paths behave as in v2 (they compile the same
#   way; the sm_75 card and the CPU interpreter run the software FP8 paths); the cold-code cost of the larger full-layer kernels (~35 KB of SASS).
# Rules: grid, split size and buffers come from shapes / config only (the page table's width bounds the context); lengths, positions, cache rows
#   and scales are read on the device; no host syncs; every key of the context is attended (splits skip only keys at or beyond seq_len); K / V are
#   read from the FP8 cache as is; q at most E4M3; accumulation FP32; output BF16.
import functools
import math

import torch
import triton
import triton.language as tl

BLOCK_N = 128           # keys per inner step (sliding layers, v2's kernel)
# Full-attention layers (window == 0), per q_len class. kernel "sep": _full_kernel (one [M, 64] problem per KV head, two heads per
# program); "cat": _full_cat_kernel (decode only: the head pair as one 16-row problem on [BN, 128] tiles).
FULL_CFG = {
    "decode": dict(kernel="cat", bn=64, stages=3, warps=2, programs=1056),   # 4 resident programs / SM -> two waves
    "verify": dict(kernel="sep", bn=64, stages=4, warps=4, programs=256),    # 2 resident programs / SM -> one wave
}
FULL_BIG = dict(kernel="sep", bn=128, stages=3, warps=4, programs=256)   # verify-full at batch >= FULL_BIG_MIN_BATCH (C27: BN128 twin of C26')
FULL_BIG_MIN_BATCH = 16
WS_CFG = dict(bn=128, nbuf=3, programs=132, rc=200, rp=96, maxnreg=168, warps=4)   # Gluon verify-full (check 0231e0c0)
WS_MIN_BATCH = 16      # measured to win at batch 16 / 32; batch 1 is faster on sonnet_v5's BN64 kernel
FULL_MIN_TILES = 4      # fewest tiles per split when a shape has few (batch x head pairs)
MAX_SPLITS = 255        # the reducer merges up to 256 partials (the splits + the token's own) in one pass (8 warps above 128)
MAX_ROWS = 128          # L * (Hq / Hkv) rows per KV head the kernel supports (verify blocks: 8 * 8 = 64)
SM_COUNT = 132          # cost model only (H200); the plan never reads the device


# --------------------------------------------------------------------------------------------- FP8 helpers
@triton.jit
def _decode_e4m3(u):
    """uint8 E4M3FN codes -> fp32 (software path for GPUs without FP8 conversions, e.g. sm_75)."""
    u = u.to(tl.int32)
    e = (u >> 3) & 15
    m = u & 7
    normal = ((e + 120) << 23) | (m << 20)
    val = tl.where(e > 0, normal.to(tl.float32, bitcast=True), m.to(tl.float32) * 0.001953125)
    return tl.where((u & 128) != 0, -val, val)


@triton.jit
def _encode_e4m3(x):
    """fp32 -> uint8 E4M3FN codes, round-to-nearest-even, saturating at 448 (software path)."""
    sign = tl.where(x < 0, 128, 0)
    a = tl.minimum(tl.abs(x), 448.0)
    bits = a.to(tl.int32, bitcast=True)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - 960
    sub = ((a * 512.0 + 12582912.0) - 12582912.0).to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    return (code | sign).to(tl.uint8)


# --------------------------------------------------------------------------------------------- RoPE helpers
@triton.jit
def _sincos(x):
    """(sin x, cos x) for fp32 |x| < ~4e6: round to a multiple of pi/2 with three FMA steps, Cephes polynomials on the
    remainder, quadrant by select. About 40 instructions per value and no slow path (libdevice's is ~1 KB of code)."""
    nf = (x * 0.6366197723675814 + 12582912.0) - 12582912.0
    r = tl.fma(nf, -1.5703125, x)
    r = tl.fma(nf, -4.837512969970703e-4, r)
    r = tl.fma(nf, -7.549789948768648e-8, r)
    z = r * r
    sr = ((-1.9515295891e-4 * z + 8.3321608736e-3) * z - 1.6666654611e-1) * z * r + r
    cr = ((2.443315711809948e-5 * z - 1.388731625493765e-3) * z + 4.166664568298827e-2) * z * z - 0.5 * z + 1.0
    q = nf.to(tl.int32) & 3
    swap = (q & 1) != 0
    s = tl.where(swap, cr, sr)
    c = tl.where(swap, sr, cr)
    s = tl.where((q & 2) != 0, -s, s)
    c = tl.where(((q + 1) & 2) != 0, -c, c)
    return s, c


@triton.jit
def _rope_tables(pos, log2_theta, d):
    """sin, cos of pos[r] * theta^(-2i/64), i = d // 2: fp32 [R, 64]."""
    inv_freq = tl.exp2(-((d // 2) * 2).to(tl.float32) * (log2_theta / 64.0))
    return _sincos(pos.to(tl.float32)[:, None] * inv_freq)


@triton.jit
def _load_norm_rope(base, ok, d, pos, scale, log2_theta, ROPE: tl.constexpr):
    """rows of 64 bf16 at base+d -> RMSNorm (eps 1e-5), x scale, interleaved RoPE if ROPE. fp32 [R, 64]."""
    x = tl.load(base + d, mask=ok, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / 64.0 + 1e-5)
    xn = x * inv[:, None]
    if ROPE:
        xp = tl.load(base + (d ^ 1), mask=ok, other=0.0).to(tl.float32) * inv[:, None]
        sin, cos = _rope_tables(pos, log2_theta, d)
        sgn = tl.where((d & 1) == 0, -1.0, 1.0)
        xn = xn * cos + sgn * xp * sin
    return xn * scale


@triton.jit
def _prep_q(qkv, positions, b, h, qk_factor, log2_theta,
            L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
            ROPE: tl.constexpr, BL: tl.constexpr, M: tl.constexpr, QK8: tl.constexpr, QK8_HW: tl.constexpr,
            CDTYPE: tl.constexpr):
    """The M = BL * GP query rows of one KV head (row r = token r // GP, head h*G + r % GP): RMSNorm, qk_factor, RoPE (tables
    computed once per token and broadcast over the group). QK8: rounded to FP8 E4M3 (round-to-nearest-even; software when
    QK8_HW is off), else CDTYPE. The logit scale is applied after the dot."""
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    d = tl.arange(0, 64)[None, :]
    r = tl.arange(0, M)
    j = r // GP
    rvalid = (j < L) & ((r % GP) < G)
    base = qkv + (b * L + j)[:, None] * QKV + (h * G + r % GP)[:, None] * 64
    x = tl.load(base + d, mask=rvalid[:, None], other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / 64.0 + 1e-5)
    xn = x * inv[:, None]
    if ROPE:
        xp = tl.load(base + (d ^ 1), mask=rvalid[:, None], other=0.0).to(tl.float32) * inv[:, None]
        jj = tl.arange(0, BL)
        pos = tl.load(positions + b * L + jj, mask=jj < L, other=0)
        sin, cos = _rope_tables(pos, log2_theta, d)
        sin = tl.reshape(tl.broadcast_to(sin[:, None, :], [BL, GP, 64]), [M, 64])
        cos = tl.reshape(tl.broadcast_to(cos[:, None, :], [BL, GP, 64]), [M, 64])
        sgn = tl.where((d & 1) == 0, -1.0, 1.0)
        xn = xn * cos + sgn * xp * sin
    xn = xn * qk_factor
    if QK8:
        if QK8_HW:
            q = xn.to(tl.float8e4nv)
        else:
            q = _encode_e4m3(xn).to(tl.float8e4nv, bitcast=True)
    else:
        q = xn.to(CDTYPE)
    return q, rvalid, j


@triton.jit
def _softmax_step(sc, c, m_i, l_i, acc, vc, CDTYPE: tl.constexpr):
    """Online softmax step on raw logits sc (times c = sm_scale * k_scale * log2(e) gives the log2-domain logits)."""
    m_new = tl.maximum(m_i, tl.max(sc, 1))
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp2(sc * c - (m_use * c)[:, None])
    alpha = tl.exp2((m_i - m_use) * c)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = tl.dot(p.to(CDTYPE), vc, acc * alpha[:, None])
    return m_new, l_i, acc


# --------------------------------------------------------------------------------------------- kernel 1
@triton.jit
def _attn_kernel(
    qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale_ptr, v_scale_ptr,
    part_o, part_m, part_l,
    sm_scale, qk_factor, log2_theta, pt_stride, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, PAGE: tl.constexpr,
    WINDOW: tl.constexpr, ROPE: tl.constexpr, EDGE: tl.constexpr, NATIVE_LOAD: tl.constexpr,
    NATIVE_STORE: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr,
    BL: tl.constexpr, M: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
):
    s = tl.program_id(0)          # 0 .. NSPLIT-1: cached keys of one split; NSPLIT: the block's own keys
    h = tl.program_id(1)
    b = tl.program_id(2)
    d = tl.arange(0, 64)[None, :]
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64

    seq = tl.load(seq_lens + b).to(tl.int32)           # cached tokens; the block starts at this position
    lo = tl.maximum(seq - WINDOW + 1, 0) if WINDOW else 0
    start = lo + s * split_len
    end = tl.minimum(seq, start + split_len)
    if (s == NSPLIT) | (start < end):
        ks = tl.load(k_scale_ptr)
        c = sm_scale * ks * 1.4426950408889634        # logits in the log2 domain
        q, rvalid, j = _prep_q(qkv, positions, b, h, qk_factor, log2_theta, L, G, GP, HQ, HKV, ROPE, BL, M, QK8, NATIVE_STORE, CDTYPE)
        m_i = tl.full([M], float("-inf"), tl.float32)
        l_i = tl.zeros([M], tl.float32)
        acc = tl.zeros([M, 64], tl.float32)

        if s < NSPLIT:
            # ---- cached keys [start, end)
            ar = tl.arange(0, BN)
            nt = (end - start + BN - 1) // BN
            prow = page_table + b * pt_stride
            for i in range(nt):
                t = start + i * BN + ar
                tm = t < end
                pg = tl.load(prow + t // PAGE, mask=tm, other=0)
                off = (pg.to(tl.int64) * PAGE + t % PAGE) * (HKV * 64) + h * 64
                ku = tl.load(k_cache + off[:, None] + d, mask=tm[:, None], other=0, eviction_policy="evict_first")
                vu = tl.load(v_cache + off[:, None] + d, mask=tm[:, None], other=0, eviction_policy="evict_first")
                if NATIVE_LOAD:
                    vc = vu.to(tl.float8e4nv, bitcast=True).to(CDTYPE)
                else:
                    vc = _decode_e4m3(vu).to(CDTYPE)
                if QK8:
                    sc = tl.dot(q, tl.trans(ku.to(tl.float8e4nv, bitcast=True)))
                elif NATIVE_LOAD:
                    sc = tl.dot(q, tl.trans(ku.to(tl.float8e4nv, bitcast=True).to(CDTYPE)))
                else:
                    sc = tl.dot(q, tl.trans(_decode_e4m3(ku).to(CDTYPE)))
                need_mask = start + (i + 1) * BN > end
                if EDGE:
                    need_mask = need_mask | ((i == 0) & (s == 0))
                if need_mask:
                    ok = tm[None, :]
                    if WINDOW:
                        ok = ok & (t[None, :] >= (seq + j - WINDOW + 1)[:, None])
                    sc = tl.where(ok, sc, float("-inf"))
                m_i, l_i, acc = _softmax_step(sc, c, m_i, l_i, acc, vc, CDTYPE)
        else:
            # ---- the block's own keys / values: norm, RoPE, FP8 round + cache write, causal tile from registers
            vs = tl.load(v_scale_ptr)
            ti = tl.arange(0, BT)
            tv = ti < L
            tpos = tl.load(positions + b * L + ti, mask=tv, other=0)
            tbase = qkv + (b * L + ti)[:, None] * QKV
            kn = _load_norm_rope(tbase + HQ * 64 + h * 64, tv[:, None], d, tpos, 1.0, log2_theta, ROPE)
            vn = tl.load(tbase + (HQ + HKV) * 64 + h * 64 + d, mask=tv[:, None], other=0.0).to(tl.float32)
            slot = tl.load(cache_loc + b * L + ti, mask=tv, other=0)
            coff = slot[:, None] * (HKV * 64) + h * 64 + d
            if NATIVE_STORE:
                k8 = (kn / ks).to(tl.float8e4nv)
                v8 = (vn / vs).to(tl.float8e4nv)
                tl.store(k_cache + coff, k8.to(tl.uint8, bitcast=True), mask=tv[:, None])
                tl.store(v_cache + coff, v8.to(tl.uint8, bitcast=True), mask=tv[:, None])
                kt = k8 if QK8 else k8.to(CDTYPE)
                vt = v8.to(CDTYPE)
            else:
                ku8 = _encode_e4m3(kn / ks)
                vu8 = _encode_e4m3(vn / vs)
                tl.store(k_cache + coff, ku8, mask=tv[:, None])
                tl.store(v_cache + coff, vu8, mask=tv[:, None])
                kt = ku8.to(tl.float8e4nv, bitcast=True) if QK8 else _decode_e4m3(ku8).to(CDTYPE)
                vt = _decode_e4m3(vu8).to(CDTYPE)
            sc = tl.dot(q, tl.trans(kt))
            ok = tv[None, :] & (ti[None, :] <= j[:, None])
            if WINDOW:
                ok = ok & (ti[None, :] > j[:, None] - WINDOW)
            sc = tl.where(ok, sc, float("-inf"))
            m_i, l_i, acc = _softmax_step(sc, c, m_i, l_i, acc, vt, CDTYPE)

        r = tl.arange(0, M)
        pidx = ((b * HKV + h) * (NSPLIT + 1) + s) * M + r
        tl.store(part_m + pidx, m_i * c, mask=rvalid)
        tl.store(part_l + pidx, l_i, mask=rvalid)
        tl.store(part_o + pidx[:, None] * 64 + d, acc, mask=rvalid[:, None])


# --------------------------------------------------------------------------------------------- full layers (v3)
@triton.jit
def _pages(prow, t, cap, PAGE: tl.constexpr):
    """Page id of every key position t (bounded by the page table's width only, so it can be issued before seq_len is known)."""
    return tl.load(prow + t // PAGE, mask=t < cap, other=0)


@triton.jit
def _gather(k_cache, v_cache, pg, t, end, hb, d, PAGE: tl.constexpr, HKV: tl.constexpr):
    """K and V rows of key positions t for the KV heads hb/64 and hb/64 + 1 (adjacent: one 128-byte line per row):
    four uint8 [BN, 64] tiles, requested together; rows at or beyond `end` are not read."""
    if PAGE == 1:
        row = pg.to(tl.int64)
    else:
        row = pg.to(tl.int64) * PAGE + t % PAGE
    off = tl.multiple_of(row * (HKV * 64) + hb, 64)
    p = off[:, None] + d
    m = (t < end)[:, None]
    k0 = tl.load(k_cache + p, mask=m, other=0, eviction_policy="evict_first")
    k1 = tl.load(k_cache + (p + 64), mask=m, other=0, eviction_policy="evict_first")
    v0 = tl.load(v_cache + p, mask=m, other=0, eviction_policy="evict_first")
    v1 = tl.load(v_cache + (p + 64), mask=m, other=0, eviction_policy="evict_first")
    return k0, k1, v0, v1


@triton.jit
def _step(q, ku, vu, m_i, l_i, acc, c, tm, need_mask, NATIVE_LOAD: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr):
    """One tile of one KV head: logits, (masked on the tail tile only) online softmax, P.V."""
    if NATIVE_LOAD:
        vc = vu.to(tl.float8e4nv, bitcast=True).to(CDTYPE)
    else:
        vc = _decode_e4m3(vu).to(CDTYPE)
    if QK8:
        sc = tl.dot(q, tl.trans(ku.to(tl.float8e4nv, bitcast=True)))
    elif NATIVE_LOAD:
        sc = tl.dot(q, tl.trans(ku.to(tl.float8e4nv, bitcast=True).to(CDTYPE)))
    else:
        sc = tl.dot(q, tl.trans(_decode_e4m3(ku).to(CDTYPE)))
    if need_mask:
        sc = tl.where(tm[None, :], sc, float("-inf"))
    return _softmax_step(sc, c, m_i, l_i, acc, vc, CDTYPE)


@triton.jit
def _own_head(qkv, positions, cache_loc, k_cache, v_cache, b, h, q, j, c, ks, vs, log2_theta, m_i, l_i, acc,
              L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr, NATIVE_STORE: tl.constexpr,
              QK8: tl.constexpr, CDTYPE: tl.constexpr, BT: tl.constexpr):
    """The block's own keys / values of one KV head (norm, RoPE, FP8 round + cache write, causal tile from registers)."""
    d = tl.arange(0, 64)[None, :]
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    ti = tl.arange(0, BT)
    tv = ti < L
    tpos = tl.load(positions + b * L + ti, mask=tv, other=0)
    tbase = qkv + (b * L + ti)[:, None] * QKV
    kn = _load_norm_rope(tbase + HQ * 64 + h * 64, tv[:, None], d, tpos, 1.0, log2_theta, ROPE)
    vn = tl.load(tbase + (HQ + HKV) * 64 + h * 64 + d, mask=tv[:, None], other=0.0).to(tl.float32)
    slot = tl.load(cache_loc + b * L + ti, mask=tv, other=0)
    coff = slot[:, None] * (HKV * 64) + h * 64 + d
    if NATIVE_STORE:
        k8 = (kn / ks).to(tl.float8e4nv)
        v8 = (vn / vs).to(tl.float8e4nv)
        tl.store(k_cache + coff, k8.to(tl.uint8, bitcast=True), mask=tv[:, None])
        tl.store(v_cache + coff, v8.to(tl.uint8, bitcast=True), mask=tv[:, None])
        kt = k8 if QK8 else k8.to(CDTYPE)
        vt = v8.to(CDTYPE)
    else:
        ku8 = _encode_e4m3(kn / ks)
        vu8 = _encode_e4m3(vn / vs)
        tl.store(k_cache + coff, ku8, mask=tv[:, None])
        tl.store(v_cache + coff, vu8, mask=tv[:, None])
        kt = ku8.to(tl.float8e4nv, bitcast=True) if QK8 else _decode_e4m3(ku8).to(CDTYPE)
        vt = _decode_e4m3(vu8).to(CDTYPE)
    sc = tl.dot(q, tl.trans(kt))
    sc = tl.where(tv[None, :] & (ti[None, :] <= j[:, None]), sc, float("-inf"))
    return _softmax_step(sc, c, m_i, l_i, acc, vt, CDTYPE)


@triton.jit
def _full_kernel(
    qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale_ptr, v_scale_ptr,
    part_o, part_m, part_l,
    sm_scale, qk_factor, log2_theta, pt_stride, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, PAGE: tl.constexpr,
    ROPE: tl.constexpr, NATIVE_LOAD: tl.constexpr, NATIVE_STORE: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr,
    BL: tl.constexpr, M: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr, S_OFFSET: tl.constexpr = 0,
):
    """Full-attention layers. One program = one split of one request for TWO adjacent KV heads. The keys of the split are
    streamed through a software pipeline kept in registers: the page ids of tile i+3 and the K / V rows of tile i+2 (both
    heads, 128-byte lines) are requested while tile i is computed, so the page-table -> row-gather chain of a tile is hidden
    behind two tiles of compute and three tiles' bytes are in flight per program."""
    s = tl.program_id(0) + S_OFFSET          # 0 .. NSPLIT-1: cached keys of one split; NSPLIT: the block's own keys
    hp = tl.program_id(1)
    b = tl.program_id(2)
    h0 = hp * 2
    d = tl.arange(0, 64)[None, :]
    ar = tl.arange(0, BN)
    cap = pt_stride * PAGE

    seq = tl.load(seq_lens + b).to(tl.int32)           # cached tokens; the block starts at this position
    start = s * split_len
    end = tl.minimum(seq, start + split_len)
    prow = page_table + b * pt_stride
    t0 = start + ar
    pgA = _pages(prow, t0, cap, PAGE)                  # page ids do not depend on seq_len: issued with its load and the q loads
    ahead = tl.load(prow + (start + BN + tl.arange(0, 2 * BN)) // PAGE, mask=(start + BN + tl.arange(0, 2 * BN)) < cap, other=0)
    ks = tl.load(k_scale_ptr)
    q0, rvalid, j = _prep_q(qkv, positions, b, h0, qk_factor, log2_theta, L, G, GP, HQ, HKV, ROPE, BL, M, QK8, NATIVE_STORE, CDTYPE)
    q1, rvalid, j = _prep_q(qkv, positions, b, h0 + 1, qk_factor, log2_theta, L, G, GP, HQ, HKV, ROPE, BL, M, QK8, NATIVE_STORE, CDTYPE)
    # `ahead` (the page ids of tiles 1 and 2) is only read to pull those lines of the page table into L1 / L2 now, in parallel with
    # the q loads, so the pipeline prologue's chain of page loads finds them hot; the * 0.0 term (IEEE: not folded) keeps the load
    # alive without changing the value, and is evaluated after q prep so nothing waits for the page table before the q loads issue.
    c = sm_scale * ks * 1.4426950408889634 + tl.sum(ahead.to(tl.float32), 0) * 0.0   # logits in the log2 domain
    m0 = tl.full([M], float("-inf"), tl.float32)
    l0 = tl.zeros([M], tl.float32)
    a0 = tl.zeros([M, 64], tl.float32)
    m1 = tl.full([M], float("-inf"), tl.float32)
    l1 = tl.zeros([M], tl.float32)
    a1 = tl.zeros([M, 64], tl.float32)

    if s < NSPLIT:
        # ---- cached keys [start, end): nothing to do (zero iterations, no bytes read) when the request ends before the split
        nt = (end - start + BN - 1) // BN
        hb = h0 * 64
        for i in range(nt):
            tc = t0 + i * BN
            kA0, kA1, vA0, vA1 = _gather(k_cache, v_cache, pgA, tc, end, hb, d, PAGE, HKV)
            pgA = _pages(prow, tc + BN, cap, PAGE)
            tm = tc < end
            need_mask = start + (i + 1) * BN > end
            m0, l0, a0 = _step(q0, kA0, vA0, m0, l0, a0, c, tm, need_mask, NATIVE_LOAD, QK8, CDTYPE)
            m1, l1, a1 = _step(q1, kA1, vA1, m1, l1, a1, c, tm, need_mask, NATIVE_LOAD, QK8, CDTYPE)
    else:
        vs = tl.load(v_scale_ptr)
        m0, l0, a0 = _own_head(qkv, positions, cache_loc, k_cache, v_cache, b, h0, q0, j, c, ks, vs, log2_theta, m0, l0, a0,
                               L, HQ, HKV, ROPE, NATIVE_STORE, QK8, CDTYPE, BT)
        m1, l1, a1 = _own_head(qkv, positions, cache_loc, k_cache, v_cache, b, h0 + 1, q1, j, c, ks, vs, log2_theta, m1, l1, a1,
                               L, HQ, HKV, ROPE, NATIVE_STORE, QK8, CDTYPE, BT)

    if (s == NSPLIT) | (start < end):
        r = tl.arange(0, M)
        pidx = ((b * HKV + h0) * M + r) * (NSPLIT + 1) + s     # [b, head, row, split]: a row's splits are contiguous for the reducer
        tl.store(part_m + pidx, m0 * c, mask=rvalid)
        tl.store(part_l + pidx, l0, mask=rvalid)
        tl.store(part_o + pidx[:, None] * 64 + d, a0, mask=rvalid[:, None])
        pidx = pidx + M * (NSPLIT + 1)
        tl.store(part_m + pidx, m1 * c, mask=rvalid)
        tl.store(part_l + pidx, l1, mask=rvalid)
        tl.store(part_o + pidx[:, None] * 64 + d, a1, mask=rvalid[:, None])


@triton.jit
def _gather_cat(k_cache, v_cache, pg, t, end, hb, c, PAGE: tl.constexpr, HKV: tl.constexpr):
    """K and V rows of key positions t for the KV heads hb/64 and hb/64 + 1 as ONE [BN, 128] tile each: the two heads' 64 bytes
    are adjacent in the cache row, so a row is one whole 128-byte line (16-byte accesses, 8 lanes per row)."""
    if PAGE == 1:
        row = pg.to(tl.int64)
    else:
        row = pg.to(tl.int64) * PAGE + t % PAGE
    p = tl.multiple_of(row * (HKV * 64) + hb, 128)[:, None] + c
    m = (t < end)[:, None]
    ku = tl.load(k_cache + p, mask=m, other=0, eviction_policy="evict_first")
    vu = tl.load(v_cache + p, mask=m, other=0, eviction_policy="evict_first")
    return ku, vu


@triton.jit
def _step_cat(q, ku, vu, m_i, l_i, acc, c, tm, need_mask, NATIVE_LOAD: tl.constexpr, QK8: tl.constexpr,
              CDTYPE: tl.constexpr):
    """One tile of a head pair, rows 0..7 = head 0, rows 8..15 = head 1 (q is zero outside its head's 64 columns): one logit
    dot and one softmax for both heads; P.V is computed for all 128 columns and only each head's own 64 are kept."""
    if NATIVE_LOAD:
        vc = vu.to(tl.float8e4nv, bitcast=True).to(CDTYPE)
    else:
        vc = _decode_e4m3(vu).to(CDTYPE)
    if QK8:
        sc = tl.dot(q, tl.trans(ku.to(tl.float8e4nv, bitcast=True)))
    elif NATIVE_LOAD:
        sc = tl.dot(q, tl.trans(ku.to(tl.float8e4nv, bitcast=True).to(CDTYPE)))
    else:
        sc = tl.dot(q, tl.trans(_decode_e4m3(ku).to(CDTYPE)))
    if need_mask:
        sc = tl.where(tm[None, :], sc, float("-inf"))
    return _softmax_step(sc, c, m_i, l_i, acc, vc, CDTYPE)


@triton.jit
def _full_cat_kernel(
    qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale_ptr, v_scale_ptr,
    part_o, part_m, part_l,
    sm_scale, qk_factor, pt_stride, split_len, NSPLIT,
    HQ: tl.constexpr, HKV: tl.constexpr, PAGE: tl.constexpr,
    NATIVE_LOAD: tl.constexpr, NATIVE_STORE: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr,
    BN: tl.constexpr, BT: tl.constexpr,
):
    """Full-attention decode (q_len 1, 8 query heads per KV head, no RoPE). One program = one split of one request for TWO
    adjacent KV heads, and the pair is ONE attention problem: 16 query rows (rows 0..7 head 0, 8..15 head 1) against
    [BN, 128] key / value tiles (both heads' 64 bytes of a cache row = one 128-byte line), with q zero outside its own
    head's columns. Same cp.async pipeline as _full_kernel (page ids one tile ahead in registers, 3 tiles in flight)."""
    s = tl.program_id(0)          # 0 .. NSPLIT-1: cached keys of one split; NSPLIT: the token's own key
    hp = tl.program_id(1)
    b = tl.program_id(2)
    h0 = hp * 2
    c128 = tl.arange(0, 128)[None, :]
    d = c128 % 64
    ar = tl.arange(0, BN)
    cap = pt_stride * PAGE
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    G: tl.constexpr = HQ // HKV

    seq = tl.load(seq_lens + b).to(tl.int32)
    start = s * split_len
    end = tl.minimum(seq, start + split_len)
    prow = page_table + b * pt_stride
    t0 = start + ar
    pgA = _pages(prow, t0, cap, PAGE)
    ahead = tl.load(prow + (start + BN + tl.arange(0, 2 * BN)) // PAGE, mask=(start + BN + tl.arange(0, 2 * BN)) < cap, other=0)
    ks = tl.load(k_scale_ptr)

    # ---- q: row r = query head h0 * G + r (G = 8), RMSNorm, qk_factor, FP8 E4M3; columns outside its own head's 64 are zero
    r = tl.arange(0, 16)
    own = (c128 // 64) == (r // G)[:, None]
    x = tl.load(qkv + b * QKV + (h0 * G + r)[:, None] * 64 + d, mask=own, other=0.0).to(tl.float32)
    xn = x * tl.rsqrt(tl.sum(x * x, axis=1) / 64.0 + 1e-5)[:, None] * qk_factor
    if QK8:
        if NATIVE_STORE:
            q = xn.to(tl.float8e4nv)
        else:
            q = _encode_e4m3(xn).to(tl.float8e4nv, bitcast=True)
    else:
        q = xn.to(CDTYPE)
    c = sm_scale * ks * 1.4426950408889634 + tl.sum(ahead.to(tl.float32), 0) * 0.0   # see _full_kernel
    m_i = tl.full([16], float("-inf"), tl.float32)
    l_i = tl.zeros([16], tl.float32)
    acc = tl.zeros([16, 128], tl.float32)

    if s < NSPLIT:
        nt = (end - start + BN - 1) // BN
        hb = h0 * 64
        for i in range(nt):
            tc = t0 + i * BN
            ku, vu = _gather_cat(k_cache, v_cache, pgA, tc, end, hb, c128, PAGE, HKV)
            pgA = _pages(prow, tc + BN, cap, PAGE)
            m_i, l_i, acc = _step_cat(q, ku, vu, m_i, l_i, acc, c, tc < end, start + (i + 1) * BN > end, NATIVE_LOAD, QK8, CDTYPE)
    else:
        # ---- the token's own key / value (both heads): norm, FP8 round + cache write, attended from registers
        vs = tl.load(v_scale_ptr)
        ti = tl.arange(0, BT)
        tv = ti < 1
        base = qkv + (b + ti)[:, None] * QKV
        kx = tl.load(base + HQ * 64 + h0 * 64 + c128, mask=tv[:, None], other=0.0).to(tl.float32)
        k2 = tl.reshape(kx, [BT, 2, 64])
        k2 = k2 * tl.rsqrt(tl.sum(k2 * k2, axis=2) / 64.0 + 1e-5)[:, :, None]
        kn = tl.reshape(k2, [BT, 128])
        vn = tl.load(base + (HQ + HKV) * 64 + h0 * 64 + c128, mask=tv[:, None], other=0.0).to(tl.float32)
        slot = tl.load(cache_loc + b + ti, mask=tv, other=0)
        coff = slot[:, None] * (HKV * 64) + h0 * 64 + c128
        if NATIVE_STORE:
            k8 = (kn / ks).to(tl.float8e4nv)
            v8 = (vn / vs).to(tl.float8e4nv)
            tl.store(k_cache + coff, k8.to(tl.uint8, bitcast=True), mask=tv[:, None])
            tl.store(v_cache + coff, v8.to(tl.uint8, bitcast=True), mask=tv[:, None])
            kt = k8 if QK8 else k8.to(CDTYPE)
            vt = v8.to(CDTYPE)
        else:
            ku8 = _encode_e4m3(kn / ks)
            vu8 = _encode_e4m3(vn / vs)
            tl.store(k_cache + coff, ku8, mask=tv[:, None])
            tl.store(v_cache + coff, vu8, mask=tv[:, None])
            kt = ku8.to(tl.float8e4nv, bitcast=True) if QK8 else _decode_e4m3(ku8).to(CDTYPE)
            vt = _decode_e4m3(vu8).to(CDTYPE)
        sc = tl.dot(q, tl.trans(kt))
        sc = tl.where(tv[None, :], sc, float("-inf"))
        m_i, l_i, acc = _softmax_step(sc, c, m_i, l_i, acc, vt, CDTYPE)

    if (s == NSPLIT) | (start < end):
        hh = r // G
        pidx = ((b * HKV + h0 + hh) * 16 + (r % G)) * (NSPLIT + 1) + s     # [b, head, row, split]
        tl.store(part_m + pidx, m_i * c)
        tl.store(part_l + pidx, l_i)
        tl.store(part_o + pidx[:, None] * 64 + d, acc, mask=own)


# --------------------------------------------------------------------------------------------- kernel 2
@triton.jit
def _reduce_kernel(
    part_o, part_m, part_l, qkv, seq_lens, v_scale_ptr, out, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, WINDOW: tl.constexpr,
    M: tl.constexpr, BS: tl.constexpr, ROWMAJOR: tl.constexpr,
):
    t = tl.program_id(0)        # b * L + j
    hq = tl.program_id(1)
    b = t // L
    j = t % L
    h = hq // G
    row = j * GP + hq % G
    d = tl.arange(0, 64)
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64

    seq = tl.load(seq_lens + b).to(tl.int32)
    lo = tl.maximum(seq - WINDOW + 1, 0) if WINDOW else 0
    n_active = (seq - lo + split_len - 1) // split_len      # cached splits with keys; the block's own one follows them
    if ROWMAJOR:
        base = ((b * HKV + h) * M + row) * (NSPLIT + 1)        # [b, head, row, split]: one contiguous run per output row
    else:
        base = (b * HKV + h) * (NSPLIT + 1) * M + row          # [b, head, split, row] (v2's kernel)
    gate = tl.load(qkv + t * QKV + (HQ + 2 * HKV) * 64 + hq * 64 + d).to(tl.float32)

    Mx = tl.full([1], float("-inf"), tl.float32)
    Lsum = tl.zeros([1], tl.float32)
    O = tl.zeros([64], tl.float32)
    for s0 in range(0, n_active + 1, BS):
        si = s0 + tl.arange(0, BS)
        sv = si <= n_active
        if ROWMAJOR:
            pidx = base + tl.where(si == n_active, NSPLIT, si)
        else:
            pidx = base + tl.where(si == n_active, NSPLIT, si) * M
        ms = tl.load(part_m + pidx, mask=sv, other=float("-inf"))
        ls = tl.load(part_l + pidx, mask=sv, other=0.0)
        os_ = tl.load(part_o + pidx[:, None] * 64 + d[None, :], mask=sv[:, None], other=0.0)
        m_new = tl.maximum(Mx, tl.max(ms, 0))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(Mx - m_use)
        w = tl.exp2(ms - m_use)
        Lsum = Lsum * alpha + tl.sum(w * ls, 0)
        O = O * alpha + tl.sum(w[:, None] * os_, 0)
        Mx = m_new

    o = O / Lsum * tl.load(v_scale_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
    o = o * (1.0 / (1.0 + tl.exp(-gate)))
    tl.store(out + t * (HQ * 64) + hq * 64 + d, o.to(tl.bfloat16))


# --------------------------------------------------------------------------------------------- host
@functools.lru_cache(maxsize=None)
def _device_mode(device_type: str, index: int):
    """(V / fallback compute dtype, FP8 loads native, FP8 stores / q rounding native, q.K in FP8): hardware only, never input
    values. sm_89+ converts FP8 in hardware and has FP8 tensor-core MMAs (q rounded to E4M3, K used as cached, FP32
    accumulation). Elsewhere (the sm_75 dev GPU) FP8 is decoded / encoded in software and q.K runs in FP16 (exact E4M3
    products); under the Triton CPU interpreter loads and FP8 dots are exact but its fp32->e4m3 cast truncates near binade
    edges, so q rounding and stores are software there."""
    if device_type != "cuda":
        return tl.float32, True, False, True
    major, minor = torch.cuda.get_device_capability(index)
    native = (major, minor) >= (8, 9)
    return tl.float16, native, native, native    # fp16 mma operands: exact for E4M3, 3 more mantissa bits than BF16


def _cdiv(a, b):
    return -(-a // b)


def _plan(batch, hkv, span, rows, bn, occ):
    """Split length for a shape (shapes only, never values). Small cost model in microseconds: a program's time is a fixed
    chain (seq_lens -> page index -> K/V rows -> write) plus ~0.8 us per tile; the whole grid can't finish before the bytes
    are moved at ~3 TB/s (64-byte random rows) scaled by how full the machine is; the last programs to start finish up to
    half a program after the rest (tail: ~0.4 of a program's share of the bytes); every split adds a partial (rows x 64 fp32, written then read back) and reduce-loop
    iterations."""
    pairs = batch * hkv
    slots = SM_COUNT * occ
    best = None
    for mult in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048):
        split_len = mult * bn
        ns = _cdiv(span, split_len)
        progs = pairs * (ns + 1)
        waves = progs / slots
        tiles = split_len / bn
        t_lat = max(1.0, waves) * (3.0 + tiles * 0.8)
        util = min(1.0, waves) if waves < 1 else waves / math.ceil(waves)
        t_bw = pairs * span * 128 / 3.0e6
        per_slot = pairs * span / bn / slots                       # tiles each slot streams in total
        t_tail = 0.4 * t_bw * min(1.0, tiles / max(per_slot, 1e-9))
        t_part = progs * rows * 264 * 2 / 8.0e6
        t_red = 1.5 + _cdiv(ns + 1, 128) * 0.6
        est = max(t_lat, t_bw / util) + t_tail + t_part + t_red
        if best is None or est < best[0] * 0.995:
            best = (est, split_len, ns)
        if ns == 1:
            break
    return best[1], best[2]


def _plan_full(batch, pairs, span, bn, programs):
    """Split length of the full-layer kernels (shapes only): about `programs` programs on the machine, at least FULL_MIN_TILES
    tiles per split, at most MAX_SPLITS splits (one reducer pass). A program streams its split through a 3-tile cp.async
    pipeline, so it is bandwidth-bound once running and costs ~3 us to start; more programs only add startup and partials."""
    ns = min(MAX_SPLITS, max(1, round(programs / (batch * pairs))))
    split_len = max(FULL_MIN_TILES * bn, _cdiv(_cdiv(span, ns), bn) * bn)
    return split_len, _cdiv(span, split_len)


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page, window = cfg["page_size"], cfg["window"]
    group = hq // hkv
    gp = triton.next_power_of_2(group)
    bl = max(triton.next_power_of_2(length), 16 // gp)
    rows = bl * gp
    if d != 64 or hq % hkv or rows > MAX_ROWS or length > BLOCK_N:
        raise ValueError(f"unsupported config: head_dim {d}, {hq} q heads / {hkv} kv heads, q_len {length}")
    block_t = max(16, triton.next_power_of_2(length))
    qkv = data["qkv"]
    dev = qkv.device
    cdtype, native_load, native_store, qk8 = _device_mode(dev.type, dev.index or 0)

    max_pages = data["page_table"].shape[1]
    cap = max_pages * page                       # most tokens a request's page table can hold
    span = max(1, min(window - 1, cap) if window else cap)
    page_table = data["page_table"]
    k_cache = data["k_cache"].view(torch.uint8)
    v_cache = data["v_cache"].view(torch.uint8)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    theta = math.log2(cfg["rope_theta"]) if cfg["rope_theta"] else 0.0

    full_layer = not window and hkv % 2 == 0
    if full_layer:
        # ---- full-attention layers: both KV heads of a pair per program, software-pipelined gathers
        fc = dict(FULL_CFG["decode" if length == 1 else "verify"])
        if length > 1 and batch >= FULL_BIG_MIN_BATCH:
            fc = dict(FULL_BIG)
        # few programs (batch 1): every SM runs one or two, nothing shares the issue slots, so a program's own per-tile chain is what
        # counts -> 4 warps share each tile; with a saturated machine the 2-warp program's lower instruction count wins
        if fc["warps"] < 4 and batch * _plan_full(batch, hkv // 2, span, fc["bn"], fc["programs"])[1] <= 2 * SM_COUNT:
            fc["warps"] = 4
        use_ws = (length > 1 and batch >= WS_MIN_BATCH and page == 1 and rows == 64 and native_load and qk8
                  and dev.type == "cuda" and torch.cuda.get_device_capability(dev.index or 0)[0] == 9)
        if use_ws:
            fc = dict(WS_CFG)
        split_len, nsplit = _plan_full(batch, hkv // 2, span, fc["bn"], fc["programs"])
        part_o = torch.empty((batch, hkv, nsplit + 1, rows, 64), dtype=torch.float32, device=dev)
        part_m = torch.empty((batch, hkv, nsplit + 1, rows), dtype=torch.float32, device=dev)
        part_l = torch.empty_like(part_m)
        if use_ws:
            kv_l = gl.NVMMASharedLayout.get_default_for([fc["bn"], 64], gl.float8e4nv)
            v16_l = gl.NVMMASharedLayout.get_default_for([fc["bn"], 64], gl.float16)
            _ws_full_verify_kernel[(nsplit, hkv // 2, batch)](
                qkv, data["seq_lens"], page_table, data["k_cache"], data["v_cache"], data["k_scale"],
                part_o, part_m, part_l, cfg["sm_scale"], cfg["qk_scale_factor"], page_table.stride(0), split_len, nsplit,
                L=length, G=group, GP=gp, HQ=hq, HKV=hkv, M=rows, BN=fc["bn"], NBUF=fc["nbuf"],
                kv_layout=kv_l, v16_layout=v16_l, REG_CONS=fc["rc"], REG_PROD=fc["rp"],
                PF=2, PMODE=0, CMODE=6, P8=False, O_K=16, KW=2, PSH=0.0, VDT=gl.float16,
                num_warps=4, maxnreg=fc["maxnreg"],
            )
            _full_kernel[(1, hkv // 2, batch)](
                qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, k_cache, v_cache,
                data["k_scale"], data["v_scale"], part_o, part_m, part_l,
                cfg["sm_scale"], cfg["qk_scale_factor"], theta, page_table.stride(0), split_len, nsplit,
                L=length, G=group, GP=gp, HQ=hq, HKV=hkv, PAGE=page, ROPE=bool(cfg["rope_theta"]),
                NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype, BL=bl, M=rows,
                BN=64, BT=block_t, S_OFFSET=nsplit, num_warps=4, num_stages=1,
            )
        elif fc["kernel"] == "cat" and length == 1 and group == 8 and not cfg["rope_theta"]:
            _full_cat_kernel[(nsplit + 1, hkv // 2, batch)](
                qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, k_cache, v_cache,
                data["k_scale"], data["v_scale"], part_o, part_m, part_l,
                cfg["sm_scale"], cfg["qk_scale_factor"], page_table.stride(0), split_len, nsplit,
                HQ=hq, HKV=hkv, PAGE=page, NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype,
                BN=fc["bn"], BT=block_t, num_warps=fc["warps"], num_stages=fc["stages"],
            )
        else:
            _full_kernel[(nsplit + 1, hkv // 2, batch)](
                qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, k_cache, v_cache,
                data["k_scale"], data["v_scale"], part_o, part_m, part_l,
                cfg["sm_scale"], cfg["qk_scale_factor"], theta, page_table.stride(0), split_len, nsplit,
                L=length, G=group, GP=gp, HQ=hq, HKV=hkv, PAGE=page, ROPE=bool(cfg["rope_theta"]),
                NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype, BL=bl, M=rows,
                BN=fc["bn"], BT=block_t, num_warps=fc["warps"], num_stages=fc["stages"],
            )
    else:
        occ = 4 if rows <= 16 else 3
        split_len, nsplit = _plan(batch, hkv, span, rows, BLOCK_N, occ)
        part_o = torch.empty((batch, hkv, nsplit + 1, rows, 64), dtype=torch.float32, device=dev)
        part_m = torch.empty((batch, hkv, nsplit + 1, rows), dtype=torch.float32, device=dev)
        part_l = torch.empty_like(part_m)
        _attn_kernel[(nsplit + 1, hkv, batch)](
            qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, k_cache, v_cache,
            data["k_scale"], data["v_scale"], part_o, part_m, part_l,
            cfg["sm_scale"], cfg["qk_scale_factor"], theta,
            page_table.stride(0), split_len, nsplit,
            L=length, G=group, GP=gp, HQ=hq, HKV=hkv, PAGE=page, WINDOW=window, ROPE=bool(cfg["rope_theta"]),
            EDGE=bool(window) and length > 1, NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype,
            BL=bl, M=rows, BN=BLOCK_N, BT=block_t,
            num_warps=4, num_stages=3 if split_len >= 3 * BLOCK_N else 1,
        )
    _reduce_kernel[(batch * length, hq)](
        part_o, part_m, part_l, qkv, data["seq_lens"], data["v_scale"], out, split_len, nsplit,
        L=length, G=group, GP=gp, HQ=hq, HKV=hkv, WINDOW=window, M=rows, ROWMAJOR=full_layer,
        BS=min(256, triton.next_power_of_2(nsplit + 1)), num_warps=8 if nsplit + 1 > 128 else 4, num_stages=1,
    )
    return out


# ================================================================================================================ G1: Gluon verify-full
from triton.experimental import gluon as _gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma as _g_wgmma, warpgroup_mma_wait as _g_wgmma_wait, fence_async_shared as _g_fence)
from triton.experimental.gluon.language.nvidia.ampere import async_copy as _g_cp


@_gluon.jit
def _g_issue(k_cache, v_cache, k_smem, v_smem, prow, t, cap, end, hb, slot, cols, HKV: gl.constexpr):
    """cp.async the K and V rows of key positions t (both heads of the pair: two 64-byte halves of one row) into ring slot `slot`."""
    pg = gl.load(prow + t, mask=t < cap, other=0)
    ok = (t < end)[:, None] & (cols < 64)[None, :]
    row = pg.to(gl.int64)[:, None] * (HKV * 64) + hb + cols[None, :]
    _g_cp.async_copy_global_to_shared(k_smem.index(2 * slot), k_cache + row, mask=ok)
    _g_cp.async_copy_global_to_shared(k_smem.index(2 * slot + 1), k_cache + (row + 64), mask=ok)
    _g_cp.async_copy_global_to_shared(v_smem.index(2 * slot), v_cache + row, mask=ok)
    _g_cp.async_copy_global_to_shared(v_smem.index(2 * slot + 1), v_cache + (row + 64), mask=ok)
    _g_cp.commit_group()


@_gluon.jit
def _g_prep_q(qkv, b, h, qk_factor, r, dq, q_op: gl.constexpr,
              L: gl.constexpr, G: gl.constexpr, GP: gl.constexpr, QKV: gl.constexpr):
    """Query rows of one KV head (row r = token r // GP, q head h*G + r % GP): RMSNorm, qk_factor, E4M3 (as sonnet_v5)."""
    j = r // GP
    rvalid = (j < L) & ((r % GP) < G)
    base = qkv + ((b * L + j) * QKV + (h * G + r % GP) * 64)[:, None] + dq[None, :]
    x = gl.load(base, mask=rvalid[:, None] & (dq < 64)[None, :], other=0.0).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, axis=1) / 64.0 + 1e-5)
    xn = x * inv[:, None]
    xn = xn * qk_factor
    return gl.convert_layout(xn.to(gl.float8e4nv), q_op)


@_gluon.jit
def _g_head(q, k_buf, v_buf, v16, m_i, l_i, acc, c, tm,
            s_l: gl.constexpr, o_l: gl.constexpr, p_op: gl.constexpr, ld: gl.constexpr, M: gl.constexpr, BN: gl.constexpr):
    """One tile of one KV head: S = q.K^T (FP8 wgmma), online softmax (log2 domain), O += P.V (FP16 wgmma, V converted via smem)."""
    sc = _g_wgmma(q, k_buf.permute((1, 0)), gl.zeros([M, BN], gl.float32, layout=s_l), use_acc=False, is_async=True)
    sc = _g_wgmma_wait(num_outstanding=0, deps=(sc,))
    sc = gl.where(tm[None, :], sc, float("-inf"))
    m_new = gl.maximum(m_i, gl.max(sc, axis=1))
    m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
    p = gl.exp2(sc * c - (m_use * c)[:, None])
    alpha = gl.exp2((m_i - m_use) * c)
    l_i = l_i * alpha + gl.sum(p, axis=1)
    gl.barrier()
    v16.store(v_buf.load(ld).to(gl.float16))
    _g_fence()
    gl.barrier()
    a_o = gl.convert_layout(alpha, gl.SliceLayout(1, o_l))
    acc = _g_wgmma(gl.convert_layout(p.to(gl.float16), p_op), v16, acc * a_o[:, None], is_async=True)
    acc = _g_wgmma_wait(num_outstanding=0, deps=(acc,))
    return m_new, l_i, acc


@_gluon.jit
def _g_full_verify_kernel(qkv, seq_lens, page_table, k_cache, v_cache, k_scale_ptr,
                          part_o, part_m, part_l, sm_scale, qk_factor, pt_stride, split_len, NSPLIT,
                          L: gl.constexpr, G: gl.constexpr, GP: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr,
                          M: gl.constexpr, BN: gl.constexpr, NBUF: gl.constexpr,
                          kv_layout: gl.constexpr, v16_layout: gl.constexpr):
    """Cached keys of one split, both KV heads of a pair (page size 1). Partials in sonnet_v5's layout for its reducer."""
    s = gl.program_id(0)
    hp = gl.program_id(1)
    b = gl.program_id(2)
    h0 = hp * 2
    QKV: gl.constexpr = (2 * HQ + 2 * HKV) * 64
    ld: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    qb: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    s_l: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    o_l: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16])
    q_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=s_l, k_width=4)
    p_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=o_l, k_width=2)

    seq = gl.load(seq_lens + b)
    start = s * split_len
    end = gl.minimum(seq, start + split_len)
    if start < end:
        ks = gl.load(k_scale_ptr)
        c = sm_scale * ks * 1.4426950408889634
        r = gl.arange(0, M, layout=gl.SliceLayout(1, qb))
        dq = gl.arange(0, 64, layout=gl.SliceLayout(0, qb))
        q0 = _g_prep_q(qkv, b, h0, qk_factor, r, dq, q_op, L, G, GP, QKV)
        q1 = _g_prep_q(qkv, b, h0 + 1, qk_factor, r, dq, q_op, L, G, GP, QKV)

        prow = page_table + b * pt_stride
        rows = gl.arange(0, BN, layout=gl.SliceLayout(1, ld))
        cols = gl.arange(0, 64, layout=gl.SliceLayout(0, ld))
        tcol = gl.arange(0, BN, layout=gl.SliceLayout(0, s_l))
        hb = h0 * 64
        k_smem = gl.allocate_shared_memory(gl.float8e4nv, [2 * NBUF, BN, 64], kv_layout)
        v_smem = gl.allocate_shared_memory(gl.float8e4nv, [2 * NBUF, BN, 64], kv_layout)
        v16 = gl.allocate_shared_memory(gl.float16, [BN, 64], v16_layout)

        for i in gl.static_range(NBUF - 1):
            _g_issue(k_cache, v_cache, k_smem, v_smem, prow, start + i * BN + rows, pt_stride, end, hb, i, cols, HKV)

        m0 = gl.full([M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, s_l))
        l0 = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, s_l))
        a0 = gl.zeros([M, 64], gl.float32, layout=o_l)
        m1 = gl.full([M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, s_l))
        l1 = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, s_l))
        a1 = gl.zeros([M, 64], gl.float32, layout=o_l)

        nt = gl.cdiv(end - start, BN)
        for i in range(nt):
            gl.barrier()
            nxt = i + NBUF - 1
            _g_issue(k_cache, v_cache, k_smem, v_smem, prow, start + nxt * BN + rows, pt_stride, end, hb, nxt % NBUF, cols, HKV)
            _g_cp.wait_group(NBUF - 1)
            gl.barrier()
            _g_fence()
            slot = i % NBUF
            tm = (start + i * BN + tcol) < end
            m0, l0, a0 = _g_head(q0, k_smem.index(2 * slot), v_smem.index(2 * slot), v16, m0, l0, a0, c, tm,
                                 s_l, o_l, p_op, ld, M, BN)
            m1, l1, a1 = _g_head(q1, k_smem.index(2 * slot + 1), v_smem.index(2 * slot + 1), v16, m1, l1, a1, c, tm,
                                 s_l, o_l, p_op, ld, M, BN)
        _g_cp.wait_group(0)

        rs = gl.arange(0, M, layout=gl.SliceLayout(1, s_l))
        rv_s = ((rs // GP) < L) & ((rs % GP) < G)
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, o_l))
        do = gl.arange(0, 64, layout=gl.SliceLayout(0, o_l))
        rv_o = (((ro // GP) < L) & ((ro % GP) < G))[:, None] & (do < 64)[None, :]
        ps = ((b * HKV + h0) * M + rs) * (NSPLIT + 1) + s
        po = ((b * HKV + h0) * M + ro) * (NSPLIT + 1) + s
        gl.store(part_m + ps, m0 * c, mask=rv_s)
        gl.store(part_l + ps, l0, mask=rv_s)
        gl.store(part_o + po[:, None] * 64 + do[None, :], a0, mask=rv_o)
        ps = ps + M * (NSPLIT + 1)
        po = po + M * (NSPLIT + 1)
        gl.store(part_m + ps, m1 * c, mask=rv_s)
        gl.store(part_l + ps, l1, mask=rv_s)
        gl.store(part_o + po[:, None] * 64 + do[None, :], a1, mask=rv_o)



# ================================================================================================================ G2: warp-specialized
from triton.experimental.gluon.language.nvidia.hopper import mbarrier as _g_mbar


@_gluon.jit
def _ws_producer(p, L: gl.constexpr, G: gl.constexpr, GP: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr,
                 M: gl.constexpr, BN: gl.constexpr, NBUF: gl.constexpr, PF: gl.constexpr, PMODE: gl.constexpr,
                 CMODE: gl.constexpr, P8: gl.constexpr, O_K: gl.constexpr, KW: gl.constexpr, PSH: gl.constexpr,
                 VDT: gl.constexpr):
    """Producer partition (4 warps): cp.async K / V rows of tile i into ring slot i % NBUF, NBUF-1 tiles ahead, page ids of
    tiles i+1, i+2 already in flight (PF). When tile k has landed: V halves to vt (FP16 converted, or FP8 transposed when P8),
    each thread touching only the bytes it copied; fence; signal ready[k].
    PMODE 1: no V conversion; PMODE 2: no copies at all (timing twins, wrong output)."""
    (qkv, prow, k_cache, v_cache, k_smem, v8_smem, vt, ready, empty, part_o, part_m, part_l,
     b, h0, s, start, end, nt, c, qk_factor, NSPLIT, cap) = p
    ld: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    rows = gl.arange(0, BN, layout=gl.SliceLayout(1, ld))
    cols = gl.arange(0, 64, layout=gl.SliceLayout(0, ld))
    hb = h0 * 64
    PRE: gl.constexpr = NBUF - 1 - (CMODE + 6) // 8
    t0 = start + rows
    pgA = gl.load(prow + t0, mask=t0 < cap, other=0)
    t1 = start + BN + rows
    pgB = gl.load(prow + t1, mask=t1 < cap, other=0)
    for i in range(nt + PRE):
        if i < nt:
            slot = i % NBUF
            _g_mbar.wait(empty.index(slot), ((i // NBUF) & 1) ^ 1)
            if PMODE != 2:
                t = start + i * BN + rows
                if PF > 0:
                    pg = pgA
                else:
                    pg = gl.load(prow + t, mask=t < cap, other=0)
                ok = (t < end)[:, None] & (cols < 64)[None, :]
                row = pg.to(gl.int64)[:, None] * (HKV * 64) + hb + cols[None, :]
                _g_cp.async_copy_global_to_shared(k_smem.index(2 * slot), k_cache + row, mask=ok)
                _g_cp.async_copy_global_to_shared(k_smem.index(2 * slot + 1), k_cache + (row + 64), mask=ok)
                _g_cp.async_copy_global_to_shared(v8_smem.index(2 * slot), v_cache + row, mask=ok)
                _g_cp.async_copy_global_to_shared(v8_smem.index(2 * slot + 1), v_cache + (row + 64), mask=ok)
                if PF > 0:
                    pgA = pgB
                    t2 = start + (i + 2) * BN + rows
                    pgB = gl.load(prow + t2, mask=t2 < cap, other=0)
        _g_cp.commit_group()
        if i >= PRE:
            kk = (i - PRE) % NBUF
            _g_cp.wait_group(PRE)
            if PMODE == 0:
                if P8:
                    vt.index(2 * kk).store(gl.permute(v8_smem.index(2 * kk).load(ld), (1, 0)))
                    vt.index(2 * kk + 1).store(gl.permute(v8_smem.index(2 * kk + 1).load(ld), (1, 0)))
                else:
                    vt.index(2 * kk).store(v8_smem.index(2 * kk).load(ld).to(gl.float16))
                    vt.index(2 * kk + 1).store(v8_smem.index(2 * kk + 1).load(ld).to(gl.float16))
            _g_fence()
            _g_mbar.arrive(ready.index(kk), count=1)


@_gluon.jit
def _ws_consume(p, HEAD: gl.constexpr, L: gl.constexpr, G: gl.constexpr, GP: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr,
                 M: gl.constexpr, BN: gl.constexpr, NBUF: gl.constexpr, PF: gl.constexpr, PMODE: gl.constexpr,
                 CMODE: gl.constexpr, P8: gl.constexpr, O_K: gl.constexpr, KW: gl.constexpr, PSH: gl.constexpr,
                 VDT: gl.constexpr):
    """Consumer partition (one warpgroup) for KV head h0 + HEAD: wait ready, S = q.K^T (FP8 wgmma), online softmax,
    O += P.V (FP16 wgmma, or FP8 with P x 2^PSH when P8), release the slot, store the split's partial.
    CMODE 1: no math at all (timing twin, wrong output)."""
    (qkv, prow, k_cache, v_cache, k_smem, v8_smem, vt, ready, empty, part_o, part_m, part_l,
     b, h0, s, start, end, nt, c, qk_factor, NSPLIT, cap) = p
    QKV: gl.constexpr = (2 * HQ + 2 * HKV) * 64
    qb: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    s_l: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    o_l: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, O_K])
    q_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=s_l, k_width=4)
    p_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=o_l, k_width=KW)
    if nt > 0:
        r = gl.arange(0, M, layout=gl.SliceLayout(1, qb))
        dq = gl.arange(0, 64, layout=gl.SliceLayout(0, qb))
        q = _g_prep_q(qkv, b, h0 + HEAD, qk_factor, r, dq, q_op, L, G, GP, QKV)
        tcol = gl.arange(0, BN, layout=gl.SliceLayout(0, s_l))
        m_i = gl.full([M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, s_l))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, s_l))
        acc = gl.zeros([M, 64], gl.float32, layout=o_l)
        for i in range(nt):
            slot = i % NBUF
            _g_mbar.wait(ready.index(slot), (i // NBUF) & 1)
            if CMODE == 0:
                sc = _g_wgmma(q, k_smem.index(2 * slot + HEAD).permute((1, 0)),
                              gl.zeros([M, BN], gl.float32, layout=s_l), use_acc=False, is_async=True)
                sc = _g_wgmma_wait(num_outstanding=0, deps=(sc,))
                tm = (start + i * BN + tcol) < end
                sc = gl.where(tm[None, :], sc, float("-inf"))
                m_new = gl.maximum(m_i, gl.max(sc, axis=1))
                m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
                pr = gl.exp2(sc * c - (m_use * c - PSH)[:, None])
                alpha = gl.exp2((m_i - m_use) * c)
                l_i = l_i * alpha + gl.sum(pr, axis=1)
                m_i = m_new
                a_o = gl.convert_layout(alpha, gl.SliceLayout(1, o_l))
                if P8:
                    bop = vt.index(2 * slot + HEAD).permute((1, 0))
                else:
                    bop = vt.index(2 * slot + HEAD)
                acc = _g_wgmma(gl.convert_layout(pr.to(VDT), p_op), bop, acc * a_o[:, None], is_async=True)
                acc = _g_wgmma_wait(num_outstanding=0, deps=(acc,))
            _g_fence()
            _g_mbar.arrive(empty.index(slot), count=1)
        if P8:
            l_i = l_i * 0.00390625
            acc = acc * 0.00390625
        rs = gl.arange(0, M, layout=gl.SliceLayout(1, s_l))
        rv_s = ((rs // GP) < L) & ((rs % GP) < G)
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, o_l))
        do = gl.arange(0, 64, layout=gl.SliceLayout(0, o_l))
        rv_o = (((ro // GP) < L) & ((ro % GP) < G))[:, None] & (do < 64)[None, :]
        ps = ((b * HKV + h0 + HEAD) * M + rs) * (NSPLIT + 1) + s
        po = ((b * HKV + h0 + HEAD) * M + ro) * (NSPLIT + 1) + s
        gl.store(part_m + ps, m_i * c, mask=rv_s)
        gl.store(part_l + ps, l_i, mask=rv_s)
        gl.store(part_o + po[:, None] * 64 + do[None, :], acc, mask=rv_o)


@_gluon.jit
def _ov_step(sc, m_i, l_i, c, MODE: gl.constexpr):
    """Online-softmax step of the overlapped consumer. MODE 3: no exp2 (timing twin); MODE 4: no softmax at all (timing twin)."""
    if MODE == 4:
        pr = sc
        alpha = gl.zeros_like(m_i) + 1.0
        m_new = m_i
    else:
        m_new = gl.maximum(m_i, gl.max(sc, axis=1))
        m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
        if MODE == 3:
            pr = sc * c - (m_use * c)[:, None]
        else:
            pr = gl.exp2(sc * c - (m_use * c)[:, None])
        alpha = gl.exp2((m_i - m_use) * c)
        l_i = l_i * alpha + gl.sum(pr, axis=1)
    return m_new, l_i, pr, alpha


@_gluon.jit
def _ws_consume_ov(p, HEAD: gl.constexpr, L: gl.constexpr, G: gl.constexpr, GP: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr,
                 M: gl.constexpr, BN: gl.constexpr, NBUF: gl.constexpr, PF: gl.constexpr, PMODE: gl.constexpr,
                 CMODE: gl.constexpr, P8: gl.constexpr, O_K: gl.constexpr, KW: gl.constexpr, PSH: gl.constexpr,
                 VDT: gl.constexpr):
    """Consumer (one warpgroup, KV head h0 + HEAD), FA3-style overlap: q.K of tile i and P.V of tile i-1 in flight together,
    softmax of tile i while P.V(i-1) runs. CMODE 2: as measured in v5; 6: mask only the tail tile; 3 / 4 / 5: timing twins
    (no exp2 / no softmax / no P.V; wrong output)."""
    (qkv, prow, k_cache, v_cache, k_smem, v8_smem, vt, ready, empty, part_o, part_m, part_l,
     b, h0, s, start, end, nt, c, qk_factor, NSPLIT, cap) = p
    QKV: gl.constexpr = (2 * HQ + 2 * HKV) * 64
    qb: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    s_l: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    o_l: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16])
    q_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=s_l, k_width=4)
    p_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=o_l, k_width=2)
    if nt > 0:
        r = gl.arange(0, M, layout=gl.SliceLayout(1, qb))
        dq = gl.arange(0, 64, layout=gl.SliceLayout(0, qb))
        q = _g_prep_q(qkv, b, h0 + HEAD, qk_factor, r, dq, q_op, L, G, GP, QKV)
        tcol = gl.arange(0, BN, layout=gl.SliceLayout(0, s_l))
        m_i = gl.full([M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, s_l))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, s_l))

        _g_mbar.wait(ready.index(0), 0)
        sc = _g_wgmma(q, k_smem.index(HEAD).permute((1, 0)), gl.zeros([M, BN], gl.float32, layout=s_l),
                      use_acc=False, is_async=True)
        sc = _g_wgmma_wait(num_outstanding=0, deps=(sc,))
        sc = gl.where(((start + tcol) < end)[None, :], sc, float("-inf"))
        m_i, l_i, pr, alpha = _ov_step(sc, m_i, l_i, c, CMODE)
        acc = gl.zeros([M, 64], gl.float32, layout=o_l)
        pp = gl.convert_layout(pr.to(gl.float16), p_op)

        for i in range(1, nt):
            slot = i % NBUF
            pslot = (i - 1) % NBUF
            _g_mbar.wait(ready.index(slot), (i // NBUF) & 1)
            sc = _g_wgmma(q, k_smem.index(2 * slot + HEAD).permute((1, 0)), gl.zeros([M, BN], gl.float32, layout=s_l),
                          use_acc=False, is_async=True)
            if CMODE == 5:
                sc = _g_wgmma_wait(num_outstanding=0, deps=(sc,))
            else:
                acc = _g_wgmma(pp, vt.index(2 * pslot + HEAD), acc, is_async=True)
                sc = _g_wgmma_wait(num_outstanding=1, deps=(sc,))
            if CMODE == 6:
                if start + (i + 1) * BN > end:
                    sc = gl.where(((start + i * BN + tcol) < end)[None, :], sc, float("-inf"))
            else:
                sc = gl.where(((start + i * BN + tcol) < end)[None, :], sc, float("-inf"))
            m_i, l_i, pr, alpha = _ov_step(sc, m_i, l_i, c, CMODE)
            if CMODE != 5:
                acc = _g_wgmma_wait(num_outstanding=0, deps=(acc,))
            _g_fence()
            _g_mbar.arrive(empty.index(pslot), count=1)
            if CMODE != 4:
                acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, o_l))[:, None]
            pp = gl.convert_layout(pr.to(gl.float16), p_op)

        lslot = (nt - 1) % NBUF
        if CMODE != 5:
            acc = _g_wgmma(pp, vt.index(2 * lslot + HEAD), acc, is_async=True)
            acc = _g_wgmma_wait(num_outstanding=0, deps=(acc,))
        _g_fence()
        _g_mbar.arrive(empty.index(lslot), count=1)

        rs = gl.arange(0, M, layout=gl.SliceLayout(1, s_l))
        rv_s = ((rs // GP) < L) & ((rs % GP) < G)
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, o_l))
        do = gl.arange(0, 64, layout=gl.SliceLayout(0, o_l))
        rv_o = (((ro // GP) < L) & ((ro % GP) < G))[:, None] & (do < 64)[None, :]
        ps = ((b * HKV + h0 + HEAD) * M + rs) * (NSPLIT + 1) + s
        po = ((b * HKV + h0 + HEAD) * M + ro) * (NSPLIT + 1) + s
        gl.store(part_m + ps, m_i * c, mask=rv_s)
        gl.store(part_l + ps, l_i, mask=rv_s)
        gl.store(part_o + po[:, None] * 64 + do[None, :], acc, mask=rv_o)


@_gluon.jit
def _ws_part0(p, L: gl.constexpr, G: gl.constexpr, GP: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr,
                 M: gl.constexpr, BN: gl.constexpr, NBUF: gl.constexpr, PF: gl.constexpr, PMODE: gl.constexpr,
                 CMODE: gl.constexpr, P8: gl.constexpr, O_K: gl.constexpr, KW: gl.constexpr, PSH: gl.constexpr,
                 VDT: gl.constexpr):
    if CMODE >= 2:
        _ws_consume_ov(p, 0, L, G, GP, HQ, HKV, M, BN, NBUF, PF, PMODE, CMODE, P8, O_K, KW, PSH, VDT)
    else:
        _ws_consume(p, 0, L, G, GP, HQ, HKV, M, BN, NBUF, PF, PMODE, CMODE, P8, O_K, KW, PSH, VDT)


@_gluon.jit
def _ws_part1(p, L: gl.constexpr, G: gl.constexpr, GP: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr,
                 M: gl.constexpr, BN: gl.constexpr, NBUF: gl.constexpr, PF: gl.constexpr, PMODE: gl.constexpr,
                 CMODE: gl.constexpr, P8: gl.constexpr, O_K: gl.constexpr, KW: gl.constexpr, PSH: gl.constexpr,
                 VDT: gl.constexpr):
    if CMODE >= 2:
        _ws_consume_ov(p, 1, L, G, GP, HQ, HKV, M, BN, NBUF, PF, PMODE, CMODE, P8, O_K, KW, PSH, VDT)
    else:
        _ws_consume(p, 1, L, G, GP, HQ, HKV, M, BN, NBUF, PF, PMODE, CMODE, P8, O_K, KW, PSH, VDT)


@_gluon.jit
def _ws_full_verify_kernel(qkv, seq_lens, page_table, k_cache, v_cache, k_scale_ptr,
                           part_o, part_m, part_l, sm_scale, qk_factor, pt_stride, split_len, NSPLIT,
                           L: gl.constexpr, G: gl.constexpr, GP: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr,
                           M: gl.constexpr, BN: gl.constexpr, NBUF: gl.constexpr,
                           kv_layout: gl.constexpr, v16_layout: gl.constexpr,
                           REG_CONS: gl.constexpr, REG_PROD: gl.constexpr, PF: gl.constexpr, PMODE: gl.constexpr,
                           CMODE: gl.constexpr, P8: gl.constexpr, O_K: gl.constexpr, KW: gl.constexpr, PSH: gl.constexpr,
                           VDT: gl.constexpr):
    s = gl.program_id(0)
    hp = gl.program_id(1)
    b = gl.program_id(2)
    h0 = hp * 2
    seq = gl.load(seq_lens + b)
    start = s * split_len
    end = gl.minimum(seq, start + split_len)
    nt = gl.maximum(gl.cdiv(end - start, BN), 0)
    ks = gl.load(k_scale_ptr)
    c = sm_scale * ks * 1.4426950408889634
    prow = page_table + b * pt_stride
    k_smem = gl.allocate_shared_memory(gl.float8e4nv, [2 * NBUF, BN, 64], kv_layout)
    v8_smem = gl.allocate_shared_memory(gl.float8e4nv, [2 * NBUF, BN, 64], kv_layout)
    if P8:
        v16_smem = gl.allocate_shared_memory(gl.float8e4nv, [2 * NBUF, 64, BN], v16_layout)
    else:
        v16_smem = gl.allocate_shared_memory(gl.float16, [2 * NBUF, BN, 64], v16_layout)
    ready = gl.allocate_shared_memory(gl.int64, [NBUF, 1], _g_mbar.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [NBUF, 1], _g_mbar.MBarrierLayout())
    for i in gl.static_range(NBUF):
        _g_mbar.init(ready.index(i), count=1)
        _g_mbar.init(empty.index(i), count=2)
    p = (qkv, prow, k_cache, v_cache, k_smem, v8_smem, v16_smem, ready, empty, part_o, part_m, part_l,
         b, h0, s, start, end, nt, c, qk_factor, NSPLIT, pt_stride)
    gl.warp_specialize([
        (_ws_part0, (p, L, G, GP, HQ, HKV, M, BN, NBUF, PF, PMODE, CMODE, P8, O_K, KW, PSH, VDT)),
        (_ws_part1, (p, L, G, GP, HQ, HKV, M, BN, NBUF, PF, PMODE, CMODE, P8, O_K, KW, PSH, VDT)),
        (_ws_producer, (p, L, G, GP, HQ, HKV, M, BN, NBUF, PF, PMODE, CMODE, P8, O_K, KW, PSH, VDT)),
    ], [4, 4], [REG_CONS, REG_PROD])
