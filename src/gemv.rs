//! Fast GEMV/GEMM for QORA-4B hybrid architecture (Gated DeltaNet + Full Attention).
//!
//! Supports two weight formats:
//! - **F16**: Half precision. Better quality.
//! - **Q4**: 4-bit symmetric quantization. Faster, lower memory.
//!
//! Q4 uses per-group (32 values) symmetric quantization:
//!   scale = absmax / 7, q = round(val/scale) + 8, packed 2 per byte.
//!   Dequant: val = (q - 8) * scale

use half::f16;
use rayon::prelude::*;
use crate::config::{Qor4bConfig, LayerType};

// ============================================================
// Weight format types
// ============================================================

const Q4_GROUP_SIZE: usize = 32;

/// A weight matrix stored as f16 with shape metadata.
pub struct F16Weight {
    pub data: Vec<f16>,
    pub k: usize, // rows (input dim)
    pub n: usize, // cols (output dim)
}

/// A weight matrix stored in 4-bit symmetric quantization.
pub struct Q4Weight {
    pub packed: Vec<u8>,
    pub scales: Vec<f16>,
    pub k: usize,
    pub n: usize,
}

/// Polymorphic weight — either f16 or Q4.
pub enum Weight {
    F16(F16Weight),
    Q4(Q4Weight),
}

impl Weight {
    pub fn n(&self) -> usize {
        match self { Weight::F16(w) => w.n, Weight::Q4(w) => w.n }
    }
    pub fn k(&self) -> usize {
        match self { Weight::F16(w) => w.k, Weight::Q4(w) => w.k }
    }
    pub fn memory_bytes(&self) -> usize {
        match self {
            Weight::F16(w) => w.data.len() * 2,
            Weight::Q4(w) => w.packed.len() + w.scales.len() * 2,
        }
    }
}

// ============================================================
// Per-layer weight structures (hybrid)
// ============================================================

/// Weights for a Gated DeltaNet layer.
pub struct DeltaNetLayerWeights {
    pub in_proj_qkv: Weight,   // [qkv_dim, hidden]
    pub in_proj_a: Weight,     // [num_v_heads, hidden]
    pub in_proj_b: Weight,     // [num_v_heads, hidden]
    pub in_proj_z: Weight,     // [v_dim, hidden]
    pub out_proj: Weight,      // [hidden, v_dim]
    pub conv1d_weight: Vec<f32>, // [qkv_dim, kernel_size] flattened
    pub a_log: Vec<f32>,       // [num_v_heads]
    pub dt_bias: Vec<f32>,     // [num_v_heads]
    pub attn_norm_weight: Vec<f16>, // [deltanet_head_dim] RMSNorm (per-head)
    // MLP
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
    pub input_norm: Vec<f16>,
    pub post_attn_norm: Vec<f16>,
}

/// Weights for a full attention (GQA) layer.
pub struct FullAttnLayerWeights {
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Vec<f16>,  // [attn_head_dim]
    pub k_norm: Vec<f16>,  // [attn_head_dim]
    // MLP
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
    pub input_norm: Vec<f16>,
    pub post_attn_norm: Vec<f16>,
}

/// Per-layer weights — one of the two layer types.
pub enum HybridLayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
}

impl HybridLayerWeights {
    pub fn memory_bytes(&self) -> usize {
        match self {
            HybridLayerWeights::DeltaNet(d) => {
                d.in_proj_qkv.memory_bytes() + d.in_proj_a.memory_bytes()
                + d.in_proj_b.memory_bytes() + d.in_proj_z.memory_bytes()
                + d.out_proj.memory_bytes()
                + d.conv1d_weight.len() * 4
                + (d.a_log.len() + d.dt_bias.len()) * 4
                + d.attn_norm_weight.len() * 2
                + d.gate_proj.memory_bytes() + d.up_proj.memory_bytes()
                + d.down_proj.memory_bytes()
                + (d.input_norm.len() + d.post_attn_norm.len()) * 2
            }
            HybridLayerWeights::FullAttn(f) => {
                f.q_proj.memory_bytes() + f.k_proj.memory_bytes()
                + f.v_proj.memory_bytes() + f.o_proj.memory_bytes()
                + (f.q_norm.len() + f.k_norm.len()) * 2
                + f.gate_proj.memory_bytes() + f.up_proj.memory_bytes()
                + f.down_proj.memory_bytes()
                + (f.input_norm.len() + f.post_attn_norm.len()) * 2
            }
        }
    }
}

// ============================================================
// DeltaNet state cache
// ============================================================

/// Per-layer DeltaNet recurrent state.
pub struct DeltaNetState {
    /// State matrices: [num_v_heads, head_k_dim, head_v_dim] flattened.
    /// Each V head has its own [128, 128] state matrix.
    pub s: Vec<f32>,
    /// Convolution buffer: last (kernel_size-1) QKV values.
    /// Layout: [qkv_dim, kernel_size-1] flattened.
    pub conv_buf: Vec<f32>,
    /// How many tokens have been fed (for conv warmup).
    pub conv_pos: usize,
}

impl DeltaNetState {
    pub fn new(config: &Qor4bConfig) -> Self {
        let num_v = config.num_v_heads; // 32
        let k_hd = config.deltanet_head_dim; // 128
        let v_hd = config.deltanet_head_dim; // 128 (same dim)
        let qkv_dim = config.deltanet_qkv_dim();
        let kernel_size = config.conv_kernel_size;
        Self {
            s: vec![0.0; num_v * k_hd * v_hd],
            conv_buf: vec![0.0; qkv_dim * (kernel_size - 1)],
            conv_pos: 0,
        }
    }
}

// ============================================================
// Hybrid cache (DeltaNet states + KV caches)
// ============================================================

/// KV cache entry for a full attention layer.
pub struct KvCacheEntry {
    pub k: Vec<f32>, // [seq_len, kv_heads, head_dim]
    pub v: Vec<f32>,
    pub seq_len: usize,
}

/// Hybrid cache for the full model.
pub struct HybridCache {
    /// One entry per layer (32 total).
    pub entries: Vec<CacheEntry>,
}

pub enum CacheEntry {
    DeltaNet(DeltaNetState),
    KvCache(KvCacheEntry),
}

impl HybridCache {
    pub fn new(config: &Qor4bConfig) -> Self {
        let mut entries = Vec::with_capacity(config.num_layers);
        for lt in &config.layer_types {
            match lt {
                LayerType::DeltaNet => entries.push(CacheEntry::DeltaNet(DeltaNetState::new(config))),
                LayerType::FullAttn => entries.push(CacheEntry::KvCache(KvCacheEntry {
                    k: Vec::with_capacity(config.num_kv_heads * 512 * config.attn_head_dim),
                    v: Vec::with_capacity(config.num_kv_heads * 512 * config.attn_head_dim),
                    seq_len: 0,
                })),
            }
        }
        Self { entries }
    }

    /// Clone this cache (deep copy of all states).
    pub fn snapshot(&self) -> Self {
        let entries = self.entries.iter().map(|e| match e {
            CacheEntry::DeltaNet(ds) => CacheEntry::DeltaNet(DeltaNetState {
                s: ds.s.clone(),
                conv_buf: ds.conv_buf.clone(),
                conv_pos: ds.conv_pos,
            }),
            CacheEntry::KvCache(kv) => CacheEntry::KvCache(KvCacheEntry {
                k: kv.k.clone(),
                v: kv.v.clone(),
                seq_len: kv.seq_len,
            }),
        }).collect();
        Self { entries }
    }

    /// Restore from a snapshot (deep copy from src).
    pub fn restore_from(&mut self, src: &HybridCache) {
        for (dst, s) in self.entries.iter_mut().zip(src.entries.iter()) {
            match (dst, s) {
                (CacheEntry::DeltaNet(d), CacheEntry::DeltaNet(s)) => {
                    d.s.copy_from_slice(&s.s);
                    d.conv_buf.copy_from_slice(&s.conv_buf);
                    d.conv_pos = s.conv_pos;
                }
                (CacheEntry::KvCache(d), CacheEntry::KvCache(s)) => {
                    d.k.clear(); d.k.extend_from_slice(&s.k);
                    d.v.clear(); d.v.extend_from_slice(&s.v);
                    d.seq_len = s.seq_len;
                }
                _ => panic!("Cache type mismatch during restore"),
            }
        }
    }
}

/// Cached system prompt state for fast re-prefill.
pub struct SystemPromptCache {
    pub tokens: Vec<u32>,
    pub cache_snapshot: HybridCache,
}

// ============================================================
// Model weights
// ============================================================

/// All model weights for QOR4B inference.
pub struct ModelWeights {
    pub layers: Vec<HybridLayerWeights>,
    pub embed: Weight,
    pub vocab: usize,
    pub hidden: usize,
    pub final_norm: Vec<f16>,
    pub rope_cos: Vec<f32>, // [max_pos, rope_dim/2] (32 entries per position)
    pub rope_sin: Vec<f32>,
    pub config: Qor4bConfig,
    pub format_name: &'static str,
    pub vision: Option<crate::vision::VisionEncoder>,
}

impl ModelWeights {
    pub fn memory_bytes(&self) -> usize {
        let mut total = self.embed.memory_bytes();
        total += self.final_norm.len() * 2;
        total += (self.rope_cos.len() + self.rope_sin.len()) * 4;
        for l in &self.layers {
            total += l.memory_bytes();
        }
        if let Some(v) = &self.vision {
            total += v.memory_bytes();
        }
        total
    }

    pub fn format_id(&self) -> u8 {
        match &self.embed {
            Weight::F16(_) => 0,
            Weight::Q4(_) => 1,
        }
    }
}

// ============================================================
// GEMV dispatch (single-token decode)
// ============================================================

/// GEMV: output = input @ weight.
#[inline]
pub fn gemv(input: &[f32], weight: &Weight) -> Vec<f32> {
    match weight {
        Weight::F16(w) => gemv_f16(input, w),
        Weight::Q4(w) => gemv_q4(input, w),
    }
}

/// GEMM: [seq_len, k] @ [k, n] -> [seq_len, n].
#[inline]
pub fn gemm(x: &[f32], seq_len: usize, weight: &Weight) -> Vec<f32> {
    match weight {
        Weight::F16(w) => gemm_f16(x, seq_len, w),
        Weight::Q4(w) => gemm_q4(x, seq_len, w),
    }
}

/// Fused gate+up GEMV: silu(gate(x)) * up(x) in one pass.
#[inline]
fn fused_gate_up_gemv(input: &[f32], gate_w: &Weight, up_w: &Weight) -> Vec<f32> {
    match (gate_w, up_w) {
        (Weight::Q4(gw), Weight::Q4(uw)) => fused_gate_up_q4(input, gw, uw),
        _ => {
            let gate = gemv(input, gate_w);
            let up = gemv(input, up_w);
            let n = gate.len();
            let mut out = vec![0.0f32; n];
            for j in 0..n {
                let g = gate[j];
                out[j] = (g / (1.0 + (-g).exp())) * up[j];
            }
            out
        }
    }
}

/// Embedding lookup.
#[inline]
pub fn embed_lookup(weight: &Weight, token_id: usize, hidden: usize) -> Vec<f32> {
    match weight {
        Weight::F16(w) => {
            let start = token_id * hidden;
            w.data[start..start + hidden].iter().map(|v| v.to_f32()).collect()
        }
        Weight::Q4(w) => embed_lookup_q4(w, token_id),
    }
}

// ============================================================
// F16 compute kernels
// ============================================================

#[inline]
fn gemv_f16(input: &[f32], weight: &F16Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    let w = &weight.data;

    // AVX-512 fast path
    if crate::simd::has_avx512() {
        if k * n < 4_000_000 {
            return unsafe { crate::simd::gemv_f16_avx512(input, w, k, n) };
        }
        // Threaded: split across k, each thread uses AVX-512
        let num_threads = rayon::current_num_threads();
        let chunk_k = (k + num_threads - 1) / num_threads;
        let partials: Vec<Vec<f32>> = (0..num_threads).into_par_iter().filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k { return None; }
            Some(unsafe { crate::simd::gemv_f16_avx512(&input[k_start..k_end], &w[k_start * n..], k_end - k_start, n) })
        }).collect();
        let mut output = vec![0.0f32; n];
        for p in &partials { for j in 0..n { output[j] += p[j]; } }
        return output;
    }

    if k * n < 4_000_000 {
        let mut output = vec![0.0f32; n];
        for ki in 0..k {
            let val = input[ki];
            let row = ki * n;
            for j in 0..n { output[j] += val * w[row + j].to_f32(); }
        }
        return output;
    }
    let num_threads = rayon::current_num_threads();
    let chunk_k = (k + num_threads - 1) / num_threads;
    let partials: Vec<Vec<f32>> = (0..num_threads).into_par_iter().filter_map(|t| {
        let k_start = t * chunk_k;
        let k_end = ((t + 1) * chunk_k).min(k);
        if k_start >= k { return None; }
        let mut out = vec![0.0f32; n];
        for ki in k_start..k_end {
            let val = input[ki];
            let row = ki * n;
            for j in 0..n { out[j] += val * w[row + j].to_f32(); }
        }
        Some(out)
    }).collect();
    let mut output = vec![0.0f32; n];
    for p in &partials { for j in 0..n { output[j] += p[j]; } }
    output
}

#[inline]
fn gemm_f16(x: &[f32], seq_len: usize, weight: &F16Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    let w = &weight.data;
    let mut output = vec![0.0f32; seq_len * n];
    output.par_chunks_mut(n).enumerate().for_each(|(t, out_row)| {
        let x_row = &x[t * k..(t + 1) * k];
        for ki in 0..k {
            let val = x_row[ki];
            let w_start = ki * n;
            for j in 0..n {
                out_row[j] += val * w[w_start + j].to_f32();
            }
        }
    });
    output
}

// ============================================================
// Q4 compute kernels
// ============================================================

#[inline]
fn gemv_q4_inner(input: &[f32], packed: &[u8], scales: &[f16],
                  _k: usize, n: usize, k_start: usize, k_end: usize) -> Vec<f32> {
    debug_assert!(n % Q4_GROUP_SIZE == 0, "Q4 GEMV: n={n} not divisible by {Q4_GROUP_SIZE}");

    // AVX-512 fast path
    if crate::simd::has_avx512() {
        return unsafe { crate::simd::gemv_q4_avx512(input, packed, scales, n, k_start, k_end) };
    }
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;
    let mut output = vec![0.0f32; n];
    for ki in k_start..k_end {
        let val = input[ki];
        if val == 0.0 { continue; }
        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;
        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * val;
            if s == 0.0 { continue; }
            let lut = [
                s * -8.0, s * -7.0, s * -6.0, s * -5.0,
                s * -4.0, s * -3.0, s * -2.0, s * -1.0,
                0.0,      s,        s * 2.0,  s * 3.0,
                s * 4.0,  s * 5.0,  s * 6.0,  s * 7.0,
            ];
            let po = pack_base + g * packed_per_group;
            let oo = g * Q4_GROUP_SIZE;
            for j in 0..packed_per_group {
                let byte = packed[po + j];
                output[oo + j * 2] += lut[(byte & 0x0F) as usize];
                output[oo + j * 2 + 1] += lut[(byte >> 4) as usize];
            }
        }
    }
    output
}

#[inline]
fn gemv_q4(input: &[f32], weight: &Q4Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    if k * n < 4_000_000 {
        return gemv_q4_inner(input, &weight.packed, &weight.scales, k, n, 0, k);
    }
    let num_threads = rayon::current_num_threads();
    let chunk_k = (k + num_threads - 1) / num_threads;
    let partials: Vec<Vec<f32>> = (0..num_threads).into_par_iter().filter_map(|t| {
        let k_start = t * chunk_k;
        let k_end = ((t + 1) * chunk_k).min(k);
        if k_start >= k { return None; }
        Some(gemv_q4_inner(input, &weight.packed, &weight.scales, k, n, k_start, k_end))
    }).collect();
    let mut output = vec![0.0f32; n];
    for p in &partials { for j in 0..n { output[j] += p[j]; } }
    output
}

fn fused_gate_up_q4(input: &[f32], gate_w: &Q4Weight, up_w: &Q4Weight) -> Vec<f32> {
    let (k, n) = (gate_w.k, gate_w.n);
    let num_threads = rayon::current_num_threads();
    let chunk_k = (k + num_threads - 1) / num_threads;
    let use_avx512 = crate::simd::has_avx512();
    let partials: Vec<(Vec<f32>, Vec<f32>)> = (0..num_threads).into_par_iter().filter_map(|t| {
        let k_start = t * chunk_k;
        let k_end = ((t + 1) * chunk_k).min(k);
        if k_start >= k { return None; }

        if use_avx512 {
            return Some(unsafe {
                crate::simd::fused_gate_up_q4_avx512(
                    input, &gate_w.packed, &gate_w.scales,
                    &up_w.packed, &up_w.scales, n, k_start, k_end,
                )
            });
        }

        // Scalar fallback
        let groups_per_row = n / Q4_GROUP_SIZE;
        let packed_per_group = Q4_GROUP_SIZE / 2;
        let mut gate_out = vec![0.0f32; n];
        let mut up_out = vec![0.0f32; n];
        for ki in k_start..k_end {
            let val = input[ki];
            if val == 0.0 { continue; }
            let sb = ki * groups_per_row;
            let pb = ki * groups_per_row * packed_per_group;
            for g in 0..groups_per_row {
                let gs = gate_w.scales[sb + g].to_f32() * val;
                let us = up_w.scales[sb + g].to_f32() * val;
                if gs == 0.0 && us == 0.0 { continue; }
                let g_lut = [gs*-8.0,gs*-7.0,gs*-6.0,gs*-5.0,gs*-4.0,gs*-3.0,gs*-2.0,gs*-1.0,
                             0.0,gs,gs*2.0,gs*3.0,gs*4.0,gs*5.0,gs*6.0,gs*7.0];
                let u_lut = [us*-8.0,us*-7.0,us*-6.0,us*-5.0,us*-4.0,us*-3.0,us*-2.0,us*-1.0,
                             0.0,us,us*2.0,us*3.0,us*4.0,us*5.0,us*6.0,us*7.0];
                let po = pb + g * packed_per_group;
                let oo = g * Q4_GROUP_SIZE;
                for j in 0..packed_per_group {
                    let gb = gate_w.packed[po + j];
                    let ub = up_w.packed[po + j];
                    gate_out[oo + j*2] += g_lut[(gb & 0x0F) as usize];
                    gate_out[oo + j*2+1] += g_lut[(gb >> 4) as usize];
                    up_out[oo + j*2] += u_lut[(ub & 0x0F) as usize];
                    up_out[oo + j*2+1] += u_lut[(ub >> 4) as usize];
                }
            }
        }
        Some((gate_out, up_out))
    }).collect();
    let mut gf = vec![0.0f32; n];
    let mut uf = vec![0.0f32; n];
    for (gp, up) in &partials { for j in 0..n { gf[j] += gp[j]; uf[j] += up[j]; } }
    let mut out = vec![0.0f32; n];
    for j in 0..n { let g = gf[j]; out[j] = (g / (1.0 + (-g).exp())) * uf[j]; }
    out
}

#[inline]
fn gemm_q4(x: &[f32], seq_len: usize, weight: &Q4Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    // Always parallelize across rows for GEMM (seq_len > 1)
    if seq_len <= 1 {
        let row = gemv_q4_inner(&x[..k], &weight.packed, &weight.scales, k, n, 0, k);
        return row;
    }
    let mut output = vec![0.0f32; seq_len * n];
    output.par_chunks_mut(n).enumerate().for_each(|(t, out_row)| {
        let row = gemv_q4_inner(&x[t*k..(t+1)*k], &weight.packed, &weight.scales, k, n, 0, k);
        out_row.copy_from_slice(&row);
    });
    output
}

fn embed_lookup_q4(weight: &Q4Weight, token_id: usize) -> Vec<f32> {
    let n = weight.n;
    debug_assert!(n % Q4_GROUP_SIZE == 0, "Q4 embed: n={n} not divisible by {Q4_GROUP_SIZE}");
    let gpr = n / Q4_GROUP_SIZE;
    let ppg = Q4_GROUP_SIZE / 2;
    let sb = token_id * gpr;
    let pb = token_id * gpr * ppg;
    let mut output = vec![0.0f32; n];
    for g in 0..gpr {
        let scale = weight.scales[sb + g].to_f32();
        let po = pb + g * ppg;
        let oo = g * Q4_GROUP_SIZE;
        for j in 0..ppg {
            let byte = weight.packed[po + j];
            output[oo + j*2] = scale * ((byte & 0x0F) as i32 - 8) as f32;
            output[oo + j*2+1] = scale * (((byte >> 4) & 0x0F) as i32 - 8) as f32;
        }
    }
    output
}

// ============================================================
// Shared compute kernels
// ============================================================

/// RmsNorm with f16 gamma. eps = 1e-6.
#[inline]
pub fn rms_norm(x: &[f32], gamma: &[f16]) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + 1e-6).sqrt();
    let mut out = vec![0.0f32; n];
    // QORA-4B uses (1 + weight) in RMSNorm
    for i in 0..n { out[i] = x[i] * inv_rms * (1.0 + gamma[i].to_f32()); }
    out
}



/// Per-head RMS norm for QK norms in full attention.
/// Input: [num_heads * head_dim], applies separate RMS norm per head using shared gamma[head_dim].
#[inline]
fn per_head_rms_norm(x: &mut [f32], num_heads: usize, head_dim: usize, gamma: &[f16]) {
    for h in 0..num_heads {
        let base = h * head_dim;
        let head = &x[base..base + head_dim];
        let sum_sq: f32 = head.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
        for d in 0..head_dim {
            // QORA-4B RMSNorm uses (1 + weight)
            x[base + d] = x[base + d] * inv_rms * (1.0 + gamma[d].to_f32());
        }
    }
}

/// Apply partial RoPE (only first rotary_dim dimensions of each head).
/// QORA-4B uses split-half pairing: pair (i, i + half) for i in 0..half, where half = rotary_dim/2.
/// cos/sin tables have `half` entries per position (since each pair shares the same frequency).
/// rotary_dim = total dims rotated per head (e.g. 64 = 256 * 0.25).
#[inline]
fn apply_partial_rope(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
    rotary_dim: usize, // total dims rotated per head (64)
    position: usize,
) {
    let half = rotary_dim / 2; // 32: number of pairs
    let cos_offset = position * half; // table has `half` entries per position
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x1 = data[base + i];        // dim i
            let x2 = data[base + half + i]; // dim i + half
            let c = cos_table[cos_offset + i];
            let s = sin_table[cos_offset + i];
            // Split-half RoPE: both elements of pair use same frequency
            data[base + i] = x1 * c - x2 * s;
            data[base + half + i] = x2 * c + x1 * s;
        }
    }
}

/// SiLU activation: x * sigmoid(x).
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: log(1 + exp(x)).
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// L2 normalize a vector in-place.
#[inline]
fn l2_normalize(x: &mut [f32]) {
    let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for v in x.iter_mut() { *v *= inv; }
    }
}

/// In-place softmax.
#[inline]
fn softmax_raw(scores: &mut [f32]) {
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() { *s = (*s - max_val).exp(); sum += *s; }
    let inv = 1.0 / sum;
    for s in scores.iter_mut() { *s *= inv; }
}

/// Parallel lm_head.
fn lm_head_parallel(input: &[f32], embed: &Weight, vocab: usize, hidden: usize) -> Vec<f32> {
    match embed {
        Weight::F16(w) => {
            let mut output = vec![0.0f32; vocab];
            output.par_chunks_mut(256).enumerate().for_each(|(ci, chunk)| {
                let start = ci * 256;
                for (i, out) in chunk.iter_mut().enumerate() {
                    let r = start + i;
                    let row = &w.data[r * hidden..(r + 1) * hidden];
                    let mut sum = 0.0f32;
                    for j in 0..hidden { sum += input[j] * row[j].to_f32(); }
                    *out = sum;
                }
            });
            output
        }
        Weight::Q4(w) => {
            let gpr = hidden / Q4_GROUP_SIZE;
            let ppg = Q4_GROUP_SIZE / 2;
            let ppr = gpr * ppg;
            let mut output = vec![0.0f32; vocab];
            output.par_chunks_mut(256).enumerate().for_each(|(ci, chunk)| {
                let start = ci * 256;
                for (i, out) in chunk.iter_mut().enumerate() {
                    let v = start + i;
                    let mut dot = 0.0f32;
                    let sb = v * gpr;
                    let pb = v * ppr;
                    for g in 0..gpr {
                        let scale = w.scales[sb + g].to_f32();
                        if scale == 0.0 { continue; }
                        let po = pb + g * ppg;
                        let io = g * Q4_GROUP_SIZE;
                        let lut = [scale*-8.0,scale*-7.0,scale*-6.0,scale*-5.0,
                                   scale*-4.0,scale*-3.0,scale*-2.0,scale*-1.0,
                                   0.0,scale,scale*2.0,scale*3.0,
                                   scale*4.0,scale*5.0,scale*6.0,scale*7.0];
                        for j in 0..ppg {
                            let byte = w.packed[po + j];
                            dot += input[io + j*2] * lut[(byte & 0x0F) as usize];
                            dot += input[io + j*2+1] * lut[(byte >> 4) as usize];
                        }
                    }
                    *out = dot;
                }
            });
            output
        }
    }
}

// ============================================================
// DeltaNet single-step forward
// ============================================================

/// Process one token through a Gated DeltaNet layer.
/// Returns residual-added output [hidden].
fn forward_deltanet_decode(
    x: &[f32],
    lw: &DeltaNetLayerWeights,
    state: &mut DeltaNetState,
    config: &Qor4bConfig,
) -> Vec<f32> {
    let hidden = config.hidden_size;
    let num_qk = config.num_qk_heads; // 16
    let num_v = config.num_v_heads;    // 32
    let hd = config.deltanet_head_dim; // 128
    let q_dim = config.deltanet_q_dim(); // 2048
    let k_dim = config.deltanet_k_dim(); // 2048
    let v_dim = config.deltanet_v_dim(); // 4096
    let qkv_dim = config.deltanet_qkv_dim(); // 8192
    let ks = config.conv_kernel_size;  // 4
    let buf_len = ks - 1; // 3
    let v_per_qk = num_v / num_qk; // 2 — GVA ratio
    let scale = 1.0 / (hd as f32).sqrt(); // 1/sqrt(128)

    // 1. Pre-attention RmsNorm
    let x_norm = rms_norm(x, &lw.input_norm);

    // 2. QKV projection
    let qkv = gemv(&x_norm, &lw.in_proj_qkv); // [8192]

    // 3. Causal Conv1d (kernel=4, depthwise across qkv_dim channels)
    //    conv_buf stores last (kernel_size-1)=3 QKV values: [qkv_dim, 3]
    //    Apply conv FIRST using old buffer + current qkv,
    //    THEN update buffer.
    let conv_w = &lw.conv1d_weight;
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    for ch in 0..qkv_dim {
        let mut sum = 0.0f32;
        for t in 0..buf_len {
            sum += state.conv_buf[ch * buf_len + t] * conv_w[ch * ks + t];
        }
        sum += qkv[ch] * conv_w[ch * ks + buf_len];
        qkv_conv[ch] = sum;
    }

    // Update buffer for next step
    {
        let buf = &mut state.conv_buf;
        for ch in 0..qkv_dim {
            for t in 0..buf_len - 1 {
                buf[ch * buf_len + t] = buf[ch * buf_len + t + 1];
            }
            buf[ch * buf_len + buf_len - 1] = qkv[ch];
        }
    }
    state.conv_pos += 1;

    // 4. SiLU on ALL channels (Q, K, V) — conv output includes activation
    for val in qkv_conv.iter_mut() { *val = silu(*val); }

    // 5. Split QKV
    let mut q = qkv_conv[..q_dim].to_vec(); // [2048]
    let mut k = qkv_conv[q_dim..q_dim + k_dim].to_vec(); // [2048]
    let v = &qkv_conv[q_dim + k_dim..]; // [4096]

    // L2 normalize Q and K per head, then scale Q by 1/sqrt(head_dim)
    for h in 0..num_qk {
        l2_normalize(&mut q[h * hd..(h + 1) * hd]);
        l2_normalize(&mut k[h * hd..(h + 1) * hd]);
        // Scale Q by 1/sqrt(head_k_dim)
        for j in 0..hd { q[h * hd + j] *= scale; }
    }

    // 6. Compute alpha and beta per V head (32 values each)
    let a_proj = gemv(&x_norm, &lw.in_proj_a); // [num_v=32]
    let b_proj = gemv(&x_norm, &lw.in_proj_b); // [num_v=32]

    let mut alpha = vec![0.0f32; num_v];
    let mut beta = vec![0.0f32; num_v];
    for vh in 0..num_v {
        // alpha = exp(-exp(A_log[vh]) * softplus(a_proj[vh] + dt_bias[vh]))
        let a_exp = lw.a_log[vh].exp();
        let sp = softplus(a_proj[vh] + lw.dt_bias[vh]);
        alpha[vh] = (-a_exp * sp).exp();
        // beta = sigmoid(b_proj[vh])
        beta[vh] = 1.0 / (1.0 + (-b_proj[vh]).exp());
    }

    // 7. GVA expansion: repeat Q and K from 16 to 32 heads
    let mut q_exp = vec![0.0f32; num_v * hd]; // [32, 128]
    let mut k_exp = vec![0.0f32; num_v * hd]; // [32, 128]
    for vh in 0..num_v {
        let qk_h = vh / v_per_qk; // which QK head this V head maps to
        q_exp[vh * hd..(vh + 1) * hd].copy_from_slice(&q[qk_h * hd..(qk_h + 1) * hd]);
        k_exp[vh * hd..(vh + 1) * hd].copy_from_slice(&k[qk_h * hd..(qk_h + 1) * hd]);
    }

    // 8. State update per V head (delta rule + gating)
    //    State S[vh] is [k_hd, v_hd] = [128, 128] per V head.
    //    Stored as [v_hd, k_hd] in row-major (our convention).
    let mut y = vec![0.0f32; v_dim]; // output [4096]
    let use_avx512 = crate::simd::has_avx512();

    for vh in 0..num_v {
        let al = alpha[vh];
        let be = beta[vh];
        let q_h = &q_exp[vh * hd..(vh + 1) * hd]; // [128]
        let k_h = &k_exp[vh * hd..(vh + 1) * hd]; // [128]
        let v_h = &v[vh * hd..(vh + 1) * hd];      // [128]
        let s_base = vh * hd * hd;
        let s = &mut state.s[s_base..s_base + hd * hd]; // [hd, hd]
        let y_base = vh * hd;

        if use_avx512 {
            unsafe {
                crate::simd::deltanet_head_update_avx512(
                    s, q_h, k_h, v_h, al, be, hd, &mut y[y_base..y_base + hd],
                );
            }
            continue;
        }

        // Scalar fallback
        // Decay state FIRST: S *= alpha
        for val in s.iter_mut() { *val *= al; }

        // Retrieve from decayed state: pred = S @ k_h → [hd]
        let mut pred = vec![0.0f32; hd];
        for r in 0..hd {
            let mut dot = 0.0f32;
            for c in 0..hd { dot += s[r * hd + c] * k_h[c]; }
            pred[r] = dot;
        }

        // Delta update: S += beta * (v_h - pred) @ k_h^T → [hd, hd]
        for r in 0..hd {
            let delta = be * (v_h[r] - pred[r]);
            for c in 0..hd {
                s[r * hd + c] += delta * k_h[c];
            }
        }

        // Output: y[vh*hd..(vh+1)*hd] = S @ q_h → [hd]
        for r in 0..hd {
            let mut dot = 0.0f32;
            for c in 0..hd { dot += s[r * hd + c] * q_h[c]; }
            y[y_base + r] = dot;
        }
    }

    // 9. Output gating: z = silu(in_proj_z @ x_norm), output = per_head_rms_norm(y) * z
    let z = gemv(&x_norm, &lw.in_proj_z); // [v_dim=4096]

    // Per-head RMS norm: norm_weight is [head_dim=128], applied to each of num_v=32 heads
    let norm_gamma: Vec<f32> = lw.attn_norm_weight.iter().map(|v| v.to_f32()).collect();
    let mut y_normed = vec![0.0f32; v_dim];
    for h in 0..num_v {
        let base = h * hd;
        let head_slice = &y[base..base + hd];
        let sum_sq: f32 = head_slice.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / hd as f32 + 1e-6).sqrt();
        for j in 0..hd {
            y_normed[base + j] = head_slice[j] * inv_rms * norm_gamma[j];
        }
    }

    let mut gated = vec![0.0f32; v_dim];
    for j in 0..v_dim {
        gated[j] = y_normed[j] * silu(z[j]);
    }

    // 10. Output projection
    let attn_out = gemv(&gated, &lw.out_proj);

    // Residual
    let mut out = vec![0.0f32; hidden];
    for j in 0..hidden { out[j] = x[j] + attn_out[j]; }

    // MLP
    let x_norm2 = rms_norm(&out, &lw.post_attn_norm);
    let intermediate = fused_gate_up_gemv(&x_norm2, &lw.gate_proj, &lw.up_proj);
    let mlp_out = gemv(&intermediate, &lw.down_proj);
    for j in 0..hidden { out[j] += mlp_out[j]; }

    out
}

// ============================================================
// Full attention single-step forward
// ============================================================

/// Process one token through a full attention (GQA) layer.
/// q_proj outputs [2 * q_dim] when attn_output_gate=true: first half Q, second half gate.
fn forward_attn_decode(
    x: &[f32],
    lw: &FullAttnLayerWeights,
    kv: &mut KvCacheEntry,
    config: &Qor4bConfig,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Vec<f32> {
    let hidden = config.hidden_size;
    let num_heads = config.num_attn_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.attn_head_dim;
    let q_dim = num_heads * head_dim; // 4096
    let num_kv_groups = config.num_kv_groups();
    let rotary_dim = config.rope_dim(); // 64: total dims rotated per head
    let offset = kv.seq_len;

    // Pre-attention RmsNorm
    let x_norm = rms_norm(x, &lw.input_norm);

    // Q projection: [8192] = interleaved [Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), ...]
    let q_gate_full = gemv(&x_norm, &lw.q_proj);
    // Split per-head: each head has [Q:head_dim, gate:head_dim]
    let mut q = vec![0.0f32; q_dim];
    let mut gate_vec = vec![0.0f32; q_dim];
    for h in 0..num_heads {
        let src = h * 2 * head_dim;
        let dst = h * head_dim;
        q[dst..dst + head_dim].copy_from_slice(&q_gate_full[src..src + head_dim]);
        gate_vec[dst..dst + head_dim].copy_from_slice(&q_gate_full[src + head_dim..src + 2 * head_dim]);
    }
    let gate = &gate_vec[..];

    // K, V projections
    let mut k_new = gemv(&x_norm, &lw.k_proj);
    let v_new = gemv(&x_norm, &lw.v_proj);

    // Per-head QK norms
    per_head_rms_norm(&mut q, num_heads, head_dim, &lw.q_norm);
    per_head_rms_norm(&mut k_new, num_kv_heads, head_dim, &lw.k_norm);

    // Partial RoPE
    apply_partial_rope(&mut q, num_heads, head_dim, rope_cos, rope_sin, rotary_dim, offset);
    apply_partial_rope(&mut k_new, num_kv_heads, head_dim, rope_cos, rope_sin, rotary_dim, offset);

    // Append to KV cache
    kv.k.extend_from_slice(&k_new);
    kv.v.extend_from_slice(&v_new);
    kv.seq_len = offset + 1;
    let kv_seq_len = kv.seq_len;

    // GQA attention
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;

    let mut attn_output: Vec<f32> = if kv_seq_len >= 64 {
        let head_results: Vec<Vec<f32>> = (0..num_heads).into_par_iter().map(|h| {
            let kv_h = h / num_kv_groups;
            let q_off = h * head_dim;
            let q_vec = &q[q_off..q_off + head_dim];
            let mut scores = vec![0.0f32; kv_seq_len];
            for s in 0..kv_seq_len {
                let k_off = s * kv_stride + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim { dot += q_vec[d] * kv.k[k_off + d]; }
                scores[s] = dot * scale;
            }
            softmax_raw(&mut scores);
            let mut head_out = vec![0.0f32; head_dim];
            for s in 0..kv_seq_len {
                let v_off = s * kv_stride + kv_h * head_dim;
                let sc = scores[s];
                for d in 0..head_dim { head_out[d] += sc * kv.v[v_off + d]; }
            }
            head_out
        }).collect();
        head_results.into_iter().flatten().collect()
    } else {
        let mut attn_out = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_h = h / num_kv_groups;
            let q_off = h * head_dim;
            let q_vec = &q[q_off..q_off + head_dim];
            let mut scores = vec![0.0f32; kv_seq_len];
            for s in 0..kv_seq_len {
                let k_off = s * kv_stride + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim { dot += q_vec[d] * kv.k[k_off + d]; }
                scores[s] = dot * scale;
            }
            softmax_raw(&mut scores);
            let out_off = h * head_dim;
            for s in 0..kv_seq_len {
                let v_off = s * kv_stride + kv_h * head_dim;
                let sc = scores[s];
                for d in 0..head_dim { attn_out[out_off + d] += sc * kv.v[v_off + d]; }
            }
        }
        attn_out
    };

    // Apply output gate: attn_output *= sigmoid(gate)
    for j in 0..q_dim {
        attn_output[j] *= 1.0 / (1.0 + (-gate[j]).exp());
    }

    // O projection + residual
    let o_out = gemv(&attn_output, &lw.o_proj);
    let mut out = vec![0.0f32; hidden];
    for j in 0..hidden { out[j] = x[j] + o_out[j]; }

    // MLP
    let x_norm2 = rms_norm(&out, &lw.post_attn_norm);
    let intermediate = fused_gate_up_gemv(&x_norm2, &lw.gate_proj, &lw.up_proj);
    let mlp_out = gemv(&intermediate, &lw.down_proj);
    for j in 0..hidden { out[j] += mlp_out[j]; }

    out
}

// ============================================================
// Hybrid forward decode (single token)
// ============================================================

/// Process one token through all layers (updates cache) but skip lm_head.
/// Use when the next token is already known (e.g. forced THINK_END).
pub fn forward_decode_no_logits(
    weights: &ModelWeights,
    token_id: usize,
    cache: &mut HybridCache,
) {
    let hidden = weights.hidden;
    let config = &weights.config;
    let mut x = embed_lookup(&weights.embed, token_id, hidden);
    for i in 0..config.num_layers {
        x = match (&weights.layers[i], &mut cache.entries[i]) {
            (HybridLayerWeights::DeltaNet(lw), CacheEntry::DeltaNet(state)) => {
                forward_deltanet_decode(&x, lw, state, config)
            }
            (HybridLayerWeights::FullAttn(lw), CacheEntry::KvCache(kv)) => {
                forward_attn_decode(&x, lw, kv, config, &weights.rope_cos, &weights.rope_sin)
            }
            _ => panic!("Layer type mismatch at layer {i}"),
        };
    }
    // Skip final_norm + lm_head — only cache updates matter
}

/// Process one token through the full hybrid model. Returns logits.
pub fn forward_decode(
    weights: &ModelWeights,
    token_id: usize,
    cache: &mut HybridCache,
) -> Vec<f32> {
    let hidden = weights.hidden;
    let config = &weights.config;
    let mut x = embed_lookup(&weights.embed, token_id, hidden);
    for i in 0..config.num_layers {
        x = match (&weights.layers[i], &mut cache.entries[i]) {
            (HybridLayerWeights::DeltaNet(lw), CacheEntry::DeltaNet(state)) => {
                forward_deltanet_decode(&x, lw, state, config)
            }
            (HybridLayerWeights::FullAttn(lw), CacheEntry::KvCache(kv)) => {
                forward_attn_decode(&x, lw, kv, config, &weights.rope_cos, &weights.rope_sin)
            }
            _ => panic!("Layer type mismatch at layer {i}"),
        };
    }
    x = rms_norm(&x, &weights.final_norm);
    lm_head_parallel(&x, &weights.embed, weights.vocab, hidden)
}

// ============================================================
// Batched DeltaNet prefill (GEMM projections + sequential state)
// ============================================================

/// Prefill a DeltaNet layer with batched GEMM projections.
/// The state update is inherently sequential, but all matrix multiplications
/// are batched into GEMM ops that parallelize across tokens.
fn prefill_deltanet_layer(
    x: &mut [f32],
    seq_len: usize,
    lw: &DeltaNetLayerWeights,
    state: &mut DeltaNetState,
    config: &Qor4bConfig,
) {
    let hidden = config.hidden_size;
    let num_qk = config.num_qk_heads;
    let num_v = config.num_v_heads;
    let hd = config.deltanet_head_dim;
    let q_dim = config.deltanet_q_dim();
    let k_dim = config.deltanet_k_dim();
    let v_dim = num_v * hd;
    let qkv_dim = config.deltanet_qkv_dim();
    let ks = config.conv_kernel_size;
    let buf_len = ks - 1;
    let v_per_qk = num_v / num_qk;
    let scale = 1.0 / (hd as f32).sqrt();

    // ── Step 1: RMSNorm all tokens ──
    let mut x_norm = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = rms_norm(&x[t * hidden..(t + 1) * hidden], &lw.input_norm);
        x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }

    // ── Step 2: Batch ALL input projections as GEMM ──
    let qkv_all = gemm(&x_norm, seq_len, &lw.in_proj_qkv);
    let a_all   = gemm(&x_norm, seq_len, &lw.in_proj_a);
    let b_all   = gemm(&x_norm, seq_len, &lw.in_proj_b);
    let z_all   = gemm(&x_norm, seq_len, &lw.in_proj_z);

    // ── Step 3: Sequential loop — conv1d + state update only ──
    let mut y_all = vec![0.0f32; seq_len * v_dim];
    let conv_w = &lw.conv1d_weight;
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    let mut q_exp = vec![0.0f32; num_v * hd];
    let mut k_exp = vec![0.0f32; num_v * hd];
    let mut pred = vec![0.0f32; hd];

    let a_exp_table: Vec<f32> = lw.a_log.iter().map(|v| v.exp()).collect();

    for t in 0..seq_len {
        let qkv_t = &qkv_all[t * qkv_dim..(t + 1) * qkv_dim];

        // Conv1d (causal, depthwise)
        for ch in 0..qkv_dim {
            let mut sum = 0.0f32;
            for k in 0..buf_len {
                sum += state.conv_buf[ch * buf_len + k] * conv_w[ch * ks + k];
            }
            sum += qkv_t[ch] * conv_w[ch * ks + buf_len];
            qkv_conv[ch] = sum;
        }
        for ch in 0..qkv_dim {
            for k in 0..buf_len - 1 {
                state.conv_buf[ch * buf_len + k] = state.conv_buf[ch * buf_len + k + 1];
            }
            state.conv_buf[ch * buf_len + buf_len - 1] = qkv_t[ch];
        }
        state.conv_pos += 1;

        for val in qkv_conv.iter_mut() { *val = silu(*val); }

        // L2 normalize Q and K per head in-place, scale Q
        for h in 0..num_qk {
            l2_normalize(&mut qkv_conv[h * hd..(h + 1) * hd]);
            l2_normalize(&mut qkv_conv[q_dim + h * hd..q_dim + (h + 1) * hd]);
            for j in 0..hd { qkv_conv[h * hd + j] *= scale; }
        }

        let a_t = &a_all[t * num_v..(t + 1) * num_v];
        let b_t = &b_all[t * num_v..(t + 1) * num_v];

        // GVA expansion
        for vh in 0..num_v {
            let qk_h = vh / v_per_qk;
            q_exp[vh * hd..(vh + 1) * hd].copy_from_slice(&qkv_conv[qk_h * hd..(qk_h + 1) * hd]);
            k_exp[vh * hd..(vh + 1) * hd].copy_from_slice(&qkv_conv[q_dim + qk_h * hd..q_dim + (qk_h + 1) * hd]);
        }

        // State update per V head (delta rule)
        let v_slice = &qkv_conv[q_dim + k_dim..];
        for vh in 0..num_v {
            let al = (-a_exp_table[vh] * softplus(a_t[vh] + lw.dt_bias[vh])).exp();
            let be = 1.0 / (1.0 + (-b_t[vh]).exp());

            let q_h = &q_exp[vh * hd..(vh + 1) * hd];
            let k_h = &k_exp[vh * hd..(vh + 1) * hd];
            let v_h = &v_slice[vh * hd..(vh + 1) * hd];
            let s_base = vh * hd * hd;
            let s = &mut state.s[s_base..s_base + hd * hd];

            for val in s.iter_mut() { *val *= al; }

            for r in 0..hd {
                let mut dot = 0.0f32;
                let row = &s[r * hd..(r + 1) * hd];
                for c in 0..hd { dot += row[c] * k_h[c]; }
                pred[r] = dot;
            }

            for r in 0..hd {
                let delta = be * (v_h[r] - pred[r]);
                let row = &mut s[r * hd..(r + 1) * hd];
                for c in 0..hd { row[c] += delta * k_h[c]; }
            }

            let y_base = t * v_dim + vh * hd;
            for r in 0..hd {
                let mut dot = 0.0f32;
                let row = &s[r * hd..(r + 1) * hd];
                for c in 0..hd { dot += row[c] * q_h[c]; }
                y_all[y_base + r] = dot;
            }
        }
    }

    // ── Step 4: Per-head norm + z gate (batched) ──
    let norm_gamma: Vec<f32> = lw.attn_norm_weight.iter().map(|v| v.to_f32()).collect();
    for t in 0..seq_len {
        let z_t = &z_all[t * v_dim..(t + 1) * v_dim];
        let y_t = &mut y_all[t * v_dim..(t + 1) * v_dim];
        for h in 0..num_v {
            let base = h * hd;
            let sum_sq: f32 = y_t[base..base + hd].iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / hd as f32 + 1e-6).sqrt();
            for j in 0..hd {
                y_t[base + j] = y_t[base + j] * inv_rms * norm_gamma[j];
            }
        }
        for j in 0..v_dim { y_t[j] *= silu(z_t[j]); }
    }

    // ── Step 5: Output projection as GEMM + residual ──
    let attn_out = gemm(&y_all, seq_len, &lw.out_proj);
    for j in 0..seq_len * hidden { x[j] += attn_out[j]; }

    // ── Step 6: MLP as GEMM ──
    let mut x_norm2 = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = rms_norm(&x[t * hidden..(t + 1) * hidden], &lw.post_attn_norm);
        x_norm2[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }
    let gate = gemm(&x_norm2, seq_len, &lw.gate_proj);
    let up   = gemm(&x_norm2, seq_len, &lw.up_proj);
    let inter_size = lw.gate_proj.n();
    let mut intermediate = vec![0.0f32; seq_len * inter_size];
    for j in 0..seq_len * inter_size {
        let g = gate[j];
        intermediate[j] = (g / (1.0 + (-g).exp())) * up[j];
    }
    let mlp_out = gemm(&intermediate, seq_len, &lw.down_proj);
    for j in 0..seq_len * hidden { x[j] += mlp_out[j]; }
}

// ============================================================
// Hybrid prefill (full prompt)
// ============================================================

/// Prefill all tokens, return logits for last token.
pub fn prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut HybridCache,
) -> Vec<f32> {
    let hidden = weights.hidden;
    let config = &weights.config;
    let seq_len = token_ids.len();

    // Embedding: [seq_len, hidden]
    let mut x = vec![0.0f32; seq_len * hidden];
    for (t, &tid) in token_ids.iter().enumerate() {
        let row = embed_lookup(&weights.embed, tid as usize, hidden);
        x[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }

    let num_layers = config.num_layers;
    for i in 0..num_layers {
        eprint!("\r  Prefill: layer {}/{num_layers}  ", i + 1);
        match (&weights.layers[i], &mut cache.entries[i]) {
            (HybridLayerWeights::DeltaNet(lw), CacheEntry::DeltaNet(state)) => {
                prefill_deltanet_layer(&mut x, seq_len, lw, state, config);
            }
            (HybridLayerWeights::FullAttn(lw), CacheEntry::KvCache(kv)) => {
                prefill_attn_layer(&mut x, seq_len, lw, kv, config, &weights.rope_cos, &weights.rope_sin);
            }
            _ => panic!("Layer type mismatch at layer {i}"),
        }
    }
    eprint!("\r                              \r");

    // Final norm (last token only)
    let last = &x[(seq_len - 1) * hidden..seq_len * hidden];
    let normed = rms_norm(last, &weights.final_norm);
    lm_head_parallel(&normed, &weights.embed, weights.vocab, hidden)
}

/// Prefill with vision embeddings injected at IMAGE_PAD positions.
/// `vision_embeds` is [num_vision_tokens, hidden] from the vision encoder.
/// The token_ids should contain IMAGE_PAD tokens at the right positions.
pub fn prefill_with_vision(
    weights: &ModelWeights,
    token_ids: &[u32],
    vision_embeds: &[f32],
    num_vision_tokens: usize,
    cache: &mut HybridCache,
) -> Vec<f32> {
    let hidden = weights.hidden;
    let config = &weights.config;
    let seq_len = token_ids.len();

    // Embedding: [seq_len, hidden]
    let mut x = vec![0.0f32; seq_len * hidden];
    let mut vision_idx = 0usize;
    for (t, &tid) in token_ids.iter().enumerate() {
        if (tid == crate::tokenizer::IMAGE_PAD || tid == crate::tokenizer::VIDEO_PAD) && vision_idx < num_vision_tokens {
            let src = &vision_embeds[vision_idx * hidden..(vision_idx + 1) * hidden];
            x[t * hidden..(t + 1) * hidden].copy_from_slice(src);
            vision_idx += 1;
        } else {
            let row = embed_lookup(&weights.embed, tid as usize, hidden);
            x[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
        }
    }
    if vision_idx != num_vision_tokens {
        eprintln!("WARNING: Expected {num_vision_tokens} vision tokens but found {vision_idx} vision pad positions");
    }

    let num_layers = config.num_layers;
    for i in 0..num_layers {
        eprint!("\r  Prefill: layer {}/{num_layers}  ", i + 1);
        match (&weights.layers[i], &mut cache.entries[i]) {
            (HybridLayerWeights::DeltaNet(lw), CacheEntry::DeltaNet(state)) => {
                prefill_deltanet_layer(&mut x, seq_len, lw, state, config);
            }
            (HybridLayerWeights::FullAttn(lw), CacheEntry::KvCache(kv)) => {
                prefill_attn_layer(&mut x, seq_len, lw, kv, config, &weights.rope_cos, &weights.rope_sin);
            }
            _ => panic!("Layer type mismatch at layer {i}"),
        }
    }
    eprint!("\r                              \r");

    // Final norm (last token only)
    let last = &x[(seq_len - 1) * hidden..seq_len * hidden];
    let normed = rms_norm(last, &weights.final_norm);
    lm_head_parallel(&normed, &weights.embed, weights.vocab, hidden)
}

/// Prefill a full attention layer with all tokens at once (efficient GEMM).
/// q_proj outputs [2 * q_dim] when attn_output_gate=true.
fn prefill_attn_layer(
    x: &mut [f32],
    seq_len: usize,
    lw: &FullAttnLayerWeights,
    kv: &mut KvCacheEntry,
    config: &Qor4bConfig,
    rope_cos: &[f32],
    rope_sin: &[f32],
) {
    let hidden = config.hidden_size;
    let num_heads = config.num_attn_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.attn_head_dim;
    let q_dim = num_heads * head_dim; // 4096
    let num_kv_groups = config.num_kv_groups();
    let rotary_dim = config.rope_dim(); // 64: total dims rotated per head

    // Pre-attention RmsNorm per token
    let mut x_norm = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &x[t * hidden..(t + 1) * hidden];
        let normed = rms_norm(row, &lw.input_norm);
        x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
    }

    // Q projection: output is [seq_len, 2*q_dim] = [seq_len, 8192]
    // Layout per token: interleaved [Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), ...]
    let q_gate_all = gemm(&x_norm, seq_len, &lw.q_proj);
    let q_gate_stride = 2 * q_dim; // 8192 per token

    // Split Q and gate per-head: each head has [Q:head_dim, gate:head_dim]
    let mut q_all = vec![0.0f32; seq_len * q_dim];
    let mut gate_all = vec![0.0f32; seq_len * q_dim];
    for t in 0..seq_len {
        let src_base = t * q_gate_stride;
        let q_base = t * q_dim;
        for h in 0..num_heads {
            let src = src_base + h * 2 * head_dim;
            let dst = q_base + h * head_dim;
            q_all[dst..dst + head_dim].copy_from_slice(&q_gate_all[src..src + head_dim]);
            gate_all[dst..dst + head_dim].copy_from_slice(&q_gate_all[src + head_dim..src + 2 * head_dim]);
        }
    }

    // K, V GEMM
    let mut k_all = gemm(&x_norm, seq_len, &lw.k_proj);
    let v_all = gemm(&x_norm, seq_len, &lw.v_proj);

    // Per-head QK norm + partial RoPE per token
    for t in 0..seq_len {
        let q_start = t * q_dim;
        per_head_rms_norm(&mut q_all[q_start..q_start + q_dim], num_heads, head_dim, &lw.q_norm);
        let k_start = t * num_kv_heads * head_dim;
        per_head_rms_norm(&mut k_all[k_start..k_start + num_kv_heads * head_dim], num_kv_heads, head_dim, &lw.k_norm);
        apply_partial_rope(&mut q_all[q_start..q_start + q_dim], num_heads, head_dim, rope_cos, rope_sin, rotary_dim, t);
        apply_partial_rope(&mut k_all[k_start..k_start + num_kv_heads * head_dim], num_kv_heads, head_dim, rope_cos, rope_sin, rotary_dim, t);
    }

    // Store in KV cache
    kv.k = k_all;
    kv.v = v_all;
    kv.seq_len = seq_len;

    // Causal attention — parallel across heads
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;

    let head_results: Vec<Vec<f32>> = (0..num_heads).into_par_iter().map(|h| {
        let kv_h = h / num_kv_groups;
        let mut head_out = vec![0.0f32; seq_len * head_dim];
        for t1 in 0..seq_len {
            let attend_len = t1 + 1;
            let q_off = t1 * q_dim + h * head_dim;
            let q_vec = &q_all[q_off..q_off + head_dim];
            let mut scores = vec![0.0f32; attend_len];
            for t2 in 0..attend_len {
                let k_off = t2 * kv_stride + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim { dot += q_vec[d] * kv.k[k_off + d]; }
                scores[t2] = dot * scale;
            }
            softmax_raw(&mut scores);
            let out_base = t1 * head_dim;
            for t2 in 0..attend_len {
                let v_off = t2 * kv_stride + kv_h * head_dim;
                let sc = scores[t2];
                for d in 0..head_dim { head_out[out_base + d] += sc * kv.v[v_off + d]; }
            }
        }
        head_out
    }).collect();

    // Interleave head results
    let mut attn_output = vec![0.0f32; seq_len * q_dim];
    for (h, hr) in head_results.iter().enumerate() {
        for t in 0..seq_len {
            let src = &hr[t * head_dim..(t + 1) * head_dim];
            let dst = t * q_dim + h * head_dim;
            attn_output[dst..dst + head_dim].copy_from_slice(src);
        }
    }

    // Apply output gate: attn_output *= sigmoid(gate) per token
    for j in 0..seq_len * q_dim {
        attn_output[j] *= 1.0 / (1.0 + (-gate_all[j]).exp());
    }

    // O projection + residual
    let o_out = gemm(&attn_output, seq_len, &lw.o_proj);
    for j in 0..seq_len * hidden { x[j] += o_out[j]; }

    // Pre-MLP RmsNorm
    let mut x_norm2 = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &x[t * hidden..(t + 1) * hidden];
        let normed = rms_norm(row, &lw.post_attn_norm);
        x_norm2[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
    }

    // MLP
    let gate = gemm(&x_norm2, seq_len, &lw.gate_proj);
    let up = gemm(&x_norm2, seq_len, &lw.up_proj);
    let inter_size = lw.gate_proj.n();
    let mut intermediate = vec![0.0f32; seq_len * inter_size];
    for j in 0..seq_len * inter_size {
        let g = gate[j];
        intermediate[j] = (g / (1.0 + (-g).exp())) * up[j];
    }
    let mlp_out = gemm(&intermediate, seq_len, &lw.down_proj);

    for j in 0..seq_len * hidden { x[j] += mlp_out[j]; }
}
