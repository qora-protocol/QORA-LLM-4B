//! AVX-512 SIMD kernels for QOR4B CPU inference.
//!
//! Optimized GEMV for Q4 and F16 weights, and DeltaNet state updates.
//! Falls back to scalar code on non-AVX-512 CPUs (dispatch in gemv.rs).

// All functions in this module are `unsafe fn` wrapping SIMD intrinsics.
// Every operation inside them is inherently unsafe, so suppress the per-op warning.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

const Q4_GROUP_SIZE: usize = 32;

/// Check if AVX-512F is available at runtime.
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ============================================================
// AVX-512 helper
// ============================================================

/// Horizontal sum of 16 f32 lanes → scalar f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hsum_ps(v: __m512) -> f32 {
    // Fold 512 → 128 via extractf32x4
    let a = _mm512_extractf32x4_ps(v, 0);
    let b = _mm512_extractf32x4_ps(v, 1);
    let c = _mm512_extractf32x4_ps(v, 2);
    let d = _mm512_extractf32x4_ps(v, 3);
    let sum = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
    // Fold 128 → scalar
    let hi = _mm_movehl_ps(sum, sum);
    let sum2 = _mm_add_ps(sum, hi);
    let hi2 = _mm_shuffle_ps(sum2, sum2, 1);
    let sum3 = _mm_add_ss(sum2, hi2);
    _mm_cvtss_f32(sum3)
}

// ============================================================
// Q4 GEMV — AVX-512
// ============================================================

/// Q4 dequant factors: (q - 8) for q = 0..15
static Q4_FACTORS: [f32; 16] = [
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
];

/// Interleave indices for lo/hi nibble → contiguous output (first 16 of 32).
static INTERLEAVE_FIRST: [i32; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
/// Interleave indices for lo/hi nibble → contiguous output (second 16 of 32).
static INTERLEAVE_SECOND: [i32; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];

/// AVX-512 Q4 GEMV inner kernel. Replaces gemv_q4_inner.
///
/// Processes k_start..k_end rows of the weight matrix, accumulating into output[0..n].
/// Uses permutexvar for 16-entry LUT lookup of dequantized Q4 nibbles.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_q4_avx512(
    input: &[f32],
    packed: &[u8],
    scales: &[f16],
    n: usize,
    k_start: usize,
    k_end: usize,
) -> Vec<f32> {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2; // 16
    let packed_per_row = groups_per_row * packed_per_group;

    let mut output = vec![0.0f32; n];

    // Load constant vectors
    let factors = _mm512_loadu_ps(Q4_FACTORS.as_ptr());
    let interleave_lo = _mm512_loadu_si512(INTERLEAVE_FIRST.as_ptr() as *const _);
    let interleave_hi = _mm512_loadu_si512(INTERLEAVE_SECOND.as_ptr() as *const _);
    let nibble_mask = _mm512_set1_epi32(0x0F);

    for ki in k_start..k_end {
        let val = input[ki];
        if val == 0.0 {
            continue;
        }

        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;

        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * val;
            if s == 0.0 {
                continue;
            }

            // Build LUT: lut[i] = s * (i - 8)
            let s_vec = _mm512_set1_ps(s);
            let lut = _mm512_mul_ps(factors, s_vec);

            // Load 16 packed bytes and zero-extend to 16 x i32
            let po = pack_base + g * packed_per_group;
            let bytes = _mm_loadu_si128(packed.as_ptr().add(po) as *const __m128i);
            let bytes_i32 = _mm512_cvtepu8_epi32(bytes);

            // Extract nibbles
            let lo_nib = _mm512_and_epi32(bytes_i32, nibble_mask);
            let hi_nib = _mm512_srli_epi32(bytes_i32, 4);

            // LUT lookup: 16 values for even positions, 16 for odd
            let lo_vals = _mm512_permutexvar_ps(lo_nib, lut);
            let hi_vals = _mm512_permutexvar_ps(hi_nib, lut);

            // Interleave to get 32 output values in contiguous order
            let first_16 = _mm512_permutex2var_ps(lo_vals, interleave_lo, hi_vals);
            let second_16 = _mm512_permutex2var_ps(lo_vals, interleave_hi, hi_vals);

            // Accumulate into output
            let oo = g * Q4_GROUP_SIZE;
            let acc1 = _mm512_loadu_ps(output.as_ptr().add(oo));
            let acc2 = _mm512_loadu_ps(output.as_ptr().add(oo + 16));
            _mm512_storeu_ps(output.as_mut_ptr().add(oo), _mm512_add_ps(acc1, first_16));
            _mm512_storeu_ps(output.as_mut_ptr().add(oo + 16), _mm512_add_ps(acc2, second_16));
        }
    }

    output
}

/// AVX-512 fused gate+up Q4 inner kernel.
/// Returns (gate_out, up_out) for k_start..k_end.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_gate_up_q4_avx512(
    input: &[f32],
    gate_packed: &[u8],
    gate_scales: &[f16],
    up_packed: &[u8],
    up_scales: &[f16],
    n: usize,
    k_start: usize,
    k_end: usize,
) -> (Vec<f32>, Vec<f32>) {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;

    let mut gate_out = vec![0.0f32; n];
    let mut up_out = vec![0.0f32; n];

    let factors = _mm512_loadu_ps(Q4_FACTORS.as_ptr());
    let interleave_lo = _mm512_loadu_si512(INTERLEAVE_FIRST.as_ptr() as *const _);
    let interleave_hi = _mm512_loadu_si512(INTERLEAVE_SECOND.as_ptr() as *const _);
    let nibble_mask = _mm512_set1_epi32(0x0F);

    for ki in k_start..k_end {
        let val = input[ki];
        if val == 0.0 {
            continue;
        }

        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;

        for g in 0..groups_per_row {
            let gs = gate_scales[scale_base + g].to_f32() * val;
            let us = up_scales[scale_base + g].to_f32() * val;
            if gs == 0.0 && us == 0.0 {
                continue;
            }

            let po = pack_base + g * packed_per_group;
            let oo = g * Q4_GROUP_SIZE;

            // Gate LUT + decode
            if gs != 0.0 {
                let gs_vec = _mm512_set1_ps(gs);
                let g_lut = _mm512_mul_ps(factors, gs_vec);

                let g_bytes = _mm_loadu_si128(gate_packed.as_ptr().add(po) as *const __m128i);
                let g_i32 = _mm512_cvtepu8_epi32(g_bytes);
                let g_lo = _mm512_and_epi32(g_i32, nibble_mask);
                let g_hi = _mm512_srli_epi32(g_i32, 4);

                let g_lo_vals = _mm512_permutexvar_ps(g_lo, g_lut);
                let g_hi_vals = _mm512_permutexvar_ps(g_hi, g_lut);

                let g_first = _mm512_permutex2var_ps(g_lo_vals, interleave_lo, g_hi_vals);
                let g_second = _mm512_permutex2var_ps(g_lo_vals, interleave_hi, g_hi_vals);

                let ga1 = _mm512_loadu_ps(gate_out.as_ptr().add(oo));
                let ga2 = _mm512_loadu_ps(gate_out.as_ptr().add(oo + 16));
                _mm512_storeu_ps(gate_out.as_mut_ptr().add(oo), _mm512_add_ps(ga1, g_first));
                _mm512_storeu_ps(gate_out.as_mut_ptr().add(oo + 16), _mm512_add_ps(ga2, g_second));
            }

            // Up LUT + decode
            if us != 0.0 {
                let us_vec = _mm512_set1_ps(us);
                let u_lut = _mm512_mul_ps(factors, us_vec);

                let u_bytes = _mm_loadu_si128(up_packed.as_ptr().add(po) as *const __m128i);
                let u_i32 = _mm512_cvtepu8_epi32(u_bytes);
                let u_lo = _mm512_and_epi32(u_i32, nibble_mask);
                let u_hi = _mm512_srli_epi32(u_i32, 4);

                let u_lo_vals = _mm512_permutexvar_ps(u_lo, u_lut);
                let u_hi_vals = _mm512_permutexvar_ps(u_hi, u_lut);

                let u_first = _mm512_permutex2var_ps(u_lo_vals, interleave_lo, u_hi_vals);
                let u_second = _mm512_permutex2var_ps(u_lo_vals, interleave_hi, u_hi_vals);

                let ua1 = _mm512_loadu_ps(up_out.as_ptr().add(oo));
                let ua2 = _mm512_loadu_ps(up_out.as_ptr().add(oo + 16));
                _mm512_storeu_ps(up_out.as_mut_ptr().add(oo), _mm512_add_ps(ua1, u_first));
                _mm512_storeu_ps(up_out.as_mut_ptr().add(oo + 16), _mm512_add_ps(ua2, u_second));
            }
        }
    }

    (gate_out, up_out)
}

// ============================================================
// F16 GEMV — AVX-512
// ============================================================

/// AVX-512 F16 GEMV: input[k] @ weight[k,n] → output[n].
/// Uses _mm512_cvtph_ps for f16→f32 and FMA for accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_f16_avx512(
    input: &[f32],
    weight_data: &[f16],
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    let n16 = n / 16 * 16; // aligned to 16

    for ki in 0..k {
        let val = input[ki];
        if val == 0.0 {
            continue;
        }
        let val_vec = _mm512_set1_ps(val);
        let row = ki * n;

        // Process 16 elements at a time
        let mut j = 0usize;
        while j < n16 {
            // Load 16 x f16 as raw bits in __m256i, convert to __m512 f32
            let w_f16 = _mm256_loadu_si256(weight_data.as_ptr().add(row + j) as *const __m256i);
            let w_f32 = _mm512_cvtph_ps(w_f16);
            let acc = _mm512_loadu_ps(output.as_ptr().add(j));
            let result = _mm512_fmadd_ps(val_vec, w_f32, acc);
            _mm512_storeu_ps(output.as_mut_ptr().add(j), result);
            j += 16;
        }

        // Scalar tail
        while j < n {
            output[j] += val * weight_data[row + j].to_f32();
            j += 1;
        }
    }

    output
}

// ============================================================
// DeltaNet state update — AVX-512
// ============================================================

/// AVX-512 per-head DeltaNet state update.
///
/// Performs: decay(S *= alpha), retrieve(pred = S @ k), delta update(S += beta*(v-pred) @ k^T),
/// output(y = S @ q). hd must be divisible by 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn deltanet_head_update_avx512(
    state: &mut [f32], // [hd * hd]
    q: &[f32],         // [hd]
    k: &[f32],         // [hd]
    v: &[f32],         // [hd]
    alpha: f32,
    beta: f32,
    hd: usize,
    y_out: &mut [f32], // [hd] output slice
) {
    let hd16 = hd / 16 * 16;

    // 1. Decay: S *= alpha
    let al_vec = _mm512_set1_ps(alpha);
    let total = hd * hd;
    let mut i = 0usize;
    while i < total {
        let s_vec = _mm512_loadu_ps(state.as_ptr().add(i));
        _mm512_storeu_ps(state.as_mut_ptr().add(i), _mm512_mul_ps(s_vec, al_vec));
        i += 16;
    }

    // 2. Retrieve: pred[r] = S[r,:] @ k  (dot product per row)
    let mut pred = vec![0.0f32; hd];
    for r in 0..hd {
        let s_row = r * hd;
        let mut acc = _mm512_setzero_ps();
        let mut c = 0usize;
        while c < hd16 {
            let s_vec = _mm512_loadu_ps(state.as_ptr().add(s_row + c));
            let k_vec = _mm512_loadu_ps(k.as_ptr().add(c));
            acc = _mm512_fmadd_ps(s_vec, k_vec, acc);
            c += 16;
        }
        pred[r] = hsum_ps(acc);
    }

    // 3. Delta update: S[r,c] += beta * (v[r] - pred[r]) * k[c]
    for r in 0..hd {
        let delta = beta * (v[r] - pred[r]);
        if delta == 0.0 {
            continue;
        }
        let delta_vec = _mm512_set1_ps(delta);
        let s_row = r * hd;
        let mut c = 0usize;
        while c < hd16 {
            let s_vec = _mm512_loadu_ps(state.as_ptr().add(s_row + c));
            let k_vec = _mm512_loadu_ps(k.as_ptr().add(c));
            _mm512_storeu_ps(
                state.as_mut_ptr().add(s_row + c),
                _mm512_fmadd_ps(delta_vec, k_vec, s_vec),
            );
            c += 16;
        }
    }

    // 4. Output: y[r] = S[r,:] @ q  (dot product per row)
    for r in 0..hd {
        let s_row = r * hd;
        let mut acc = _mm512_setzero_ps();
        let mut c = 0usize;
        while c < hd16 {
            let s_vec = _mm512_loadu_ps(state.as_ptr().add(s_row + c));
            let q_vec = _mm512_loadu_ps(q.as_ptr().add(c));
            acc = _mm512_fmadd_ps(s_vec, q_vec, acc);
            c += 16;
        }
        y_out[r] = hsum_ps(acc);
    }
}
