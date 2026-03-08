//! GPU inference for QORA-4B.
//!
//! Implements single-token decode using Cortex GPU tensors for the hybrid
//! DeltaNet + Full Attention architecture.

use cortex::prelude::*;
use cortex::tensor::activation;
use std::io::Write;
use std::time::Instant;

use crate::config::Qor4bConfig;
use crate::gemv::ModelWeights;
use crate::generate::GenerateParams;
use crate::gpu_loader::{GpuModel, GpuHybridLayer, GpuDeltaNetLayer, GpuFullAttnLayer, load_model_gpu};
use crate::tokenizer::QoraTokenizer;

// ============================================================
// GPU Cache
// ============================================================

pub struct GpuHybridCache<B: Backend> {
    pub entries: Vec<GpuCacheEntry<B>>,
}

pub enum GpuCacheEntry<B: Backend> {
    DeltaNet(GpuDeltaNetState<B>),
    KvCache(GpuKvCacheEntry<B>),
}

pub struct GpuDeltaNetState<B: Backend> {
    /// State matrices: [num_v_heads, head_dim, head_dim]
    pub s: Tensor<B, 3>,
    /// Conv buffer: [qkv_dim, kernel_size-1]
    pub conv_buf: Tensor<B, 2>,
    pub conv_pos: usize,
}

pub struct GpuKvCacheEntry<B: Backend> {
    pub k: Option<Tensor<B, 3>>,  // [kv_heads, seq_len, head_dim]
    pub v: Option<Tensor<B, 3>>,
    pub seq_len: usize,
}

impl<B: Backend> GpuHybridCache<B> {
    pub fn new(config: &Qor4bConfig, device: &B::Device) -> Self {
        let mut entries = Vec::with_capacity(config.num_layers);
        for lt in &config.layer_types {
            match lt {
                crate::config::LayerType::DeltaNet => {
                    let num_v = config.num_v_heads;
                    let hd = config.deltanet_head_dim;
                    let qkv_dim = config.deltanet_qkv_dim();
                    let buf_len = config.conv_kernel_size - 1;
                    entries.push(GpuCacheEntry::DeltaNet(GpuDeltaNetState {
                        s: Tensor::zeros([num_v, hd, hd], device),
                        conv_buf: Tensor::zeros([qkv_dim, buf_len], device),
                        conv_pos: 0,
                    }));
                }
                crate::config::LayerType::FullAttn => {
                    entries.push(GpuCacheEntry::KvCache(GpuKvCacheEntry {
                        k: None,
                        v: None,
                        seq_len: 0,
                    }));
                }
            }
        }
        Self { entries }
    }
}

// ============================================================
// Helper ops
// ============================================================

/// RMS norm with (1 + gamma) scaling — used for all layer norms.
fn rms_norm_1plus<B: Backend>(x: Tensor<B, 2>, gamma: &Tensor<B, 1>) -> Tensor<B, 2> {
    let eps = 1e-6;
    let x_sq = x.clone().powf_scalar(2.0);
    let mean_sq = x_sq.mean_dim(1);
    let rms = (mean_sq + eps).sqrt();
    let normed = x / rms;
    // (1 + gamma) scaling
    let g = gamma.clone().unsqueeze::<2>() + 1.0;
    normed * g
}

/// RMS norm with plain gamma scaling — used for DeltaNet per-head attention norm.
fn rms_norm_plain<B: Backend>(x: Tensor<B, 2>, gamma: &Tensor<B, 1>) -> Tensor<B, 2> {
    let eps = 1e-6;
    let x_sq = x.clone().powf_scalar(2.0);
    let mean_sq = x_sq.mean_dim(1);
    let rms = (mean_sq + eps).sqrt();
    let normed = x / rms;
    normed * gamma.clone().unsqueeze::<2>()
}

/// L2 normalize each row: x / ||x||
fn l2_normalize_rows<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let eps = 1e-12;
    let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt() + eps;
    x / norm
}

/// SiLU activation: x * sigmoid(x)
fn silu_2d<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    activation::silu(x)
}

/// Sigmoid
fn sigmoid_2d<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    activation::sigmoid(x)
}

/// Softplus: log(1 + exp(x))
fn softplus_1d<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    (x.exp() + 1.0).log()
}

// ============================================================
// DeltaNet GPU forward (single token)
// ============================================================

fn forward_deltanet_decode_gpu<B: Backend>(
    x: Tensor<B, 2>,           // [1, hidden]
    layer: &GpuDeltaNetLayer<B>,
    state: &mut GpuDeltaNetState<B>,
    config: &Qor4bConfig,
) -> Tensor<B, 2> {
    let _device = x.device();
    let num_qk = config.num_qk_heads;
    let num_v = config.num_v_heads;
    let hd = config.deltanet_head_dim;
    let q_dim = config.deltanet_q_dim();
    let k_dim = config.deltanet_k_dim();
    let v_dim = config.deltanet_v_dim();
    let qkv_dim = config.deltanet_qkv_dim();
    let ks = config.conv_kernel_size;
    let v_per_qk = num_v / num_qk;
    let scale = 1.0 / (hd as f32).sqrt();

    // 1. Pre-attention RmsNorm (1+gamma)
    let x_norm = rms_norm_1plus(x.clone(), &layer.input_norm_gamma);

    // 2. QKV projection: [1, hidden] @ [hidden, qkv_dim] → [1, qkv_dim]
    let qkv = x_norm.clone().matmul(layer.in_proj_qkv.clone());

    // 3. Conv1d: depthwise causal conv over qkv_dim channels
    //    conv_buf: [qkv_dim, buf_len=3], qkv: [1, qkv_dim] → [qkv_dim, 1]
    let qkv_col = qkv.clone().reshape([qkv_dim, 1]); // current token as column
    // Cat with buffer: [qkv_dim, buf_len] cat [qkv_dim, 1] → [qkv_dim, ks=4]
    let conv_input = Tensor::cat(vec![state.conv_buf.clone(), qkv_col.clone()], 1);
    // Elementwise multiply with conv weights and sum: [qkv_dim, ks] * [qkv_dim, ks] → sum dim 1 → [qkv_dim]
    let qkv_conv = (conv_input.clone() * layer.conv1d_weight.clone()).sum_dim(1).reshape([1, qkv_dim]);

    // Update conv buffer: shift left, append new
    state.conv_buf = conv_input.slice([0..qkv_dim, 1..ks]); // take last buf_len columns
    state.conv_pos += 1;

    // 4. SiLU on all channels
    let qkv_act = activation::silu(qkv_conv);

    // 5. Split Q, K, V
    let q = qkv_act.clone().slice([0..1, 0..q_dim]);           // [1, 2048]
    let k = qkv_act.clone().slice([0..1, q_dim..q_dim + k_dim]); // [1, 2048]
    let v = qkv_act.slice([0..1, q_dim + k_dim..qkv_dim]);     // [1, 4096]

    // 6. L2 normalize Q, K per head, scale Q
    let q = q.reshape([num_qk, hd]);
    let k = k.reshape([num_qk, hd]);
    let q = l2_normalize_rows(q).mul_scalar(scale);
    let k = l2_normalize_rows(k);

    // 7. Alpha and beta
    let a_proj = x_norm.clone().matmul(layer.in_proj_a.clone()).reshape([num_v]); // [32]
    let b_proj = x_norm.clone().matmul(layer.in_proj_b.clone()).reshape([num_v]); // [32]

    // alpha = exp(-exp(a_log) * softplus(a_proj + dt_bias))
    let a_exp = layer.a_log.clone().exp(); // [32]
    let sp = softplus_1d(a_proj + layer.dt_bias.clone()); // [32]
    let alpha = (a_exp * sp).neg().exp(); // [32]
    // beta = sigmoid(b_proj)
    let beta = activation::sigmoid(b_proj); // [32]

    // 8. GVA expansion: repeat QK heads from num_qk to num_v
    //    Each QK head maps to v_per_qk V heads
    //    [num_qk, hd] → [num_qk, 1, hd] → [num_qk, v_per_qk, hd] → [num_v, hd]
    let q_exp: Tensor<B, 3> = q.unsqueeze_dim::<3>(1).repeat_dim(1, v_per_qk);
    let q_exp = q_exp.reshape([num_v, hd]);
    let k_exp: Tensor<B, 3> = k.unsqueeze_dim::<3>(1).repeat_dim(1, v_per_qk);
    let k_exp = k_exp.reshape([num_v, hd]);
    let v_2d = v.reshape([num_v, hd]); // [32, 128]

    // 9. Delta rule state update (batched across all 32 V heads)
    // S: [32, 128, 128], K: [32, 128], Q: [32, 128], V: [32, 128]

    // Decay: S *= alpha (broadcast [32,1,1])
    let alpha_3d: Tensor<B, 3> = alpha.clone().reshape([num_v, 1, 1]);
    state.s = state.s.clone() * alpha_3d;

    // Retrieve: pred = S @ K → [32, 128]
    // S: [32, 128, 128], K: [32, 128, 1] → bmm → [32, 128, 1] → squeeze → [32, 128]
    let k_col: Tensor<B, 3> = k_exp.clone().unsqueeze_dim::<3>(2); // [32, 128, 1]
    let pred = state.s.clone().matmul(k_col).reshape([num_v, hd]); // [32, 128]

    // Delta update: S += beta * outer(V - pred, K)
    // delta_v = beta * (V - pred): [32, 128]
    let beta_2d: Tensor<B, 2> = beta.reshape([num_v, 1]);
    let delta_v = (v_2d - pred) * beta_2d;
    // outer: [32, 128, 1] @ [32, 1, 128] → [32, 128, 128]
    let delta_col: Tensor<B, 3> = delta_v.unsqueeze_dim::<3>(2); // [32, 128, 1]
    let k_row: Tensor<B, 3> = k_exp.unsqueeze_dim::<3>(1); // [32, 1, 128]
    let outer = delta_col.matmul(k_row); // [32, 128, 128]
    state.s = state.s.clone() + outer;

    // Output: y = S @ Q → [32, 128]
    let q_col: Tensor<B, 3> = q_exp.unsqueeze_dim::<3>(2); // [32, 128, 1]
    let y = state.s.clone().matmul(q_col).reshape([num_v, hd]); // [32, 128]

    // 10. Per-head RMS norm (plain gamma, NOT 1+gamma)
    let y_normed = rms_norm_plain(y, &layer.attn_norm_weight); // [32, 128]

    // 11. Output gating: z = silu(z_proj(x)), output = y_normed * z
    let z = x_norm.clone().matmul(layer.in_proj_z.clone()); // [1, v_dim=4096]
    let z = activation::silu(z).reshape([num_v, hd]); // [32, 128]
    let gated = y_normed * z; // [32, 128]

    // 12. Out projection + residual
    let gated_flat = gated.reshape([1, v_dim]); // [1, 4096]
    let attn_out = gated_flat.matmul(layer.out_proj.clone()); // [1, hidden]
    let out = x + attn_out;

    // 13. MLP: silu(gate) * up → down
    let x_norm2 = rms_norm_1plus(out.clone(), &layer.post_attn_norm_gamma);
    let gate = x_norm2.clone().matmul(layer.gate_proj.clone());
    let up = x_norm2.matmul(layer.up_proj.clone());
    let mlp_out = silu_2d(gate).mul(up).matmul(layer.down_proj.clone());

    out + mlp_out
}

// ============================================================
// Full Attention GPU forward (single token)
// ============================================================

fn forward_attn_decode_gpu<B: Backend>(
    x: Tensor<B, 2>,           // [1, hidden]
    layer: &GpuFullAttnLayer<B>,
    kv: &mut GpuKvCacheEntry<B>,
    config: &Qor4bConfig,
    rope_cos: &Tensor<B, 2>,   // [max_pos, half_dim]
    rope_sin: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let num_heads = config.num_attn_heads; // 16
    let num_kv_heads = config.num_kv_heads; // 4
    let head_dim = config.attn_head_dim; // 256
    let q_dim = config.attn_q_dim(); // 4096
    let num_kv_groups = config.num_kv_groups(); // 4
    let rotary_dim = config.rope_dim(); // 64
    let half_rot = rotary_dim / 2; // 32
    let offset = kv.seq_len;

    // 1. Pre-attention RmsNorm (1+gamma)
    let x_norm = rms_norm_1plus(x.clone(), &layer.input_norm_gamma);

    // 2. Q+Gate projection: [1, hidden] → [1, 2*q_dim=8192]
    //    Interleaved: [Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), ...]
    let q_gate = x_norm.clone().matmul(layer.q_proj.clone()); // [1, 8192]
    // Reshape to [num_heads, 2*head_dim] → split
    let q_gate_3d = q_gate.reshape([num_heads, 2 * head_dim]);
    let q_raw = q_gate_3d.clone().slice([0..num_heads, 0..head_dim]); // [16, 256]
    let gate = q_gate_3d.slice([0..num_heads, head_dim..2 * head_dim]); // [16, 256]

    // 3. K, V projections
    let k_raw = x_norm.clone().matmul(layer.k_proj.clone()); // [1, 1024]
    let k_raw = k_raw.reshape([num_kv_heads, head_dim]); // [4, 256]
    let v_new = x_norm.matmul(layer.v_proj.clone()); // [1, 1024]
    let v_new = v_new.reshape([num_kv_heads, head_dim]); // [4, 256]

    // 4. Per-head QK-norm (1+gamma)
    let q_normed = rms_norm_1plus(q_raw, &layer.q_norm); // [16, 256]
    let k_normed = rms_norm_1plus(k_raw, &layer.k_norm); // [4, 256]

    // 5. Partial RoPE: rotate first rotary_dim=64 dims of each head
    let q_rot = q_normed.clone().slice([0..num_heads, 0..rotary_dim]); // [16, 64]
    let q_pass = q_normed.slice([0..num_heads, rotary_dim..head_dim]); // [16, 192]
    let k_rot = k_normed.clone().slice([0..num_kv_heads, 0..rotary_dim]); // [4, 64]
    let k_pass = k_normed.slice([0..num_kv_heads, rotary_dim..head_dim]); // [4, 192]

    // Split-half RoPE on rotary part
    let q_rotated = apply_partial_rope_gpu(q_rot, rope_cos, rope_sin, num_heads, half_rot, offset);
    let k_rotated = apply_partial_rope_gpu(k_rot, rope_cos, rope_sin, num_kv_heads, half_rot, offset);

    // Reassemble: [rotated, pass_through]
    let q_final = Tensor::cat(vec![q_rotated, q_pass], 1); // [16, 256]
    let k_final = Tensor::cat(vec![k_rotated, k_pass], 1); // [4, 256]

    // 6. KV cache update: reshape to [heads, 1, head_dim] and cat
    let k_new_3d: Tensor<B, 3> = k_final.unsqueeze_dim::<3>(1); // [4, 1, 256]
    let v_new_3d: Tensor<B, 3> = v_new.unsqueeze_dim::<3>(1);   // [4, 1, 256]

    let (k_cache, v_cache) = if let (Some(prev_k), Some(prev_v)) =
        (kv.k.take(), kv.v.take())
    {
        (Tensor::cat(vec![prev_k, k_new_3d], 1),
         Tensor::cat(vec![prev_v, v_new_3d], 1))
    } else {
        (k_new_3d, v_new_3d)
    };
    kv.seq_len = offset + 1;
    let kv_seq_len = kv.seq_len;

    // 7. GQA attention
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q: [16, 256] → [16, 1, 256]
    let q_3d: Tensor<B, 3> = q_final.unsqueeze_dim::<3>(1);

    // Expand KV heads: [4, seq, 256] → [16, seq, 256]
    let k_exp = if num_kv_groups > 1 {
        let k4: Tensor<B, 4> = k_cache.clone().unsqueeze_dim::<4>(1).repeat_dim(1, num_kv_groups);
        k4.reshape([num_heads, kv_seq_len, head_dim])
    } else {
        k_cache.clone()
    };
    let v_exp = if num_kv_groups > 1 {
        let v4: Tensor<B, 4> = v_cache.clone().unsqueeze_dim::<4>(1).repeat_dim(1, num_kv_groups);
        v4.reshape([num_heads, kv_seq_len, head_dim])
    } else {
        v_cache.clone()
    };

    // scores: [16, 1, 256] @ [16, 256, seq] → [16, 1, seq]
    let scores = q_3d.matmul(k_exp.swap_dims(1, 2)).mul_scalar(scale);
    let attn_weights = activation::softmax(scores, 2);
    // context: [16, 1, seq] @ [16, seq, 256] → [16, 1, 256]
    let context = attn_weights.matmul(v_exp);

    // Store cache
    kv.k = Some(k_cache);
    kv.v = Some(v_cache);

    // Reshape: [16, 1, 256] → [1, 4096]
    let attn_out = context.reshape([1, q_dim]);

    // 8. Output gate: attn_out *= sigmoid(gate)
    let gate_flat = gate.reshape([1, q_dim]);
    let attn_gated = attn_out * sigmoid_2d(gate_flat);

    // 9. O projection + residual
    let o_out = attn_gated.matmul(layer.o_proj.clone());
    let out = x + o_out;

    // 10. MLP
    let x_norm2 = rms_norm_1plus(out.clone(), &layer.post_attn_norm_gamma);
    let gate_mlp = x_norm2.clone().matmul(layer.gate_proj.clone());
    let up = x_norm2.matmul(layer.up_proj.clone());
    let mlp_out = silu_2d(gate_mlp).mul(up).matmul(layer.down_proj.clone());

    out + mlp_out
}

/// Apply split-half RoPE on rotary portion.
/// x: [num_heads, rotary_dim], cos/sin tables: [max_pos, half_rot]
fn apply_partial_rope_gpu<B: Backend>(
    x: Tensor<B, 2>,
    cos_table: &Tensor<B, 2>,
    sin_table: &Tensor<B, 2>,
    num_heads: usize,
    half_rot: usize,
    offset: usize,
) -> Tensor<B, 2> {
    let rotary_dim = half_rot * 2;
    // Split into first half and second half per head
    let x_first = x.clone().slice([0..num_heads, 0..half_rot]);       // [heads, 32]
    let x_second = x.slice([0..num_heads, half_rot..rotary_dim]);      // [heads, 32]

    // cos, sin for this position: [1, half_rot] → broadcast over heads
    let cos = cos_table.clone().slice([offset..offset + 1, 0..half_rot]); // [1, 32]
    let sin = sin_table.clone().slice([offset..offset + 1, 0..half_rot]); // [1, 32]

    let new_first = x_first.clone() * cos.clone() - x_second.clone() * sin.clone();
    let new_second = x_second * cos + x_first * sin;

    Tensor::cat(vec![new_first, new_second], 1) // [heads, rotary_dim]
}

// ============================================================
// Forward decode (full model, single token)
// ============================================================

fn forward_decode_gpu<B: Backend>(
    model: &GpuModel<B>,
    token_id: usize,
    cache: &mut GpuHybridCache<B>,
) -> Vec<f32> {
    let device = model.final_norm_gamma.device();
    let hidden = model.embed_hidden;
    let config = &model.config;

    // Embed: lookup from CPU, upload to GPU
    let embed_start = token_id * hidden;
    let embed_slice = &model.embed_f32[embed_start..embed_start + hidden];
    let mut x: Tensor<B, 2> = Tensor::from_data(
        cortex::tensor::TensorData::new(embed_slice.to_vec(), [1, hidden]),
        &device,
    );

    for (i, (layer, entry)) in model.layers.iter().zip(cache.entries.iter_mut()).enumerate() {
        x = match (layer, entry) {
            (GpuHybridLayer::DeltaNet(dl), GpuCacheEntry::DeltaNet(ds)) => {
                forward_deltanet_decode_gpu(x, dl, ds, config)
            }
            (GpuHybridLayer::FullAttn(fl), GpuCacheEntry::KvCache(kv)) => {
                forward_attn_decode_gpu(x, fl, kv, config, &model.rope_cos, &model.rope_sin)
            }
            _ => panic!("Layer/cache type mismatch at layer {i}"),
        };
    }

    // Final norm (1+gamma)
    x = rms_norm_1plus(x, &model.final_norm_gamma);

    // lm_head on CPU (vocab=248K too large for VRAM)
    let x_data = x.to_data().to_vec::<f32>().unwrap();
    lm_head_cpu(&x_data, &model.embed_f32, model.embed_vocab, hidden)
}

/// Batched DeltaNet prefill: GPU batch matmuls + CPU sequential state update.
///
/// Strategy: batch all expensive projections on GPU, download results,
/// run the lightweight conv1d + recurrent state update on CPU, then
/// upload outputs back to GPU for the remaining batch ops (gating, out_proj, MLP).
/// This avoids seq_len GPU round-trips and achieves ~10x fewer PCIe transfers.
fn prefill_deltanet_layer_gpu<B: Backend>(
    x: Tensor<B, 2>,           // [seq_len, hidden] on GPU
    layer: &GpuDeltaNetLayer<B>,
    state: &mut GpuDeltaNetState<B>,
    config: &Qor4bConfig,
) -> Tensor<B, 2> {
    let device = x.device();
    let seq_len = x.dims()[0];
    let _hidden = config.hidden_size;
    let num_qk = config.num_qk_heads;
    let num_v = config.num_v_heads;
    let hd = config.deltanet_head_dim;
    let q_dim = config.deltanet_q_dim();
    let k_dim = config.deltanet_k_dim();
    let v_dim = config.deltanet_v_dim();
    let qkv_dim = config.deltanet_qkv_dim();
    let ks = config.conv_kernel_size;
    let buf_len = ks - 1;
    let v_per_qk = num_v / num_qk;
    let scale = 1.0 / (hd as f32).sqrt();

    // ==== GPU BATCH PHASE 1: all projections ====
    let x_norm = rms_norm_1plus(x.clone(), &layer.input_norm_gamma);
    let qkv_all = x_norm.clone().matmul(layer.in_proj_qkv.clone()); // [seq, qkv_dim]
    let a_all = x_norm.clone().matmul(layer.in_proj_a.clone());       // [seq, num_v]
    let b_all = x_norm.clone().matmul(layer.in_proj_b.clone());       // [seq, num_v]
    let z_all = x_norm.matmul(layer.in_proj_z.clone());               // [seq, v_dim]

    // Download batch results + small layer weights to CPU
    let qkv_cpu: Vec<f32> = qkv_all.to_data().to_vec::<f32>().unwrap();
    let a_cpu: Vec<f32> = a_all.to_data().to_vec::<f32>().unwrap();
    let b_cpu: Vec<f32> = b_all.to_data().to_vec::<f32>().unwrap();

    let conv_w: Vec<f32> = layer.conv1d_weight.clone().to_data().to_vec::<f32>().unwrap();
    let a_log_data: Vec<f32> = layer.a_log.clone().to_data().to_vec::<f32>().unwrap();
    let dt_bias_data: Vec<f32> = layer.dt_bias.clone().to_data().to_vec::<f32>().unwrap();
    let attn_norm_w: Vec<f32> = layer.attn_norm_weight.clone().to_data().to_vec::<f32>().unwrap();

    // Download state to CPU
    let mut s: Vec<f32> = state.s.clone().to_data().to_vec::<f32>().unwrap();
    let mut conv_buf: Vec<f32> = state.conv_buf.clone().to_data().to_vec::<f32>().unwrap();

    // Precompute a_exp table
    let a_exp_table: Vec<f32> = a_log_data.iter().map(|&al| al.exp()).collect();

    // ==== CPU SEQUENTIAL PHASE: conv1d + state update ====
    let mut output_y = vec![0.0f32; seq_len * v_dim];
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    let mut q_exp = vec![0.0f32; num_v * hd];
    let mut k_exp = vec![0.0f32; num_v * hd];
    let mut pred = vec![0.0f32; num_v * hd];

    for t in 0..seq_len {
        let qkv_t = &qkv_cpu[t * qkv_dim..(t + 1) * qkv_dim];
        let a_t = &a_cpu[t * num_v..(t + 1) * num_v];
        let b_t = &b_cpu[t * num_v..(t + 1) * num_v];

        // Conv1d: depthwise causal convolution
        for ch in 0..qkv_dim {
            let buf_off = ch * buf_len;
            let w_off = ch * ks;
            let mut sum = 0.0f32;
            for j in 0..buf_len {
                sum += conv_buf[buf_off + j] * conv_w[w_off + j];
            }
            sum += qkv_t[ch] * conv_w[w_off + buf_len];
            qkv_conv[ch] = sum;
            // Shift buffer left, append new token
            for j in 0..buf_len - 1 {
                conv_buf[buf_off + j] = conv_buf[buf_off + j + 1];
            }
            conv_buf[buf_off + buf_len - 1] = qkv_t[ch];
        }

        // SiLU
        for v in qkv_conv.iter_mut() {
            *v = *v * (1.0 / (1.0 + (-*v).exp()));
        }

        // Split Q, K, V and L2 normalize + GVA expand
        let q = &qkv_conv[0..q_dim];
        let k = &qkv_conv[q_dim..q_dim + k_dim];
        let v_data = &qkv_conv[q_dim + k_dim..qkv_dim];

        for qk_h in 0..num_qk {
            let qs = qk_h * hd;
            let ks_off = qk_h * hd;
            let mut q_sq = 0.0f32;
            let mut k_sq = 0.0f32;
            for d in 0..hd {
                q_sq += q[qs + d] * q[qs + d];
                k_sq += k[ks_off + d] * k[ks_off + d];
            }
            let q_inv = scale / (q_sq.sqrt() + 1e-12);
            let k_inv = 1.0 / (k_sq.sqrt() + 1e-12);
            for g in 0..v_per_qk {
                let vh = qk_h * v_per_qk + g;
                let o = vh * hd;
                for d in 0..hd {
                    q_exp[o + d] = q[qs + d] * q_inv;
                    k_exp[o + d] = k[ks_off + d] * k_inv;
                }
            }
        }

        // State update per V head
        for h in 0..num_v {
            let sp_in = a_t[h] + dt_bias_data[h];
            let sp = (1.0 + sp_in.exp()).ln();
            let alpha = (-a_exp_table[h] * sp).exp();
            let beta = 1.0 / (1.0 + (-b_t[h]).exp());

            let s_off = h * hd * hd;
            let ho = h * hd;

            // Decay
            for i in 0..hd * hd { s[s_off + i] *= alpha; }

            // pred = S @ K
            for i in 0..hd {
                let mut dot = 0.0f32;
                for j in 0..hd { dot += s[s_off + i * hd + j] * k_exp[ho + j]; }
                pred[ho + i] = dot;
            }

            // S += outer(beta*(V - pred), K)
            for i in 0..hd {
                let delta_i = beta * (v_data[ho + i] - pred[ho + i]);
                for j in 0..hd {
                    s[s_off + i * hd + j] += delta_i * k_exp[ho + j];
                }
            }

            // y = S @ Q
            for i in 0..hd {
                let mut dot = 0.0f32;
                for j in 0..hd { dot += s[s_off + i * hd + j] * q_exp[ho + j]; }
                pred[ho + i] = dot;
            }
        }

        // Per-head RMS norm (plain gamma, NOT 1+gamma)
        let y_out = &mut output_y[t * v_dim..(t + 1) * v_dim];
        for h in 0..num_v {
            let ho = h * hd;
            let mut sum_sq = 0.0f32;
            for d in 0..hd { sum_sq += pred[ho + d] * pred[ho + d]; }
            let inv_rms = 1.0 / (sum_sq / hd as f32 + 1e-6).sqrt();
            for d in 0..hd {
                y_out[ho + d] = pred[ho + d] * inv_rms * attn_norm_w[d];
            }
        }
    }

    // ==== GPU BATCH PHASE 2: gating + out_proj + MLP ====
    // Upload y and state back to GPU
    let y_gpu: Tensor<B, 2> = Tensor::from_data(
        cortex::tensor::TensorData::new(output_y, [seq_len, v_dim]),
        &device,
    );
    state.s = Tensor::from_data(
        cortex::tensor::TensorData::new(s, [num_v, hd, hd]),
        &device,
    );
    state.conv_buf = Tensor::from_data(
        cortex::tensor::TensorData::new(conv_buf, [qkv_dim, buf_len]),
        &device,
    );
    state.conv_pos += seq_len;

    // Output gating: y * silu(z)
    let z_act = activation::silu(z_all);
    let gated = y_gpu * z_act;

    // Out projection + residual
    let attn_out = gated.matmul(layer.out_proj.clone());
    let out = x + attn_out;

    // MLP: norm → silu(gate) * up → down → residual
    let x_norm2 = rms_norm_1plus(out.clone(), &layer.post_attn_norm_gamma);
    let gate = x_norm2.clone().matmul(layer.gate_proj.clone());
    let up = x_norm2.matmul(layer.up_proj.clone());
    let mlp_out = silu_2d(gate).mul(up).matmul(layer.down_proj.clone());

    out + mlp_out
}

/// Prefill on GPU — keeps x as GPU tensor throughout all layers.
fn forward_prefill_gpu<B: Backend>(
    model: &GpuModel<B>,
    token_ids: &[u32],
    cache: &mut GpuHybridCache<B>,
) -> Vec<f32> {
    let device = model.final_norm_gamma.device();
    let hidden = model.embed_hidden;
    let config = &model.config;
    let seq_len = token_ids.len();

    // Embed all tokens on CPU, upload to GPU once
    let mut embed_data = vec![0.0f32; seq_len * hidden];
    for (t, &tid) in token_ids.iter().enumerate() {
        let start = tid as usize * hidden;
        embed_data[t * hidden..(t + 1) * hidden]
            .copy_from_slice(&model.embed_f32[start..start + hidden]);
    }
    let mut x: Tensor<B, 2> = Tensor::from_data(
        cortex::tensor::TensorData::new(embed_data, [seq_len, hidden]),
        &device,
    );

    // Process layer by layer (x stays on GPU)
    for (i, (layer, entry)) in model.layers.iter().zip(cache.entries.iter_mut()).enumerate() {
        x = match (layer, entry) {
            (GpuHybridLayer::DeltaNet(dl), GpuCacheEntry::DeltaNet(ds)) => {
                prefill_deltanet_layer_gpu(x, dl, ds, config)
            }
            (GpuHybridLayer::FullAttn(fl), GpuCacheEntry::KvCache(kv)) => {
                prefill_attn_layer_gpu(x, fl, kv, config, &model.rope_cos, &model.rope_sin)
            }
            _ => panic!("Layer/cache type mismatch at layer {i}"),
        };
        eprint!("\r  Prefill: layer {}/{}  ", i + 1, config.num_layers);
        std::io::stderr().flush().ok();
    }
    eprintln!();

    // Final norm (last token only)
    let last = x.slice([seq_len - 1..seq_len, 0..hidden]);
    let normed = rms_norm_1plus(last, &model.final_norm_gamma);
    let x_data = normed.to_data().to_vec::<f32>().unwrap();
    lm_head_cpu(&x_data, &model.embed_f32, model.embed_vocab, hidden)
}

/// Prefill one full attention layer with all tokens (causal masked attention).
fn prefill_attn_layer_gpu<B: Backend>(
    x: Tensor<B, 2>,           // [seq_len, hidden]
    layer: &GpuFullAttnLayer<B>,
    kv: &mut GpuKvCacheEntry<B>,
    config: &Qor4bConfig,
    rope_cos: &Tensor<B, 2>,
    rope_sin: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let seq_len = x.dims()[0];
    let num_heads = config.num_attn_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.attn_head_dim;
    let q_dim = config.attn_q_dim();
    let num_kv_groups = config.num_kv_groups();
    let rotary_dim = config.rope_dim();
    let half_rot = rotary_dim / 2;

    // 1. RmsNorm
    let x_norm = rms_norm_1plus(x.clone(), &layer.input_norm_gamma);

    // 2. Q+Gate projection
    let q_gate = x_norm.clone().matmul(layer.q_proj.clone()); // [seq, 8192]
    let q_gate_4d = q_gate.reshape([seq_len, num_heads, 2, head_dim]);
    let q_raw = q_gate_4d.clone().slice([0..seq_len, 0..num_heads, 0..1, 0..head_dim])
        .reshape([seq_len * num_heads, head_dim]); // [seq*16, 256]
    let gate = q_gate_4d.slice([0..seq_len, 0..num_heads, 1..2, 0..head_dim])
        .reshape([seq_len, q_dim]); // [seq, 4096]

    // 3. K, V projections
    let k_raw = x_norm.clone().matmul(layer.k_proj.clone())
        .reshape([seq_len * num_kv_heads, head_dim]); // [seq*4, 256]
    let v_new = x_norm.matmul(layer.v_proj.clone()); // [seq, kv_dim]

    // 4. Per-head QK-norm
    let q_normed = rms_norm_1plus(q_raw, &layer.q_norm); // [seq*16, 256]
    let k_normed = rms_norm_1plus(k_raw, &layer.k_norm); // [seq*4, 256]

    // 5. Partial RoPE for each position
    let q_3d = q_normed.reshape([seq_len, num_heads, head_dim]);
    let k_3d = k_normed.reshape([seq_len, num_kv_heads, head_dim]);

    // Apply RoPE position by position across sequence
    let q_rot = q_3d.clone().slice([0..seq_len, 0..num_heads, 0..rotary_dim])
        .reshape([seq_len * num_heads, rotary_dim]);
    let q_pass = q_3d.slice([0..seq_len, 0..num_heads, rotary_dim..head_dim])
        .reshape([seq_len * num_heads, head_dim - rotary_dim]);
    let k_rot = k_3d.clone().slice([0..seq_len, 0..num_kv_heads, 0..rotary_dim])
        .reshape([seq_len * num_kv_heads, rotary_dim]);
    let k_pass = k_3d.slice([0..seq_len, 0..num_kv_heads, rotary_dim..head_dim])
        .reshape([seq_len * num_kv_heads, head_dim - rotary_dim]);

    // RoPE for prefill: each position gets its own cos/sin
    let q_rotated = apply_prefill_rope_gpu(q_rot, rope_cos, rope_sin, seq_len, num_heads, half_rot);
    let k_rotated = apply_prefill_rope_gpu(k_rot, rope_cos, rope_sin, seq_len, num_kv_heads, half_rot);

    let q_final = Tensor::cat(vec![q_rotated, q_pass], 1)
        .reshape([seq_len, num_heads, head_dim]);
    let k_final = Tensor::cat(vec![k_rotated, k_pass], 1)
        .reshape([seq_len, num_kv_heads, head_dim]);

    // 6. Store KV cache
    let k_cache = k_final.swap_dims(0, 1); // [kv_heads, seq, head_dim]
    let v_cache = v_new.reshape([seq_len, num_kv_heads, head_dim]).swap_dims(0, 1);
    kv.k = Some(k_cache.clone());
    kv.v = Some(v_cache.clone());
    kv.seq_len = seq_len;

    // 7. GQA with causal mask
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_4d = q_final.swap_dims(0, 1); // [num_heads, seq, head_dim]

    let k_exp = if num_kv_groups > 1 {
        let k4: Tensor<B, 4> = k_cache.unsqueeze_dim::<4>(1).repeat_dim(1, num_kv_groups);
        k4.reshape([num_heads, seq_len, head_dim])
    } else {
        k_cache
    };
    let v_exp = if num_kv_groups > 1 {
        let v4: Tensor<B, 4> = v_cache.unsqueeze_dim::<4>(1).repeat_dim(1, num_kv_groups);
        v4.reshape([num_heads, seq_len, head_dim])
    } else {
        v_cache
    };

    let scores = q_4d.matmul(k_exp.swap_dims(1, 2)).mul_scalar(scale);
    // Causal mask
    let mask = Tensor::<B, 2>::ones([seq_len, seq_len], &scores.device())
        .triu(1)
        .mul_scalar(-1e9f32);
    let scores = scores + mask.unsqueeze_dim::<3>(0);
    let attn_weights = activation::softmax(scores, 2);
    let context = attn_weights.matmul(v_exp); // [num_heads, seq, head_dim]

    let attn_out = context.swap_dims(0, 1).reshape([seq_len, q_dim]); // [seq, 4096]

    // 8. Output gate
    let attn_gated = attn_out * sigmoid_2d(gate);

    // 9. O projection + residual
    let o_out = attn_gated.matmul(layer.o_proj.clone());
    let out = x + o_out;

    // 10. MLP
    let x_norm2 = rms_norm_1plus(out.clone(), &layer.post_attn_norm_gamma);
    let gate_mlp = x_norm2.clone().matmul(layer.gate_proj.clone());
    let up = x_norm2.matmul(layer.up_proj.clone());
    let mlp_out = silu_2d(gate_mlp).mul(up).matmul(layer.down_proj.clone());

    out + mlp_out
}

/// Apply split-half RoPE for prefill (multiple positions).
/// x: [seq*heads, rotary_dim]
fn apply_prefill_rope_gpu<B: Backend>(
    x: Tensor<B, 2>,
    cos_table: &Tensor<B, 2>,
    sin_table: &Tensor<B, 2>,
    seq_len: usize,
    num_heads: usize,
    half_rot: usize,
) -> Tensor<B, 2> {
    let rotary_dim = half_rot * 2;
    let total = seq_len * num_heads;

    let x_first = x.clone().slice([0..total, 0..half_rot]);
    let x_second = x.slice([0..total, half_rot..rotary_dim]);

    // Build cos/sin for each (position, head) pair: positions 0..seq_len, repeated per head
    // cos_table: [max_pos, half_rot] → slice [0..seq_len, :] → [seq_len, half_rot]
    let cos_seq = cos_table.clone().slice([0..seq_len, 0..half_rot]); // [seq, half_rot]
    let sin_seq = sin_table.clone().slice([0..seq_len, 0..half_rot]);

    // Repeat for each head: [seq, 1, half_rot] → [seq, heads, half_rot] → [seq*heads, half_rot]
    let cos: Tensor<B, 3> = cos_seq.unsqueeze_dim::<3>(1).repeat_dim(1, num_heads);
    let cos = cos.reshape([total, half_rot]);
    let sin: Tensor<B, 3> = sin_seq.unsqueeze_dim::<3>(1).repeat_dim(1, num_heads);
    let sin = sin.reshape([total, half_rot]);

    let new_first = x_first.clone() * cos.clone() - x_second.clone() * sin.clone();
    let new_second = x_second * cos + x_first * sin;

    Tensor::cat(vec![new_first, new_second], 1)
}

// ============================================================
// CPU lm_head
// ============================================================

fn lm_head_cpu(x: &[f32], embed: &[f32], vocab: usize, hidden: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let chunk_size = 4096;
    let mut logits = vec![0.0f32; vocab];

    logits
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * chunk_size;
            for (j, out) in chunk.iter_mut().enumerate() {
                let v = start + j;
                if v >= vocab { break; }
                let row = &embed[v * hidden..(v + 1) * hidden];
                let mut dot = 0.0f32;
                for d in 0..hidden {
                    dot += x[d] * row[d];
                }
                *out = dot;
            }
        });

    logits
}

// ============================================================
// Sampling
// ============================================================

struct Rng { state: u64 }

impl Rng {
    fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self { state: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / 16777216.0
    }
}

fn sample_token_top_k(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32).unwrap_or(0);
    }

    let inv_temp = 1.0 / temperature;

    // Top-k: find k largest logits
    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &l)| (i as u32, l)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    // Softmax over top-k
    let max_logit = indexed[0].1;
    let mut probs: Vec<(u32, f32)> = indexed.iter()
        .map(|&(i, l)| (i, ((l - max_logit) * inv_temp).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() { *p /= sum; }

    // Top-p within top-k
    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p { cutoff = i + 1; break; }
    }
    let candidates = &probs[..cutoff];
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let rand_val = rng.next_f32() * total;
    let mut accum = 0.0f32;
    for &(token_id, prob) in candidates {
        accum += prob;
        if accum >= rand_val { return token_id; }
    }
    candidates[0].0
}

fn apply_presence_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 0.0 || generated.is_empty() { return; }
    let mut seen = vec![false; logits.len()];
    for &tok in generated {
        let idx = tok as usize;
        if idx < seen.len() { seen[idx] = true; }
    }
    for (idx, &was_seen) in seen.iter().enumerate() {
        if was_seen { logits[idx] -= penalty; }
    }
}

fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 1.0 { return; }
    let window = generated.len().min(64);
    let recent = &generated[generated.len() - window..];
    for &tok in recent {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 { logits[idx] /= penalty; }
            else { logits[idx] *= penalty; }
        }
    }
}

// ============================================================
// Main entry point
// ============================================================

pub fn generate_gpu(
    weights: &ModelWeights,
    tokenizer: &QoraTokenizer,
    prompt: &str,
    params: &GenerateParams,
) -> Result<(), String> {
    let result = std::cell::RefCell::new(None);

    std::thread::scope(|s| {
        let builder = std::thread::Builder::new()
            .name("gpu-llm".into())
            .stack_size(128 * 1024 * 1024);
        let handle = builder.spawn_scoped(s, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                generate_gpu_inner(weights, tokenizer, prompt, params)
            }))
        });
        match handle {
            Ok(h) => {
                let join_result = h.join();
                *result.borrow_mut() = Some(match join_result {
                    Ok(Ok(inner)) => inner,
                    Ok(Err(panic_payload)) => {
                        let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                            format!("GPU panic: {s}")
                        } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            format!("GPU panic: {s}")
                        } else {
                            "GPU panic: unknown error (likely out of VRAM)".to_string()
                        };
                        Err(msg)
                    }
                    Err(_) => Err("GPU thread panicked unexpectedly".to_string()),
                });
            }
            Err(e) => {
                *result.borrow_mut() = Some(Err(format!("Failed to spawn GPU thread: {e}")));
            }
        }
    });

    result.into_inner().unwrap_or(Err("No result from GPU thread".into()))
}

fn generate_gpu_inner(
    weights: &ModelWeights,
    tokenizer: &QoraTokenizer,
    prompt: &str,
    params: &GenerateParams,
) -> Result<(), String> {
    type B = cortex::backend::Wgpu;

    let device: <B as Backend>::Device = Default::default();
    let _test: Tensor<B, 1> = Tensor::zeros([1], &device);
    eprintln!("GPU initialized successfully");

    // VRAM probe
    {
        let probe_size = 256 * 1024 * 1024 / 4;
        let probe_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _probe: Tensor<B, 1> = Tensor::zeros([probe_size], &device);
            let _ = _probe.slice([0..1]).to_data();
        }));
        if probe_result.is_err() {
            return Err("GPU has insufficient VRAM (failed 256MB probe)".to_string());
        }
        eprintln!("VRAM probe: 256MB OK");
    }

    let model_mb = weights.memory_bytes() / (1024 * 1024);
    eprintln!("Estimated VRAM needed: ~{}MB", model_mb + 800);

    // Load to GPU
    let t0 = Instant::now();
    let model = load_model_gpu::<B>(weights, &device);
    eprintln!("Weights loaded to GPU in {:.1?}", t0.elapsed());

    // Tokenize
    let tokens = tokenizer.format_chat(prompt, params.think, params.max_new_tokens);
    eprintln!("Prompt tokens: {}", tokens.len());

    let config = &model.config;
    let mut cache = GpuHybridCache::<B>::new(config, &device);
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut in_think_block = params.think;  // prompt already contains <think>\n
    let mut think_content = String::new();
    let mut think_tokens = 0u32;
    let mut rng = Rng::new();

    // Prefill
    let t0 = Instant::now();
    let mut logits = forward_prefill_gpu(&model, &tokens, &mut cache);
    let prefill_time = t0.elapsed();
    let seq_len = tokens.len();
    eprintln!("Prefill: {seq_len} tokens in {prefill_time:.1?} ({:.1} tok/s)",
        seq_len as f64 / prefill_time.as_secs_f64());

    // Sample first token
    if !in_think_block && params.temperature > 0.0 {
        apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
    }
    let mut next_token_id = sample_token_top_k(&logits, params.temperature, params.top_p, params.top_k, &mut rng);
    eprintln!("First token: {} (id={})", tokenizer.decode(&[next_token_id]), next_token_id);

    let mut decode_tokens = 0u32;
    let decode_start = Instant::now();

    // Decode loop
    for step in 0..params.max_new_tokens {
        if next_token_id == params.eos_token_id
            || next_token_id == crate::tokenizer::IM_END
            || next_token_id == crate::tokenizer::ENDOFTEXT
        {
            eprintln!("\n[EOS after {step} tokens (id={next_token_id})]");
            break;
        }

        generated_tokens.push(next_token_id);

        // Handle think blocks
        if next_token_id == crate::tokenizer::THINK_START {
            in_think_block = true;
            if params.show_think {
                eprint!("<think>");
                std::io::stderr().flush().ok();
            }
        } else if next_token_id == crate::tokenizer::THINK_END {
            in_think_block = false;
            eprintln!("\n[thinking done: {} tokens, {} chars]", think_tokens, think_content.len());
            if params.show_think { eprintln!("</think>"); }
            think_content.clear();
            think_tokens = 0;
        } else {
            let token_text = tokenizer.decode(&[next_token_id]);
            if in_think_block {
                think_content.push_str(&token_text);
                think_tokens += 1;
                if params.show_think {
                    eprint!("{token_text}");
                    std::io::stderr().flush().ok();
                } else if think_tokens % 50 == 0 {
                    eprint!("[thinking: {think_tokens} tokens...] ");
                    std::io::stderr().flush().ok();
                }
            } else {
                print!("{token_text}");
                std::io::stdout().flush().ok();
            }
        }

        // Sentence-boundary stop near token budget (85%) to avoid mid-sentence cutoff
        if !in_think_block && step >= params.max_new_tokens * 85 / 100 {
            let piece = tokenizer.decode(&[next_token_id]);
            let trimmed = piece.trim_end();
            if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
                eprintln!("\n[clean stop near token limit at step {step}]");
                break;
            }
        }

        // Loop detection
        if crate::generate::is_stuck_in_loop(&generated_tokens) {
            eprintln!("\n[loop detected at step {}, forcing EOS]", decode_tokens);
            break;
        }

        // GPU decode
        let mut logits = forward_decode_gpu(&model, next_token_id as usize, &mut cache);

        // Penalties
        if in_think_block {
            apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
            logits[params.eos_token_id as usize] = f32::NEG_INFINITY;
        } else if params.temperature > 0.0 {
            apply_repetition_penalty(&mut logits, &generated_tokens, params.repetition_penalty);
            apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
        }

        // Think budget
        if in_think_block && think_tokens >= params.think_budget as u32 {
            next_token_id = crate::tokenizer::THINK_END;
            eprintln!("\n[think budget reached ({} tokens), forcing </think>]", params.think_budget);
        } else {
            next_token_id = sample_token_top_k(&logits, params.temperature, params.top_p, params.top_k, &mut rng);
        }
        decode_tokens += 1;

        if decode_tokens % 50 == 0 {
            let elapsed = decode_start.elapsed();
            let tps = decode_tokens as f64 / elapsed.as_secs_f64();
            eprint!("[{decode_tokens} tokens, {tps:.1} tok/s] ");
            std::io::stderr().flush().ok();
        }
    }

    if in_think_block {
        eprintln!("\n[WARNING: thinking did not finish within {} tokens]", params.max_new_tokens);
    }

    println!();

    let decode_elapsed = decode_start.elapsed();
    if decode_tokens > 0 {
        eprintln!("Decode: {} tokens in {decode_elapsed:.1?} ({:.2} tok/s)",
            decode_tokens,
            decode_tokens as f64 / decode_elapsed.as_secs_f64());
    }

    Ok(())
}
