//! GPU weight loading for QOR4B.
//!
//! Converts CPU ModelWeights (Q4 or F16) into Cortex GPU tensors.
//! Q4 weights are uploaded as packed quantized tensors with on-the-fly
//! GPU dequantization during matmul.

use cortex::prelude::*;
use cortex::tensor::{DType, TensorData};
use cortex::tensor::quantization::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use half::f16;

use crate::config::Qor4bConfig;
use crate::gemv::{ModelWeights, HybridLayerWeights, DeltaNetLayerWeights, FullAttnLayerWeights, Weight};

// ============================================================
// GPU model structures
// ============================================================

pub struct GpuModel<B: Backend> {
    pub layers: Vec<GpuHybridLayer<B>>,
    pub embed_f32: Vec<f32>,
    pub embed_vocab: usize,
    pub embed_hidden: usize,
    pub final_norm_gamma: Tensor<B, 1>,
    pub rope_cos: Tensor<B, 2>,
    pub rope_sin: Tensor<B, 2>,
    pub config: Qor4bConfig,
}

pub enum GpuHybridLayer<B: Backend> {
    DeltaNet(GpuDeltaNetLayer<B>),
    FullAttn(GpuFullAttnLayer<B>),
}

pub struct GpuDeltaNetLayer<B: Backend> {
    pub in_proj_qkv: Tensor<B, 2>,
    pub in_proj_a: Tensor<B, 2>,
    pub in_proj_b: Tensor<B, 2>,
    pub in_proj_z: Tensor<B, 2>,
    pub out_proj: Tensor<B, 2>,
    pub conv1d_weight: Tensor<B, 2>,  // [qkv_dim, kernel_size]
    pub a_log: Tensor<B, 1>,          // [num_v_heads]
    pub dt_bias: Tensor<B, 1>,        // [num_v_heads]
    pub attn_norm_weight: Tensor<B, 1>, // [head_dim] — plain gamma (NOT 1+gamma)
    pub gate_proj: Tensor<B, 2>,
    pub up_proj: Tensor<B, 2>,
    pub down_proj: Tensor<B, 2>,
    pub input_norm_gamma: Tensor<B, 1>,
    pub post_attn_norm_gamma: Tensor<B, 1>,
}

pub struct GpuFullAttnLayer<B: Backend> {
    pub q_proj: Tensor<B, 2>,
    pub k_proj: Tensor<B, 2>,
    pub v_proj: Tensor<B, 2>,
    pub o_proj: Tensor<B, 2>,
    pub q_norm: Tensor<B, 1>,
    pub k_norm: Tensor<B, 1>,
    pub gate_proj: Tensor<B, 2>,
    pub up_proj: Tensor<B, 2>,
    pub down_proj: Tensor<B, 2>,
    pub input_norm_gamma: Tensor<B, 1>,
    pub post_attn_norm_gamma: Tensor<B, 1>,
}

// ============================================================
// Q4 format conversion (ported from QOR3B)
// ============================================================

fn q4_scheme() -> QuantScheme {
    QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([32])),
        mode: QuantMode::Symmetric,
    }
}

fn convert_q4_packed(packed: &[u8]) -> Vec<u8> {
    packed.iter().map(|&b| b ^ 0x88).collect()
}

fn q4_weight_to_gpu<B: Backend>(
    packed: &[u8],
    scales: &[f16],
    k: usize,
    n: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let burn_packed = convert_q4_packed(packed);
    let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();
    let scale_bytes: Vec<u8> = scales_f32.iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();

    let mut combined = Vec::with_capacity(burn_packed.len() + scale_bytes.len());
    combined.extend_from_slice(&burn_packed);
    combined.extend_from_slice(&scale_bytes);

    let scheme = q4_scheme();
    let data = TensorData::from_bytes_vec(combined, vec![k, n], DType::QFloat(scheme));
    Tensor::from_data(data, device)
}

fn f16_weight_to_gpu<B: Backend>(
    data: &[f16],
    k: usize,
    n: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    Tensor::from_data(TensorData::new(f32_data, [k, n]), device)
}

fn weight_to_gpu<B: Backend>(w: &Weight, device: &B::Device) -> Tensor<B, 2> {
    match w {
        Weight::F16(fw) => f16_weight_to_gpu::<B>(&fw.data, fw.k, fw.n, device),
        Weight::Q4(qw) => q4_weight_to_gpu::<B>(&qw.packed, &qw.scales, qw.k, qw.n, device),
    }
}

fn norm_to_gpu<B: Backend>(gamma: &[f16], device: &B::Device) -> Tensor<B, 1> {
    let f32_data: Vec<f32> = gamma.iter().map(|v| v.to_f32()).collect();
    Tensor::from_data(TensorData::new(f32_data, [gamma.len()]), device)
}

fn f32_vec_to_gpu_1d<B: Backend>(data: &[f32], device: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
}

fn f32_vec_to_gpu_2d<B: Backend>(data: &[f32], rows: usize, cols: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(data.to_vec(), [rows, cols]), device)
}

// ============================================================
// Model loading
// ============================================================

pub fn load_model_gpu<B: Backend>(
    weights: &ModelWeights,
    device: &B::Device,
) -> GpuModel<B> {
    let config = &weights.config;

    eprintln!("Loading {} layers to GPU ({})...", config.num_layers, weights.format_name);

    let mut layers = Vec::with_capacity(config.num_layers);
    for (i, lw) in weights.layers.iter().enumerate() {
        if i % 6 == 0 {
            eprintln!("  GPU layer {i}/{}...", config.num_layers);
        }
        match lw {
            HybridLayerWeights::DeltaNet(d) => {
                layers.push(GpuHybridLayer::DeltaNet(load_deltanet_layer::<B>(d, config, device)));
            }
            HybridLayerWeights::FullAttn(f) => {
                layers.push(GpuHybridLayer::FullAttn(load_fullattn_layer::<B>(f, device)));
            }
        }
    }
    eprintln!("  GPU layer {}/{}... done", config.num_layers, config.num_layers);

    // Embedding: keep on CPU (vocab=248K is too large for VRAM)
    let embed_f32 = match &weights.embed {
        Weight::F16(fw) => fw.data.iter().map(|v| v.to_f32()).collect(),
        Weight::Q4(qw) => dequant_q4_to_f32(&qw.packed, &qw.scales, qw.k, qw.n),
    };

    let final_norm_gamma = norm_to_gpu::<B>(&weights.final_norm, device);

    // RoPE tables for full attention (partial RoPE, rope_dim/2 = 32 entries per position)
    let rope_half = config.rope_dim() / 2; // 32
    let max_pos = weights.rope_cos.len() / rope_half;
    let rope_cos = Tensor::from_data(
        TensorData::new(weights.rope_cos.clone(), [max_pos, rope_half]),
        device,
    );
    let rope_sin = Tensor::from_data(
        TensorData::new(weights.rope_sin.clone(), [max_pos, rope_half]),
        device,
    );

    GpuModel {
        layers,
        embed_f32,
        embed_vocab: config.vocab_size,
        embed_hidden: config.hidden_size,
        final_norm_gamma,
        rope_cos,
        rope_sin,
        config: config.clone(),
    }
}

fn load_deltanet_layer<B: Backend>(
    d: &DeltaNetLayerWeights,
    config: &Qor4bConfig,
    device: &B::Device,
) -> GpuDeltaNetLayer<B> {
    let qkv_dim = config.deltanet_qkv_dim();
    let kernel = config.conv_kernel_size;
    GpuDeltaNetLayer {
        in_proj_qkv: weight_to_gpu::<B>(&d.in_proj_qkv, device),
        in_proj_a: weight_to_gpu::<B>(&d.in_proj_a, device),
        in_proj_b: weight_to_gpu::<B>(&d.in_proj_b, device),
        in_proj_z: weight_to_gpu::<B>(&d.in_proj_z, device),
        out_proj: weight_to_gpu::<B>(&d.out_proj, device),
        conv1d_weight: f32_vec_to_gpu_2d::<B>(&d.conv1d_weight, qkv_dim, kernel, device),
        a_log: f32_vec_to_gpu_1d::<B>(&d.a_log, device),
        dt_bias: f32_vec_to_gpu_1d::<B>(&d.dt_bias, device),
        attn_norm_weight: norm_to_gpu::<B>(&d.attn_norm_weight, device),
        gate_proj: weight_to_gpu::<B>(&d.gate_proj, device),
        up_proj: weight_to_gpu::<B>(&d.up_proj, device),
        down_proj: weight_to_gpu::<B>(&d.down_proj, device),
        input_norm_gamma: norm_to_gpu::<B>(&d.input_norm, device),
        post_attn_norm_gamma: norm_to_gpu::<B>(&d.post_attn_norm, device),
    }
}

fn load_fullattn_layer<B: Backend>(
    f: &FullAttnLayerWeights,
    device: &B::Device,
) -> GpuFullAttnLayer<B> {
    GpuFullAttnLayer {
        q_proj: weight_to_gpu::<B>(&f.q_proj, device),
        k_proj: weight_to_gpu::<B>(&f.k_proj, device),
        v_proj: weight_to_gpu::<B>(&f.v_proj, device),
        o_proj: weight_to_gpu::<B>(&f.o_proj, device),
        q_norm: norm_to_gpu::<B>(&f.q_norm, device),
        k_norm: norm_to_gpu::<B>(&f.k_norm, device),
        gate_proj: weight_to_gpu::<B>(&f.gate_proj, device),
        up_proj: weight_to_gpu::<B>(&f.up_proj, device),
        down_proj: weight_to_gpu::<B>(&f.down_proj, device),
        input_norm_gamma: norm_to_gpu::<B>(&f.input_norm, device),
        post_attn_norm_gamma: norm_to_gpu::<B>(&f.post_attn_norm, device),
    }
}

fn dequant_q4_to_f32(packed: &[u8], scales: &[f16], k: usize, n: usize) -> Vec<f32> {
    let groups_per_row = n / 32;
    let mut out = vec![0.0f32; k * n];

    for ki in 0..k {
        for g in 0..groups_per_row {
            let group_idx = ki * groups_per_row + g;
            let scale = scales[group_idx].to_f32();
            let col_start = g * 32;
            let pack_start = group_idx * 16;

            for j in (0..32).step_by(2) {
                let byte = packed[pack_start + j / 2];
                let q0 = (byte & 0x0F) as f32 - 8.0;
                let q1 = (byte >> 4) as f32 - 8.0;
                out[ki * n + col_start + j] = q0 * scale;
                out[ki * n + col_start + j + 1] = q1 * scale;
            }
        }
    }
    out
}
