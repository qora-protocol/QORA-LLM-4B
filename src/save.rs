//! Save/load QORA-4B ModelWeights to compact binary format (.qor4b).
//!
//! File format:
//!   Header: magic "QOR4" + version(u32) + format(u8: 0=F16, 1=Q4)
//!   Config: all architecture params
//!   Per-layer: type byte + layer-specific weights
//!   Global: embedding + final norm + RoPE tables
//!   Vision: (optional) patch_embed, pos_embed, 24 blocks, merger

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use half::f16;

use crate::config::{Qor4bConfig, LayerType};
use crate::gemv::*;

const MAGIC: &[u8; 4] = b"QOR4";
const VERSION: u32 = 2; // v2: vision weights stored as f16 (was f32 in v1)

// ============================================================
// I/O helpers
// ============================================================

fn write_u8(w: &mut impl Write, val: u8) -> io::Result<()> { w.write_all(&[val]) }
fn write_u32(w: &mut impl Write, val: u32) -> io::Result<()> { w.write_all(&val.to_le_bytes()) }
fn write_u64(w: &mut impl Write, val: u64) -> io::Result<()> { w.write_all(&val.to_le_bytes()) }
fn write_f32(w: &mut impl Write, val: f32) -> io::Result<()> { w.write_all(&val.to_le_bytes()) }
fn write_f64(w: &mut impl Write, val: f64) -> io::Result<()> { w.write_all(&val.to_le_bytes()) }

fn write_f16_vec(w: &mut impl Write, data: &[f16]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn write_f32_vec(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn write_bytes(w: &mut impl Write, data: &[u8]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    w.write_all(data)
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0])
}
fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b))
}
fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(f64::from_le_bytes(b))
}

fn read_f16_vec(r: &mut impl Read) -> io::Result<Vec<f16>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 2];
    r.read_exact(&mut bytes)?;
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f16, len).to_vec() })
}

fn read_f32_vec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 4];
    r.read_exact(&mut bytes)?;
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() })
}

/// Write f32 data as f16 on disk (half the size, minimal quality loss).
fn write_f32_as_f16(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    write_f16_vec(w, &f16_data)
}

/// Read f16 data from disk and convert to f32 for computation.
fn read_f16_as_f32(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let f16_data = read_f16_vec(r)?;
    Ok(f16_data.iter().map(|v| v.to_f32()).collect())
}

fn read_bytes_vec(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ============================================================
// Weight I/O
// ============================================================

fn write_weight(w: &mut impl Write, weight: &Weight) -> io::Result<()> {
    write_u64(w, weight.k() as u64)?;
    write_u64(w, weight.n() as u64)?;
    let fmt = match weight { Weight::F16(_) => 0u8, Weight::Q4(_) => 1u8 };
    write_u8(w, fmt)?;
    match weight {
        Weight::F16(fw) => write_f16_vec(w, &fw.data)?,
        Weight::Q4(qw) => { write_bytes(w, &qw.packed)?; write_f16_vec(w, &qw.scales)?; }
    }
    Ok(())
}

fn read_weight(r: &mut impl Read, _format_id: u8) -> io::Result<Weight> {
    let k = read_u64(r)? as usize;
    let n = read_u64(r)? as usize;
    let per_weight_fmt = read_u8(r)?;
    match per_weight_fmt {
        0 => {
            let data = read_f16_vec(r)?;
            Ok(Weight::F16(F16Weight { data, k, n }))
        }
        1 => {
            let packed = read_bytes_vec(r)?;
            let scales = read_f16_vec(r)?;
            Ok(Weight::Q4(Q4Weight { packed, scales, k, n }))
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unknown weight format {per_weight_fmt}")))
    }
}

// ============================================================
// Save
// ============================================================

pub fn save_model(weights: &ModelWeights, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
    let config = &weights.config;
    let format_id = weights.format_id();

    // Header
    w.write_all(MAGIC)?;
    write_u32(&mut w, VERSION)?;
    write_u8(&mut w, format_id)?;

    // Config
    write_u32(&mut w, config.vocab_size as u32)?;
    write_u32(&mut w, config.hidden_size as u32)?;
    write_u32(&mut w, config.num_layers as u32)?;
    write_u32(&mut w, config.num_attn_heads as u32)?;
    write_u32(&mut w, config.num_kv_heads as u32)?;
    write_u32(&mut w, config.attn_head_dim as u32)?;
    write_u32(&mut w, config.num_qk_heads as u32)?;
    write_u32(&mut w, config.num_v_heads as u32)?;
    write_u32(&mut w, config.deltanet_head_dim as u32)?;
    write_u32(&mut w, config.conv_kernel_size as u32)?;
    write_u32(&mut w, config.intermediate_size as u32)?;
    write_f64(&mut w, config.rope_theta)?;
    write_f32(&mut w, config.partial_rotary_factor)?;
    write_f64(&mut w, config.rms_norm_eps)?;
    write_u32(&mut w, config.eos_token_id)?;
    write_u8(&mut w, config.tie_word_embeddings as u8)?;
    write_u8(&mut w, config.has_vision as u8)?;

    // Layer types
    for lt in &config.layer_types {
        write_u8(&mut w, *lt as u8)?;
    }

    // Per-layer weights
    for (i, layer) in weights.layers.iter().enumerate() {
        match layer {
            HybridLayerWeights::DeltaNet(lw) => {
                write_weight(&mut w, &lw.in_proj_qkv)?;
                write_weight(&mut w, &lw.in_proj_a)?;
                write_weight(&mut w, &lw.in_proj_b)?;
                write_weight(&mut w, &lw.in_proj_z)?;
                write_weight(&mut w, &lw.out_proj)?;
                write_f32_vec(&mut w, &lw.conv1d_weight)?;
                write_f32_vec(&mut w, &lw.a_log)?;
                write_f32_vec(&mut w, &lw.dt_bias)?;
                write_f16_vec(&mut w, &lw.attn_norm_weight)?;
                write_weight(&mut w, &lw.gate_proj)?;
                write_weight(&mut w, &lw.up_proj)?;
                write_weight(&mut w, &lw.down_proj)?;
                write_f16_vec(&mut w, &lw.input_norm)?;
                write_f16_vec(&mut w, &lw.post_attn_norm)?;
            }
            HybridLayerWeights::FullAttn(lw) => {
                write_weight(&mut w, &lw.q_proj)?;
                write_weight(&mut w, &lw.k_proj)?;
                write_weight(&mut w, &lw.v_proj)?;
                write_weight(&mut w, &lw.o_proj)?;
                write_f16_vec(&mut w, &lw.q_norm)?;
                write_f16_vec(&mut w, &lw.k_norm)?;
                write_weight(&mut w, &lw.gate_proj)?;
                write_weight(&mut w, &lw.up_proj)?;
                write_weight(&mut w, &lw.down_proj)?;
                write_f16_vec(&mut w, &lw.input_norm)?;
                write_f16_vec(&mut w, &lw.post_attn_norm)?;
            }
        }
        if i % 6 == 0 { eprintln!("  Saved layer {i}/{}...", config.num_layers); }
    }

    // Global
    write_weight(&mut w, &weights.embed)?;
    write_f16_vec(&mut w, &weights.final_norm)?;
    write_f32_vec(&mut w, &weights.rope_cos)?;
    write_f32_vec(&mut w, &weights.rope_sin)?;

    // Vision weights (stored as f16 on disk, loaded as f32 for compute)
    if let Some(vision) = &weights.vision {
        eprintln!("  Saving vision encoder (f16)...");
        write_f32_as_f16(&mut w, &vision.patch_proj_weight)?;
        write_f32_as_f16(&mut w, &vision.patch_proj_bias)?;
        write_f32_as_f16(&mut w, &vision.pos_embed)?;
        for b in &vision.blocks {
            write_f32_as_f16(&mut w, &b.norm1_weight)?;
            write_f32_as_f16(&mut w, &b.norm1_bias)?;
            write_f32_as_f16(&mut w, &b.qkv_weight)?;
            write_f32_as_f16(&mut w, &b.qkv_bias)?;
            write_f32_as_f16(&mut w, &b.proj_weight)?;
            write_f32_as_f16(&mut w, &b.proj_bias)?;
            write_f32_as_f16(&mut w, &b.norm2_weight)?;
            write_f32_as_f16(&mut w, &b.norm2_bias)?;
            write_f32_as_f16(&mut w, &b.fc1_weight)?;
            write_f32_as_f16(&mut w, &b.fc1_bias)?;
            write_f32_as_f16(&mut w, &b.fc2_weight)?;
            write_f32_as_f16(&mut w, &b.fc2_bias)?;
        }
        write_f32_as_f16(&mut w, &vision.merger.norm_weight)?;
        write_f32_as_f16(&mut w, &vision.merger.norm_bias)?;
        write_f32_as_f16(&mut w, &vision.merger.fc1_weight)?;
        write_f32_as_f16(&mut w, &vision.merger.fc1_bias)?;
        write_f32_as_f16(&mut w, &vision.merger.fc2_weight)?;
        write_f32_as_f16(&mut w, &vision.merger.fc2_bias)?;
    }

    w.flush()?;
    Ok(())
}

// ============================================================
// Load
// ============================================================

pub fn load_model(path: &Path) -> io::Result<ModelWeights> {
    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);

    // Header
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic (expected QOR4)"));
    }
    let version = read_u32(&mut r)?;
    if version != 1 && version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unsupported version {version} (expected 1 or {VERSION})")));
    }
    let format_id = read_u8(&mut r)?;

    // Config
    let vocab_size = read_u32(&mut r)? as usize;
    let hidden_size = read_u32(&mut r)? as usize;
    let num_layers = read_u32(&mut r)? as usize;
    let num_attn_heads = read_u32(&mut r)? as usize;
    let num_kv_heads = read_u32(&mut r)? as usize;
    let attn_head_dim = read_u32(&mut r)? as usize;
    let num_qk_heads = read_u32(&mut r)? as usize;
    let num_v_heads = read_u32(&mut r)? as usize;
    let deltanet_head_dim = read_u32(&mut r)? as usize;
    let conv_kernel_size = read_u32(&mut r)? as usize;
    let intermediate_size = read_u32(&mut r)? as usize;
    let rope_theta = read_f64(&mut r)?;
    let partial_rotary_factor = read_f32(&mut r)?;
    let rms_norm_eps = read_f64(&mut r)?;
    let eos_token_id = read_u32(&mut r)?;
    let tie_word_embeddings = read_u8(&mut r)? != 0;
    let has_vision = read_u8(&mut r)? != 0;

    // Layer types
    let mut layer_types = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let lt = read_u8(&mut r)?;
        layer_types.push(if lt == 0 { LayerType::DeltaNet } else { LayerType::FullAttn });
    }

    let config = Qor4bConfig {
        vocab_size, hidden_size, num_layers, num_attn_heads, num_kv_heads,
        attn_head_dim, num_qk_heads, num_v_heads, deltanet_head_dim,
        conv_kernel_size, intermediate_size, rope_theta, partial_rotary_factor,
        rms_norm_eps, eos_token_id, tie_word_embeddings,
        layer_types: layer_types.clone(),
        has_vision,
        // Vision defaults (read from binary later if has_vision)
        vision_hidden: 1024, vision_layers: 24, vision_heads: 16,
        vision_ffn: 4096, patch_size: 16, spatial_merge_size: 2,
        num_position_embeddings: 2304,
    };

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for (i, lt) in layer_types.iter().enumerate() {
        if i % 6 == 0 { eprintln!("  Loading layer {i}/{num_layers}..."); }
        match lt {
            LayerType::DeltaNet => {
                layers.push(HybridLayerWeights::DeltaNet(DeltaNetLayerWeights {
                    in_proj_qkv: read_weight(&mut r, format_id)?,
                    in_proj_a: read_weight(&mut r, format_id)?,
                    in_proj_b: read_weight(&mut r, format_id)?,
                    in_proj_z: read_weight(&mut r, format_id)?,
                    out_proj: read_weight(&mut r, format_id)?,
                    conv1d_weight: read_f32_vec(&mut r)?,
                    a_log: read_f32_vec(&mut r)?,
                    dt_bias: read_f32_vec(&mut r)?,
                    attn_norm_weight: read_f16_vec(&mut r)?,
                    gate_proj: read_weight(&mut r, format_id)?,
                    up_proj: read_weight(&mut r, format_id)?,
                    down_proj: read_weight(&mut r, format_id)?,
                    input_norm: read_f16_vec(&mut r)?,
                    post_attn_norm: read_f16_vec(&mut r)?,
                }));
            }
            LayerType::FullAttn => {
                layers.push(HybridLayerWeights::FullAttn(FullAttnLayerWeights {
                    q_proj: read_weight(&mut r, format_id)?,
                    k_proj: read_weight(&mut r, format_id)?,
                    v_proj: read_weight(&mut r, format_id)?,
                    o_proj: read_weight(&mut r, format_id)?,
                    q_norm: read_f16_vec(&mut r)?,
                    k_norm: read_f16_vec(&mut r)?,
                    gate_proj: read_weight(&mut r, format_id)?,
                    up_proj: read_weight(&mut r, format_id)?,
                    down_proj: read_weight(&mut r, format_id)?,
                    input_norm: read_f16_vec(&mut r)?,
                    post_attn_norm: read_f16_vec(&mut r)?,
                }));
            }
        }
    }

    // Global
    let embed = read_weight(&mut r, format_id)?;
    let final_norm = read_f16_vec(&mut r)?;
    let rope_cos = read_f32_vec(&mut r)?;
    let rope_sin = read_f32_vec(&mut r)?;

    // Vision weights: v1=f32 on disk, v2=f16 on disk → both loaded as f32 for compute
    let vision = if has_vision {
        eprintln!("  Loading vision encoder...");
        use crate::vision::*;
        // Helper closure: read vision vec based on file version
        let read_vis = |r: &mut BufReader<File>| -> io::Result<Vec<f32>> {
            if version >= 2 { read_f16_as_f32(r) } else { read_f32_vec(r) }
        };
        let patch_proj_weight = read_vis(&mut r)?;
        let patch_proj_bias = read_vis(&mut r)?;
        let pos_embed = read_vis(&mut r)?;
        let mut blocks = Vec::with_capacity(config.vision_layers);
        for _ in 0..config.vision_layers {
            blocks.push(VisionBlock {
                norm1_weight: read_vis(&mut r)?,
                norm1_bias: read_vis(&mut r)?,
                qkv_weight: read_vis(&mut r)?,
                qkv_bias: read_vis(&mut r)?,
                proj_weight: read_vis(&mut r)?,
                proj_bias: read_vis(&mut r)?,
                norm2_weight: read_vis(&mut r)?,
                norm2_bias: read_vis(&mut r)?,
                fc1_weight: read_vis(&mut r)?,
                fc1_bias: read_vis(&mut r)?,
                fc2_weight: read_vis(&mut r)?,
                fc2_bias: read_vis(&mut r)?,
            });
        }
        let merger = Merger {
            norm_weight: read_vis(&mut r)?,
            norm_bias: read_vis(&mut r)?,
            fc1_weight: read_vis(&mut r)?,
            fc1_bias: read_vis(&mut r)?,
            fc2_weight: read_vis(&mut r)?,
            fc2_bias: read_vis(&mut r)?,
        };
        Some(VisionEncoder { patch_proj_weight, patch_proj_bias, pos_embed, blocks, merger })
    } else {
        None
    };

    let format_name = match format_id { 0 => "F16", 1 => "Q4", _ => "unknown" };
    eprintln!("  Loaded {format_name} model: {num_layers} layers, vocab={vocab_size}, hidden={hidden_size}");

    Ok(ModelWeights {
        layers, embed, vocab: vocab_size, hidden: hidden_size,
        final_norm, rope_cos, rope_sin, config, format_name, vision,
    })
}
