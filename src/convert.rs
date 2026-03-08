//! Convert QORA-4B safetensors → .qor4b binary format (one-time conversion tool).
//!
//! Usage: convert --input <safetensors_dir> --output model.qor4b [--format q4|f16]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::io::{BufWriter, Write};
use std::fs::File;

use half::f16;
use memmap2::Mmap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input_dir = PathBuf::from(".");
    let mut output_path = PathBuf::from("model.qor4b");
    let mut format = "q4".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { if i+1 < args.len() { input_dir = PathBuf::from(&args[i+1]); i += 1; } }
            "--output" => { if i+1 < args.len() { output_path = PathBuf::from(&args[i+1]); i += 1; } }
            "--format" => { if i+1 < args.len() { format = args[i+1].clone(); i += 1; } }
            _ => {}
        }
        i += 1;
    }

    let format_id: u8 = if format == "f16" { 0 } else { 1 };
    eprintln!("QOR4B Converter");
    eprintln!("  Input:  {}", input_dir.display());
    eprintln!("  Output: {}", output_path.display());
    eprintln!("  Format: {} (id={})", format, format_id);

    // Load safetensors index
    let index_path = input_dir.join("model.safetensors.index.json");
    let weight_map = if index_path.exists() {
        load_index(&index_path)
    } else {
        // Single safetensors file
        let single = input_dir.join("model.safetensors");
        if !single.exists() {
            eprintln!("ERROR: No safetensors found in {}", input_dir.display());
            std::process::exit(1);
        }
        let mut map = HashMap::new();
        map.insert("__single__".to_string(), single.to_string_lossy().to_string());
        map
    };

    // Collect unique safetensors files
    let mut shard_files: Vec<String> = weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();

    // Load all shards
    eprintln!("Loading {} safetensors shard(s)...", shard_files.len());
    let mut all_tensors: HashMap<String, TensorData> = HashMap::new();
    for shard_name in &shard_files {
        let shard_path = input_dir.join(shard_name);
        eprintln!("  Loading {}...", shard_name);
        let tensors = load_safetensors(&shard_path);
        all_tensors.extend(tensors);
    }
    eprintln!("Loaded {} tensors total", all_tensors.len());

    // Build config
    let config = qor4b::config::Qor4bConfig::default_4b();
    let _hidden = config.hidden_size;
    let num_layers = config.num_layers;

    // Open output
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&output_path).unwrap());

    // Header
    w.write_all(b"QOR4").unwrap();
    w.write_all(&2u32.to_le_bytes()).unwrap(); // version 2: f16 vision
    w.write_all(&[format_id]).unwrap();

    // Config
    write_u32(&mut w, config.vocab_size as u32);
    write_u32(&mut w, config.hidden_size as u32);
    write_u32(&mut w, config.num_layers as u32);
    write_u32(&mut w, config.num_attn_heads as u32);
    write_u32(&mut w, config.num_kv_heads as u32);
    write_u32(&mut w, config.attn_head_dim as u32);
    write_u32(&mut w, config.num_qk_heads as u32);
    write_u32(&mut w, config.num_v_heads as u32);
    write_u32(&mut w, config.deltanet_head_dim as u32);
    write_u32(&mut w, config.conv_kernel_size as u32);
    write_u32(&mut w, config.intermediate_size as u32);
    write_f64(&mut w, config.rope_theta);
    write_f32(&mut w, config.partial_rotary_factor);
    write_f64(&mut w, config.rms_norm_eps);
    write_u32(&mut w, config.eos_token_id);
    w.write_all(&[config.tie_word_embeddings as u8]).unwrap();
    w.write_all(&[config.has_vision as u8]).unwrap();

    // Layer types
    for lt in &config.layer_types {
        w.write_all(&[*lt as u8]).unwrap();
    }

    // Per-layer weights
    for i in 0..num_layers {
        let prefix = format!("model.language_model.layers.{i}");
        let lt = &config.layer_types[i];

        match lt {
            qor4b::config::LayerType::DeltaNet => {
                eprintln!("  Layer {i}/{num_layers} (DeltaNet)...");
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.linear_attn.in_proj_qkv.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.linear_attn.in_proj_a.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.linear_attn.in_proj_b.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.linear_attn.in_proj_z.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.linear_attn.out_proj.weight"), format_id);
                // conv1d, A_log, dt_bias as f32
                write_tensor_f32(&mut w, &all_tensors, &format!("{prefix}.linear_attn.conv1d.weight"));
                write_tensor_f32(&mut w, &all_tensors, &format!("{prefix}.linear_attn.A_log"));
                write_tensor_f32(&mut w, &all_tensors, &format!("{prefix}.linear_attn.dt_bias"));
                // norm as f16
                write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.linear_attn.norm.weight"));
                // MLP
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.mlp.gate_proj.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.mlp.up_proj.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.mlp.down_proj.weight"), format_id);
                // Norms
                write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.input_layernorm.weight"));
                write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.post_attention_layernorm.weight"));
            }
            qor4b::config::LayerType::FullAttn => {
                eprintln!("  Layer {i}/{num_layers} (FullAttn)...");
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.self_attn.q_proj.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.self_attn.k_proj.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.self_attn.v_proj.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.self_attn.o_proj.weight"), format_id);
                write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.self_attn.q_norm.weight"));
                write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.self_attn.k_norm.weight"));
                // MLP
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.mlp.gate_proj.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.mlp.up_proj.weight"), format_id);
                write_tensor_weight(&mut w, &all_tensors, &format!("{prefix}.mlp.down_proj.weight"), format_id);
                // Norms
                write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.input_layernorm.weight"));
                write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.post_attention_layernorm.weight"));
            }
        }
    }

    // Embedding (tied to lm_head)
    // Embedding needs k=vocab, n=hidden (transposed from standard GEMV layout)
    // because embed_lookup indexes by token_id to get one hidden-dim row.
    // Also written as-is (not transposed) for lm_head GEMV where k=hidden, n=vocab.
    eprintln!("  Embedding...");
    write_embedding_weight(&mut w, &all_tensors, "model.language_model.embed_tokens.weight", format_id);

    // Final norm
    write_tensor_f16(&mut w, &all_tensors, "model.language_model.norm.weight");

    // RoPE tables (precompute partial RoPE)
    let rope_dim = config.rope_dim(); // 64 (256 * 0.25)
    let half_dim = rope_dim / 2;      // 32
    let max_pos = 8192; // reasonable default
    let mut rope_cos = vec![0.0f32; max_pos * half_dim];
    let mut rope_sin = vec![0.0f32; max_pos * half_dim];
    for pos in 0..max_pos {
        for i in 0..half_dim {
            let freq = 1.0 / (config.rope_theta as f32).powf(2.0 * i as f32 / rope_dim as f32);
            let angle = pos as f32 * freq;
            rope_cos[pos * half_dim + i] = angle.cos();
            rope_sin[pos * half_dim + i] = angle.sin();
        }
    }
    write_f32_vec(&mut w, &rope_cos);
    write_f32_vec(&mut w, &rope_sin);

    // Vision weights (stored as f16 for compact size, source is bf16 anyway)
    if config.has_vision {
        eprintln!("  Vision encoder (f16)...");
        // Patch embedding
        write_tensor_f16(&mut w, &all_tensors, "model.visual.patch_embed.proj.weight");
        write_tensor_f16(&mut w, &all_tensors, "model.visual.patch_embed.proj.bias");
        // Positional embedding
        write_tensor_f16(&mut w, &all_tensors, "model.visual.pos_embed.weight");
        // 24 vision blocks
        for b in 0..config.vision_layers {
            let prefix = format!("model.visual.blocks.{b}");
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.norm1.weight"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.norm1.bias"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.attn.qkv.weight"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.attn.qkv.bias"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.attn.proj.weight"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.attn.proj.bias"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.norm2.weight"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.norm2.bias"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.mlp.linear_fc1.weight"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.mlp.linear_fc1.bias"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.mlp.linear_fc2.weight"));
            write_tensor_f16(&mut w, &all_tensors, &format!("{prefix}.mlp.linear_fc2.bias"));
            if b % 6 == 0 { eprintln!("    Vision block {b}/{}...", config.vision_layers); }
        }
        // Merger
        write_tensor_f16(&mut w, &all_tensors, "model.visual.merger.norm.weight");
        write_tensor_f16(&mut w, &all_tensors, "model.visual.merger.norm.bias");
        write_tensor_f16(&mut w, &all_tensors, "model.visual.merger.linear_fc1.weight");
        write_tensor_f16(&mut w, &all_tensors, "model.visual.merger.linear_fc1.bias");
        write_tensor_f16(&mut w, &all_tensors, "model.visual.merger.linear_fc2.weight");
        write_tensor_f16(&mut w, &all_tensors, "model.visual.merger.linear_fc2.bias");
    }

    w.flush().unwrap();
    let file_size = std::fs::metadata(&output_path).unwrap().len();
    eprintln!("Done! Output: {} ({:.1} MB)", output_path.display(), file_size as f64 / 1024.0 / 1024.0);
}

// ============================================================
// Safetensors loading
// ============================================================

struct TensorData {
    data: Vec<u8>,
    dtype: String,
    shape: Vec<usize>,
}

impl TensorData {
    fn to_f32(&self) -> Vec<f32> {
        match self.dtype.as_str() {
            "BF16" => {
                let count = self.data.len() / 2;
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let bits = u16::from_le_bytes([self.data[i*2], self.data[i*2+1]]);
                    let f32_bits = (bits as u32) << 16;
                    out.push(f32::from_bits(f32_bits));
                }
                out
            }
            "F16" => {
                let count = self.data.len() / 2;
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let bits = u16::from_le_bytes([self.data[i*2], self.data[i*2+1]]);
                    out.push(f16::from_bits(bits).to_f32());
                }
                out
            }
            "F32" => {
                let count = self.data.len() / 4;
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let bytes = [self.data[i*4], self.data[i*4+1], self.data[i*4+2], self.data[i*4+3]];
                    out.push(f32::from_le_bytes(bytes));
                }
                out
            }
            _ => panic!("Unsupported dtype: {}", self.dtype),
        }
    }

    fn to_f16(&self) -> Vec<f16> {
        self.to_f32().iter().map(|&v| f16::from_f32(v)).collect()
    }

    fn _numel(&self) -> usize {
        self.shape.iter().product()
    }
}

fn load_index(path: &Path) -> HashMap<String, String> {
    let content = std::fs::read_to_string(path).unwrap();
    // Simple JSON parsing for weight_map
    let mut map = HashMap::new();
    if let Some(wm_start) = content.find("\"weight_map\"") {
        let wm_str = &content[wm_start..];
        if let Some(brace_start) = wm_str.find('{') {
            if let Some(brace_end) = wm_str.find('}') {
                let inner = &wm_str[brace_start+1..brace_end];
                for line in inner.split(',') {
                    let parts: Vec<&str> = line.split(':').collect();
                    if parts.len() == 2 {
                        let key = parts[0].trim().trim_matches('"').trim();
                        let val = parts[1].trim().trim_matches('"').trim();
                        if !key.is_empty() && !val.is_empty() {
                            map.insert(key.to_string(), val.to_string());
                        }
                    }
                }
            }
        }
    }
    map
}

fn load_safetensors(path: &Path) -> HashMap<String, TensorData> {
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let data = &mmap[..];

    // Header: 8 bytes LE u64 = header_size, then JSON header, then raw data
    let header_size = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header_json = std::str::from_utf8(&data[8..8 + header_size]).unwrap();
    let data_start = 8 + header_size;

    // Parse header JSON (simple parsing)
    let mut tensors = HashMap::new();
    // The header is a JSON object with tensor names as keys
    // Each value has: dtype, shape, data_offsets: [start, end]
    let parsed: serde_json::Value = serde_json::from_str(header_json).unwrap();
    if let serde_json::Value::Object(map) = parsed {
        for (name, info) in map {
            if name == "__metadata__" { continue; }
            let dtype = info["dtype"].as_str().unwrap().to_string();
            let shape: Vec<usize> = info["shape"].as_array().unwrap()
                .iter().map(|v| v.as_u64().unwrap() as usize).collect();
            let offsets = info["data_offsets"].as_array().unwrap();
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            let tensor_data = data[data_start + start..data_start + end].to_vec();
            tensors.insert(name, TensorData { data: tensor_data, dtype, shape });
        }
    }
    tensors
}

// ============================================================
// Weight writing helpers
// ============================================================

/// Write a weight matrix (f16 or Q4 format).
/// PyTorch Linear weights are [out_dim, in_dim] row-major.
/// GEMV needs [in_dim, out_dim] row-major (transposed).
fn write_tensor_weight(w: &mut impl Write, tensors: &HashMap<String, TensorData>, name: &str, format_id: u8) {
    let t = tensors.get(name).unwrap_or_else(|| panic!("Missing tensor: {name}"));
    let raw = t.to_f32();
    let shape = &t.shape;

    let (k, n, f32_data) = if shape.len() == 2 {
        // Transpose: [out_dim, in_dim] → [in_dim, out_dim]
        let out_dim = shape[0];
        let in_dim = shape[1];
        let mut transposed = vec![0.0f32; in_dim * out_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                transposed[i * out_dim + o] = raw[o * in_dim + i];
            }
        }
        (in_dim, out_dim, transposed) // k=in_dim, n=out_dim
    } else if shape.len() == 1 {
        (1, shape[0], raw)
    } else {
        let total: usize = shape.iter().product();
        (1, total, raw)
    };

    write_u64(w, k as u64);
    write_u64(w, n as u64);

    // For Q4: fall back to f16 if n not divisible by 32 (can't quantize small matrices)
    let effective_format = if format_id == 1 && (n % 32 != 0) {
        eprintln!("    {name}: n={n} not divisible by 32, storing as F16 instead of Q4");
        0u8
    } else {
        format_id
    };

    // Write per-weight format byte so loader knows
    w.write_all(&[effective_format]).unwrap();

    match effective_format {
        0 => {
            // F16
            let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
            write_f16_vec(w, &f16_data);
        }
        1 => {
            // Q4 symmetric quantization, group_size=32
            let (packed, scales) = quantize_q4(&f32_data, k, n);
            write_bytes(w, &packed);
            write_f16_vec(w, &scales);
        }
        _ => panic!("Unknown format {format_id}"),
    }
}

/// Write embedding weight: shape [vocab, hidden].
/// For embed_lookup: stored with k=vocab, n=hidden (each row = one token).
/// The data is NOT transposed — stored in original [vocab, hidden] layout.
fn write_embedding_weight(w: &mut impl Write, tensors: &HashMap<String, TensorData>, name: &str, format_id: u8) {
    let t = tensors.get(name).unwrap_or_else(|| panic!("Missing tensor: {name}"));
    let f32_data = t.to_f32();
    let shape = &t.shape;
    // Embedding shape: [vocab_size, hidden_size]
    let (k, n) = (shape[0], shape[1]); // k=vocab (rows), n=hidden (cols)
    eprintln!("    Embedding: shape [{}, {}] → k={k}, n={n}", shape[0], shape[1]);

    write_u64(w, k as u64);
    write_u64(w, n as u64);

    // For embedding: n=hidden=2560, always divisible by 32
    w.write_all(&[format_id]).unwrap();
    match format_id {
        0 => {
            let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
            write_f16_vec(w, &f16_data);
        }
        1 => {
            let (packed, scales) = quantize_q4(&f32_data, k, n);
            write_bytes(w, &packed);
            write_f16_vec(w, &scales);
        }
        _ => panic!("Unknown format {format_id}"),
    }
}

/// Write a tensor as f32 (for small tensors like A_log, dt_bias, conv1d).
fn write_tensor_f32(w: &mut impl Write, tensors: &HashMap<String, TensorData>, name: &str) {
    let t = tensors.get(name).unwrap_or_else(|| panic!("Missing tensor: {name}"));
    let data = t.to_f32();
    write_f32_vec(w, &data);
}

/// Write a tensor as f16 (for norms).
fn write_tensor_f16(w: &mut impl Write, tensors: &HashMap<String, TensorData>, name: &str) {
    let t = tensors.get(name).unwrap_or_else(|| panic!("Missing tensor: {name}"));
    let data = t.to_f16();
    write_f16_vec(w, &data);
}

// ============================================================
// Q4 quantization
// ============================================================

/// Quantize f32 data to Q4 symmetric (group_size=32).
/// Returns (packed_bytes, scales).
fn quantize_q4(data: &[f32], k: usize, n: usize) -> (Vec<u8>, Vec<f16>) {
    let group_size = 32;
    assert!(n % group_size == 0, "n={n} not divisible by group_size={group_size}");
    let groups_per_row = n / group_size;
    let packed_per_group = group_size / 2;

    let mut packed = Vec::with_capacity(k * groups_per_row * packed_per_group);
    let mut scales = Vec::with_capacity(k * groups_per_row);

    for ki in 0..k {
        for g in 0..groups_per_row {
            let offset = ki * n + g * group_size;
            let group = &data[offset..offset + group_size];

            // Find absmax
            let absmax = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if absmax > 0.0 { absmax / 7.0 } else { 0.0 };
            scales.push(f16::from_f32(scale));

            // Quantize and pack
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for j in (0..group_size).step_by(2) {
                let q0 = ((group[j] * inv_scale).round() as i32).clamp(-8, 7) + 8;
                let q1 = ((group[j+1] * inv_scale).round() as i32).clamp(-8, 7) + 8;
                packed.push((q0 as u8) | ((q1 as u8) << 4));
            }
        }
    }

    (packed, scales)
}

// ============================================================
// I/O helpers
// ============================================================

fn write_u32(w: &mut impl Write, val: u32) { w.write_all(&val.to_le_bytes()).unwrap(); }
fn write_u64(w: &mut impl Write, val: u64) { w.write_all(&val.to_le_bytes()).unwrap(); }
fn write_f32(w: &mut impl Write, val: f32) { w.write_all(&val.to_le_bytes()).unwrap(); }
fn write_f64(w: &mut impl Write, val: f64) { w.write_all(&val.to_le_bytes()).unwrap(); }

fn write_f16_vec(w: &mut impl Write, data: &[f16]) {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    write_u64(w, data.len() as u64);
    w.write_all(bytes).unwrap();
}

fn write_f32_vec(w: &mut impl Write, data: &[f32]) {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    write_u64(w, data.len() as u64);
    w.write_all(bytes).unwrap();
}

fn write_bytes(w: &mut impl Write, data: &[u8]) {
    write_u64(w, data.len() as u64);
    w.write_all(data).unwrap();
}
