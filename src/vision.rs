//! Vision encoder for QORA-4B (ViT + merger).
//!
//! Architecture:
//! - Patch embedding: Conv3d(3, 1024, kernel=[2,16,16]), temporal_patch_size=2
//! - Learned positional embedding: nn.Embedding(2304, 1024) with bilinear interpolation
//! - 24 ViT blocks: LayerNorm + MHA (fused QKV [3072,1024]) + LayerNorm + MLP (gelu_pytorch_tanh)
//! - Spatial merge (2x2) → per-patch LayerNorm(1024) → concat 4x1024 → Merger MLP → 2560
//! - Supports both image (single frame, duplicated) and video (frame pairs via Conv3d)
//!
//! Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
//! Weight layout: stored as f16 on disk, loaded as f32 for compute.

/// Vision encoder weights.
pub struct VisionEncoder {
    pub patch_proj_weight: Vec<f32>, // [1024, 3, 2, 16, 16] (Conv3d, temporal_patch_size=2)
    pub patch_proj_bias: Vec<f32>,   // [1024]
    pub pos_embed: Vec<f32>,         // [2304, 1024] (learned positional embedding)
    pub blocks: Vec<VisionBlock>,
    pub merger: Merger,
}

pub struct VisionBlock {
    pub norm1_weight: Vec<f32>,  // [1024]
    pub norm1_bias: Vec<f32>,    // [1024]
    pub qkv_weight: Vec<f32>,   // [3072, 1024] (fused Q+K+V, PyTorch layout)
    pub qkv_bias: Vec<f32>,     // [3072]
    pub proj_weight: Vec<f32>,  // [1024, 1024]
    pub proj_bias: Vec<f32>,    // [1024]
    pub norm2_weight: Vec<f32>, // [1024]
    pub norm2_bias: Vec<f32>,   // [1024]
    pub fc1_weight: Vec<f32>,   // [4096, 1024] (plain MLP, NOT GEGLU)
    pub fc1_bias: Vec<f32>,     // [4096]
    pub fc2_weight: Vec<f32>,   // [1024, 4096]
    pub fc2_bias: Vec<f32>,     // [1024]
}

pub struct Merger {
    pub norm_weight: Vec<f32>,  // [1024] (per-patch norm BEFORE concat)
    pub norm_bias: Vec<f32>,    // [1024]
    pub fc1_weight: Vec<f32>,   // [4096, 4096] (PyTorch layout)
    pub fc1_bias: Vec<f32>,     // [4096]
    pub fc2_weight: Vec<f32>,   // [2560, 4096] (PyTorch layout)
    pub fc2_bias: Vec<f32>,     // [2560]
}

impl VisionEncoder {
    pub fn memory_bytes(&self) -> usize {
        let mut total = (self.patch_proj_weight.len() + self.patch_proj_bias.len() + self.pos_embed.len()) * 4;
        for b in &self.blocks {
            total += (b.norm1_weight.len() + b.norm1_bias.len()
                + b.qkv_weight.len() + b.qkv_bias.len()
                + b.proj_weight.len() + b.proj_bias.len()
                + b.norm2_weight.len() + b.norm2_bias.len()
                + b.fc1_weight.len() + b.fc1_bias.len()
                + b.fc2_weight.len() + b.fc2_bias.len()) * 4;
        }
        total += (self.merger.norm_weight.len() + self.merger.norm_bias.len()
            + self.merger.fc1_weight.len() + self.merger.fc1_bias.len()
            + self.merger.fc2_weight.len() + self.merger.fc2_bias.len()) * 4;
        total
    }
}

/// QORA-4B vision normalization constants.
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Load an image from file, resize to fit within vision encoder limits,
/// return (pixels_rgb_f32_01, height, width).
pub fn load_image(path: &std::path::Path) -> Result<(Vec<f32>, usize, usize), String> {
    let img = image::open(path).map_err(|e| format!("Failed to load image: {e}"))?;
    let img = img.to_rgb8();
    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
    eprintln!("  Image: {}x{}", orig_w, orig_h);

    // Resize to fit within max dimensions, keeping aspect ratio.
    // Dims must be divisible by patch_size * spatial_merge_size = 16 * 2 = 32.
    // Max patches per side: sqrt(2304) = 48, so max pixels = 48 * 16 = 768.
    let max_side = 768usize;
    let align = 32usize;

    let scale = if orig_w.max(orig_h) > max_side {
        max_side as f32 / orig_w.max(orig_h) as f32
    } else {
        1.0
    };

    let mut new_w = ((orig_w as f32 * scale) as usize).max(align);
    let mut new_h = ((orig_h as f32 * scale) as usize).max(align);
    // Round to nearest multiple of align
    new_w = ((new_w + align / 2) / align) * align;
    new_h = ((new_h + align / 2) / align) * align;

    // Check total patches doesn't exceed pos_embed limit
    let max_patches = 2304;
    let patches = (new_h / 16) * (new_w / 16);
    if patches > max_patches {
        // Scale down further
        let factor = (max_patches as f32 / patches as f32).sqrt();
        new_w = (((new_w as f32 * factor) as usize) / align) * align;
        new_h = (((new_h as f32 * factor) as usize) / align) * align;
        if new_w < align { new_w = align; }
        if new_h < align { new_h = align; }
    }

    eprintln!("  Resized: {}x{} ({} patches)", new_w, new_h, (new_h / 16) * (new_w / 16));

    // Resize using the image crate
    let resized = image::imageops::resize(
        &img, new_w as u32, new_h as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // Convert to f32 [0, 1] in [H, W, 3] layout
    let mut pixels = vec![0.0f32; new_h * new_w * 3];
    for y in 0..new_h {
        for x in 0..new_w {
            let p = resized.get_pixel(x as u32, y as u32);
            let idx = (y * new_w + x) * 3;
            pixels[idx + 0] = p[0] as f32 / 255.0;
            pixels[idx + 1] = p[1] as f32 / 255.0;
            pixels[idx + 2] = p[2] as f32 / 255.0;
        }
    }

    Ok((pixels, new_h, new_w))
}

/// Load video frames from a directory of images, resize them to uniform dimensions,
/// and return normalized pixel data.
/// Returns (pixels_flat, num_frames, height, width) where pixels_flat is [num_frames * H * W * 3].
pub fn load_video_frames(dir: &std::path::Path) -> Result<(Vec<f32>, usize, usize, usize), String> {
    // Collect image files from directory, sorted by name
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read video frames directory: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_lowercase();
            name.ends_with(".png") || name.ends_with(".jpg") || name.ends_with(".jpeg") || name.ends_with(".bmp")
        })
        .collect();
    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    if entries.is_empty() {
        return Err("No image files found in video frames directory".into());
    }

    eprintln!("  Found {} frame images", entries.len());

    // Load all frames as RGB8
    let mut frames: Vec<image::RgbImage> = Vec::new();
    for entry in &entries {
        let img = image::open(entry.path())
            .map_err(|e| format!("Failed to load frame {}: {e}", entry.file_name().to_string_lossy()))?;
        frames.push(img.to_rgb8());
    }

    // Determine target size from first frame, then resize all to same dimensions
    let (orig_w, orig_h) = (frames[0].width() as usize, frames[0].height() as usize);
    eprintln!("  Frame size: {}x{}", orig_w, orig_h);

    let max_side = 768usize;
    let align = 32usize;

    let scale = if orig_w.max(orig_h) > max_side {
        max_side as f32 / orig_w.max(orig_h) as f32
    } else {
        1.0
    };

    let mut new_w = ((orig_w as f32 * scale) as usize).max(align);
    let mut new_h = ((orig_h as f32 * scale) as usize).max(align);
    new_w = ((new_w + align / 2) / align) * align;
    new_h = ((new_h + align / 2) / align) * align;

    // Check total patches doesn't exceed pos_embed limit (per temporal step)
    let max_patches_per_frame = 2304;
    let patches = (new_h / 16) * (new_w / 16);
    if patches > max_patches_per_frame {
        let factor = (max_patches_per_frame as f32 / patches as f32).sqrt();
        new_w = (((new_w as f32 * factor) as usize) / align) * align;
        new_h = (((new_h as f32 * factor) as usize) / align) * align;
        if new_w < align { new_w = align; }
        if new_h < align { new_h = align; }
    }

    let num_frames = frames.len();
    eprintln!("  Resized: {}x{} ({} patches/frame, {} frames)", new_w, new_h, (new_h / 16) * (new_w / 16), num_frames);

    // Resize and convert all frames to f32 [0, 1]
    let mut pixels = vec![0.0f32; num_frames * new_h * new_w * 3];
    for (fi, frame) in frames.iter().enumerate() {
        let resized = image::imageops::resize(
            frame, new_w as u32, new_h as u32,
            image::imageops::FilterType::Lanczos3,
        );
        let frame_offset = fi * new_h * new_w * 3;
        for y in 0..new_h {
            for x in 0..new_w {
                let p = resized.get_pixel(x as u32, y as u32);
                let idx = frame_offset + (y * new_w + x) * 3;
                pixels[idx + 0] = p[0] as f32 / 255.0;
                pixels[idx + 1] = p[1] as f32 / 255.0;
                pixels[idx + 2] = p[2] as f32 / 255.0;
            }
        }
    }

    Ok((pixels, num_frames, new_h, new_w))
}

impl VisionEncoder {
    /// Encode an image (raw RGB pixels) into LLM-compatible embeddings.
    /// Input: pixels [height, width, 3] as f32 in [0, 1].
    /// Output: flattened [num_merged_tokens * 2560] embeddings ready for LLM.
    /// Also returns num_merged_tokens.
    pub fn encode_image(&self, pixels: &[f32], height: usize, width: usize) -> (Vec<f32>, usize) {
        let patch_size = 16;
        let hidden = 1024;
        let num_heads = 16;
        let head_dim = hidden / num_heads; // 64

        // Normalize: (pixel - 0.5) / 0.5 = pixel * 2 - 1  (maps to [-1, 1])
        let mut normalized = vec![0.0f32; height * width * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                for c in 0..3 {
                    normalized[idx + c] = (pixels[idx + c] - MEAN[c]) / STD[c];
                }
            }
        }

        // Patch embedding: Conv3d [1024, 3, 2, 16, 16] with temporal_patch_size=2
        // For a single image, we duplicate the frame along temporal axis.
        // Since both frames are identical, Conv3d reduces to Conv2d with:
        //   weight_2d[oc,c,ky,kx] = weight_3d[oc,c,0,ky,kx] + weight_3d[oc,c,1,ky,kx]
        let patches_h = height / patch_size;
        let patches_w = width / patch_size;
        let num_patches = patches_h * patches_w;
        let temporal_size = 2;
        eprintln!("  Patches: {}x{} = {} total", patches_w, patches_h, num_patches);

        let mut patch_embeds = vec![0.0f32; num_patches * hidden];
        for py in 0..patches_h {
            for px in 0..patches_w {
                let patch_idx = py * patches_w + px;
                for oc in 0..hidden {
                    let mut sum = self.patch_proj_bias[oc];
                    for c in 0..3 {
                        for ky in 0..patch_size {
                            for kx in 0..patch_size {
                                let iy = py * patch_size + ky;
                                let ix = px * patch_size + kx;
                                let pixel = normalized[(iy * width + ix) * 3 + c];
                                // Weight layout: [oc, c, t, ky, kx] → sum over t=0,1
                                let mut w_sum = 0.0f32;
                                for t in 0..temporal_size {
                                    let w_idx = ((((oc * 3 + c) * temporal_size + t) * patch_size) + ky) * patch_size + kx;
                                    w_sum += self.patch_proj_weight[w_idx];
                                }
                                sum += pixel * w_sum;
                            }
                        }
                    }
                    patch_embeds[patch_idx * hidden + oc] = sum;
                }
            }
        }
        eprintln!("  Patch embedding done");

        // Add learned positional embeddings via bilinear interpolation from 48×48 grid.
        // pos_embed is nn.Embedding(2304, 1024) stored as [2304 * 1024] flat.
        // Grid is 48×48 (num_grid_per_side = sqrt(2304) = 48).
        let grid_side = 48usize; // num_grid_per_side
        let h_idxs: Vec<f32> = (0..patches_h)
            .map(|i| if patches_h <= 1 { 0.0 } else { i as f32 * (grid_side - 1) as f32 / (patches_h - 1) as f32 })
            .collect();
        let w_idxs: Vec<f32> = (0..patches_w)
            .map(|i| if patches_w <= 1 { 0.0 } else { i as f32 * (grid_side - 1) as f32 / (patches_w - 1) as f32 })
            .collect();

        for py in 0..patches_h {
            for px in 0..patches_w {
                let pi = py * patches_w + px;
                let h_idx = h_idxs[py];
                let w_idx = w_idxs[px];
                let h_floor = h_idx as usize;
                let w_floor = w_idx as usize;
                let h_ceil = (h_floor + 1).min(grid_side - 1);
                let w_ceil = (w_floor + 1).min(grid_side - 1);
                let dh = h_idx - h_floor as f32;
                let dw = w_idx - w_floor as f32;

                // Bilinear interpolation from 4 corners of the 48×48 grid
                let w00 = (1.0 - dh) * (1.0 - dw);
                let w01 = (1.0 - dh) * dw;
                let w10 = dh * (1.0 - dw);
                let w11 = dh * dw;
                let i00 = (h_floor * grid_side + w_floor) * hidden;
                let i01 = (h_floor * grid_side + w_ceil) * hidden;
                let i10 = (h_ceil * grid_side + w_floor) * hidden;
                let i11 = (h_ceil * grid_side + w_ceil) * hidden;

                let base = pi * hidden;
                for j in 0..hidden {
                    patch_embeds[base + j] += w00 * self.pos_embed[i00 + j]
                        + w01 * self.pos_embed[i01 + j]
                        + w10 * self.pos_embed[i10 + j]
                        + w11 * self.pos_embed[i11 + j];
                }
            }
        }

        // Vision RoPE: dim=head_dim//2=32, theta=10000.0
        // Position IDs are 2D (row, col) per patch. Frequencies are duplicated to fill head_dim.
        let rope_dim = head_dim / 2; // 32
        let half_rope = rope_dim / 2; // 16 frequency entries
        let theta = 10000.0f64;
        let inv_freq: Vec<f64> = (0..half_rope)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / rope_dim as f64))
            .collect();

        // Compute cos/sin tables for each patch position [num_patches, head_dim]
        // Layout: [freqs_h(16), freqs_w(16), freqs_h(16), freqs_w(16)] = 64
        let mut rope_cos = vec![0.0f32; num_patches * head_dim];
        let mut rope_sin = vec![0.0f32; num_patches * head_dim];
        for py in 0..patches_h {
            for px in 0..patches_w {
                let pi = py * patches_w + px;
                let row = py as f64;
                let col = px as f64;
                for i in 0..half_rope {
                    let freq_h = row * inv_freq[i];
                    let freq_w = col * inv_freq[i];
                    let (sh, ch) = freq_h.sin_cos();
                    let (sw, cw) = freq_w.sin_cos();
                    let base = pi * head_dim;
                    rope_cos[base + i] = ch as f32;
                    rope_cos[base + half_rope + i] = cw as f32;
                    rope_cos[base + 2 * half_rope + i] = ch as f32;
                    rope_cos[base + 3 * half_rope + i] = cw as f32;
                    rope_sin[base + i] = sh as f32;
                    rope_sin[base + half_rope + i] = sw as f32;
                    rope_sin[base + 2 * half_rope + i] = sh as f32;
                    rope_sin[base + 3 * half_rope + i] = sw as f32;
                }
            }
        }

        // ViT blocks
        for (blk_idx, block) in self.blocks.iter().enumerate() {
            // Pre-norm (LayerNorm with bias)
            let mut x_norm = vec![0.0f32; num_patches * hidden];
            for t in 0..num_patches {
                layer_norm(
                    &patch_embeds[t * hidden..(t + 1) * hidden],
                    &mut x_norm[t * hidden..(t + 1) * hidden],
                    &block.norm1_weight, &block.norm1_bias,
                );
            }

            // QKV projection (fused): [3072, 1024] in PyTorch layout → output [num_patches, 3072]
            let qkv = matmul_bias_t(&x_norm, num_patches, hidden, &block.qkv_weight, &block.qkv_bias, 3 * hidden);

            // Per-token split: for token t, Q=qkv[t*3H..t*3H+H], K=..+H..+2H, V=..+2H..+3H
            let mut q_all = vec![0.0f32; num_patches * hidden];
            let mut k_all = vec![0.0f32; num_patches * hidden];
            let mut v_all = vec![0.0f32; num_patches * hidden];
            for t in 0..num_patches {
                let base = t * 3 * hidden;
                q_all[t * hidden..(t + 1) * hidden].copy_from_slice(&qkv[base..base + hidden]);
                k_all[t * hidden..(t + 1) * hidden].copy_from_slice(&qkv[base + hidden..base + 2 * hidden]);
                v_all[t * hidden..(t + 1) * hidden].copy_from_slice(&qkv[base + 2 * hidden..base + 3 * hidden]);
            }

            // Apply vision RoPE to Q and K (same cos/sin for all heads)
            for t in 0..num_patches {
                let rope_off = t * head_dim;
                let cos = &rope_cos[rope_off..rope_off + head_dim];
                let sin = &rope_sin[rope_off..rope_off + head_dim];
                let half = head_dim / 2; // 32
                for h in 0..num_heads {
                    let off = t * hidden + h * head_dim;
                    apply_rope_vision(&mut q_all[off..off + head_dim], cos, sin, half);
                    apply_rope_vision(&mut k_all[off..off + head_dim], cos, sin, half);
                }
            }

            // Multi-head attention (no causal mask for vision)
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_output = vec![0.0f32; num_patches * hidden];
            for h in 0..num_heads {
                for t1 in 0..num_patches {
                    let q_off = t1 * hidden + h * head_dim;
                    let q = &q_all[q_off..q_off + head_dim];
                    let mut scores = vec![0.0f32; num_patches];
                    for t2 in 0..num_patches {
                        let k_off = t2 * hidden + h * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim { dot += q[d] * k_all[k_off + d]; }
                        scores[t2] = dot * scale;
                    }
                    softmax(&mut scores);
                    let out_off = t1 * hidden + h * head_dim;
                    for t2 in 0..num_patches {
                        let v_off = t2 * hidden + h * head_dim;
                        let sc = scores[t2];
                        for d in 0..head_dim {
                            attn_output[out_off + d] += sc * v_all[v_off + d];
                        }
                    }
                }
            }

            // Output projection + residual
            let proj_out = matmul_bias_t(&attn_output, num_patches, hidden, &block.proj_weight, &block.proj_bias, hidden);
            for j in 0..num_patches * hidden { patch_embeds[j] += proj_out[j]; }

            // FFN: LayerNorm + gelu_pytorch_tanh MLP (NOT GEGLU!)
            let mut x_norm2 = vec![0.0f32; num_patches * hidden];
            for t in 0..num_patches {
                layer_norm(
                    &patch_embeds[t * hidden..(t + 1) * hidden],
                    &mut x_norm2[t * hidden..(t + 1) * hidden],
                    &block.norm2_weight, &block.norm2_bias,
                );
            }

            // MLP: fc1(1024→4096) → gelu_tanh → fc2(4096→1024)
            let fc1_out_dim = block.fc1_bias.len(); // [4096]
            let fc1_out = matmul_bias_t(&x_norm2, num_patches, hidden, &block.fc1_weight, &block.fc1_bias, fc1_out_dim);

            // Apply gelu_pytorch_tanh (plain, not gated)
            let mut fc1_act = vec![0.0f32; num_patches * fc1_out_dim];
            for j in 0..num_patches * fc1_out_dim {
                fc1_act[j] = gelu_tanh(fc1_out[j]);
            }

            let fc2_out = matmul_bias_t(&fc1_act, num_patches, fc1_out_dim, &block.fc2_weight, &block.fc2_bias, hidden);
            for j in 0..num_patches * hidden { patch_embeds[j] += fc2_out[j]; }

            if blk_idx % 6 == 0 {
                eprintln!("  ViT block {}/{}...", blk_idx + 1, self.blocks.len());
            }
        }
        eprintln!("  All ViT blocks done");

        // Spatial merge (2×2 → 1)
        // Step 1: Apply per-patch LayerNorm(1024) BEFORE concatenation
        let mut normed_patches = vec![0.0f32; num_patches * hidden];
        for t in 0..num_patches {
            layer_norm(
                &patch_embeds[t * hidden..(t + 1) * hidden],
                &mut normed_patches[t * hidden..(t + 1) * hidden],
                &self.merger.norm_weight, &self.merger.norm_bias,
            );
        }

        // Step 2: Spatial merge — group 2×2 patches and concatenate
        let merge = 2;
        let merged_h = patches_h / merge;
        let merged_w = patches_w / merge;
        let num_merged = merged_h * merged_w;
        let merged_input_dim = merge * merge * hidden; // 4 * 1024 = 4096
        eprintln!("  Merged: {}x{} = {} tokens ({}→2560 dim)", merged_w, merged_h, num_merged, merged_input_dim);

        let mut merged = vec![0.0f32; num_merged * merged_input_dim];
        for my in 0..merged_h {
            for mx in 0..merged_w {
                let mi = my * merged_w + mx;
                let mut offset = 0;
                for dy in 0..merge {
                    for dx in 0..merge {
                        let py = my * merge + dy;
                        let px = mx * merge + dx;
                        let pi = py * patches_w + px;
                        let src = &normed_patches[pi * hidden..(pi + 1) * hidden];
                        merged[mi * merged_input_dim + offset..mi * merged_input_dim + offset + hidden]
                            .copy_from_slice(src);
                        offset += hidden;
                    }
                }
            }
        }

        // Step 3: Merger MLP — fc1(4096→4096) + GELU + fc2(4096→2560)
        let merger_hidden = self.merger.fc1_bias.len(); // 4096
        let fc1 = matmul_bias_t(&merged, num_merged, merged_input_dim,
            &self.merger.fc1_weight, &self.merger.fc1_bias, merger_hidden);
        let mut fc1_act = vec![0.0f32; num_merged * merger_hidden];
        for j in 0..num_merged * merger_hidden {
            fc1_act[j] = gelu_plain(fc1[j]); // Merger uses plain GELU (not tanh approx)
        }
        let out_dim = self.merger.fc2_bias.len(); // 2560
        let output = matmul_bias_t(&fc1_act, num_merged, merger_hidden,
            &self.merger.fc2_weight, &self.merger.fc2_bias, out_dim);
        eprintln!("  Merger done: {} tokens × {} dim", num_merged, out_dim);

        (output, num_merged)
    }

    /// Encode video frames into LLM-compatible embeddings.
    /// Input: pixels [num_frames, height, width, 3] as f32 in [0, 1].
    /// Conv3d processes frame PAIRS: N frames → ceil(N/2) temporal patches.
    /// Output: flattened [num_merged_tokens * 2560] embeddings ready for LLM.
    pub fn encode_video(&self, pixels: &[f32], num_frames: usize, height: usize, width: usize) -> (Vec<f32>, usize) {
        let patch_size = 16;
        let hidden = 1024;
        let num_heads = 16;
        let head_dim = hidden / num_heads; // 64
        let temporal_size = 2;
        let frame_pixels = height * width * 3;

        // Pad to even number of frames by duplicating the last frame
        let actual_frames = if num_frames % 2 != 0 { num_frames + 1 } else { num_frames };
        let temporal_steps = actual_frames / 2; // T = number of temporal patches

        // Normalize all frames: (pixel - 0.5) / 0.5 = pixel * 2 - 1
        let mut normalized = vec![0.0f32; actual_frames * height * width * 3];
        for f in 0..num_frames {
            for i in 0..frame_pixels {
                let c = i % 3;
                normalized[f * frame_pixels + i] = (pixels[f * frame_pixels + i] - MEAN[c]) / STD[c];
            }
        }
        // If padded, copy last frame
        if actual_frames > num_frames {
            let last_start = (num_frames - 1) * frame_pixels;
            let pad_start = num_frames * frame_pixels;
            for i in 0..frame_pixels {
                normalized[pad_start + i] = normalized[last_start + i];
            }
        }

        let patches_h = height / patch_size;
        let patches_w = width / patch_size;
        let patches_per_frame = patches_h * patches_w;
        let total_patches = temporal_steps * patches_per_frame;
        eprintln!("  Video: {} frames → {} temporal steps, {}x{} = {} patches/step, {} total patches",
            num_frames, temporal_steps, patches_w, patches_h, patches_per_frame, total_patches);

        // Conv3d patch embedding over frame PAIRS
        let mut patch_embeds = vec![0.0f32; total_patches * hidden];
        for t in 0..temporal_steps {
            let f0 = t * 2;     // first frame in pair
            let f1 = t * 2 + 1; // second frame in pair
            for py in 0..patches_h {
                for px in 0..patches_w {
                    let patch_idx = t * patches_per_frame + py * patches_w + px;
                    for oc in 0..hidden {
                        let mut sum = self.patch_proj_bias[oc];
                        for c in 0..3 {
                            for ky in 0..patch_size {
                                for kx in 0..patch_size {
                                    let iy = py * patch_size + ky;
                                    let ix = px * patch_size + kx;
                                    let pixel_idx = (iy * width + ix) * 3 + c;
                                    let pixel_f0 = normalized[f0 * frame_pixels + pixel_idx];
                                    let pixel_f1 = normalized[f1 * frame_pixels + pixel_idx];
                                    // Weight layout: [oc, c, t, ky, kx]
                                    let w_idx_t0 = ((((oc * 3 + c) * temporal_size + 0) * patch_size) + ky) * patch_size + kx;
                                    let w_idx_t1 = ((((oc * 3 + c) * temporal_size + 1) * patch_size) + ky) * patch_size + kx;
                                    sum += pixel_f0 * self.patch_proj_weight[w_idx_t0]
                                         + pixel_f1 * self.patch_proj_weight[w_idx_t1];
                                }
                            }
                        }
                        patch_embeds[patch_idx * hidden + oc] = sum;
                    }
                }
            }
            eprintln!("  Conv3d temporal step {}/{} done", t + 1, temporal_steps);
        }
        eprintln!("  Patch embedding done ({} patches)", total_patches);

        // Add learned positional embeddings via bilinear interpolation from 48×48 grid.
        // Same spatial embeddings, repeated for each temporal step.
        let grid_side = 48usize;
        let h_idxs: Vec<f32> = (0..patches_h)
            .map(|i| if patches_h <= 1 { 0.0 } else { i as f32 * (grid_side - 1) as f32 / (patches_h - 1) as f32 })
            .collect();
        let w_idxs: Vec<f32> = (0..patches_w)
            .map(|i| if patches_w <= 1 { 0.0 } else { i as f32 * (grid_side - 1) as f32 / (patches_w - 1) as f32 })
            .collect();

        for t in 0..temporal_steps {
            for py in 0..patches_h {
                for px in 0..patches_w {
                    let pi = t * patches_per_frame + py * patches_w + px;
                    let h_idx = h_idxs[py];
                    let w_idx = w_idxs[px];
                    let h_floor = h_idx as usize;
                    let w_floor = w_idx as usize;
                    let h_ceil = (h_floor + 1).min(grid_side - 1);
                    let w_ceil = (w_floor + 1).min(grid_side - 1);
                    let dh = h_idx - h_floor as f32;
                    let dw = w_idx - w_floor as f32;

                    let w00 = (1.0 - dh) * (1.0 - dw);
                    let w01 = (1.0 - dh) * dw;
                    let w10 = dh * (1.0 - dw);
                    let w11 = dh * dw;
                    let i00 = (h_floor * grid_side + w_floor) * hidden;
                    let i01 = (h_floor * grid_side + w_ceil) * hidden;
                    let i10 = (h_ceil * grid_side + w_floor) * hidden;
                    let i11 = (h_ceil * grid_side + w_ceil) * hidden;

                    let base = pi * hidden;
                    for j in 0..hidden {
                        patch_embeds[base + j] += w00 * self.pos_embed[i00 + j]
                            + w01 * self.pos_embed[i01 + j]
                            + w10 * self.pos_embed[i10 + j]
                            + w11 * self.pos_embed[i11 + j];
                    }
                }
            }
        }

        // Vision RoPE: same spatial frequencies, repeated for each temporal step
        let rope_dim = head_dim / 2; // 32
        let half_rope = rope_dim / 2; // 16 frequency entries
        let theta = 10000.0f64;
        let inv_freq: Vec<f64> = (0..half_rope)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / rope_dim as f64))
            .collect();

        let mut rope_cos = vec![0.0f32; total_patches * head_dim];
        let mut rope_sin = vec![0.0f32; total_patches * head_dim];
        for t in 0..temporal_steps {
            for py in 0..patches_h {
                for px in 0..patches_w {
                    let pi = t * patches_per_frame + py * patches_w + px;
                    let row = py as f64;
                    let col = px as f64;
                    for i in 0..half_rope {
                        let freq_h = row * inv_freq[i];
                        let freq_w = col * inv_freq[i];
                        let (sh, ch) = freq_h.sin_cos();
                        let (sw, cw) = freq_w.sin_cos();
                        let base = pi * head_dim;
                        rope_cos[base + i] = ch as f32;
                        rope_cos[base + half_rope + i] = cw as f32;
                        rope_cos[base + 2 * half_rope + i] = ch as f32;
                        rope_cos[base + 3 * half_rope + i] = cw as f32;
                        rope_sin[base + i] = sh as f32;
                        rope_sin[base + half_rope + i] = sw as f32;
                        rope_sin[base + 2 * half_rope + i] = sh as f32;
                        rope_sin[base + 3 * half_rope + i] = sw as f32;
                    }
                }
            }
        }

        // ViT blocks — full attention across ALL temporal+spatial patches
        for (blk_idx, block) in self.blocks.iter().enumerate() {
            let mut x_norm = vec![0.0f32; total_patches * hidden];
            for t in 0..total_patches {
                layer_norm(
                    &patch_embeds[t * hidden..(t + 1) * hidden],
                    &mut x_norm[t * hidden..(t + 1) * hidden],
                    &block.norm1_weight, &block.norm1_bias,
                );
            }

            let qkv = matmul_bias_t(&x_norm, total_patches, hidden, &block.qkv_weight, &block.qkv_bias, 3 * hidden);

            let mut q_all = vec![0.0f32; total_patches * hidden];
            let mut k_all = vec![0.0f32; total_patches * hidden];
            let mut v_all = vec![0.0f32; total_patches * hidden];
            for t in 0..total_patches {
                let base = t * 3 * hidden;
                q_all[t * hidden..(t + 1) * hidden].copy_from_slice(&qkv[base..base + hidden]);
                k_all[t * hidden..(t + 1) * hidden].copy_from_slice(&qkv[base + hidden..base + 2 * hidden]);
                v_all[t * hidden..(t + 1) * hidden].copy_from_slice(&qkv[base + 2 * hidden..base + 3 * hidden]);
            }

            // Apply vision RoPE
            for t in 0..total_patches {
                let rope_off = t * head_dim;
                let cos = &rope_cos[rope_off..rope_off + head_dim];
                let sin = &rope_sin[rope_off..rope_off + head_dim];
                let half = head_dim / 2;
                for h in 0..num_heads {
                    let off = t * hidden + h * head_dim;
                    apply_rope_vision(&mut q_all[off..off + head_dim], cos, sin, half);
                    apply_rope_vision(&mut k_all[off..off + head_dim], cos, sin, half);
                }
            }

            // Multi-head attention (no causal mask)
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_output = vec![0.0f32; total_patches * hidden];
            for h in 0..num_heads {
                for t1 in 0..total_patches {
                    let q_off = t1 * hidden + h * head_dim;
                    let q = &q_all[q_off..q_off + head_dim];
                    let mut scores = vec![0.0f32; total_patches];
                    for t2 in 0..total_patches {
                        let k_off = t2 * hidden + h * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim { dot += q[d] * k_all[k_off + d]; }
                        scores[t2] = dot * scale;
                    }
                    softmax(&mut scores);
                    let out_off = t1 * hidden + h * head_dim;
                    for t2 in 0..total_patches {
                        let v_off = t2 * hidden + h * head_dim;
                        let sc = scores[t2];
                        for d in 0..head_dim {
                            attn_output[out_off + d] += sc * v_all[v_off + d];
                        }
                    }
                }
            }

            let proj_out = matmul_bias_t(&attn_output, total_patches, hidden, &block.proj_weight, &block.proj_bias, hidden);
            for j in 0..total_patches * hidden { patch_embeds[j] += proj_out[j]; }

            // FFN
            let mut x_norm2 = vec![0.0f32; total_patches * hidden];
            for t in 0..total_patches {
                layer_norm(
                    &patch_embeds[t * hidden..(t + 1) * hidden],
                    &mut x_norm2[t * hidden..(t + 1) * hidden],
                    &block.norm2_weight, &block.norm2_bias,
                );
            }

            let fc1_out_dim = block.fc1_bias.len();
            let fc1_out = matmul_bias_t(&x_norm2, total_patches, hidden, &block.fc1_weight, &block.fc1_bias, fc1_out_dim);
            let mut fc1_act = vec![0.0f32; total_patches * fc1_out_dim];
            for j in 0..total_patches * fc1_out_dim {
                fc1_act[j] = gelu_tanh(fc1_out[j]);
            }
            let fc2_out = matmul_bias_t(&fc1_act, total_patches, fc1_out_dim, &block.fc2_weight, &block.fc2_bias, hidden);
            for j in 0..total_patches * hidden { patch_embeds[j] += fc2_out[j]; }

            if blk_idx % 6 == 0 {
                eprintln!("  ViT block {}/{} ({} patches)...", blk_idx + 1, self.blocks.len(), total_patches);
            }
        }
        eprintln!("  All ViT blocks done");

        // Spatial merge (2×2 → 1) per temporal step
        let merge = 2;
        let merged_h = patches_h / merge;
        let merged_w = patches_w / merge;
        let merged_per_step = merged_h * merged_w;
        let num_merged = temporal_steps * merged_per_step;
        let merged_input_dim = merge * merge * hidden; // 4 * 1024 = 4096
        eprintln!("  Merged: {} temporal steps × {}x{} = {} tokens ({}→2560 dim)",
            temporal_steps, merged_w, merged_h, num_merged, merged_input_dim);

        // Apply per-patch LayerNorm BEFORE concatenation
        let mut normed_patches = vec![0.0f32; total_patches * hidden];
        for t in 0..total_patches {
            layer_norm(
                &patch_embeds[t * hidden..(t + 1) * hidden],
                &mut normed_patches[t * hidden..(t + 1) * hidden],
                &self.merger.norm_weight, &self.merger.norm_bias,
            );
        }

        // Spatial merge per temporal step
        let mut merged = vec![0.0f32; num_merged * merged_input_dim];
        for ts in 0..temporal_steps {
            let patch_base = ts * patches_per_frame;
            let merge_base = ts * merged_per_step;
            for my in 0..merged_h {
                for mx in 0..merged_w {
                    let mi = merge_base + my * merged_w + mx;
                    let mut offset = 0;
                    for dy in 0..merge {
                        for dx in 0..merge {
                            let py = my * merge + dy;
                            let px = mx * merge + dx;
                            let pi = patch_base + py * patches_w + px;
                            let src = &normed_patches[pi * hidden..(pi + 1) * hidden];
                            merged[mi * merged_input_dim + offset..mi * merged_input_dim + offset + hidden]
                                .copy_from_slice(src);
                            offset += hidden;
                        }
                    }
                }
            }
        }

        // Merger MLP: fc1(4096→4096) + GELU + fc2(4096→2560)
        let merger_hidden = self.merger.fc1_bias.len();
        let fc1 = matmul_bias_t(&merged, num_merged, merged_input_dim,
            &self.merger.fc1_weight, &self.merger.fc1_bias, merger_hidden);
        let mut fc1_act = vec![0.0f32; num_merged * merger_hidden];
        for j in 0..num_merged * merger_hidden {
            fc1_act[j] = gelu_plain(fc1[j]);
        }
        let out_dim = self.merger.fc2_bias.len(); // 2560
        let output = matmul_bias_t(&fc1_act, num_merged, merger_hidden,
            &self.merger.fc2_weight, &self.merger.fc2_bias, out_dim);
        eprintln!("  Merger done: {} tokens × {} dim", num_merged, out_dim);

        (output, num_merged)
    }
}

// ============================================================
// Helper functions
// ============================================================

fn layer_norm(input: &[f32], output: &mut [f32], weight: &[f32], bias: &[f32]) {
    let n = input.len();
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-6).sqrt();
    for i in 0..n {
        output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

fn softmax(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
    let inv = 1.0 / sum;
    for s in scores.iter_mut() { *s *= inv; }
}

/// GELU with tanh approximation (gelu_pytorch_tanh) — used in ViT blocks.
fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + ((0.7978845608 * (x + 0.044715 * x * x * x)) as f64).tanh() as f32)
}

/// Plain GELU (erf-based) — used in merger MLP.
fn gelu_plain(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Approximate erf for f32.
fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun approximation
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;
    let p = 0.3275911f32;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Apply vision RoPE (rotate_half style: split-half, [-x2, x1]).
/// cos/sin: [head_dim], half = head_dim/2
fn apply_rope_vision(x: &mut [f32], cos: &[f32], sin: &[f32], half: usize) {
    // rotate_half: [-x[half..], x[..half]]
    // out[d] = x[d]*cos[d] + rotate_half(x)[d]*sin[d]
    // For d < half:  out[d] = x[d]*cos[d] - x[d+half]*sin[d]
    // For d >= half: out[d] = x[d]*cos[d] + x[d-half]*sin[d]
    let hd = x.len();
    debug_assert_eq!(hd, 2 * half);
    // Save original first half (needed since we modify in-place)
    let mut first_half = [0.0f32; 64]; // max head_dim
    first_half[..half].copy_from_slice(&x[..half]);
    for d in 0..half {
        x[d] = x[d] * cos[d] - x[d + half] * sin[d];
    }
    for d in half..hd {
        x[d] = x[d] * cos[d] + first_half[d - half] * sin[d];
    }
}

/// Matrix multiply with transposed weight: y = x @ weight.T + bias.
/// Weight is stored in PyTorch layout [out_dim, in_dim] (row-major).
/// x: [seq_len, in_dim], weight: [out_dim, in_dim], bias: [out_dim]
/// output: [seq_len, out_dim]
fn matmul_bias_t(x: &[f32], seq_len: usize, in_dim: usize, weight: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * out_dim];
    for t in 0..seq_len {
        let x_row = &x[t * in_dim..(t + 1) * in_dim];
        let out_row = &mut output[t * out_dim..(t + 1) * out_dim];
        for j in 0..out_dim {
            let mut sum = bias[j];
            let w_row = &weight[j * in_dim..(j + 1) * in_dim];
            for i in 0..in_dim {
                sum += x_row[i] * w_row[i];
            }
            out_row[j] = sum;
        }
    }
    output
}
