

# QORA-4B

Pure Rust multimodal inference engine based on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). No Python, no CUDA, no external ML frameworks. Single executable + model weights = portable AI that runs on any machine.

**GPU accelerated** — auto-detects Vulkan (Windows/Linux) or Metal (macOS) GPU and runs inference on it. Falls back to CPU if no GPU available. **Smart system awareness** — detects RAM and CPU at startup and adjusts generation limits automatically.

## License

This project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The base model [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) is released by the Qwen team under Apache 2.0.

## What It Does

QORA-4B is a 4-billion parameter language model with built-in vision. It can:

- **Text generation** — answer questions, write code, reason through problems
- **Image understanding** — describe photos, answer questions about images
- **Video understanding** — analyze frame sequences, describe motion and temporal changes
- **Thinking mode** — extended chain-of-thought reasoning with configurable budget

## Architecture

QORA-4B uses a hybrid architecture combining two attention mechanisms:

| Component | Details |
|-----------|---------|
| **Parameters** | 4B total |
| **Hidden dim** | 2560 |
| **Layers** | 32 (24 DeltaNet + 8 Full Attention) |
| **Layer pattern** | 3x DeltaNet + 1x Full Attention, repeated 8 times |
| **Vocabulary** | 248,320 tokens |
| **Context** | 262K tokens natively |

### DeltaNet Layers (24 of 32)
- Gated linear attention with delta rule state updates
- 16 QK heads + 32 V heads, head_dim=128
- Causal Conv1d (kernel=4) + SiLU activation
- O(1) memory per token (recurrent state, no KV cache needed)

### Full Attention Layers (8 of 32)
- Grouped Query Attention (16Q / 4KV heads), head_dim=256
- QK-norm + partial RoPE (64/256 dims rotated), theta=10M
- Output gating (sigmoid gate on attention output)
- Standard KV cache

### Vision Encoder
- 24-layer ViT, hidden=1024, 16 heads
- Conv3d patch embedding [1024, 3, 2, 16, 16] (temporal_patch_size=2)
- Learned positional embedding with bilinear interpolation from 48x48 grid
- 2D spatial RoPE (dim=32, theta=10000)
- 2x2 spatial merger: LayerNorm → concat → MLP(4096 → 2560)
- **Images**: single frame duplicated along temporal axis
- **Video**: actual Conv3d over consecutive frame pairs (N frames → N/2 temporal patches)

## GPU Support

QORA-4B automatically detects and uses your GPU via the Burn framework's wgpu backend:

- **Windows/Linux**: Vulkan
- **macOS**: Metal (Apple Silicon and Intel)

```
Attempting GPU inference...
GPU initialized successfully
VRAM probe: 256MB OK
Weights loaded to GPU in 5.9s
Prefill: 97 tokens in 21.8s (4.5 tok/s)
Decode: 96 tokens in 29.3s (3.27 tok/s)
```

| Mode | Decode Speed | Prefill Speed |
|------|-------------|---------------|
| **GPU** | ~3.3 tok/s | ~4.5 tok/s |
| **CPU** | ~1.3 tok/s | ~1.9 tok/s |

GPU gives a **~2.5x speedup** over CPU. Use `--cpu` to force CPU-only inference.

**VRAM requirements**: ~2 GB for Q4 weights + cache. Fits in 4+ GB GPUs. Embedding and lm_head stay on CPU (vocab=248K is too large for VRAM).

**GPU prefill optimization**: DeltaNet layers use a hybrid approach — batch all matrix projections on GPU, then run the lightweight sequential state update on CPU. This avoids per-token GPU round-trips and achieves near-optimal throughput.

## Smart System Awareness

QORA-4B detects your system at startup and automatically adjusts generation limits:

```
QORA-4B - Pure Rust Multimodal Inference Engine
System: 16101 MB RAM (8271 MB free), 12 threads
```

| Available RAM | Think Budget | Max Tokens | Behavior |
|---------------|-------------|------------|----------|
| < 4 GB | 128 (cap 256) | 256 (cap 512) | Minimal generation, warning displayed |
| 4-8 GB | 256 (cap 1024) | 512 (cap 1024) | Constrained, warning displayed |
| 8-12 GB | 1024 (cap 2048) | 1024 (cap 2048) | Normal operation |
| >= 12 GB | 2048 (cap 8192) | 2048 (cap 8192) | Full capability |

Hard caps apply even to explicit user values. Supports **Windows** (wmic), **Linux** (/proc/meminfo), and **macOS** (sysctl/vm_stat).

## Weight Formats

| Format | Size | Quality | Speed (GPU) | Speed (CPU) |
|--------|------|---------|-------------|-------------|
| **Q4** (default) | ~3.5 GB | Good | ~3.3 tok/s | ~1.3 tok/s |
| **F16** | ~7.5 GB | Best | — | ~0.5 tok/s |

Q4 uses 4-bit symmetric quantization with group_size=32 and LUT-optimized dequantization. Multi-threaded GEMV/GEMM via rayon for large matrices.

### AVX-512 SIMD Acceleration

On CPUs with AVX-512 support (Intel 11th gen+, AMD Zen 4+), QORA-4B automatically uses hand-written AVX-512 SIMD kernels for a **~2.5x CPU speedup**:

| Kernel | Technique | Speedup |
|--------|-----------|---------|
| **Q4 GEMV** | `permutexvar_ps` 16-entry LUT lookup, nibble extract via `cvtepu8_epi32` | ~2.5x |
| **F16 GEMV** | `cvtph_ps` f16→f32 + `fmadd_ps` FMA accumulation | ~2.5x |
| **DeltaNet state** | Vectorized decay/retrieve/delta/output over 128-dim heads | ~3x |
| **Fused gate+up** | Parallel gate & up SIMD LUT decode in MLP | ~2.5x |

Detection is automatic at runtime — falls back to scalar code on non-AVX-512 CPUs with zero overhead.

## Platform Support

| Platform | Binary | GPU Backend | Status |
|----------|--------|-------------|--------|
| **Windows x86_64** | `qor4b.exe` | Vulkan | Tested |
| **Linux x86_64** | `qor4b` | Vulkan | Supported |
| **macOS aarch64** | `qor4b` | Metal | Supported |

Pre-built binaries are available on the [Releases](https://github.com/qora-protocol/QORA-LLM-4B/releases) page. GPU is auto-detected — falls back to CPU if unavailable.

## Quick Start

1. Download from the [Releases](https://github.com/qora-protocol/QORA-LLM-4B/releases) page:
   - `model.qor4b.part_aa` + `model.qor4b.part_ab` (reassemble below)
   - `tokenizer.json`
   - `qor4b.exe` (Windows) or build from source (Linux/macOS)

2. Reassemble the model file:

**Windows (cmd):**
```cmd
copy /b model.qor4b.part_aa + model.qor4b.part_ab model.qor4b
```
**Linux / macOS:**
```bash
cat model.qor4b.part_aa model.qor4b.part_ab > model.qor4b
```

3. Run:

```bash
# Text generation (auto-detects GPU)
qor4b --prompt "Explain quantum computing" --max-tokens 500

# Force CPU-only
qor4b --prompt "Hello" --cpu

# Image understanding
qor4b --prompt "What's in this image?" --image photo.jpg

# Video understanding (directory of frame images)
qor4b --prompt "What happens in this video?" --video frames_dir/

# Thinking mode (default, extended reasoning)
qor4b --prompt "Solve: integral of x^2 * e^x dx" --think-budget 2048

# No-think mode (faster, direct answers)
qor4b --prompt "What is 2+2?" --no-think

# Greedy decoding (deterministic output)
qor4b --prompt "Hello" --greedy
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Input prompt (default: "Hello, how are you?") |
| `--image PATH` | Path to an image file (PNG/JPG) |
| `--video PATH` | Path to directory of frame images (PNG/JPG, sorted by name) |
| `--max-tokens N` | Max tokens to generate (default: 1024) |
| `--think-budget N` | Max thinking tokens before forcing answer (default: 1024) |
| `--no-think` | Disable thinking mode (direct answers) |
| `--show-think` | Display thinking tokens on stderr |
| `--greedy` | Greedy decoding (temperature=0, not recommended with thinking mode) |
| `--cpu` | Force CPU inference (skip GPU auto-detection) |

### Sampling Defaults

| Parameter | Think mode | No-think mode |
|-----------|-----------|---------------|
| temperature | 1.0 | 0.7 |
| top_k | 20 | 20 |
| top_p | 0.95 | 0.95 |
| presence_penalty | 1.5 | 1.5 |

### Video Input

Video is provided as a directory of frame images (not a video file). Extract frames however you like:

```bash
# Example: extract 4 frames from a video with ffmpeg
ffmpeg -i video.mp4 -vf "select=not(mod(n\,30))" -frames:v 4 frames/frame_%02d.png

# Then run
qor4b --prompt "Describe what happens" --video frames/
```

Frames are loaded in alphabetical order, resized to uniform dimensions (max 768px, divisible by 32), and processed as temporal pairs via Conv3d. Odd frame counts are padded by duplicating the last frame.

## Building from Source

```bash
# CPU-only (all platforms)
cargo build --release

# GPU — Windows/Linux (Vulkan)
cargo build --release --features gpu

# GPU — macOS (Metal)
cargo build --release --features gpu-metal
```

### Dependencies

- **Language**: Pure Rust (2024 edition)
- `cortex` — Rust deep learning framework (GPU via wgpu/Vulkan/Metal backend)
- `rayon` — Thread pool for parallel GEMV, attention, lm_head
- `half` — F16 support
- `image` — Image loading (PNG/JPG)
- `tokenizers` — HuggingFace tokenizer
- `memmap2` — Memory-mapped I/O for converter
- `serde_json` — Config parsing
- **No ML framework** for CPU inference — all matrix ops are hand-written Rust with AVX-512 SIMD
- **Cortex framework** used for GPU tensor operations and binary format types

### Cross-Platform Releases

Pre-built binaries are automatically built via GitHub Actions for:
- **Windows x86_64** — CPU + GPU (Vulkan)
- **Linux x86_64** — CPU + GPU (Vulkan)
- **macOS aarch64** — CPU + GPU (Metal)

Create a git tag (e.g. `v0.1.0`) and push to trigger a release build.

## File Structure

```
src/
  main.rs           — CLI entry point, argument parsing
  config.rs         — Model architecture configuration
  gemv.rs           — GEMV/GEMM kernels (F16 + Q4), hybrid forward pass, batched prefill
  simd.rs           — AVX-512 SIMD kernels (Q4/F16 GEMV, DeltaNet, fused MLP)
  generate.rs       — Text generation loop (text, image, video modes)
  tokenizer.rs      — Tokenizer wrapper and chat templates
  vision.rs         — Vision encoder (ViT + merger), image/video loading
  save.rs           — Binary model format (.qor4b) save/load
  convert.rs        — One-time safetensors → .qor4b converter
  system.rs         — System awareness (RAM detection, smart limits)
  gpu_loader.rs     — CPU → GPU weight conversion (Q4/F16 → Burn tensors)
  gpu_inference.rs  — GPU forward pass (DeltaNet + Full Attention), prefill + decode
  lib.rs            — Module exports
```

## Model Binary Format (.qor4b)

Custom binary format for fast loading:

```
Header:  "QOR4" magic + version(u32) + format(u8: 0=F16, 1=Q4)
Config:  Architecture params (vocab, hidden, layers, heads, etc.)
Layers:  32 layers, each with type byte + layer-specific weights
Global:  Embedding + final norm + precomputed RoPE tables
Vision:  Conv3d patch embed + pos_embed + 24 ViT blocks + merger MLP
```

Loading is ~25s for the Q4 model (~3.5 GB) via buffered sequential reads.

## Performance

Tested on i5-11500 (6C/12T), 16GB RAM, GTX 1660 SUPER (6GB):

| Task | GPU | CPU |
|------|-----|-----|
| Text decode | **~3.3 tok/s** | ~1.3 tok/s (AVX-512) / ~0.54 tok/s (scalar) |
| Text prefill (89 tok) | **~4.5 tok/s** | ~3.9 tok/s (AVX-512) / ~1.9 tok/s (scalar) |
| Image encode (256x256) | — | ~90s |
| Video encode (4 frames) | — | ~180s |
| Model load (Q4) | ~25s | ~25s |
| GPU weight upload | ~6s | — |

## Comparison with QORA-0.8B

| | QORA-4B | QORA-0.8B |
|---|---------|-----------|
| Parameters | 4B | 0.8B |
| Model size (Q4) | 3.5 GB | 600 MB |
| Load time | ~25s | ~500ms |
| Decode speed (GPU) | ~3.3 tok/s | N/A |
| Decode speed (CPU) | ~1.3 tok/s | ~3.9 tok/s |
| RAM usage | ~3.5 GB | ~791 MB |
| GPU support | Yes (Vulkan/Metal) | No (not needed) |
| Vision | 24L ViT (1024) | 12L ViT (768) |
| Best for | Desktop, complex reasoning | Mobile, edge, quick tasks |

