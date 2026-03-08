/// QORA-4B model configuration (hybrid DeltaNet + Full Attention architecture).

#[derive(Debug, Clone)]
pub struct Qor4bConfig {
    // Language model
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attn_heads: usize,      // full attention Q heads
    pub num_kv_heads: usize,        // full attention KV heads
    pub attn_head_dim: usize,       // full attention head dim
    pub num_qk_heads: usize,        // DeltaNet QK heads
    pub num_v_heads: usize,         // DeltaNet V heads
    pub deltanet_head_dim: usize,   // DeltaNet head dim
    pub conv_kernel_size: usize,    // DeltaNet conv1d kernel
    pub intermediate_size: usize,
    pub rope_theta: f64,
    pub partial_rotary_factor: f32, // fraction of head_dim that gets RoPE
    pub rms_norm_eps: f64,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<LayerType>,
    // Vision
    pub has_vision: bool,
    pub vision_hidden: usize,
    pub vision_layers: usize,
    pub vision_heads: usize,
    pub vision_ffn: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    DeltaNet = 0,
    FullAttn = 1,
}

impl Qor4bConfig {
    /// Default config for QORA-4B.
    pub fn default_4b() -> Self {
        // Layer pattern: 3 DeltaNet + 1 FullAttn, repeated 8 times
        let mut layer_types = Vec::with_capacity(32);
        for _ in 0..8 {
            layer_types.push(LayerType::DeltaNet);
            layer_types.push(LayerType::DeltaNet);
            layer_types.push(LayerType::DeltaNet);
            layer_types.push(LayerType::FullAttn);
        }

        Self {
            vocab_size: 248_320,
            hidden_size: 2560,
            num_layers: 32,
            num_attn_heads: 16,
            num_kv_heads: 4,
            attn_head_dim: 256,
            num_qk_heads: 16,
            num_v_heads: 32,
            deltanet_head_dim: 128,
            conv_kernel_size: 4,
            intermediate_size: 9216,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            eos_token_id: 248044, // <|endoftext|> (config.json)
            tie_word_embeddings: true,
            layer_types,
            has_vision: true,
            vision_hidden: 1024,
            vision_layers: 24,
            vision_heads: 16,
            vision_ffn: 4096,
            patch_size: 16,
            spatial_merge_size: 2,
            num_position_embeddings: 2304,
        }
    }

    /// Rotary dimension for full attention (partial_rotary_factor * attn_head_dim).
    pub fn rope_dim(&self) -> usize {
        (self.attn_head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// Number of KV groups for GQA in full attention layers.
    pub fn num_kv_groups(&self) -> usize {
        self.num_attn_heads / self.num_kv_heads
    }

    /// DeltaNet QKV projection output size: Q + K + V.
    pub fn deltanet_qkv_dim(&self) -> usize {
        let q = self.num_qk_heads * self.deltanet_head_dim;
        let k = self.num_qk_heads * self.deltanet_head_dim;
        let v = self.num_v_heads * self.deltanet_head_dim;
        q + k + v
    }

    /// DeltaNet Q dimension.
    pub fn deltanet_q_dim(&self) -> usize {
        self.num_qk_heads * self.deltanet_head_dim
    }

    /// DeltaNet K dimension (same as Q).
    pub fn deltanet_k_dim(&self) -> usize {
        self.num_qk_heads * self.deltanet_head_dim
    }

    /// DeltaNet V dimension.
    pub fn deltanet_v_dim(&self) -> usize {
        self.num_v_heads * self.deltanet_head_dim
    }

    /// Full attention Q dimension.
    pub fn attn_q_dim(&self) -> usize {
        self.num_attn_heads * self.attn_head_dim
    }

    /// Full attention KV dimension.
    pub fn attn_kv_dim(&self) -> usize {
        self.num_kv_heads * self.attn_head_dim
    }

    /// Vision head dimension.
    pub fn vision_head_dim(&self) -> usize {
        self.vision_hidden / self.vision_heads
    }
}
