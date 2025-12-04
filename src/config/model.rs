use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,

    /// Total parameters in the model (all parameters, including inactive experts in MoE)
    pub num_parameters: u64,

    /// Active parameters used during inference (for MoE models with sparse activation)
    /// If not specified, defaults to num_parameters (dense models)
    #[serde(default)]
    pub num_active_parameters: Option<u64>,

    /// Number of transformer layers
    pub num_layers: u32,

    /// Hidden dimension
    pub hidden_dim: u32,

    /// Number of attention heads
    pub num_heads: u32,

    /// Number of KV heads (for GQA/MQA). If not specified, defaults to num_heads (MHA)
    #[serde(default)]
    pub num_kv_heads: Option<u32>,

    /// Maximum sequence length supported
    pub max_seq_len: u32,

    /// KV cache size per token per layer (in bytes)
    /// For GQA: 2 * num_kv_heads * head_dim * bytes_per_param * num_layers
    /// For MHA: 2 * num_heads * head_dim * bytes_per_param * num_layers
    #[serde(skip)]
    pub kv_cache_bytes_per_token: u64,
}

impl ModelConfig {
    /// Get the number of active parameters (defaults to total parameters for dense models)
    pub fn active_parameters(&self) -> u64 {
        self.num_active_parameters.unwrap_or(self.num_parameters)
    }

    /// Calculate and set the KV cache size per token
    pub fn compute_kv_cache_size(&mut self, bytes_per_param: u32) {
        // Use num_kv_heads if specified (GQA/MQA), otherwise use num_heads (MHA)
        let kv_heads = self.num_kv_heads.unwrap_or(self.num_heads);
        let head_dim = self.hidden_dim / self.num_heads;

        // 2 for key and value vectors
        // kv_heads * head_dim gives the total KV dimension per layer
        // Multiply by num_layers for all layers
        self.kv_cache_bytes_per_token =
            2 * kv_heads as u64 * head_dim as u64 * bytes_per_param as u64 * self.num_layers as u64;
    }

    /// Initialize with KV cache size pre-computed
    pub fn with_kv_cache_size(mut self, bytes_per_param: u32) -> Self {
        self.compute_kv_cache_size(bytes_per_param);
        self
    }

    /// Calculate total KV cache size for a sequence
    pub fn kv_cache_size_for_sequence(&self, seq_len: u32) -> u64 {
        self.kv_cache_bytes_per_token * seq_len as u64
    }
}
