"""
Complete Engram Module Implementation

This module integrates all components of the Engram architecture into a single
plug-and-play module that can be inserted into any Transformer model.

Reference: DeepSeek's "Conditional Memory via Scalable Lookup" Paper (arXiv:2601.07372)

Architecture Overview:
    1. Tokenizer Compression: x'_t = P(x_t)
    2. N-gram Extraction: g_{t,n} = (x'_{t-n+1}, ..., x'_t)
    3. Multi-Head Hashing: z_{t,n,k} = φ_{n,k}(g_{t,n})
    4. Embedding Lookup: e_{t,n,k} = E_{n,k}[z_{t,n,k}]
    5. Concatenation: e_t = ||_{n,k} e_{t,n,k}
    6. Context-Aware Gating: α_t = σ(h_t · k_t / √d), ṽ_t = α_t * v_t
    7. Fusion: Y = SiLU(Conv1D(RMSNorm(ṽ_t))) + ṽ_t
    8. Residual: H ← H + Y
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

# Import all components
from tokenizer_compression import TokenizerCompressor, NGramExtractor
from multi_head_hashing import MultiHeadHasher, EngramEmbeddingTable
from context_aware_gating import ContextAwareGating, RMSNorm
from fusion import EngramFusion


class EngramConfig:
    """Configuration class for Engram module."""
    
    def __init__(
        self,
        # Vocabulary
        vocab_size: int = 100_000,
        compressed_vocab_size: Optional[int] = None,
        
        # N-gram settings
        min_n: int = 2,
        max_n: int = 4,
        
        # Hashing
        num_hash_heads: int = 4,
        embedding_table_size: int = 1_000_000,
        embedding_dim: int = 64,
        
        # Model dimensions
        hidden_dim: int = 2048,
        
        # Multi-branch (for DeepSeek-V3 style architectures)
        num_branches: int = 1,
        shared_value: bool = True,
        
        # Fusion
        conv_kernel_size: int = 3,
        causal: bool = True,
        dropout: float = 0.0,
        
        # Inference optimization
        offload_to_cpu: bool = False,
        enable_prefetch: bool = True,
    ):
        self.vocab_size = vocab_size
        self.compressed_vocab_size = compressed_vocab_size
        self.min_n = min_n
        self.max_n = max_n
        self.num_hash_heads = num_hash_heads
        self.embedding_table_size = embedding_table_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_branches = num_branches
        self.shared_value = shared_value
        self.conv_kernel_size = conv_kernel_size
        self.causal = causal
        self.dropout = dropout
        self.offload_to_cpu = offload_to_cpu
        self.enable_prefetch = enable_prefetch
        
        # Computed values
        self.num_ngram_orders = max_n - min_n + 1
        self.memory_dim = self.num_ngram_orders * num_hash_heads * embedding_dim


class EngramModule(nn.Module):
    """
    Complete Engram Module: Conditional Memory via Scalable Lookup
    
    A plug-and-play module that can be inserted into Transformer layers
    (typically at Layer 2) to offload knowledge retrieval from neural
    computation to hash-based lookup.
    
    Key Features:
    - O(1) lookup complexity (vs O(N) for neural forward pass)
    - Token-dependent addressing enables prefetching
    - Can offload embedding tables to CPU/SSD
    - Multi-head hashing reduces collision probability
    - Context-aware gating suppresses irrelevant retrievals
    """
    
    def __init__(self, config: EngramConfig):
        super().__init__()
        self.config = config
        
        # 1. Tokenizer Compression
        self.compressor = TokenizerCompressor(
            vocab_size=config.vocab_size,
            compressed_vocab_size=config.compressed_vocab_size
        )
        
        # 2. N-gram Extraction
        self.ngram_extractor = NGramExtractor(max_n=config.max_n)
        
        # 3. Multi-Head Hasher
        self.hasher = MultiHeadHasher(
            num_hash_heads=config.num_hash_heads,
            embedding_table_size=config.embedding_table_size
        )
        
        # 4. Embedding Tables
        self.embedding_table = EngramEmbeddingTable(
            min_n=config.min_n,
            max_n=config.max_n,
            num_hash_heads=config.num_hash_heads,
            embedding_table_size=config.embedding_table_size,
            embedding_dim=config.embedding_dim
        )
        
        # 5. Context-Aware Gating
        self.gating = ContextAwareGating(
            hidden_dim=config.hidden_dim,
            memory_dim=config.memory_dim,
            num_branches=config.num_branches,
            shared_value=config.shared_value
        )
        
        # 6. Fusion Layer
        self.fusion = EngramFusion(
            hidden_dim=config.hidden_dim,
            conv_kernel_size=config.conv_kernel_size,
            causal=config.causal,
            dropout=config.dropout
        )
        
        # CPU offload buffer (for inference)
        self._cpu_embedding_cache = None
        self._prefetch_indices = None
        
    def compute_indices(
        self,
        input_ids: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Pre-compute hash indices for all positions (enables prefetching).
        
        This is the key to zero-overhead prefetching: indices only depend
        on input tokens, not on hidden states.
        
        Args:
            input_ids: Original token IDs, shape (batch, seq_len)
            
        Returns:
            Dictionary mapping n -> hash indices (batch, seq_len, num_hash_heads)
        """
        # Step 1: Compress tokens
        compressed_ids = self.compressor.compress(input_ids)
        
        # Step 2 & 3: Extract N-grams and compute hash indices
        all_indices = {}
        for n in range(self.config.min_n, self.config.max_n + 1):
            ngrams = self.ngram_extractor(compressed_ids, n)
            indices = self.hasher.hash_all_heads(ngrams)
            all_indices[n] = indices
            
        return all_indices
    
    def prefetch_embeddings(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Prefetch embeddings to GPU (call this while computing other layers).
        
        In production, this would be called asynchronously while the GPU
        is computing previous layers.
        
        Args:
            input_ids: Original token IDs
            
        Returns:
            Retrieved memory embeddings e_t
        """
        # Compute indices (can be done on CPU)
        all_indices = self.compute_indices(input_ids)
        
        # Look up embeddings
        memory_embeddings = self.embedding_table(all_indices)
        
        # Cache for later use
        self._prefetch_indices = all_indices
        
        return memory_embeddings
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        memory_embeddings: Optional[torch.Tensor] = None,
        branch_idx: int = 0,
        return_gating: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full Engram forward pass.
        
        Args:
            hidden_states: Transformer hidden states, shape (batch, seq, hidden_dim)
            input_ids: Original token IDs (if memory_embeddings not provided)
            memory_embeddings: Pre-computed memory embeddings (from prefetch)
            branch_idx: Index of the branch (for multi-branch architectures)
            return_gating: Whether to return gating coefficients
            
        Returns:
            Updated hidden states
            Optionally, gating coefficients
        """
        # Step 1-4: Get memory embeddings (either from prefetch or compute now)
        if memory_embeddings is None:
            if input_ids is None:
                raise ValueError("Either input_ids or memory_embeddings must be provided")
            memory_embeddings = self.prefetch_embeddings(input_ids)
        
        # Step 5: Context-aware gating
        gated_values, alpha = self.gating(
            hidden_states=hidden_states,
            memory_embeddings=memory_embeddings,
            branch_idx=branch_idx,
            return_gating=True
        )
        
        # Step 6: Fusion with local convolution
        fused_output = self.fusion(gated_values)
        
        # Step 7: Add to residual stream
        updated_hidden_states = hidden_states + fused_output
        
        if return_gating:
            return updated_hidden_states, alpha
        return updated_hidden_states, None
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for the module."""
        # Embedding table stats
        emb_stats = self.embedding_table.get_memory_footprint()
        
        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "embedding_tables_gb": emb_stats["memory_gb"],
            "total_params_m": total_params / 1e6,
            "trainable_params_m": trainable_params / 1e6,
            "embedding_table_size": self.config.embedding_table_size,
            "num_tables": emb_stats["num_tables"]
        }
    
    def offload_to_cpu(self) -> None:
        """Offload embedding tables to CPU memory."""
        self.embedding_table = self.embedding_table.cpu()
        self.config.offload_to_cpu = True
        
    def load_to_gpu(self, device: torch.device) -> None:
        """Load embedding tables back to GPU."""
        self.embedding_table = self.embedding_table.to(device)
        self.config.offload_to_cpu = False


class EngramTransformerLayer(nn.Module):
    """
    Example: A Transformer layer with integrated Engram module.
    
    This shows how to integrate Engram into an existing Transformer architecture.
    Typically inserted at Layer 2 of the model.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        engram_config: Optional[EngramConfig] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Standard Transformer components
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Engram module (optional)
        self.engram = EngramModule(engram_config) if engram_config else None
        if self.engram:
            self.norm_engram = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        memory_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            attn_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        
        # Engram (if enabled)
        if self.engram is not None:
            residual = hidden_states
            hidden_states = self.norm_engram(hidden_states)
            hidden_states, _ = self.engram(
                hidden_states=hidden_states,
                input_ids=input_ids,
                memory_embeddings=memory_embeddings
            )
            # Note: Engram already adds to residual internally
        
        # FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


if __name__ == "__main__":
    print("=== Complete Engram Module Demo ===\n")
    
    # Configuration
    config = EngramConfig(
        vocab_size=50_000,
        min_n=2,
        max_n=4,
        num_hash_heads=4,
        embedding_table_size=100_000,
        embedding_dim=64,
        hidden_dim=256,
        num_branches=1,
        conv_kernel_size=3,
        causal=True,
        dropout=0.1
    )
    
    print("Engram Configuration:")
    print(f"  Memory dimension: {config.memory_dim}")
    print(f"  N-gram orders: {config.min_n} to {config.max_n}")
    print(f"  Hash heads: {config.num_hash_heads}")
    print(f"  Embedding table size: {config.embedding_table_size:,}")
    
    # Create module
    engram = EngramModule(config)
    
    # Memory stats
    print("\nMemory Statistics:")
    stats = engram.get_memory_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Simulate inputs
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
    
    # Forward pass
    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  hidden_states: {hidden_states.shape}")
    
    # Option 1: Compute everything in forward pass
    updated_states, alpha = engram(
        hidden_states=hidden_states,
        input_ids=input_ids,
        return_gating=True
    )
    
    print(f"\nOutput shapes:")
    print(f"  updated_states: {updated_states.shape}")
    print(f"  gating coefficients: {alpha.shape}")
    print(f"  gating stats: min={alpha.min():.3f}, max={alpha.max():.3f}, mean={alpha.mean():.3f}")
    
    # Option 2: Prefetch embeddings (simulating pipeline)
    print("\n--- Prefetch Demo ---")
    memory_embeddings = engram.prefetch_embeddings(input_ids)
    print(f"Prefetched memory shape: {memory_embeddings.shape}")
    
    # Later, use prefetched embeddings
    updated_states_2, _ = engram(
        hidden_states=hidden_states,
        memory_embeddings=memory_embeddings
    )
    print(f"Using prefetch - output shape: {updated_states_2.shape}")
    
    # Verify both methods give same result
    diff = (updated_states - updated_states_2).abs().max()
    print(f"Difference between methods: {diff:.6f}")
    
    # Transformer layer integration demo
    print("\n=== Transformer Layer Integration Demo ===\n")
    
    layer = EngramTransformerLayer(
        hidden_dim=config.hidden_dim,
        num_heads=8,
        ffn_dim=config.hidden_dim * 4,
        engram_config=config,
        dropout=0.1
    )
    
    output = layer(hidden_states, input_ids=input_ids)
    print(f"Layer output shape: {output.shape}")
    
    # Total parameters
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"Total layer parameters: {total_params / 1e6:.2f}M")
