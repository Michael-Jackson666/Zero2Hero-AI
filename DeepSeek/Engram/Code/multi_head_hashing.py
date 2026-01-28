"""
Multi-Head Hashing Module for Engram Architecture

This module implements the core N-gram hashing mechanism that maps N-gram contexts
to embedding table indices using multiple hash functions (hash heads) to reduce
collision probability.

Reference: DeepSeek's "Conditional Memory via Scalable Lookup" Paper (arXiv:2601.07372)

Key equations:
    z_{t,n,k} = φ_{n,k}(g_{t,n})           # Hash index
    e_{t,n,k} = E_{n,k}[z_{t,n,k}]         # Embedding lookup
    e_t = ||_{n=2}^N ||_{k=1}^K e_{t,n,k}  # Concatenation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import math


class MultiHeadHasher(nn.Module):
    """
    Multi-Head Hashing: Maps N-grams to embedding indices using K independent hash functions.
    
    Similar to Bloom Filters, using multiple hash heads reduces collision probability.
    """
    
    def __init__(
        self,
        num_hash_heads: int = 4,
        embedding_table_size: int = 1_000_000,
        seed: int = 42
    ):
        """
        Args:
            num_hash_heads: Number of independent hash functions (K)
            embedding_table_size: Size of each embedding table
            seed: Random seed for hash function initialization
        """
        super().__init__()
        self.num_hash_heads = num_hash_heads
        self.embedding_table_size = embedding_table_size
        
        # Initialize hash function parameters (using polynomial rolling hash)
        # Each hash head has different prime bases and moduli
        torch.manual_seed(seed)
        
        # Prime bases for polynomial hashing
        self.register_buffer(
            'hash_bases', 
            torch.tensor([self._get_prime(i * 100 + 31) for i in range(num_hash_heads)])
        )
        
        # Large prime moduli
        self.register_buffer(
            'hash_moduli',
            torch.tensor([self._get_prime(embedding_table_size + i * 1000) 
                         for i in range(num_hash_heads)])
        )
        
    def _get_prime(self, n: int) -> int:
        """Get a prime number >= n (simple implementation)."""
        def is_prime(x):
            if x < 2:
                return False
            for i in range(2, int(x ** 0.5) + 1):
                if x % i == 0:
                    return False
            return True
        
        while not is_prime(n):
            n += 1
        return n
    
    def forward(
        self,
        ngrams: torch.Tensor,
        head_idx: int
    ) -> torch.Tensor:
        """
        Compute hash indices for a specific hash head.
        
        z_{t,n,k} = φ_{n,k}(g_{t,n})
        
        Uses polynomial rolling hash: hash(g) = sum(g[i] * base^i) mod modulus
        
        Args:
            ngrams: N-gram tensor, shape (batch_size, seq_len, n)
            head_idx: Index of the hash head (k in [0, K-1])
            
        Returns:
            Hash indices, shape (batch_size, seq_len)
        """
        batch_size, seq_len, n = ngrams.shape
        device = ngrams.device
        
        base = self.hash_bases[head_idx].item()
        modulus = self.hash_moduli[head_idx].item()
        
        # Compute polynomial hash
        # hash = g[0] * base^0 + g[1] * base^1 + ... + g[n-1] * base^(n-1)
        powers = torch.tensor([base ** i for i in range(n)], device=device, dtype=torch.long)
        
        # Compute hash values
        hash_values = (ngrams * powers.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        hash_values = hash_values % modulus
        
        # Map to embedding table size
        indices = hash_values % self.embedding_table_size
        
        return indices.long()
    
    def hash_all_heads(
        self,
        ngrams: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hash indices for all hash heads.
        
        Args:
            ngrams: N-gram tensor, shape (batch_size, seq_len, n)
            
        Returns:
            Hash indices for all heads, shape (batch_size, seq_len, num_hash_heads)
        """
        indices = [self.forward(ngrams, k) for k in range(self.num_hash_heads)]
        return torch.stack(indices, dim=-1)


class EngramEmbeddingTable(nn.Module):
    """
    Engram Embedding Tables: E_{n,k}
    
    Stores embeddings for each (N-gram order, hash head) pair.
    Supports efficient lookup and can be offloaded to CPU/SSD for inference.
    """
    
    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 4,
        num_hash_heads: int = 4,
        embedding_table_size: int = 1_000_000,
        embedding_dim: int = 64,
        init_std: float = 0.02
    ):
        """
        Args:
            min_n: Minimum N-gram order (typically 2)
            max_n: Maximum N-gram order (typically 4)
            num_hash_heads: Number of hash heads (K)
            embedding_table_size: Size of each embedding table
            embedding_dim: Dimension of each embedding vector (d_e)
            init_std: Standard deviation for weight initialization
        """
        super().__init__()
        self.min_n = min_n
        self.max_n = max_n
        self.num_hash_heads = num_hash_heads
        self.embedding_dim = embedding_dim
        
        # Create embedding tables for each (n, k) pair
        # E_{n,k} has shape (embedding_table_size, embedding_dim)
        self.embeddings = nn.ModuleDict()
        
        for n in range(min_n, max_n + 1):
            for k in range(num_hash_heads):
                table = nn.Embedding(embedding_table_size, embedding_dim)
                nn.init.normal_(table.weight, mean=0.0, std=init_std)
                self.embeddings[f"E_{n}_{k}"] = table
        
        # Total output dimension after concatenation
        self.total_dim = (max_n - min_n + 1) * num_hash_heads * embedding_dim
        
    def lookup(
        self,
        indices: torch.Tensor,
        n: int,
        k: int
    ) -> torch.Tensor:
        """
        Look up embeddings from table E_{n,k}.
        
        e_{t,n,k} = E_{n,k}[z_{t,n,k}]
        
        Args:
            indices: Hash indices, shape (batch_size, seq_len)
            n: N-gram order
            k: Hash head index
            
        Returns:
            Embeddings, shape (batch_size, seq_len, embedding_dim)
        """
        table = self.embeddings[f"E_{n}_{k}"]
        return table(indices)
    
    def forward(
        self,
        all_indices: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Look up and concatenate embeddings for all (n, k) pairs.
        
        e_t = ||_{n=2}^N ||_{k=1}^K e_{t,n,k}
        
        Args:
            all_indices: Dictionary mapping n -> indices tensor of shape 
                        (batch_size, seq_len, num_hash_heads)
            
        Returns:
            Concatenated embeddings, shape (batch_size, seq_len, total_dim)
        """
        embeddings_list = []
        
        for n in range(self.min_n, self.max_n + 1):
            indices_for_n = all_indices[n]  # (batch_size, seq_len, num_hash_heads)
            
            for k in range(self.num_hash_heads):
                idx = indices_for_n[:, :, k]  # (batch_size, seq_len)
                emb = self.lookup(idx, n, k)  # (batch_size, seq_len, embedding_dim)
                embeddings_list.append(emb)
        
        # Concatenate all embeddings
        return torch.cat(embeddings_list, dim=-1)
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """Calculate memory footprint of embedding tables."""
        num_tables = (self.max_n - self.min_n + 1) * self.num_hash_heads
        params_per_table = next(iter(self.embeddings.values())).weight.numel()
        bytes_per_param = 4  # float32
        
        return {
            "num_tables": num_tables,
            "params_per_table": params_per_table,
            "total_params": num_tables * params_per_table,
            "memory_bytes": num_tables * params_per_table * bytes_per_param,
            "memory_gb": num_tables * params_per_table * bytes_per_param / (1024 ** 3)
        }


class MultiHeadHashRetrieval(nn.Module):
    """
    Complete Multi-Head Hash Retrieval Module.
    
    Combines tokenizer compression, N-gram extraction, hashing, and embedding lookup.
    """
    
    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 4,
        num_hash_heads: int = 4,
        embedding_table_size: int = 1_000_000,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.min_n = min_n
        self.max_n = max_n
        self.num_hash_heads = num_hash_heads
        
        # Multi-head hasher
        self.hasher = MultiHeadHasher(
            num_hash_heads=num_hash_heads,
            embedding_table_size=embedding_table_size
        )
        
        # Embedding tables
        self.embedding_table = EngramEmbeddingTable(
            min_n=min_n,
            max_n=max_n,
            num_hash_heads=num_hash_heads,
            embedding_table_size=embedding_table_size,
            embedding_dim=embedding_dim
        )
        
    def forward(
        self,
        ngrams_dict: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Full retrieval pipeline.
        
        Args:
            ngrams_dict: Dictionary mapping n -> n-gram tensor of shape (batch, seq, n)
            
        Returns:
            Retrieved memory embeddings e_t, shape (batch, seq, total_dim)
        """
        # Step 1: Hash all n-grams
        all_indices = {}
        for n, ngrams in ngrams_dict.items():
            all_indices[n] = self.hasher.hash_all_heads(ngrams)
        
        # Step 2: Look up and concatenate embeddings
        return self.embedding_table(all_indices)
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension of retrieved embeddings."""
        return self.embedding_table.total_dim


if __name__ == "__main__":
    print("=== Multi-Head Hash Retrieval Demo ===\n")
    
    # Configuration
    batch_size = 2
    seq_len = 16
    min_n, max_n = 2, 4
    num_hash_heads = 4
    embedding_table_size = 100_000
    embedding_dim = 64
    
    # Create module
    retrieval = MultiHeadHashRetrieval(
        min_n=min_n,
        max_n=max_n,
        num_hash_heads=num_hash_heads,
        embedding_table_size=embedding_table_size,
        embedding_dim=embedding_dim
    )
    
    # Simulate N-gram inputs (in practice, from NGramExtractor)
    ngrams_dict = {
        n: torch.randint(0, 10000, (batch_size, seq_len, n))
        for n in range(min_n, max_n + 1)
    }
    
    # Forward pass
    memory_embeddings = retrieval(ngrams_dict)
    
    print(f"Input N-gram shapes:")
    for n, ng in ngrams_dict.items():
        print(f"  {n}-gram: {ng.shape}")
    
    print(f"\nOutput embedding shape: {memory_embeddings.shape}")
    print(f"Expected dimension: {retrieval.output_dim}")
    
    # Memory footprint
    print("\nEmbedding Table Memory Footprint:")
    footprint = retrieval.embedding_table.get_memory_footprint()
    for k, v in footprint.items():
        print(f"  {k}: {v}")
