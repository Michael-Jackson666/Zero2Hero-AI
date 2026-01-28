"""
Tokenizer Compression Module for Engram Architecture

This module implements the tokenizer compression step that normalizes semantically
equivalent tokens (e.g., case variations, leading spaces) to reduce vocabulary size
and improve N-gram lookup accuracy.

Reference: DeepSeek's "Conditional Memory via Scalable Lookup" Paper (arXiv:2601.07372)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class TokenizerCompressor:
    """
    Tokenizer Compression: x'_t = P(x_t)
    
    Maps original token IDs to canonical IDs by normalizing semantically equivalent tokens.
    This step can reduce effective vocabulary size by ~23%.
    """
    
    def __init__(self, vocab_size: int, compressed_vocab_size: Optional[int] = None):
        """
        Args:
            vocab_size: Original vocabulary size V
            compressed_vocab_size: Compressed vocabulary size V' (if None, learned from data)
        """
        self.vocab_size = vocab_size
        self.compressed_vocab_size = compressed_vocab_size or vocab_size
        
        # Mapping table: V -> V'
        # Initialize as identity mapping, then learn/build the compression
        self.mapping = torch.arange(vocab_size)
        
        # Reverse mapping for analysis (optional)
        self.reverse_mapping: Dict[int, List[int]] = {}
        
    def build_compression_table(
        self, 
        tokenizer,
        normalization_rules: Optional[List[str]] = None
    ) -> None:
        """
        Build the compression mapping based on normalization rules.
        
        Common normalization rules:
        - 'case': Ignore case differences (Apple -> apple)
        - 'space': Ignore leading space (_Apple -> Apple) 
        - 'unicode': Normalize unicode variants
        
        Args:
            tokenizer: The tokenizer with decode capability
            normalization_rules: List of normalization rules to apply
        """
        if normalization_rules is None:
            normalization_rules = ['case', 'space']
            
        # Dictionary to group equivalent tokens
        canonical_to_ids: Dict[str, List[int]] = {}
        
        for token_id in range(self.vocab_size):
            try:
                # Decode token to string
                token_str = tokenizer.decode([token_id])
                
                # Apply normalization
                canonical = self._normalize(token_str, normalization_rules)
                
                if canonical not in canonical_to_ids:
                    canonical_to_ids[canonical] = []
                canonical_to_ids[canonical].append(token_id)
                
            except Exception:
                # Handle special tokens that can't be decoded
                canonical_to_ids[f"__special_{token_id}__"] = [token_id]
        
        # Build mapping: all tokens in a group map to the same canonical ID
        new_id = 0
        for canonical, token_ids in canonical_to_ids.items():
            for orig_id in token_ids:
                self.mapping[orig_id] = new_id
            self.reverse_mapping[new_id] = token_ids
            new_id += 1
            
        self.compressed_vocab_size = new_id
        print(f"Vocabulary compressed: {self.vocab_size} -> {self.compressed_vocab_size} "
              f"({100 * (1 - self.compressed_vocab_size / self.vocab_size):.1f}% reduction)")
    
    def _normalize(self, token_str: str, rules: List[str]) -> str:
        """Apply normalization rules to a token string."""
        result = token_str
        
        if 'space' in rules:
            # Remove leading special space characters (common in BPE)
            result = result.lstrip(' \u0120\u2581')  # Various space representations
            
        if 'case' in rules:
            result = result.lower()
            
        if 'unicode' in rules:
            import unicodedata
            result = unicodedata.normalize('NFKC', result)
            
        return result if result else token_str  # Fallback to original if empty
    
    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply compression mapping: x'_t = P(x_t)
        
        Args:
            token_ids: Original token IDs, shape (batch_size, seq_len)
            
        Returns:
            Compressed token IDs, shape (batch_size, seq_len)
        """
        device = token_ids.device
        mapping = self.mapping.to(device)
        return mapping[token_ids]
    
    def get_compression_ratio(self) -> float:
        """Return the vocabulary compression ratio."""
        return 1 - self.compressed_vocab_size / self.vocab_size


class NGramExtractor(nn.Module):
    """
    Extract N-gram suffixes from compressed token sequences.
    
    For position t, constructs suffix N-gram: g_{t,n} = (x'_{t-n+1}, ..., x'_t)
    """
    
    def __init__(self, max_n: int = 4):
        """
        Args:
            max_n: Maximum N-gram order (typically 2 to N where N=4)
        """
        super().__init__()
        self.max_n = max_n
        
    def forward(
        self, 
        compressed_ids: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Extract n-gram suffixes for each position.
        
        Args:
            compressed_ids: Compressed token IDs, shape (batch_size, seq_len)
            n: N-gram order (e.g., 2 for bigram, 3 for trigram)
            
        Returns:
            N-gram tensor, shape (batch_size, seq_len, n)
            Position t contains tokens [t-n+1, ..., t]
            Padded with 0 for positions < n-1
        """
        assert 2 <= n <= self.max_n, f"N-gram order must be in [2, {self.max_n}]"
        
        batch_size, seq_len = compressed_ids.shape
        device = compressed_ids.device
        
        # Create output tensor
        ngrams = torch.zeros(batch_size, seq_len, n, dtype=torch.long, device=device)
        
        # Fill in the n-gram for each position
        for i in range(n):
            # Position in the n-gram (0 = oldest, n-1 = current)
            offset = n - 1 - i
            if offset == 0:
                ngrams[:, :, i] = compressed_ids
            else:
                # Shift and pad
                ngrams[:, offset:, i] = compressed_ids[:, :-offset]
                
        return ngrams
    
    def extract_all_ngrams(
        self, 
        compressed_ids: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Extract all n-grams from order 2 to max_n.
        
        Args:
            compressed_ids: Compressed token IDs
            
        Returns:
            Dictionary mapping n -> n-gram tensors
        """
        return {n: self.forward(compressed_ids, n) for n in range(2, self.max_n + 1)}


if __name__ == "__main__":
    # Demo usage
    print("=== Tokenizer Compression Demo ===\n")
    
    # Simulate a small vocabulary
    vocab_size = 1000
    compressor = TokenizerCompressor(vocab_size)
    
    # Simulate compression (in practice, use real tokenizer)
    # Here we just demonstrate the interface
    batch_size, seq_len = 2, 10
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Original tokens shape: {token_ids.shape}")
    compressed = compressor.compress(token_ids)
    print(f"Compressed tokens shape: {compressed.shape}")
    
    # N-gram extraction
    extractor = NGramExtractor(max_n=4)
    
    for n in [2, 3, 4]:
        ngrams = extractor.forward(compressed, n)
        print(f"\n{n}-gram shape: {ngrams.shape}")
        print(f"Sample {n}-gram at position 5: {ngrams[0, 5, :]}")
