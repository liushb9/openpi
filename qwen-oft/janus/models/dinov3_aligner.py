"""
DINOv3 Aligner Module

This module provides alignment layers to match DINOv3 features with Qwen3VL features.
The aligner uses a two-stage linear projection strategy with layer normalization:
1. Linear1: Align head_dim (dinov3_head_dim -> qwen_head_dim)
2. Linear2: Align num_heads (dinov3_num_heads -> qwen_num_heads)
3. LayerNorm: Stabilize output distribution
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class DINOv3QKVAligner(nn.Module):
    """
    Aligner for DINOv3 QKV states to match Qwen3VL dimensions.
    
    Supports Grouped Query Attention (GQA) where K/V have fewer heads than Q.
    
    Uses a two-stage alignment:
    1. Align head dimension: dinov3_head_dim -> qwen_head_dim
    2. Align number of heads: dinov3_num_heads -> qwen_q_num_heads (for Q) or qwen_kv_num_heads (for K/V)
    3. Apply LayerNorm for stability
    """
    
    def __init__(
        self,
        dinov3_num_heads: int,
        dinov3_head_dim: int,
        qwen_q_num_heads: int,
        qwen_kv_num_heads: int,
        qwen_head_dim: int,
        layer_mapping: Dict[int, int],  # {qwen_layer: dinov3_layer}
    ):
        """
        Args:
            dinov3_num_heads: Number of attention heads in DINOv3
            dinov3_head_dim: Head dimension in DINOv3
            qwen_q_num_heads: Number of query heads in Qwen (e.g., 32)
            qwen_kv_num_heads: Number of key/value heads in Qwen (e.g., 8)
            qwen_head_dim: Head dimension in Qwen
            layer_mapping: Mapping from Qwen layer indices to DINOv3 layer indices
        """
        super().__init__()
        self.dinov3_num_heads = dinov3_num_heads
        self.dinov3_head_dim = dinov3_head_dim
        self.qwen_q_num_heads = qwen_q_num_heads
        self.qwen_kv_num_heads = qwen_kv_num_heads
        self.qwen_head_dim = qwen_head_dim
        self.layer_mapping = layer_mapping
        
        # Create linear layers and layer norms for each (layer, qkv_type) combination
        # 📌 重要：Q 和 K/V 使用不同数量的头（GQA）
        self.linear1_dict = nn.ModuleDict()  # Align head_dim
        self.linear2_dict = nn.ModuleDict()  # Align num_heads
        self.layernorm_dict = nn.ModuleDict()  # Normalize output
        
        for qwen_layer in layer_mapping.keys():
            for qkv_type in ['q', 'k', 'v']:  # query, key, value
                key = f'layer_{qwen_layer}_{qkv_type}'
                
                # Linear1: [dinov3_head_dim] -> [qwen_head_dim]
                self.linear1_dict[key] = nn.Linear(dinov3_head_dim, qwen_head_dim, bias=False)
                
                # Linear2: [dinov3_num_heads] -> [qwen_q_num_heads or qwen_kv_num_heads]
                # Query 使用 qwen_q_num_heads，Key/Value 使用 qwen_kv_num_heads
                target_num_heads = qwen_q_num_heads if qkv_type == 'q' else qwen_kv_num_heads
                self.linear2_dict[key] = nn.Linear(dinov3_num_heads, target_num_heads, bias=False)
                
                # LayerNorm: normalize over qwen_head_dim
                self.layernorm_dict[key] = nn.LayerNorm(qwen_head_dim)
    
    def align_qkv(
        self,
        qkv_tensor: torch.Tensor,
        qwen_layer: int,
        qkv_type: str  # 'query', 'key', or 'value'
    ) -> torch.Tensor:
        """
        Align a single QKV tensor from DINOv3 to Qwen dimensions.
        
        Supports Grouped Query Attention (GQA):
        - Query: aligned to qwen_q_num_heads heads
        - Key/Value: aligned to qwen_kv_num_heads heads
        
        Args:
            qkv_tensor: Shape [batch, dinov3_num_heads, seq_len, dinov3_head_dim]
            qwen_layer: Target Qwen layer index
            qkv_type: One of 'query', 'key', 'value'
        
        Returns:
            Aligned tensor of shape:
            - Query: [batch, qwen_q_num_heads, seq_len, qwen_head_dim]
            - Key/Value: [batch, qwen_kv_num_heads, seq_len, qwen_head_dim]
        """
        # Get the corresponding modules
        qkv_initial = qkv_type[0]  # 'q', 'k', or 'v'
        key = f'layer_{qwen_layer}_{qkv_initial}'
        
        if key not in self.linear1_dict:
            # No aligner for this layer, return original
            return qkv_tensor
        
        linear1 = self.linear1_dict[key]
        linear2 = self.linear2_dict[key]
        layernorm = self.layernorm_dict[key]
        
        # Input shape: [batch, dinov3_num_heads, seq_len, dinov3_head_dim]
        batch_size, dinov3_num_heads, seq_len, dinov3_head_dim = qkv_tensor.shape
        
        # Step 1: Align head_dim with Linear1
        # [batch, dinov3_num_heads, seq_len, dinov3_head_dim] -> [batch, dinov3_num_heads, seq_len, qwen_head_dim]
        qkv_step1 = linear1(qkv_tensor)
        
        # Step 2: Permute to apply Linear2 on num_heads dimension
        # [batch, dinov3_num_heads, seq_len, qwen_head_dim] -> [batch, seq_len, qwen_head_dim, dinov3_num_heads]
        qkv_step2 = qkv_step1.permute(0, 2, 3, 1)
        
        # Step 3: Align num_heads with Linear2
        # [batch, seq_len, qwen_head_dim, dinov3_num_heads] -> [batch, seq_len, qwen_head_dim, qwen_num_heads]
        qkv_step3 = linear2(qkv_step2)
        
        # Step 4: Permute back to correct format
        # [batch, seq_len, qwen_head_dim, qwen_num_heads] -> [batch, seq_len, qwen_num_heads, qwen_head_dim]
        qkv_step4 = qkv_step3.permute(0, 1, 3, 2)
        
        # Step 5: Apply LayerNorm on the last dimension
        qkv_step5 = layernorm(qkv_step4)
        
        # Step 6: Final permute to [batch, qwen_num_heads, seq_len, qwen_head_dim]
        qkv_final = qkv_step5.permute(0, 2, 1, 3)
        
        return qkv_final
    
    def forward(
        self,
        dinov3_qkv_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Align all QKV states from DINOv3 to Qwen dimensions.
        
        Args:
            dinov3_qkv_states: Dict mapping DINOv3 layer indices to QKV dicts
                               Each QKV dict has keys 'query', 'key', 'value'
                               with tensors of shape [batch, dinov3_num_heads, seq_len, dinov3_head_dim]
        
        Returns:
            Aligned QKV states with Qwen layer indices as keys
            Output shapes (GQA):
            - query: [batch, qwen_q_num_heads, seq_len, qwen_head_dim]
            - key: [batch, qwen_kv_num_heads, seq_len, qwen_head_dim]
            - value: [batch, qwen_kv_num_heads, seq_len, qwen_head_dim]
        """
        aligned_qkv_states = {}
        
        for qwen_layer, dinov3_layer in self.layer_mapping.items():
            if dinov3_layer not in dinov3_qkv_states:
                continue
            
            qkv_dict = dinov3_qkv_states[dinov3_layer]
            aligned_qkv = {}
            
            for qkv_type in ['query', 'key', 'value']:
                if qkv_type in qkv_dict:
                    aligned_qkv[qkv_type] = self.align_qkv(
                        qkv_dict[qkv_type],
                        qwen_layer,
                        qkv_type
                    )
            
            aligned_qkv_states[qwen_layer] = aligned_qkv
        
        return aligned_qkv_states
