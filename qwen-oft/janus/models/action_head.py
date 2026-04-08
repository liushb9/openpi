"""
Action Head Module for Continuous Action Prediction

This module contains MLP-based architectures for predicting continuous actions
from transformer hidden states, using L1 regression.
"""

import torch
import torch.nn as nn


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        """
        Args:
            num_blocks: Number of residual blocks
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        action_chunk=1,
    ):
        """
        Args:
            input_dim: Hidden dimension from transformer
            hidden_dim: Hidden dimension for MLP
            action_dim: Action space dimension (e.g., 7 for robotics: 3D position + 3D rotation + gripper)
            action_chunk: Number of action steps to predict (for action chunking)
        """
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.model = MLPResNet(
            num_blocks=2, 
            input_dim=input_dim * action_dim * action_chunk, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim * action_chunk
        )

    def predict_action(self, actions_hidden_states):
        """
        Predict continuous actions from hidden states.
        
        Args:
            actions_hidden_states: Last hidden states of Transformer corresponding to action tokens
                                   Shape: (batch_size, action_chunk * action_dim, hidden_dim)
        
        Returns:
            Predicted actions of shape (batch_size, action_chunk, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        # Flatten the hidden states: (batch_size, action_chunk * action_dim * hidden_dim)
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, -1)
        # Predict actions: (batch_size, action_chunk * action_dim)
        action = self.model(rearranged_actions_hidden_states)
        # Reshape to: (batch_size, action_chunk, action_dim)
        action = action.reshape(batch_size, self.action_chunk, self.action_dim)
        return action
    
    def forward(self, actions_hidden_states):
        """
        Forward pass (alias for predict_action for compatibility).
        
        Args:
            actions_hidden_states: Shape (batch_size, action_chunk * action_dim, hidden_dim)
        
        Returns:
            Predicted actions of shape (batch_size, action_chunk, action_dim)
        """
        return self.predict_action(actions_hidden_states)
