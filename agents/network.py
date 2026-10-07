# agents/networks.py

import torch
import torch.nn as nn


class Network(nn.Module):
    """DQN Network for Q-value estimation.
    
    Args:
        in_dim (int): Input dimension (state size)
        out_dim (int): Output dimension (number of actions)
        view (int): Field of view size (unused, but kept for compatibility)
        hidden_dim (int): Hidden layer dimension
        blind (bool): Whether to use attention mask (for ablation studies)
    """
    
    def __init__(self, in_dim: int, out_dim: int, view, hidden_dim, blind=False):
        """Initialization."""
        super(Network, self).__init__()
        self.h = hidden_dim
        self.blind = blind
        
        # Attention mask for "blind" ablation
        if self.blind: 
            self.mask = nn.Parameter(0.1 + torch.zeros(1, in_dim))
        
        # Neural network layers
        self.layers = nn.Sequential(
            nn.Linear(in_dim, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.
        
        Args:
            x (torch.Tensor): Input state
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        if self.blind: 
            x = x * self.mask  # Apply attention mask
        return self.layers(x)