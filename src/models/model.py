"""MNISTNet: 784 -> hidden -> 10 fully connected network with ReLU activation."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class MNISTNet(nn.Module):
    """
    Fully connected network for MNIST classification.
    Architecture: 784 (input) -> hidden_dim (ReLU) -> 10 (output)
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 16,
        output_dim: int = 10,
        activation: str = "relu",
        bias: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only logits.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784)
            
        Returns:
            Logits of shape (batch, 10)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        hidden = self.activation(self.fc1(x))
        return self.fc2(hidden)

    def forward_with_hidden(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning both logits and hidden activations.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784)
            
        Returns:
            Tuple of (logits, hidden_activations)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        hidden = self.activation(self.fc1(x))
        return self.fc2(hidden), hidden

    def forward_with_all_layers(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning all intermediate representations.
        
        Args:
            x: Input tensor
            
        Returns:
            Dict with keys: 'input', 'fc1_pre_activation', 'fc1_post_activation', 'output'
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        fc1_pre = self.fc1(x)
        fc1_post = self.activation(fc1_pre)
        output = self.fc2(fc1_post)
        return {
            'input': x,
            'fc1_pre_activation': fc1_pre,
            'fc1_post_activation': fc1_post,
            'output': output
        }

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Extract hidden layer activations (post-ReLU)."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.activation(self.fc1(x))

    def get_weight_info(self) -> Dict[str, Any]:
        """Return weight statistics for analysis."""
        with torch.no_grad():
            fc1_w = self.fc1.weight.data
            fc2_w = self.fc2.weight.data
            return {
                'fc1': {
                    'shape': tuple(fc1_w.shape),
                    'mean': fc1_w.mean().item(),
                    'std': fc1_w.std().item(),
                    'min': fc1_w.min().item(),
                    'max': fc1_w.max().item(),
                },
                'fc2': {
                    'shape': tuple(fc2_w.shape),
                    'mean': fc2_w.mean().item(),
                    'std': fc2_w.std().item(),
                    'min': fc2_w.min().item(),
                    'max': fc2_w.max().item(),
                },
                'total_params': sum(p.numel() for p in self.parameters())
            }

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """Save model checkpoint with optional metadata."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'activation': self.activation_name
            }
        }
        if metadata:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None) -> 'MNISTNet':
        """Load model from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        arch = checkpoint['architecture']
        
        model = cls(
            input_dim=arch['input_dim'],
            hidden_dim=arch['hidden_dim'],
            output_dim=arch['output_dim'],
            activation=arch['activation']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)