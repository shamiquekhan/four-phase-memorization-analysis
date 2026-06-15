import torch
import torch.nn as nn
from typing import Optional, Dict


class CIFARNet(nn.Module):
    """
    Three-layer fully connected network for CIFAR-10.
    Input: 3072 (32x32x3 flattened)
    Architecture: 3072 -> 256 -> 128 -> 10
    Parameter count: ~820k
    """
    def __init__(self, hidden1=256, hidden2=128, num_classes=10, bias=True):
        super().__init__()
        self.input_dim = 3072
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = num_classes

        self.fc1 = nn.Linear(3072, hidden1, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2, bias=bias)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes, bias=bias)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h1_post = self.relu1(self.fc1(x))
        h2_post = self.relu2(self.fc2(h1_post))
        return self.fc3(h2_post)

    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.view(x.size(0), -1)
        h1_pre = self.fc1(x)
        h1_post = self.relu1(h1_pre)
        h2_pre = self.fc2(h1_post)
        h2_post = self.relu2(h2_pre)
        out = self.fc3(h2_post)
        return {
            'input':    x,
            'fc1_pre':  h1_pre,
            'fc1_post': h1_post,
            'fc2_pre':  h2_pre,
            'fc2_post': h2_post,
            'output':   out,
        }

    def forward_with_all_layers(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.get_activations(x)

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))

    def get_weight_info(self) -> Dict:
        with torch.no_grad():
            return {
                'fc1': {'shape': tuple(self.fc1.weight.shape)},
                'fc2': {'shape': tuple(self.fc2.weight.shape)},
                'fc3': {'shape': tuple(self.fc3.weight.shape)},
                'total_params': sum(p.numel() for p in self.parameters())
            }

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'architecture': {
                'hidden1': self.hidden1,
                'hidden2': self.hidden2,
                'output_dim': self.output_dim,
            }
        }
        if metadata:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = None) -> 'CIFARNet':
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        arch = checkpoint['architecture']
        model = cls(
            hidden1=arch['hidden1'],
            hidden2=arch['hidden2'],
            num_classes=arch['output_dim'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
