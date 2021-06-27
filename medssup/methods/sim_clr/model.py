from typing import Tuple

import torch
from torch import Tensor, nn
from vit_pytorch import ViT


class SimCLRViT(nn.Module):
    def __init__(self, vit: ViT, projection_dim: int):
        super().__init__()

        vit_hidden_features = vit.mlp_head[-1].in_features

        self.transformer = vit
        self.transformer.mlp_head = nn.Identity()
        self.transformer.pool = "cls"

        self.transformer.to_latent = nn.Sequential(
            nn.Linear(vit_hidden_features, vit_hidden_features, bias=False),
            nn.ReLU(),
            nn.Linear(vit_hidden_features, projection_dim, bias=False),
        )

    def forward(self, inputs0: Tensor, inputs1: torch.Tensor) -> Tuple[Tensor, Tensor]:
        projection0 = self.transformer(inputs0)
        projection1 = self.transformer(inputs1)
        return projection0, projection1
