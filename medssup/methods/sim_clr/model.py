from typing import Tuple

import torch
from torch import Tensor, nn
from vit_pytorch import ViT


#  TODO make vit support inputs with num channels != 3
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

    def forward(self, inputs: Tensor) -> Tensor:
        return self.transformer(inputs)

    @classmethod
    def from_scratch(
        cls,
        image_size: int,
        patch_size: int,
        trf_dim: int,
        trf_depth: int,
        trf_heads: int,
        trf_dropout: float,
        trf_emb_dropout: float,
        projection_dim: int,
    ):
        vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=trf_dim,
            depth=trf_depth,
            heads=trf_heads,
            dropout=trf_dropout,
            emb_dropout=trf_emb_dropout,
            num_classes=1,  # unused
            mlp_dim=1024,  # unused
        )
        return cls(vit, projection_dim)
