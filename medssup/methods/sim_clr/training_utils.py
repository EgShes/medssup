from argparse import Namespace
from itertools import chain
from typing import List, Optional

import albumentations
import cv2
import torch
import torch.nn.functional as F
from albumentations import (
    Compose,
    ElasticTransform,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    OneOf,
    ShiftScaleRotate,
)
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from vit_pytorch import ViT
from vit_pytorch.recorder import Recorder

from medssup.data.dataset import RSNASiameseDataset
from medssup.data.preprocessing import Preprocessing


def get_loaders(
    args: Namespace, preprocessing: Preprocessing, augmentations: Optional[albumentations.Compose]
) -> List[DataLoader]:
    loaders = []
    for set_ in ["train", "val", "test"]:
        dataset = RSNASiameseDataset(
            data_path=args.data_path,
            annotation_path=args.annotations_path / f"{set_}.csv",
            preprocessing=preprocessing,
            augmentations=augmentations,
        )
        loader = DataLoader(
            dataset,
            args.bs,
            shuffle=set_ == "train",
            num_workers=args.num_workers,
            collate_fn=RSNASiameseDataset.collate_fn,
        )
        loaders.append(loader)

    return loaders


def get_augmentations() -> albumentations.Compose:
    augs = Compose(
        [
            HorizontalFlip(),
            ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            ElasticTransform(alpha_affine=30, p=0.3),
            OneOf([GaussNoise(var_limit=(0.019, 0.02), p=1), GaussianBlur(p=1)], p=0.1),
        ]
    )
    return augs


def write_attention_maps(
    model: ViT,
    loader: DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    head2visualize: int = 5,
    n2write: int = 10,
    tag: str = "Val_attn",
    iteration: int = 0,
):
    def _make_figure(img: Tensor, attns: Tensor) -> plt.Figure:
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes[0][0].imshow(img, cmap="gray")
        axes[0][0].set_title("Image")
        for i, attn in enumerate(attns):
            row, col = (i + 1) // 3, (i + 1) % 3
            if row >= rows or col >= cols:
                break
            axes[row, col].imshow(attn, cmap="gray")
            axes[row, col].set_title(f"Attention {i}")
        # remove axis
        [ax.axis("off") for ax in chain.from_iterable(zip(*axes))]
        return fig

    print("Logging attentions")
    model.eval()
    model = Recorder(model)
    cntr = 0
    for batch in loader:
        _, attns = model(batch["pos"].to(device))

        for image, attn in zip(batch["pos"], attns):
            image = image[1].detach().to("cpu")
            attn = attn[head2visualize].detach().to("cpu")
            attn = F.interpolate(attn.unsqueeze(1), image.shape).squeeze(1)
            fig = _make_figure(image, attn)
            writer.add_figure(f"{tag}/{cntr}", fig, iteration)
            if cntr == n2write:
                return
            cntr += 1
