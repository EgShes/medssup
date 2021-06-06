from pathlib import Path
from typing import List

import numpy as np
import torch

from medssup.data.dataset import CtichDataset
from medssup.data.preprocessing import Preprocessing
from medssup.data.volume import Volume
from medssup.methods.dino.utils import DataAugmentationDINO


class DinoCtichDefaultDataset(CtichDataset):
    def __init__(
        self, data_path: Path, annotation_path: Path, preprocessing: Preprocessing, augmentations: DataAugmentationDINO
    ):
        super().__init__(data_path, annotation_path, preprocessing, augmentations, [], "binary")

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        image = self._load_annotation(index)

        image = self._preprocessing.preprocess(image)
        image = torch.from_numpy(image.squeeze(0))
        crops = self._augmentations(image)

        return crops

    def _load_annotation(self, index: int) -> np.ndarray:
        row = self._annotation_df.iloc[index]

        volume = Volume(self._data_path / row["volume_path"], load_pixels=False)
        image = volume[int(row["slice_number"])]

        return image
