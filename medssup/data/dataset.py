import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import albumentations
import numpy as np
import pandas as pd
import torch

from medssup.data.preprocessing import Preprocessing
from medssup.data.volume import Volume


@dataclass
class Labels:
    binary: Optional[torch.Tensor] = None
    multilabel: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        if self.binary is not None:
            self.binary = self.binary.to(device)
        if self.multilabel is not None:
            self.multilabel = self.multilabel.to(device)
        return self

    def as_dict(self) -> Dict[str, torch.Tensor]:
        result = {}
        if self.binary is not None:
            result["binary_label"] = self.binary
        if self.multilabel is not None:
            result["multilabel_label"] = self.multilabel
        return result

    def __len__(self):
        if self.binary is None and self.multilabel is None:
            return 0
        else:
            return len(self.binary) if self.binary is not None else len(self.multilabel)

    def __iter__(self):
        for i in range(len(self)):
            bin_elem = self.binary[i] if self.binary is not None else None
            mul_elem = self.multilabel[i] if self.multilabel is not None else None
            yield bin_elem, mul_elem


class CtichDataset:
    def __init__(
        self,
        data_path: Path,
        annotation_path: Path,
        preprocessing: Preprocessing,
        augmentations: Optional[albumentations.Compose],
        labels: List[str],
        clf_task: Literal["binary", "multilabel", "both"] = "binary",
    ):
        assert clf_task in ["binary", "multilabel", "both"]
        self._data_path = data_path
        self._annotation_df = pd.read_csv(annotation_path, converters={"labels": ast.literal_eval})
        self._preprocessing = preprocessing
        self._augmentations = augmentations
        self._labels = labels
        self._clf_task = clf_task
        self._used_labels = set(labels).intersection(self._annotation_df.columns)

    def __len__(self) -> int:
        return len(self._annotation_df)

    @property
    def labels(self) -> pd.Series:
        """Binary labels of the entire dataset"""
        return self._annotation_df["Hemorrhage"]

    def _get_labels(self, row: pd.Series) -> Labels:
        result = Labels()
        create_binary = self._clf_task == "both" or self._clf_task == "binary"
        create_multilabel = self._clf_task == "both" or self._clf_task == "multilabel"
        if create_binary:
            result.binary = torch.tensor(row["Hemorrhage"]).float()
        if create_multilabel:
            # ['lbl1', 'lbl2', ...] or []
            labels = self._used_labels.intersection(row["labels"])
            if len(labels) == 0:
                label = torch.zeros(len(self._labels))
            else:
                label = [self._labels.index(label) for label in labels]
                # tensor([0, 0, 1, 0, 1, ...])
                label = torch.zeros(len(self._labels)).index_fill_(0, torch.tensor(label), 1)
            result.multilabel = label.float()

        return result

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
        image, label, mask, meta = self._load_annotation(index)

        image, mask = self._preprocessing.preprocess(image, mask)
        image, mask = image.squeeze(0).transpose(1, 2, 0), mask.squeeze(0)

        if self._augmentations:
            augmented = self._augmentations(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image, mask = torch.FloatTensor(image).permute(2, 0, 1), torch.FloatTensor(mask).unsqueeze(0)
        return {"image": image, "label": label, "mask": mask, "meta": meta}

    def _load_annotation(self, index: int) -> Tuple[np.ndarray, Labels, np.ndarray, Dict[str, Any]]:
        row = self._annotation_df.iloc[index]
        meta = {"study_id": row["patient_number"], "slice_id": row["slice_number"], "file_path": row["volume_path"]}
        label = self._get_labels(row)

        volume = Volume(self._data_path / row["volume_path"])
        mask_volume = Volume(self._data_path / row["mask_path"])
        image = volume[int(row["slice_number"])]
        mask = mask_volume[int(row["slice_number"])]
        mask = self._preprocessing.normalize_mask(mask)

        return image, label, mask, meta
