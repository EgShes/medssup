from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

# prevents deadlocks in pytorch dataloaders
cv2.setNumThreads(0)


class BrainWindows:
    STROKE = (40, 40)
    BRAIN = (40, 80)
    SUBDURAL = (100, 254)
    BRAIN_BONE = (600, 2800)


class Preprocessing:

    allowed_windows = ["bb", "br", "je", "st", "su"]
    name2window = {
        "bb": BrainWindows.BRAIN_BONE,
        "br": BrainWindows.BRAIN,
        "st": BrainWindows.STROKE,
        "su": BrainWindows.SUBDURAL,
    }

    def __init__(self, windows_string: str, size_hw: Tuple[int, int]):
        self.init_params = {key: value for key, value in locals().items() if key != "self" and not key.startswith("__")}
        windows = windows_string.split("|")
        assert all(
            [w in self.allowed_windows for w in windows]
        ), f"Unknown windows type. Must be one of {self.allowed_windows}"
        assert windows == sorted(windows), "Windows must be in alphabetic order to prevent mistakes on inference"
        self.jeremy_presents = "je" in windows
        if self.jeremy_presents:
            windows.remove("je")
        self.windows = windows
        self.size_hw = size_hw
        self.level = torch.FloatTensor([self.name2window[window_name][0] for window_name in windows])
        self.width = torch.FloatTensor([self.name2window[window_name][1] for window_name in windows])
        self.level = self.level.view(1, -1, 1, 1)
        self.width = self.width.view(1, -1, 1, 1)

    @torch.no_grad()
    def preprocess(
        self, img: Union[np.ndarray, torch.Tensor], mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[
        Union[torch.Tensor, np.ndarray], Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]
    ]:
        self._check_shape(img)
        self._check_shape(mask)
        self._img_is_torch, self._mask_is_torch = isinstance(img, torch.Tensor), isinstance(mask, torch.Tensor)
        if self._img_is_torch:
            img = img.cpu().numpy()
        if self._mask_is_torch:
            mask = mask.cpu().numpy()

        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, 0)
        if img.ndim == 2:
            img = np.expand_dims(img, 0)

        img = self.resize(img, self.size_hw)
        if mask is not None:
            mask = self.resize(mask, self.size_hw)

        windowed_img = self._apply_windows(img)
        jeremied_img = self._apply_jeremy(img) if self.jeremy_presents else None
        processed_img = self._merge_processed(windowed_img, jeremied_img)

        processed_img = torch.from_numpy(processed_img) if self._img_is_torch else processed_img

        if mask is None:
            return processed_img
        else:
            mask = torch.from_numpy(mask) if self._img_is_torch else mask
            return processed_img, mask

    def resize(self, img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
        h, w = size_hw
        if img.shape[1:] == (h, w):
            return img
        slices = []
        for slice_ in img.astype(np.float):
            slices.append(cv2.resize(slice_, (w, h), interpolation=cv2.INTER_AREA).astype("float32"))
        img = np.stack(slices)
        return img

    def _apply_windows(self, img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        assert (
            3 <= img.ndim <= 4
        ), f"Image must have 3 or 4 dimentions. Got {img.ndim} dimentions with shape {img.shape}"
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 3:
            img = img.unsqueeze(1)
        img = img.expand(-1, len(self.windows), -1, -1)
        w_min = self.level - 0.5 * self.width
        w_max = self.level + 0.5 * self.width
        img = torch.max(img, w_min)
        img = torch.min(img, w_max)
        img = (img - w_min) / (w_max - w_min)
        return img.numpy()

    def _apply_jeremy(self, img: np.ndarray) -> np.ndarray:
        slices = []
        for slice_ in img:
            slices.append(self._apply_jeremy_preprocessing(slice_)[0])
        img = np.stack(slices)
        img = np.expand_dims(img, 1)
        return img

    def _merge_processed(self, windowed_img: np.ndarray, jeremied_img: Optional[np.ndarray]) -> np.ndarray:
        if jeremied_img is None:
            return windowed_img
        else:
            jeremy_idx = sorted(self.windows + ["je"]).index("je")
            parts2cat = [windowed_img[:, :jeremy_idx, :, :], jeremied_img, windowed_img[:, jeremy_idx:, :, :]]
            img = np.concatenate(parts2cat, axis=1)
            return img

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        (h, w), c = self.size_hw, len(self.windows) + int(self.jeremy_presents)
        return c, h, w

    @staticmethod
    def _apply_jeremy_preprocessing(
        img: np.ndarray, pixel_groups: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Jeremy preprocessing
        explanation: https://www.kaggle.com/jhoward/don-t-see-like-a-radiologist-fastai

        Arguments:
            img : Input image, array of some shape
            pixel_groups : Precomputed groups of pixel intensity. Each group has around the same number of pixels
        """

        def make_semiequal_groups(img: np.ndarray, n_groups: int = 1000) -> np.ndarray:
            """
            A function to split the range of pixel values into groups,
            such that each group has around the same number of pixels
            """
            imsd = np.sort(img.flatten())
            t = np.array([0.001])
            t = np.append(t, np.arange(n_groups) / n_groups + (1 / 2 / n_groups))
            t = np.append(t, 0.999)
            t = (len(imsd) * t + 0.5).astype(int)
            return np.unique(imsd[t])

        if pixel_groups is None:
            pixel_groups = make_semiequal_groups(img)
        ys = np.linspace(0.0, 1.0, len(pixel_groups))
        x = np.interp(img.flatten(), pixel_groups, ys)
        return x.reshape(img.shape).clip(0.0, 1.0).astype(np.float32), pixel_groups

    @staticmethod
    def normalize_mask(mask: np.ndarray) -> np.ndarray:
        return (mask - mask.min()) / (mask.max() - mask.min() + 1e-12)

    def _check_shape(self, inputs: Optional[Union[np.ndarray, torch.Tensor]]):
        if inputs is not None:
            assert (
                2 <= inputs.ndim <= 3
            ), f"Inputs must have 2 or 3 dimensions. Got {inputs.ndim} dimensions with shape {inputs.shape}"
