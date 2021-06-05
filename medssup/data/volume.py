from pathlib import Path
from typing import List, Tuple, Union

import SimpleITK as sitk
import SimpleITK
import numpy as np


class IsDicomOrNiftiVolume(Exception):
    pass


class Volume:
    def __init__(self, path: Union[Path, str], load_pixels: bool = False):
        self._volume, self._paths = self._read_volume(Path(path))
        self._data = SimpleITK.GetArrayFromImage(self._volume) if load_pixels else None

    def _read_volume(self, path: Path) -> Tuple[sitk.Image, List[str]]:
        if path.suffix == ".nii":
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            reader.LoadPrivateTagsOn()
            paths = [str(path)]
        elif path.is_dir():
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            reader.SetFileNames(dicom_names)
            paths = dicom_names
        else:
            raise IsDicomOrNiftiVolume(f"Expected '{path}' to be Dicom or Nifti study")
        try:
            image = reader.Execute()
        except RuntimeError as e:
            if "File names information is empty" in str(e):
                raise IsDicomOrNiftiVolume(f"Expected '{path}' to be Dicom or Nifti study")
            else:
                raise e
        return image, paths

    def __getitem__(self, item: int) -> np.ndarray:
        if not isinstance(item, int) and self._data is None:
            raise NotImplementedError("Slice indexing when pixels were not loaded is not implemented in Volume")
        if self._data is None:
            return self._load_slice(item)
        else:
            return self._data[item]

    def _load_slice(self, slice_index: int) -> np.ndarray:
        reader = sitk.ImageFileReader()
        if len(self._paths) == 1:
            reader.SetFileName(self._paths[0])
        else:
            reader.SetFileName(self._paths[slice_index])
            slice_index = 0
        reader.ReadImageInformation()
        volume_shape = reader.GetSize()
        reader.SetExtractIndex((0, 0, slice_index))
        reader.SetExtractSize((*volume_shape[:-1], 1))
        slice_ = reader.Execute()
        slice_ = sitk.GetArrayFromImage(slice_)
        return slice_
