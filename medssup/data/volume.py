import shutil
from pathlib import Path
from typing import List, Literal, Tuple, TypeVar, Union

import numpy as np
import SimpleITK as sitk

VolumeType = TypeVar("VolumeType", bound="Volume")


class IsDicomDirError(Exception):
    pass


class Volume:
    def __init__(self, dicom_path: Union[str, Path]):
        self._dicom_path = str(dicom_path)
        self._file_names = self._get_file_names(str(dicom_path))

    def _get_file_names(self, path: str) -> List[str]:
        reader = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(path)
        if len(file_names) == 0:
            if not Path(path).exists():
                raise FileExistsError(f"Dicom directory you provided does not exist {path}")
            raise IsDicomDirError(f"Is a path a dicom directory? {path}")
        slice_idxs = []
        reader = sitk.ImageFileReader()
        for file_name in file_names:
            reader.SetFileName(file_name)
            reader.ReadImageInformation()
            slice_idxs.append(int(reader.GetMetaData("0020|0013")))  # read slice index in dicom
        file_names = [file_names[i] for i in np.argsort(slice_idxs)]
        return file_names

    def _get_reader(self, type_: Literal["slice", "volume"]) -> sitk.ImageFileReader:
        if type_ == "slice":
            reader = sitk.ImageFileReader()
            reader.SetFileName(self._file_names[0])
            reader.ReadImageInformation()
            return reader
        elif type_ == "volume":
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(self._file_names)
            return reader.Execute()
        else:
            raise ValueError(f"Wrong reader type. Must be slice or volume. Got {type_}")

    def __getitem__(self, item) -> np.ndarray:
        if isinstance(item, int):
            reader = sitk.ImageFileReader()
            reader.SetFileName(self._file_names[item])
        elif isinstance(item, slice):
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(self._file_names[item])
        elif isinstance(item, (list, tuple)):
            reader = sitk.ImageSeriesReader()
            file_names = [self._file_names[i] for i in item]
            reader.SetFileNames(file_names)
        else:
            raise TypeError(f"Wrong indexing type. Must be int, slice or iterable of ints. Got {type(item)}")
        data = sitk.GetArrayFromImage(reader.Execute())
        return data

    def __len__(self) -> int:
        return len(self._file_names)

    @property
    def shape(self) -> Tuple[int, int, int]:
        h, w, _ = self._get_reader("slice").GetSize()
        return len(self), h, w

    @property
    def study_uid(self) -> str:
        return self._get_reader("slice").GetMetaData("0020|000d")  # Study UID

    @property
    def spacing(self) -> np.ndarray:
        return self._get_reader("volume").GetSpacing()

    @property
    def data(self) -> np.ndarray:
        return self[:]

    def get_slice_ids(self) -> List[str]:
        reader = sitk.ImageFileReader()
        slice_ids = []

        for slice_path in self._file_names:
            reader.SetFileName(str(slice_path))
            reader.ReadImageInformation()

            slice_id = reader.GetMetaData("0008|0018")  # SOP Instance UID
            slice_ids.append(slice_id)

        return slice_ids

    def delete_files(self):
        shutil.rmtree(self._dicom_path)
