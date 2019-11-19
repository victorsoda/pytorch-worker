from .nlp.JsonFromFiles import JsonFromFilesDataset
from .others.FilenameOnly import FilenameOnlyDataset
from .cv.ImageFromJson import ImageFromJsonDataset
from .cv.Cifar10 import Cifar10Dataset

dataset_list = {
    "ImageFromJson": ImageFromJsonDataset,
    "JsonFromFiles": JsonFromFilesDataset,
    "FilenameOnly": FilenameOnlyDataset,
    "Cifar10": Cifar10Dataset
}
