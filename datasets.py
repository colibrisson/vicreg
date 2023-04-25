import re
import random
import pathlib
from typing import Callable

import torch
import torchvision
import PIL as pil

class ImgPair(torchvision.datasets.VisionDataset):
    """Dataset of that returns a pair of images from the same folder."""

    def __init__(self, root: str, transforms: Callable) -> None:
        super().__init__(root)
        self.root = pathlib.Path(root)
        self.transforms = transforms

        img_by_folder_l = []
        for folder_path in self.root.iterdir():
            img_path_l = [img_path for img_path in folder_path.iterdir() if re.match(r'.*\.(jpg|png|jpeg)', img_path.name)]
            if len(img_path_l) > 1:
                img_by_folder_l.append(img_path_l)

    def __getitem__(self, index: int) -> torch.Tensor:
        folder_idx = random.randint(0, len(self.img_by_folder_l) - 1)
        img_path_l = self.image_folder_path_l[folder_idx]
        img_1_path, img_2_path = random.sample(img_path_l, 2)
        img_1 = pil.Image.open(img_1_path)
        img_2 = pil.Image.open(img_2_path)
        view_1 = self.transforms.transform(img_1)
        view_2 = self.transforms.transform(img_2)
        return view_1, view_2
    
    def __len__(self) -> int:
        return 100_000


def get_dataset(dataset_type: str) -> torchvision.datasets.VisionDataset:
    dataset_dict = {'ImgPair': ImgPair,
                    'ImageFolder': torchvision.datasets.ImageFolder,}
    return dataset_dict.get(dataset_type, None)