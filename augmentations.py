# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import cv2 as cv
from PIL import Image
from skimage.filters import threshold_sauvola
class Binarize(object):
    def __init__(self):
        pass

    def __call__(self, pil_img):
        gray_img = pil_img.convert('L')
        gray_array = np.array(gray_img)
        thresh_array = threshold_sauvola(gray_array, window_size=75, k=0.2)
        bin_array = gray_array < thresh_array
        # footprint = disk(6)
        # bin_array = closing(bin_array, footprint)
        bin_img = Image.fromarray(bin_array.astype(np.uint8))
        return bin_img
    
class Erosion(object):
    def __init__(self, kernel_size):
        self.k = kernel_size
    
    def __call__(self, img):
        img = np.array(img)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.k, self.k))
        img = cv.erode(img, kernel, iterations=1)
        return Image.fromarray(img)
    
class RandomErosion(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img)
        k = np.random.randint(1, 6)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        img = cv.erode(img, kernel, iterations=1)
        return Image.fromarray(img)
    
class Opening(object):
    def __init__(self, kernel_size):
        self.k = kernel_size
    
    def __call__(self, img):
        img = np.array(img)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.k, self.k))
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return Image.fromarray(img)
    
class Bin2RGB(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        return Image.fromarray(img)

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
    
class GlyphTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomAffine((-10,10), translate=None, scale=(1,1), shear=(-20,20), interpolation=InterpolationMode.BICUBIC, fill=255, center=None),
                transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.2), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.3),
                Solarization(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                Binarize(),
                transforms.RandomAffine((-10,10), translate=None, scale=(1,1), shear=(-20,20), interpolation=InterpolationMode.BICUBIC, fill=255, center=None),
                transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.2), ratio=(0.5, 1.5), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomApply(
                    [
                        Erosion(2),
                    ],
                    p=0.5,
                ),
                Bin2RGB(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2

class LinearEvalGlyphTrainTransform(object):

    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.2), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            GaussianBlur(p=0.3),
            transforms.RandomGrayscale(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    
    def __call__(self, sample):
        x = self.transform(sample)
        return x
class LinearEvalGlyphValTransform(object):
    
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        x = self.transform(sample)
        return x