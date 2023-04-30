# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any
from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import cv2 as cv
from PIL import Image
from skimage.filters import threshold_sauvola
class SauvolaBinarize(object):
    def __init__(self):
        pass

    def __call__(self, pil_img):
        gray_img = pil_img.convert('L')
        gray_array = np.array(gray_img)
        thresh_array = threshold_sauvola(gray_array, window_size=25, k=0.2)
        bin_array = gray_array < thresh_array
        bin_img = Image.fromarray(bin_array.astype(np.uint8) * 255)
        return bin_img
    
class RandomCropNotAllBlack(object):
    def __init__(self, size, pad_if_needed, padding_mode, fill) -> None:
        self.crop = transforms.RandomCrop(size, pad_if_needed=pad_if_needed, padding_mode=padding_mode, fill=0)

    def __call__(self, pil_img: Image) -> Image:
        error_c = 0
        while True:            
            cropped_img = self.crop(pil_img)
            gray_img = cropped_img.convert('L')
            gray_array = np.array(gray_img)
            thresh_array = cv.threshold(gray_array, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
            if thresh_array.mean() > 0.3:
                return cropped_img
            elif error_c > 100:
                return cropped_img
            else:
                error_c += 1

class Scale(object):

    def __init__(self, scale_factors: tuple) -> None:
        self.width_scale_factor = scale_factors[0]
        self.height_scale_factor = scale_factors[1]

    def __call__(self, img: Image):
        width, height = img.size
        scale_factors = [self.width_scale_factor, self.height_scale_factor]
        if scale_factors[0] == 0:
            scale_factors[0] = scale_factors[1]
        if scale_factors[1] == 0:
            scale_factors[1] = scale_factors[0]
        new_width = int(width * scale_factors[0])
        new_height = int(height * scale_factors[1])
        return img.resize((new_width, new_height), Image.Resampling.BICUBIC)

class OtsuBinarize(object):
    def __init__(self):
        pass

    def __call__(self, pil_img):
        gray_img = pil_img.convert('L')
        gray_array = np.array(gray_img)
        thresh_array = cv.threshold(gray_array, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        bin_img = Image.fromarray(thresh_array.astype(np.uint8))
        return bin_img

class Erosion(object):
    def __init__(self, kernel_size):
        self.k = kernel_size
    
    def __call__(self, img):
        img = np.array(img)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.k, self.k))
        img = cv.erode(img, kernel, iterations=1)
        return Image.fromarray(img)
    
class RandomCropLine(object):
    def __init__(self, output_size: tuple):
        self.output_width = output_size[0]
        self.output_height = output_size[1]

    def __call__(self, img):
        img_width, img_height = img.size
        scale_factor = self.output_height / img_height
        crop_length = int(self.output_width / scale_factor)
        crop_start = np.random.randint(0, img_width - crop_length)
        croped_img = img.crop((crop_start, 0, crop_start + crop_length, img_height))
        resized_img = croped_img.resize((self.output_width, self.output_height))
        return resized_img
    
class RandomCropFragment(object):
    pass
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
        if img.mode == 'RGB':
            return img
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
                SauvolaBinarize(),
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
    
class WITransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                RandomCropLine((256, 256)),
                transforms.RandomCrop((224, 224)),
                Bin2RGB(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.5),
                Solarization(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        x = self.transform(sample)
        return x
    
class WITransform2(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
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
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
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
    
class Icdar_2020_WITransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                Scale((0, 4)),
                RandomCropNotAllBlack((224, 224), pad_if_needed=True, padding_mode='constant', fill=0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, img):
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2

def get_transform(transform_type):
    transform_dict = {'SIMCLR': TrainTransform,
                      'GlyphTransform': GlyphTransform,
                      'LinearEvalGlyphTrainTransform': LinearEvalGlyphTrainTransform,
                      'LinearEvalGlyphValTransform': LinearEvalGlyphValTransform,
                      'WITransform': WITransform,
                        'Icdar_2020_WITransform': Icdar_2020_WITransform,
                        'WITransform2': WITransform2,
                      }
    return transform_dict.get(transform_type, None)