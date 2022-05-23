import torch
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        else:
            raise TypeError('Data should be ndarray.')
        return img


class Normalize(object):
    def __init__(self):
        self.mean = [-1.43, -0.36]
        self.std = [3.05, 3.36]

    def __call__(self, img):
        for i in range(2):
            img[i] = (img[i] - self.mean[i]) / self.std[i]
        return img
