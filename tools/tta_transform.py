from megengine.data import transform as T
import albumentations as A
from megengine.data.transform.vision import functional as F
import numpy as np
import random
import math
import itertools

class TTAOriginScaleMultiCropForY(T.VisionTransform):
    def __init__(self, y_min, y_max, output_size, order=None):
        super().__init__(order=order)
        self.y_min = y_min # ratio
        self.y_max = y_max # ratio
        self.output_size = output_size  # (h, w)

    def _apply_image(self, image):
        _h, _w = image.shape[:2]
        start_y, end_y = int(self.y_min * _h), int(self.y_max * _h)
        end_y = max(end_y, start_y + self.output_size[0])
        x_value = [(x, x + self.output_size[1]) for x in range(0, _w, self.output_size[1])][:-1] + [(_w - self.output_size[1], _w)]
        y_value = [(y, y + self.output_size[0]) for y in range(start_y, end_y, self.output_size[0])][:-1] + [(end_y - self.output_size[0], end_y)]
        return_images = [(image, (0, 0))] + [(image[y1: y2, x1: x2, :], (x1, y1)) for (x1, x2), (y1, y2) in list(itertools.product(x_value, y_value))]
        return return_images

class MultiScaleCrop(T.VisionTransform):
    def __init__(self, output_sizes, use_origin_image=True, order=None):
        super().__init__(order=order)
        self.output_sizes = output_sizes  # (h, w)
        self.use_origin_image = use_origin_image
    
    def _apply_image_single_scale(self, image, output_size):
        _h, _w = image.shape[:2]
        
        x_value = [(x, x + output_size[1]) for x in range(0, _w, output_size[1])][:-1] + [(_w - output_size[1], _w)]
        y_value = [(y, y + output_size[0]) for y in range(0, _h, output_size[0])][:-1] + [(_h - output_size[0], _h)]
        return_images = [(image[y1: y2, x1: x2, :], (x1, y1)) for (x1, x2), (y1, y2) in list(itertools.product(x_value, y_value))]
        return return_images
    
    def _apply_image(self, image):
        return_images = [(image, (0, 0))] if self.use_origin_image else []
        for output_size in self.output_sizes:
            return_images.extend(self._apply_image_single_scale(image, output_size))
        
        return return_images