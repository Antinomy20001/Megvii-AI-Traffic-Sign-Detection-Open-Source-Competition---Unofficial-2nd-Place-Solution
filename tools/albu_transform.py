from megengine.data import transform as T
from megengine.data.transform.vision import functional as F
import albumentations as A
import cv2
from typing import List
import numpy as np
import random
from .dataset import BitmapMasks


def clip_filter_valid_boxes(boxes, hw, overlap_threshold=0.5, size=0):
    box_x1 = np.clip(boxes[:, 0::4], 0, hw[1])
    box_y1 = np.clip(boxes[:, 1::4], 0, hw[0])
    box_x2 = np.clip(boxes[:, 2::4], 0, hw[1])
    box_y2 = np.clip(boxes[:, 3::4], 0, hw[0])

    clip_box = (np.concatenate([box_x1, box_y1, box_x2, box_y2], axis=1))

    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    keep = (width > size) & (height > size)

    origin_area = width * height
    filter_area = (clip_box[:, 2] - clip_box[:, 0]) * (clip_box[:, 3] - clip_box[:, 1])

    overlap = filter_area / (origin_area + 1e-5)

    keep &= (overlap >= overlap_threshold)

    return (clip_box), (keep)


def ignore_category(boxes_category, ignore_class=[]):
    keep = np.ones_like(boxes_category).astype(np.bool)

    for ignore in ignore_class:
        keep &= (boxes_category != ignore)

    return (keep)


class MaskShortestEdgeResize(T.VisionTransform):
    r"""
    Resize the input data with specified shortset edge.
    """

    def __init__(
        self,
        min_size,
        max_size,
        sample_style="range",
        interpolation=cv2.INTER_LINEAR,
        *,
        order=None
    ):
        super().__init__(order)
        if sample_style not in ("range", "choice"):
            raise NotImplementedError(
                "{} is unsupported sample style".format(sample_style)
            )
        self.sample_style = sample_style
        if isinstance(min_size, int):
            min_size = (min_size, min_size)
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def apply(self, input):
        self._shape_info = self._get_shape(self._get_image(input))
        return super().apply(input)

    def _apply_image(self, image):
        h, w, th, tw = self._shape_info
        if h == th and w == tw:
            return image
        return F.resize(image, (th, tw), self.interpolation)

    def _apply_coords(self, coords):
        h, w, th, tw = self._shape_info
        if h == th and w == tw:
            return coords
        coords[:, 0] = coords[:, 0] * (tw / w)
        coords[:, 1] = coords[:, 1] * (th / h)
        return coords

    def _apply_mask(self, mask):
        h, w, th, tw = self._shape_info
        if h == th and w == tw:
            return mask

        out_shape = (th, tw)
        return [mask_.resize(out_shape) for mask_ in mask]

    def _get_shape(self, image):
        h, w, _ = image.shape
        if self.sample_style == "range":
            size = np.random.randint(self.min_size[0], self.min_size[1] + 1)
        else:
            size = np.random.choice(self.min_size)

        scale = size / min(h, w)
        if h < w:
            th, tw = size, scale * w
        else:
            th, tw = scale * h, size
        if max(th, tw) > self.max_size:
            scale = self.max_size / max(th, tw)
            th = th * scale
            tw = tw * scale
        th = int(round(th))
        tw = int(round(tw))
        return h, w, th, tw


class CustomOneOf(T.VisionTransform):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms, p=0.5, order=None):
        super().__init__(order=order)
        self.p = p
        self.transforms = transforms
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def apply(self, **data):
        if self.transforms_ps and random.random() < self.p:
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms.transforms, p=self.transforms_ps)
            data = t.apply(**data)
        return data


class WarpBoxRandomCrop(T.VisionTransform):
    def __init__(self,
                 output_size,
                 ignore_class=[],
                 overlap_threshold=0.8,
                 p=0.5,
                 padding_size=0,
                 padding_value=[0, 0, 0],
                 padding_maskvalue=0,
                 *,
                 order=None):
        super().__init__(order)
        self.ignore_class = ignore_class
        self.overlap_threshold = overlap_threshold
        self.p = p
        self.flag = False

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.pad = T.Pad(padding_size, padding_value, order=self.order)
        self.padding_value = padding_value
        self.padding_maskvalue = padding_maskvalue

    def _apply_mask(self, mask):
        masks = mask[0].masks.transpose(1, 2, 0)
        if self._th > self._h:
            masks = F.pad(masks, (self._th - self._h, 0), self.padding_value)
        if self._tw > self._w:
            masks = F.pad(masks, (0, self._tw - self._w), self.padding_value)
        masks = masks[self._y: self._y + self._th, self._x: self._x + self._tw].transpose(2, 0, 1)
        return [BitmapMasks(masks=masks, height=masks.shape[1], width=masks.shape[2])]

    def _apply_image(self, image):
        if self._th > self._h:
            image = F.pad(image, (self._th - self._h, 0), self.padding_value)
        if self._tw > self._w:
            image = F.pad(image, (0, self._tw - self._w), self.padding_value)
        new_image = image[self._y: self._y + self._th, self._x: self._x + self._tw]
        return new_image

    def _apply_coords(self, coords):
        coords[:, 0] -= self._x
        coords[:, 1] -= self._y
        return coords

    def apply(self, input):
        self.flag = (np.random.random() < self.p)
        
        if not self.flag:
            return input

        self._h, self._w, _ = self._get_image(input).shape
        self._th, self._tw = self.output_size

        input = self.pad.apply(input)

        boxes_id = self.order.index('boxes')
        boxes_category_id = self.order.index('boxes_category')

        boxes_h = (input[boxes_id][:, 3] - input[boxes_id][:, 1])
        boxes_w = (input[boxes_id][:, 2] - input[boxes_id][:, 0])

        center_y = input[boxes_id][:, 1] + boxes_h / 2
        center_x = input[boxes_id][:, 0] + boxes_w / 2

        start_y = center_y - self._th / 2
        start_x = center_x - self._tw / 2
        
        h_diff = random.randint(-(self._th // 3), (self._th // 3))
        w_diff = random.randint(-(self._tw // 3), (self._tw // 3))

        start_y += h_diff
        start_x += w_diff

        start_y = np.minimum(self._h - self._th, start_y)
        start_x = np.minimum(self._w - self._tw, start_x)

        start_y = np.maximum(start_y, 0).astype(np.int)
        start_x = np.maximum(start_x, 0).astype(np.int)

        choice = random.choice(range(input[boxes_id].shape[0]))

        # self._x = np.random.randint(0, max(0, self._w - self._tw) + 1)
        # self._y = self.start_y
        self._x = start_x[choice]
        self._y = start_y[choice]
        output = list(super().apply(input))

        boxes = output[boxes_id]
        boxes_category = output[boxes_category_id]

        clip_boxes, keep = clip_filter_valid_boxes(boxes, [self._th, self._tw])
        keep &= ignore_category(boxes_category, self.ignore_class)

        output[self.order.index('image')] = (output[self.order.index('image')])
        output[boxes_id] = (clip_boxes[keep])
        output[boxes_category_id] = (boxes_category[keep])

        return tuple(output)

class MultiScaleWarpBoxRandomCrop(WarpBoxRandomCrop):
    def __init__(self, output_sizes, ignore_class=[], p=1.0, overlap_threshold=0.8, padding_size=0, padding_value=[0, 0, 0], padding_maskvalue=0, *, order=None):
        super().__init__(output_sizes[0], ignore_class=ignore_class, overlap_threshold=overlap_threshold, p=p, padding_size=padding_size, padding_value=padding_value, padding_maskvalue=padding_maskvalue, order=order)    
        
        self.output_sizes = output_sizes
        delattr(self, 'output_size')

    def apply(self, input):
        self.output_size = random.choice(self.output_sizes)
        return super().apply(input)

def draw(image, boxes):
    img = image
    for box in boxes:
        img = cv2.rectangle(img, box[:2].astype(int), box[2:].astype(int), (255,0,0), 5)
    return img

class ClassAwareRandomHorizontalFlip(T.VisionTransform):
    def __init__(self, p, ignore_class=[], order=None):
        super().__init__(order=order)
        self.ignore_class = ignore_class
        self.p = p
        self.method = A.HorizontalFlip(always_apply=False, p=1.0)
        self.flag = False
        self._w = -1

    def apply(self, input):
        boxes_category_id = self.order.index('boxes_category')
        category_set = set(input[boxes_category_id])
        
        self.flag = not (set(self.ignore_class).issubset(category_set))
        self.flag &= (np.random.random() <= self.p)

        if not self.flag:
            return input

        self._w = self._get_image(input).shape[1]
        output = list(super().apply(input))

        boxes_id = self.order.index('boxes')
        boxes_category_id = self.order.index('boxes_category')
        boxes = output[boxes_id]
        boxes_category = output[boxes_category_id]

        keep = ignore_category(boxes_category, self.ignore_class)

        output[self.order.index('image')] = (output[self.order.index('image')])
        output[boxes_id] = (boxes[keep])
        output[boxes_category_id] = (boxes_category[keep])

        return tuple(output)

    def _apply_image(self, image):
        if self.flag:
            return self.method(image=image)['image']
        else:
            return image

    def _apply_coords(self, coords):
        if self.flag:
            coords[:, 0] = self._w - coords[:, 0]
        return coords


class ConvertRGB(T.VisionTransform):
    def __init__(self, order=None):
        super().__init__(order=order)

    def _apply_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.

    def _apply_coords(self, coords):
        return coords

    def _apply_mask(self, mask):
        return mask

class AlbuTransform(T.VisionTransform):
    '''
    NOTE: only support image transform
    '''

    def __init__(self, method, order=None):
        super().__init__(order=order)
        self.method = method

    def _apply_image(self, image):
        return self.method(image=image)['image']

    def _apply_coords(self, coords):
        return coords

    def _apply_mask(self, mask):
        return mask

class AlbuOneOf(T.VisionTransform):
    '''
    NOTE: only support image transform
    '''

    def __init__(self, transform: List[AlbuTransform], p=1.0, order=None):
        super().__init__(order=order)
        self.method = A.OneOf(transforms=[i.method for i in transform], p=p)

    def _apply_image(self, image):
        return self.method(image=image)['image']

    def _apply_coords(self, coords):
        return coords

    def _apply_mask(self, mask):
        return mask

class AlbuBlur(AlbuTransform):
    def __init__(self, order=None, blur_limit=7, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.Blur(
            blur_limit=blur_limit,
            always_apply=always_apply,
            p=p
        ))


class AlbuDownscale(AlbuTransform):
    def __init__(self, order=None, scale_min=0.25, scale_max=0.25, interpolation=cv2.INTER_NEAREST, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.Downscale(
            scale_min=scale_min,
            scale_max=scale_max,
            interpolation=interpolation,
            always_apply=always_apply,
            p=p
        ))


class AlbuEqualize(AlbuTransform):
    def __init__(self, order=None, mode="cv", by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5):
        super().__init__(order=order, method=A.Equalize(
            mode=mode,
            by_channels=by_channels,
            mask=mask,
            mask_params=mask_params,
            always_apply=always_apply,
            p=p
        ))


class AlbuGaussNoise(AlbuTransform):
    def __init__(self, order=None, var_limit=(10, 50), mean=0, per_channel=True, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.GaussNoise(
            var_limit=var_limit,
            mean=mean,
            per_channel=per_channel,
            always_apply=always_apply,
            p=p
        ))


class AlbuGaussianBlur(AlbuTransform):
    def __init__(self, order=None, blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.GaussianBlur(
            blur_limit=blur_limit,
            sigma_limit=sigma_limit,
            always_apply=always_apply,
            p=p
        ))


class AlbuMotionBlur(AlbuTransform):
    def __init__(self, order=None, blur_limit=7, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.MotionBlur(
            blur_limit=blur_limit,
            always_apply=always_apply,
            p=p
        ))


class AlbuHueSaturationValue(AlbuTransform):
    def __init__(self, order=None, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.HueSaturationValue(
            hue_shift_limit=hue_shift_limit,
            sat_shift_limit=sat_shift_limit,
            val_shift_limit=val_shift_limit,
            always_apply=always_apply,
            p=p
        ))


class AlbuJpegCompression(AlbuTransform):
    def __init__(self, order=None, quality_lower=99, quality_upper=100, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.JpegCompression(
            quality_lower=quality_lower,
            quality_upper=quality_upper,
            always_apply=always_apply,
            p=p
        ))


class AlbuRandomBrightnessContrast(AlbuTransform):
    def __init__(self, order=None, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            brightness_by_max=brightness_by_max,
            always_apply=always_apply,
            p=p
        ))


class AlbuRandomRain(AlbuTransform):
    def __init__(self, order=None, slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.RandomRain(
            slant_lower=slant_lower,
            slant_upper=slant_upper,
            drop_length=drop_length,
            drop_width=drop_width,
            drop_color=drop_color,
            blur_value=blur_value,
            brightness_coefficient=brightness_coefficient,
            rain_type=rain_type,
            always_apply=always_apply,
            p=p,
        ))


class AlbuRandomGamma(AlbuTransform):
    def __init__(self, order=None, gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5):
        super().__init__(order=order, method=A.RandomGamma(
            gamma_limit=gamma_limit,
            eps=eps,
            always_apply=always_apply,
            p=p
        ))

class MaskToMode(T.ToMode):
    def _apply_mask(self, mask):
        return (mask[0].masks, mask[0].height, mask[0].width)