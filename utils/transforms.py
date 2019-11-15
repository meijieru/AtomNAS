"""Additional transforms."""
import math
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms

imagenet_pca = {
    'eigval':
        np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec':
        np.asarray([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
}


class Lighting(object):
    """From https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/transforms.py"""

    def __init__(self,
                 alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterCropPadding(object):
    """Tensorflow style `CenterCrop`.

    This class try to mimic MnasNet's training schedule.

    NOTE: `torchvision.utils.transforms.CenterCropPadding` doesn't take image
    size into consideration when cropping. This behavior also influences
    `torchvision.utils.transforms.RandomResizedCrop`.
    """

    def __init__(self, size, crop_padding=0):
        self.size = size
        self.crop_padding = crop_padding

    def __call__(self, img):
        width, height = img.size
        padded_center_crop_size = int(
            self.size / (self.size + self.crop_padding) * min(width, height))
        return F.center_crop(img, padded_center_crop_size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_padding={1})'.format(
            self.size, self.crop_padding)


class RandomResizedCropPadding(object):
    """Tensorflow style `RandomResizedCrop`.

    This class try to mimic MnasNet's training schedule.

    NOTE: Several differences compared with `torchvision.utils.transforms.RandomResizedCrop`.
    1. `torchvision` use `log_ratio=True`, thus the ratio will bias towards 1.
    2. `torchvision` doesn't try to recover from failure trial.
    In short, `torchvision` will fall back to `CenterCrop` with higher probability.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 min_object_covered=None,
                 ratio=(3. / 4., 4. / 3.),
                 log_ratio=True,
                 interpolation=Image.BILINEAR,
                 max_attempts=10,
                 crop_padding=0):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        assert (scale[0] < scale[1]) and (ratio[0] < ratio[1])

        self.interpolation = interpolation
        self.max_attempts = max_attempts
        self.scale = scale
        self.min_object_covered = min_object_covered or scale[0]
        self.ratio = ratio
        self.log_ratio = log_ratio
        self.crop_resize = transforms.Compose([
            CenterCropPadding(size, crop_padding=crop_padding),
            transforms.Resize(size, interpolation)
        ])

    def get_params(self, img):
        original_width, original_height = img.size
        original_area = original_width * original_height
        min_area, max_area = [original_area * scale for scale in self.scale]
        for attempt in range(self.max_attempts):
            if self.log_ratio:
                log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
                aspect_ratio = math.exp(random.uniform(*log_ratio))
            else:
                aspect_ratio = random.uniform(*self.ratio)

            min_height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))
            if max_height * aspect_ratio > original_width:
                max_height = int(
                    (original_width + 0.5 - 0.0000001) / aspect_ratio)
            max_height = min(max_height, original_height)
            min_height = min(max_height, min_height)
            height = random.randint(min_height, max_height)
            width = int(round(height * aspect_ratio))
            assert width <= original_width

            # try to fix rounding errors
            area = height * width
            if area < min_area:
                height += 1
            if area > max_area:
                height -= 1
            width = int(round(height * aspect_ratio))
            area = height * width

            if area < min_area or area > max_area:
                continue
            if area < self.min_object_covered * original_area:
                continue
            if width > original_width or height > original_height \
                    or width < 0 or height < 0:
                continue

            if width <= original_width and height <= original_height:
                i = random.randint(0, original_height - height)
                j = random.randint(0, original_width - width)
                return i, j, height, width, True
        return None, None, None, None, False

    def __call__(self, img):
        i, j, h, w, success = self.get_params(img)
        if success:
            return F.resized_crop(img, i, j, h, w, self.size,
                                  self.interpolation)
        else:
            return self.crop_resize(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(
            tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(
            tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(self.interpolation)
        format_string += ', crop_padding={0})'.format(self.crop_padding)
        return format_string
