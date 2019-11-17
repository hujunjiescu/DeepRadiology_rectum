import torch, cv2
import math
import numbers
import random, pdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            sample["image"] = img
            sample["label"] = mask
            return sample
        
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            sample["image"] = img
            sample["label"] = mask
            return sample

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        sample["image"] = img
        sample["label"] = mask
        return sample


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        sample["image"] = img
        sample["label"] = mask
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        sample["image"] = img
        sample["label"] = mask
        return sample


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        sample["image"] = img
        sample["label"] = mask
        return sample

class Normalize_divide(object):
    def __init__(self, denominator = 255.0):
        self.denominator = float(denominator)
    
    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        
        sample["image"] = img
        sample["label"] = mask
        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass
    
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.expand_dims(np.array(sample['image']).astype(np.float32), -1).transpose((2, 0, 1))
        mask = np.array(sample['label']).astype(np.float32)
        
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        sample["image"] = img
        sample["label"] = mask
        return sample


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        sample["image"] = img
        sample["label"] = mask
        return sample


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)
                
                sample["image"] = img
                sample["label"] = mask
                return sample

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        sample["image"] = img
        sample["label"] = mask
        return sample

class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            sample["image"] = img
            sample["label"] = mask
            return sample
        
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        sample["image"] = img
        sample["label"] = mask
        return sample

class Padding(object):
    """padding zero to image to match the maximum value of target width or height"""
    def __init__(self, size, fill=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill
    
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        target_h, target_w = self.size
        
        left = top = right = bottom = 0
        doit = False
        if target_w > w:
            delta = target_w - w
            left = delta // 2
            right = delta - left
            doit = True
            
        if target_h > h:
            delta = target_h - h
            top = delta // 2
            bottom = delta - top
            doit = True
        if doit:
            img = ImageOps.expand(img, border=(left, top, right, bottom), fill=self.fill)
            mask = ImageOps.expand(mask, border=(left, top, right, bottom), fill=self.fill)
            
        sample["image"] = img
        sample["label"] = mask
        return sample
    
class RandomSized(object):
    def __init__(self, size, scale_min, scale_max):
        self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.padding = Padding(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        
        scale = random.uniform(self.scale_min, self.scale_max)
        
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])
        
        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample["image"] = img
        sample["label"] = mask
        
        padded = self.padding(sample)
        cropped = self.crop(padded)
        return cropped

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        sample["image"] = img
        sample["label"] = mask
        return sample

class Filter_HU(object):
    def __init__(self, HU_min, HU_max):
        self.HU_min = HU_min
        self.HU_max = HU_max
    
    def __call__(self, sample):
        image = sample["image"].astype(np.int32)
        mask = sample["label"]
        
        np.place(image, image > self.HU_max, self.HU_max)
        np.place(image, image < self.HU_min, self.HU_min)
        
        sample["image"] = image
        sample["label"] = mask
        return sample

class Normalize_maxmin(object):
    def __init__(self, HU_min, HU_max):
        self.HU_min = HU_min
        self.HU_max = HU_max
    
    def __call__(self, sample):
        image = sample["image"]
        mask = sample["label"]
        
        image = (np.array(image) - float(self.HU_min)) / (self.HU_max - self.HU_min)
        
        sample["image"] = image
        sample["label"] = mask
        return sample   
    
class Arr2image(object):
    def __call__(self, sample):
        image = sample["image"]
        mask = sample["label"]
        
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        
        if type(mask) == np.ndarray:
            mask = Image.fromarray(mask)
        
        sample["image"] = image
        sample["label"] = mask
        return sample