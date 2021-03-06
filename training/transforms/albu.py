import random

import cv2
import numpy as np
#Library for image augmentation
#Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models
#DualTransform - Transform for segmentation task
#ImageOnlyTransform - Transform applied to image only
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.functional import crop


#INTER_AREA – resampling using pixel area relation
#INTER_CUBIC - a bicubic interpolation over 4×4 pixel neighborhood
def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    """
    This function gets an image and resized it by increasing the quality of the pixels accordingly
    """
    # initialize the height and width of the image based on its shape
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    # check if width is greater than height then scale up the height
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    # check if height is greater than width then scale up the width
    else:
        scale = size / h
        w = w * scale
        h = size
    # to increase the quantity of pixels, so that when we zoom an image, we will see more detail
    interpolation = interpolation_up if scale > 1 else interpolation_down
    # resize the image
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized



#Isotropic scaling is a linear transformation that enlarges (increases) or shrinks (diminishes) objects by a scale factor that is the same in all directions
class IsotropicResize(DualTransform):

    """
    Resize the image so that maximum side is equal to max_size
    interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
    """
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up
    
    #**params - parameter represents all the keyword arguments passed to the function as a dictionary
    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        """
        This function return an image which is resized and increase the quality of the pixels accordingly
        """
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        #INTER_NEAREST - a nearest-neighbor interpolation

        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        """
        This function return the arguments defined in the constructeur
        """
        return ("max_side", "interpolation_down", "interpolation_up")


class Resize4xAndBack(ImageOnlyTransform):
     """
        It resizes the image by scaling it down and resizes back by applying a random interpolation between INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST
     """
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h),
                         interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):
    
    """
        Crop a random part of the input and rescale it to some size if the input mask exists.
        min_max_height ((int, int)): crop size limits
        w2h_ratio (float): aspect ratio of crop
    """
    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists, self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1]))
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"
