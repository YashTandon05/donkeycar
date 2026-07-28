import random
import albumentations.core.transforms_interface
import logging
import albumentations as A
import cv2
import numpy as np
from albumentations import GaussianBlur, RandomGamma, GaussNoise, ToGray
from albumentations.augmentations import RandomBrightnessContrast
from albumentations.core.transforms_interface import ImageOnlyTransform


from donkeycar.config import Config


logger = logging.getLogger(__name__)


class RandomShadow(ImageOnlyTransform):
    """ Adds one or more randomly shaped, randomly placed shadows to an
        image, so the model sees the kind of partial, irregular shadows
        (trees, buildings, etc.) that cross the lane at different times of
        day. Only the lightness is reduced (via the HLS color space), so hue
        and saturation - and hence the road / lane colors - are preserved,
        rather than the whole image being darkened. """

    def __init__(self,
                num_shadows_range=(1, 2),
                darkness_range=(0.4, 0.7),
                shadow_dimension=5,
                shadow_roi=(0.0, 0.3, 1.0, 1.0),
                blur_ksize=21,
                p=0.5):
        super().__init__(p=p)
        self.num_shadows_range = num_shadows_range
        self.darkness_range = darkness_range
        self.shadow_dimension = shadow_dimension
        self.shadow_roi = shadow_roi
        self.blur_ksize = blur_ksize

    def apply(self, img, **params):
        # Compatibility guard: shadow simulation reduces lightness via the
        # HLS color space, which requires a 3-channel (RGB/BGR) image.
        # Skip (no-op) on grayscale or other channel counts (e.g. if a
        # config sets IMAGE_DEPTH = 1) instead of letting cv2 raise.
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        height, width = img.shape[:2]
        x_min, y_min, x_max, y_max = self.shadow_roi
        roi_x_min, roi_x_max = int(x_min * width), int(x_max * width)
        roi_y_min, roi_y_max = int(y_min * height), int(y_max * height)

        # Build a single 0..1 mask that can hold several shadow shapes.
        # Each shape is a random, irregular polygon (not a rectangle), so
        # shadows look like real cast shadows rather than a hard box.
        mask = np.zeros((height, width), dtype=np.float32)
        num_shadows = random.randint(*self.num_shadows_range)
        for _ in range(num_shadows):
            vertices = np.array(
                [[random.randint(roi_x_min, roi_x_max),
                 random.randint(roi_y_min, roi_y_max)]
                 for _ in range(self.shadow_dimension)], dtype=np.int32)
            darkness = random.uniform(*self.darkness_range)
            polygon_mask = np.zeros((height, width), dtype=np.float32)
            cv2.fillPoly(polygon_mask, [vertices], 1.0)
            mask = np.maximum(mask, polygon_mask * darkness)

        # Blur the mask edges so the shadow fades into the image instead of
        # appearing as a hard-edged cutout.
        if self.blur_ksize and self.blur_ksize > 1:
            k = self.blur_ksize | 1  # cv2 requires an odd kernel size
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Reduce lightness only. cv2's RGB<->HLS and BGR<->HLS conversions
        # produce the same L channel (it only depends on per-pixel max/min
        # across channels), and we convert back with the matching HLS2RGB
        # code, so this is correct whether img is in RGB or BGR order.
        orig_dtype = img.dtype
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
        hls[:, :, 1] *= (1.0 - mask)
        hls = np.clip(hls, 0, 255).astype(np.uint8)
        shadowed = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        return shadowed.astype(orig_dtype)

    def get_transform_init_args_names(self):
        return ('num_shadows_range', 'darkness_range', 'shadow_dimension',
                'shadow_roi', 'blur_ksize')


class RandomHighPass(ImageOnlyTransform):
    """ Applies a high-pass filter: subtracts a blurred ("low-frequency")
        version of the image from the original, so smooth regions (overall
        color, broad brightness gradients - the things that change a lot
        between different times of day) collapse toward flat mid-gray,
        while edges and texture (lane lines, track boundaries - the things
        that stay comparatively stable across lighting) stand out. This is
        a genuine high-pass filter (image minus its own low-pass/blurred
        version), not a mild sharpening/unsharp-mask effect - a fully
        high-pass image with no original blended back in looks like edges
        on a flat gray background, since flat/low-frequency content is
        removed almost entirely regardless of whether it was originally
        bright, dark, or brightly colored.

        blend_range controls how much of the original image is mixed back
        in on top of the high-pass result (0 = pure high-pass "edges on
        gray", higher = a more subtle edge-enhanced version of the
        original), so this can range from a strong "structure only" effect
        to a mild sharpening-like effect depending on how it's tuned. """

    def __init__(self,
                blur_sigma_range=(3.0, 8.0),
                strength_range=(0.7, 1.3),
                blend_range=(0.0, 0.15),
                p=0.5):
        super().__init__(p=p)
        self.blur_sigma_range = blur_sigma_range
        self.strength_range = strength_range
        self.blend_range = blend_range

    def apply(self, img, **params):
        orig_dtype = img.dtype
        img_f = img.astype(np.float32)

        sigma = random.uniform(*self.blur_sigma_range)
        strength = random.uniform(*self.strength_range)
        blend = random.uniform(*self.blend_range)

        k = int(sigma * 3) | 1  # odd kernel size derived from sigma
        k = max(3, k)
        blurred = cv2.GaussianBlur(img_f, (k, k), sigma)

        # High-pass = original - blurred. Values naturally center around 0,
        # so we scale by strength and re-center at mid-gray (128) to get a
        # valid, viewable uint8 image.
        high_pass = (img_f - blurred) * strength + 128.0

        # Blend a little of the original back in if requested (0 = pure
        # high-pass).
        result = high_pass * (1 - blend) + img_f * blend
        result = np.clip(result, 0, 255).astype(orig_dtype)
        return result

    def get_transform_init_args_names(self):
        return ('blur_sigma_range', 'strength_range', 'blend_range')


class ImageAugmentation:
    def __init__(self, cfg, key, prob=0.5):
        aug_list = getattr(cfg, key, [])
        augmentations = [ImageAugmentation.create(a, cfg, prob)
                         for a in aug_list]
        self.augmentations = A.Compose(augmentations)

    @classmethod
    def create(cls, aug_type: str, config: Config, prob) -> \
            albumentations.core.transforms_interface.BasicTransform:
        """ Augmentation factory. Cropping and trapezoidal mask are
            transformations which should be applied in training, validation
            and inference. Multiply, Blur and similar are augmentations
            which should be used only in training. """

        if aug_type == 'BRIGHTNESS':
            b_limit = getattr(config, 'AUG_BRIGHTNESS_RANGE', 0.2)
            # AUG_CONTRAST_RANGE is optional and defaults to the brightness
            # range, so existing configs that only set AUG_BRIGHTNESS_RANGE
            # keep their exact previous behaviour (brightness and contrast
            # limits equal).
            c_limit = getattr(config, 'AUG_CONTRAST_RANGE', b_limit)
            logger.info(f'Creating augmentation {aug_type} '
                       f'brightness={b_limit} contrast={c_limit}')
            return RandomBrightnessContrast(brightness_limit=b_limit,
                                            contrast_limit=c_limit,
                                            p=prob)

        elif aug_type == 'BLUR':
            b_range = getattr(config, 'AUG_BLUR_RANGE', 3)
            logger.info(f'Creating augmentation {aug_type} {b_range}')
            return GaussianBlur(sigma_limit=b_range, blur_limit=(13, 13),
                                p=prob)

        elif aug_type == 'GAMMA':
            # Nonlinear brightness change. gamma_limit is a percentage
            # around 100: values below 100 brighten the image, values above
            # 100 darken it, so a single range like (60, 160) simulates
            # both glare/overexposure and dim/nighttime conditions without
            # needing separate augmentations for each direction.
            gamma_limit = getattr(config, 'AUG_GAMMA_RANGE', (80, 120))
            gamma_prob = getattr(config, 'AUG_GAMMA_PROBABILITY', prob)
            logger.info(f'Creating augmentation {aug_type} {gamma_limit} '
                       f'p={gamma_prob}')
            return RandomGamma(gamma_limit=gamma_limit, p=gamma_prob)

        elif aug_type == 'NOISE':
            # Simulates sensor grain, most noticeable in low-light /
            # nighttime footage. std_range/mean_range are fractions of the
            # image's max pixel value (e.g. 0.1 ~= 10% of 255 for uint8
            # images).
            std_range = getattr(config, 'AUG_NOISE_STD_RANGE', (0.005, 0.03))
            mean_range = getattr(config, 'AUG_NOISE_MEAN_RANGE', (0.0, 0.0))
            noise_prob = getattr(config, 'AUG_NOISE_PROBABILITY', prob)
            logger.info(f'Creating augmentation {aug_type} std={std_range} '
                       f'p={noise_prob}')

            try:
                # Albumentations 2.x
                return GaussNoise(std_range=std_range,
                                  mean_range=mean_range,
                                  p=noise_prob)
            except TypeError:
                # Albumentations 1.x used in the DSMLP GPU env
                var_limit = tuple((s * 255.0) ** 2 for s in std_range)
                mean = sum(mean_range) / 2.0 * 255.0

                return GaussNoise(var_limit=var_limit,
                                  mean=mean,
                                  p=noise_prob)

        elif aug_type == 'GRAYSCALE':
            # Converts the image to grayscale (dropping hue/saturation) then
            # replicates it back to 3 channels, so the model keeps its
            # normal 3-channel input shape but only ever sees luminance/
            # contrast for this image - color values can shift a lot
            # between different times of day (white balance, sodium vs LED
            # lighting, etc.), so training on some grayscale samples
            # discourages the model from leaning on color as a cue.
            # NOTE: this is a training-time-only AUGMENTATION (applied
            # probabilistically, like BRIGHTNESS/SHADOW/etc.) - it does NOT
            # change what the deployed car sees at inference. If you later
            # want the live camera feed to always be grayscale too (a
            # stronger, permanent form of this same idea), that needs to be
            # done as a TRANSFORMATIONS entry instead, since only
            # TRANSFORMATIONS run at inference as well as training.
            grayscale_prob = getattr(config, 'AUG_GRAYSCALE_PROBABILITY',
                                     prob)
            grayscale_method = getattr(config, 'AUG_GRAYSCALE_METHOD',
                                       'weighted_average')
            logger.info(f'Creating augmentation {aug_type} '
                       f'method={grayscale_method} p={grayscale_prob}')
            return ToGray(num_output_channels=3, method=grayscale_method,
                          p=grayscale_prob)

        elif aug_type == 'SHADOW':
            shadow_prob = getattr(config, 'AUG_SHADOW_PROBABILITY', prob)
            num_shadows_range = getattr(config, 'AUG_SHADOW_COUNT_RANGE',
                                        (1, 2))
            darkness_range = getattr(config, 'AUG_SHADOW_DARKNESS_RANGE',
                                     (0.4, 0.7))
            shadow_dimension = getattr(config, 'AUG_SHADOW_DIMENSION', 5)
            shadow_roi = getattr(config, 'AUG_SHADOW_ROI',
                                 (0.0, 0.3, 1.0, 1.0))
            blur_ksize = getattr(config, 'AUG_SHADOW_BLUR_KSIZE', 21)
            logger.info(f'Creating augmentation {aug_type} '
                       f'darkness={darkness_range} p={shadow_prob}')
            return RandomShadow(num_shadows_range=num_shadows_range,
                                darkness_range=darkness_range,
                                shadow_dimension=shadow_dimension,
                                shadow_roi=shadow_roi,
                                blur_ksize=blur_ksize,
                                p=shadow_prob)

        elif aug_type == 'HIGHPASS':
            # Edge/structure emphasis: subtracts a blurred version of the
            # image from itself, so flat/smooth regions (overall color and
            # brightness - the parts that vary a lot across times of day)
            # collapse toward mid-gray, while edges (lane lines, track
            # boundaries) stand out. See RandomHighPass docstring for
            # details. Training on some high-pass images encourages the
            # model to rely on structure rather than absolute color or
            # brightness.
            blur_sigma_range = getattr(config, 'AUG_HIGHPASS_BLUR_SIGMA_RANGE',
                                       (3.0, 8.0))
            strength_range = getattr(config, 'AUG_HIGHPASS_STRENGTH_RANGE',
                                     (0.7, 1.3))
            blend_range = getattr(config, 'AUG_HIGHPASS_BLEND_RANGE',
                                  (0.0, 0.15))
            highpass_prob = getattr(config, 'AUG_HIGHPASS_PROBABILITY', prob)
            logger.info(f'Creating augmentation {aug_type} '
                       f'sigma={blur_sigma_range} p={highpass_prob}')
            return RandomHighPass(blur_sigma_range=blur_sigma_range,
                                  strength_range=strength_range,
                                  blend_range=blend_range,
                                  p=highpass_prob)


    # Parts interface
    def run(self, img_arr):
        if len(self.augmentations) == 0:
            return img_arr
        aug_img_arr = self.augmentations(image=img_arr)["image"]
        return aug_img_arr

