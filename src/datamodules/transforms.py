import random
import numpy as np
import torch
import elasticdeform


from skimage.transform import rotate

from monai.transforms.transform import Transform
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils.enums import TransformBackends
from monai.utils.module import look_up_option
from monai.transforms.croppad.array import Pad
from monai.utils import (
    InterpolateMode,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple_rep
)
from typing import Any, List, Optional, Sequence, Tuple, Union
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type


random.seed(260520)

class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class ToTensor:
    def __init__(self, mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target_mask']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            mask = np.transpose(mask, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            sample['input'], sample['target_mask'] = img, mask

        else:  # if self.mode == 'test'
            img = sample['input']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            sample['input'] = img

        return sample        


class Mirroring:
    def __init__(self, p=0.5):
        self.p = p
        random.seed(260520)

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target_mask']

            n_axes = random.randint(0, 3)
            random_axes = random.sample(range(3), n_axes)

            img = np.flip(img, axis=tuple(random_axes))
            mask = np.flip(mask, axis=tuple(random_axes))

            sample['input'], sample['target_mask'] = img.copy(), mask.copy()

        return sample


class NormalizeIntensity:

    def __call__(self, sample):
        img = sample['input']
        img[:, :, :, 0] = self.normalize_ct(img[:, :, :, 0])
        img[:, :, :, 1] = self.normalize_pt(img[:, :, :, 1])

        sample['input'] = img
        return sample

    @staticmethod
    def normalize_ct(img):
        norm_img = np.clip(img, -1024, 1024) / 1024
        return norm_img

    @staticmethod
    def normalize_pt(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-3)


class RandomRotation:

    def __init__(self, p=0.5, angle_range=[5, 15]):
        self.p = p
        self.angle_range = angle_range
        random.seed(260520)

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target_mask']

            num_of_seqs = img.shape[-1]
            n_axes = random.randint(1, 3)
            random_axes = random.sample([0, 1, 2], n_axes)

            for axis in random_axes:

                angle = random.randrange(*self.angle_range)
                angle = -angle if random.random() < 0.5 else angle

                for i in range(num_of_seqs):
                    img[:, :, :, i] = RandomRotation.rotate_3d_along_axis(img[:, :, :, i], angle, axis, 1)

                mask[:, :, :, 0] = RandomRotation.rotate_3d_along_axis(mask[:, :, :, 0], angle, axis, 0)

            sample['input'], sample['target_mask'] = img, mask
        return sample

    @staticmethod
    def rotate_3d_along_axis(img, angle, axis, order):

        if axis == 0:
            rot_img = rotate(img, angle, order=order, preserve_range=True)

        if axis == 1:
            rot_img = np.transpose(img, axes=(1, 2, 0))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(2, 0, 1))

        if axis == 2:
            rot_img = np.transpose(img, axes=(2, 0, 1))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(1, 2, 0))

        return rot_img


class ZeroPadding:

    def __init__(self, target_shape, mode='train'):
        self.target_shape = np.array(target_shape)  # without channel dimension
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target_mask']

            input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
            d_x, d_y, d_z = self.target_shape - input_shape
            d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

            if not all(i == 0 for i in (d_x, d_y, d_z)):
                positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
                negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

                # padding for positive values:
                img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))
                mask = np.pad(mask, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

                # cropping for negative values:
                img = img[: negative[0], : negative[1], : negative[2], :].copy()
                mask = mask[: negative[0], : negative[1], : negative[2], :].copy()

                assert img.shape[:-1] == mask.shape[:-1], f'Shape mismatch for the image {img.shape[:-1]} and mask {mask.shape[:-1]}'

                sample['input'], sample['target_mask'] = img, mask

            return sample

        else:  # if self.mode == 'test'
            img = sample['input']

            input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
            d_x, d_y, d_z = self.target_shape - input_shape
            d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

            if not all(i == 0 for i in (d_x, d_y, d_z)):
                positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
                negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

                # padding for positive values:
                img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

                # cropping for negative values:
                img = img[: negative[0], : negative[1], : negative[2], :].copy()

                sample['input'] = img

            return sample


class ExtractPatch:
    """Extracts a patch of a given size from an image (4D numpy array)."""

    def __init__(self, patch_size, p_tumor=0.5):
        self.patch_size = patch_size  # without channel dimension!
        self.p_tumor = p_tumor  # probs to extract a patch with a tumor

    def __call__(self, sample):
        img = sample['input']
        mask = sample['target_mask']

        assert all(x <= y for x, y in zip(self.patch_size, img.shape[:-1])), \
            f"Cannot extract the patch with the shape {self.patch_size} from  " \
                f"the image with the shape {img.shape}."

        # patch_size components:
        ps_x, ps_y, ps_z = self.patch_size

        if random.random() < self.p_tumor:
            # coordinates of the tumor's center:
            xs, ys, zs, _ = np.where(mask != 0)
            tumor_center_x = np.min(xs) + (np.max(xs) - np.min(xs)) // 2
            tumor_center_y = np.min(ys) + (np.max(ys) - np.min(ys)) // 2
            tumor_center_z = np.min(zs) + (np.max(zs) - np.min(zs)) // 2

            # compute the origin of the patch:
            patch_org_x = random.randint(tumor_center_x - ps_x, tumor_center_x)
            patch_org_x = np.clip(patch_org_x, 0, img.shape[0] - ps_x)

            patch_org_y = random.randint(tumor_center_y - ps_y, tumor_center_y)
            patch_org_y = np.clip(patch_org_y, 0, img.shape[1] - ps_y)

            patch_org_z = random.randint(tumor_center_z - ps_z, tumor_center_z)
            patch_org_z = np.clip(patch_org_z, 0, img.shape[2] - ps_z)
        else:
            patch_org_x = random.randint(0, img.shape[0] - ps_x)
            patch_org_y = random.randint(0, img.shape[1] - ps_y)
            patch_org_z = random.randint(0, img.shape[2] - ps_z)

        # extract the patch:
        patch_img = img[patch_org_x: patch_org_x + ps_x,
                    patch_org_y: patch_org_y + ps_y,
                    patch_org_z: patch_org_z + ps_z,
                    :].copy()

        patch_mask = mask[patch_org_x: patch_org_x + ps_x,
                     patch_org_y: patch_org_y + ps_y,
                     patch_org_z: patch_org_z + ps_z,
                     :].copy()

        assert patch_img.shape[:-1] == self.patch_size, \
            f"Shape mismatch for the patch with the shape {patch_img.shape[:-1]}, " \
                f"whereas the required shape is {self.patch_size}."

        sample['input'] = patch_img
        sample['target_mask'] = patch_mask

        return sample


class InverseToTensor:
    def __call__(self, sample):
        output = sample['output']

        output = torch.squeeze(output)  # squeeze the batch and channel dimensions
        output = output.numpy()

        sample['output'] = output
        return sample


class CheckOutputShape:
    def __init__(self, shape=(144, 144, 144)):
        self.shape = shape

    def __call__(self, sample):
        output = sample['output']
        assert output.shape == self.shape, \
            f'Received wrong output shape. Must be {self.shape}, but received {output.shape}.'
        return sample


class ProbsToLabels:
    def __call__(self, sample):
        output = sample['output']
        output = (output > 0.5).astype(int)  # get binary label
        sample['output'] = output
        return sample

class AdjustContrast(Transform):
    """
    Changes image intensity by gamma. Each pixel/voxel intensity is updated as::
        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min
    Args:
        gamma: gamma value to adjust the contrast as function.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, gamma: float, random=True) -> None:
        if not isinstance(gamma, (int, float)):
            raise ValueError("gamma must be a float or int number.")
        self.gamma = gamma
        self.random = random

    def __call__(self, sample):
        """
        Apply the transform to `img`.
        """
        if self.random:
            self.gamma = np.random.uniform(0.5, 2.0)

        images, mask = sample['input'], sample['target_mask']
        ct_img = images[:,:,:,0]
        pet_img = images[:,:,:,1]
        
        
        epsilon = 1e-7
        img_min = pet_img.min()
        img_range = pet_img.max() - img_min
        
        ret: NdarrayOrTensor = ((pet_img - img_min) / float(img_range + epsilon)) ** self.gamma * img_range + img_min
        img = np.stack([ct_img, ret], axis=-1)

        sample['input'] = img

        return sample
class AdjustContrastCT(Transform):
    """
    Changes image intensity by gamma. Each pixel/voxel intensity is updated as::
        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min
    Args:
        gamma: gamma value to adjust the contrast as function.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, gamma: float, p=0.5, random=True) -> None:
        if not isinstance(gamma, (int, float)):
            raise ValueError("gamma must be a float or int number.")
        self.p = p
        self.gamma = gamma
        self.random = random

    def __call__(self, sample):
        """
        Apply the transform to `img`.
        """
        if random.random() <self.p:
            if self.random:
                self.gamma = np.random.uniform(0.5, 2.0)

            images, mask = sample['input'], sample['target_mask']
            ct_img = images[:,:,:,0]
            pet_img = images[:,:,:,1]
            
            
            epsilon = 1e-7
            img_min = ct_img.min()
            img_range = ct_img.max() - img_min
            
            ret: NdarrayOrTensor = ((ct_img - img_min) / float(img_range + epsilon)) ** self.gamma * img_range + img_min
            img = np.stack([ret, pet_img], axis=-1)

            sample['input'] = img

        return sample

class Zoom(Transform):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        images, mask = sample['input'], sample['target_mask']
        ct_img = images[:,:,:,0]
        pet = images[:,:,:,1]
        mask = mask.squeeze(-1)
        
        zoomed_ct = zoom(ct_img, self.factor)
        zoomed_pet = zoom(pet, self.factor)
        
        img = np.stack([zoomed_ct, zoomed_pet], axis=-1)
        sample['input'] = img


        zoomed_mask = zoom(mask, self.factor)
        zoomed_mask[zoomed_mask<0.5] = 0
        zoomed_mask[zoomed_mask>0] = 1  
        sample['target_mask'] = np.expand_dims(zoomed_mask, axis=-1)

        return sample


def zoom(
    img,
    factor,
    padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
    align_corners: Optional[bool] = True,
    keep_size = True,

) -> NdarrayOrTensor:
    """
    Args:
        img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """
    
    img_t: torch.Tensor
    img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float32)  # type: ignore
    mode = InterpolateMode('bilinear')
    _zoom = ensure_tuple_rep(factor, img.ndim - 1)  # match the spatial image dim
    zoomed: NdarrayOrTensor = torch.nn.functional.interpolate(  # type: ignore
        recompute_scale_factor=True,
        input=img_t.unsqueeze(0),
        scale_factor=list(_zoom),
        mode=look_up_option(mode if mode is None else mode, InterpolateMode).value,
        align_corners=align_corners if align_corners is None else align_corners,
    )
    zoomed = zoomed.squeeze(0)

    if keep_size and not np.allclose(img_t.shape, zoomed.shape):

        pad_vec = [(0, 0)] * len(img_t.shape)
        slice_vec = [slice(None)] * len(img_t.shape)
        for idx, (od, zd) in enumerate(zip(img_t.shape, zoomed.shape)):
            diff = od - zd
            half = abs(diff) // 2
            if diff > 0:  # need padding
                pad_vec[idx] = (half, diff - half)
            elif diff < 0:  # need slicing
                slice_vec[idx] = slice(half, half + od)

        padder = Pad(pad_vec, padding_mode or padding_mode)
        zoomed = padder(zoomed)
        zoomed = zoomed[tuple(slice_vec)]

    out, *_ = convert_to_dst_type(zoomed, dst=img)
    return out



class ElasticDeformation():
    def __init__(self, p = 0.5):
        self.p = p
    def __call__(self, sample):
        images, mask = sample['input'], sample['target_mask']
        ct_img = images[:,:,:,0]
        pet_img = images[:,:,:,1]
        if random.random()<self.p:
            new_ct, new_pet, new_mask = elasticdeform.deform_random_grid([ct_img,pet_img,mask],sigma = random.randint(5, 10), points =  random.randint(1,3),axis=(0, 1, 2))
            new_mask = (new_mask - np.min(new_mask))/(np.max(new_mask) - np.min(new_mask))
            new_mask[new_mask<0.5] = 0
            new_mask[new_mask>0] = 1
            img = np.stack([new_ct, new_pet], axis=-1)
            sample['input'], sample['target_mask'] =img,new_mask
        return sample
