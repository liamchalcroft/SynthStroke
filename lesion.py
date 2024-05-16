import torch
from monai.transforms.transform import MapTransform
from monai.utils.enums import TransformBackends
from typing import (
    Dict,
    Hashable,
    Mapping,
)
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import KeysCollection
from copy import deepcopy
from random import shuffle
from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    Compose,
    ToTensorD,
    GaussianSmooth,
    ResizeWithPadOrCropD,
    RandBiasField,
    ScaleIntensityRange
)


def paste_lesion(label, seg):
    return torch.cat([label * (1.0 - seg), seg])


def dilate(label):
    return torch.nn.functional.conv3d(
        label[None],
        torch.ones((1, 1, 3, 3, 3), dtype=label.dtype, device=label.device) / (3**3),
        padding=2,
    )[0]


def erode(label):
    return 1.0 - torch.nn.functional.conv3d(
        1.0 - label[None],
        torch.ones((1, 1, 3, 3, 3), dtype=label.dtype, device=label.device) / (3**3),
        padding=0,
    )[0]


def mask_fill(label):
    label = dilate(label)
    label = (label > 0.5).float()
    label = erode(label)
    return label


def softcp(label, k_erode=1, k_dilate=1, alpha=0.9):
    """
    Implemented from https://arxiv.org/abs/2203.10507
    """
    for i in range(k_erode):
        label = erode(label)

    for j in range(k_dilate):
        label = (label > 1e-5).float()
        label = dilate(label)
        label = (1.0 - label) * (label * alpha ** (j + 1)) + label

    return label


class LesionPasteD(MapTransform):
    """ """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        new_keys: str = "seg",
        label_list: list = [],
        smooth_lesion=True,
        fill_lesion=False,
        allow_missing_keys=False,
        fs_healthy=False,
        mb_healthy=False,
        lesion_fading=False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_list = label_list
        self.new_keys = new_keys
        self.smooth = smooth_lesion
        self.fill = fill_lesion
        self.fs_healthy = fs_healthy
        self.mb_healthy = mb_healthy
        self.lesion_fade = Compose(
            transforms=[
            RandBiasField(prob=1, coeff_range=(2, 3), degree=7),
            ScaleIntensityRange(a_min=0, a_max=1, b_min=0, b_max=1, clip=True),
            ]
        ) if lesion_fading else None
        self.load = Compose(
            transforms=[
                LoadImageD(keys=["label"], image_only=True),
                EnsureChannelFirstD(keys=["label"]),
            ]
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        shuffle(self.label_list)

        for i, key in enumerate(self.key_iterator(d)):
            pth = {"label": self.label_list[i]}
            lesion = self.load(pth)
            lesion = ToTensorD(
                keys=["label"], dtype=d[key].dtype, device=d[key].device
            )(lesion)
            lesion = ResizeWithPadOrCropD(
                keys=["label"], spatial_size=d[key].shape[1:], value=0
            )(lesion)
            lesion = lesion["label"].to(d[key].device)
            if self.fill:
                lesion = mask_fill(lesion)
            lesion = (lesion > 0.5).int()
            lesion_mask = softcp(lesion.float()) if self.smooth else lesion
            if self.lesion_fade is not None:
                lesion_mask = self.lesion_fade(lesion_mask)
                d[key] = paste_lesion(d[key], lesion_mask)
                lesion = (lesion_mask > 0.2).int()
            else:
                d[key] = paste_lesion(d[key], lesion_mask)
                lesion = (lesion_mask > 0.5).int()
            if self.fs_healthy:
                d[self.new_keys[i]] = paste_lesion(d['healthy'], lesion)
                if "healthy_meta_dict" in list(d.keys()):
                    d[self.new_keys[i] + "_meta_dict"] = deepcopy(d["healthy_meta_dict"])
            elif self.mb_healthy:
                brain_tissue = d[key][[0,1,2,3]]
                brain_tissue = ((brain_tissue.sum(0,keepdim=True) * GaussianSmooth(3)((d["healthy"]>0).float())) > 0.2) * brain_tissue
                brain_tissue[brain_tissue<0] = 0
                brain_tissue = brain_tissue / brain_tissue.sum(0) # normalize to [0,1]
                brain_tissue = torch.nan_to_num(brain_tissue, nan=0., posinf=0., neginf=0.)
                brain_tissue = torch.cat([1. - brain_tissue.sum(0,keepdim=True), brain_tissue],0)
                d[self.new_keys[i]] = paste_lesion(brain_tissue, lesion)
                if key + "_meta_dict" in list(d.keys()):
                    d[self.new_keys[i] + "_meta_dict"] = deepcopy(d[key + "_meta_dict"])
                del lesion
            else:
                d[self.new_keys[i]] = lesion
                if key + "_meta_dict" in list(d.keys()):
                    d[self.new_keys[i] + "_meta_dict"] = deepcopy(d[key + "_meta_dict"])
                del lesion
            del lesion_mask
        return d
    