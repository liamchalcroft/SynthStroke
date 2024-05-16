import monai as mn
import glob
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from random import seed

# Add your ATLAS data paths here (or list of whatever stroke label niftis you have)
atlas_train_txt = "/home/lchalcroft/git/lab-vae/atlas_train.txt"
atlas_val_txt = "/home/lchalcroft/git/lab-vae/atlas_val.txt"

seed(0)

def printshape(x):
    print(x.shape)
    print(x.min(), x.max())
    return x

def get_loaders(
    batch_size=1,
    device='cpu',
    ptch=128,
    lowres=False,
):
    train_label_list = list(np.loadtxt(atlas_train_txt, dtype=str))
    val_label_list = list(np.loadtxt(atlas_val_txt, dtype=str))

    train_dict = [{'image': f.replace('label-L_desc-T1lesion_mask', 'T1w'), 'label': f} for f in train_label_list]
    val_dict = [{'image': f.replace('label-L_desc-T1lesion_mask', 'T1w'), 'label': f} for f in val_label_list]

    for pair in train_dict:
        if not os.path.exists(pair['image']):
            txt = pair['image']
            txt = txt.split('/')
            txt[-1] = txt[-1][:10]+'*.nii*'
            txt = '/'.join(txt)
            pair['image'] = glob.glob(txt)[0]
    for pair in val_dict:
        if not os.path.exists(pair['image']):
            txt = pair['image']
            txt = txt.split('/')
            txt[-1] = txt[-1][:10]+'*.nii*'
            txt = '/'.join(txt)
            pair['image'] = glob.glob(txt)[0]
    train_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(keys=["label","image"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["label","image"]),
            mn.transforms.SpacingD(keys=["label","image"], pixdim=1 if not lowres else 2),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["label","image"], spatial_size=(256, 256, 256) if not lowres else (128, 128, 128)
            ),
            mn.transforms.ToTensorD(dtype=float, keys=["label","image"], device=device),
            mn.transforms.RandSpatialCropD(
                keys=["image", "label",], roi_size=(ptch, ptch, ptch), random_size=False
            ) if not lowres else mn.transforms.IdentityD(keys="label"),
            mn.transforms.Rand3DElasticD(keys=["image", "label"], sigma_range=(5,7), magnitude_range=(50,150),
                                        rotate_range=15, shear_range=0.012, scale_range=0.15,
                                        padding_mode='zeros', prob=0.8),
            mn.transforms.LambdaD(keys=["image", "label"], func=mn.transforms.SignalFillEmpty()),
            mn.transforms.HistogramNormalizeD(keys="image"),
            mn.transforms.RandHistogramShiftD(keys="image", prob=0.8),
            mn.transforms.RandBiasFieldD(keys="image", prob=0.8),
            mn.transforms.RandAdjustContrastD(keys="image", prob=0.8),
            mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
            mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
            mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.RandGaussianNoiseD(keys="image", prob=0.8),
            mn.transforms.ResizeD(keys=["image", "label"], spatial_size=(ptch, ptch, ptch)) if not lowres else mn.transforms.IdentityD(keys="label"),
            mn.transforms.ScaleIntensityRangeD(keys=["label"], a_min=0, a_max=1, b_min=0, b_max=1, clip=True),
            mn.transforms.AsDiscreteD(keys=["label"], threshold=0.5, to_onehot=2),
            mn.transforms.ToTensorD(dtype=torch.float32, keys="image"),
        ]
    )

    train_data = mn.data.Dataset(train_dict, transform=train_transform)
    val_data = mn.data.Dataset(val_dict, transform=train_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
    )

    return train_loader, val_loader