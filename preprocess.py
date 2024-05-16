import monai as mn
import glob
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from random import shuffle, seed
import lesion
import custom
import custom_cc

fslab = [0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,77,80,192,786]
targetlab = list(range(len(fslab)))
labmap = {}
for i, l in enumerate(fslab):
    labmap[l] = i

fsnames = ["Background"
,"Left cerebral white matter"
,"Left cerebral cortex"
,"Left lateral ventricle"
,"Left inferior lateral ventricle"
,"Left cerebellum white matter"
,"Left cerebellum cortex"
,"Left thalamus"
,"Left caudate"
,"Left putamen"
,"Left pallidum"
,"3rd ventricle"
,"4th ventricle"
,"Brain-stem"
,"Left hippocampus"
,"Left amygdala"
,"CSF"
,"Left accumbens area"
,"Left ventral DC"
# ,"Right cerebral white matter"
# ,"Right cerebral cortex"
# ,"Right lateral ventricle"
# ,"Right inferior lateral ventricle"
# ,"Right cerebellum white matter"
# ,"Right cerebellum cortex"
# ,"Right thalamus"
# ,"Right caudate"
# ,"Right putamen"
# ,"Right pallidum"
# ,"Right hippocampus"
# ,"Right amygdala"
# ,"Right accumbens area"
# ,"Right ventral DC"
,"WM Anomaly"
,"Non-WM Anomaly"
,"Corpus Callopsum"
,"Stroke lesion"]

seed(0)

# Add your ATLAS data paths here (or list of whatever stroke label niftis you have)
atlas_train_txt = "/your/path/atlas_train.txt"
atlas_val_txt = "/your/path/atlas_val.txt"
oasis_path = "/your/path/here/"

def printshape(x):
    print(x.shape)
    print(x.min(), x.max())
    return x


def get_loaders(
    batch_size=1,
    fs_healthy=False,
    mb_healthy=False,
    device='cpu',
    fade=False,
    lowres=False,
    ptch=128,
):
    train_files = glob.glob(os.path.join(oasis_path, "OAS*/OAS*_Freesurfer*/DATA/OAS*/mri/mni_1mm_healthy_symmetric.nii.gz"))
    train_dict = [
        {"healthy": f, "label": f.replace("healthy_symmetric", "mb_labels")}
        for f in train_files
    ]

    shuffle(train_dict)

    train_dict, val_dict = (
        train_dict[: -100],
        train_dict[-100 :],
    )

    print(f"\nHealthy labels: Train {len(train_dict)} Val {len(val_dict)}\n")

    train_label_list = list(np.loadtxt(atlas_train_txt, dtype=str))
    val_label_list = list(np.loadtxt(atlas_val_txt, dtype=str))

    print(f"\nLesion labels: Train {len(train_label_list)} Val {len(val_label_list)}\n")

    if lowres:
        train_label_list = [lst.replace('1mm','2mm') for lst in train_label_list]
        val_label_list = [lst.replace('1mm','2mm') for lst in val_label_list]
    
    train_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(keys=["label","healthy"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["label","healthy"]),
            mn.transforms.SpacingD(keys=["label","healthy"], pixdim=1 if not lowres else 2),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["label","healthy"], spatial_size=(256, 256, 256) if not lowres else (128, 128, 128)
            ),
            mn.transforms.ToTensorD(dtype=float, keys=["label","healthy"], device=device),
            mn.transforms.LambdaD(keys=["healthy"], 
                                func=mn.transforms.MapLabelValue(orig_labels=fslab, 
                                target_labels=targetlab)) if fs_healthy else mn.transforms.AsDiscreteD(keys="healthy", threshold=0.5),
            mn.transforms.AsDiscreteD(keys="healthy", to_onehot=len(fslab)-1) if fs_healthy else mn.transforms.IdentityD(keys="label"),
            lesion.LesionPasteD(
                keys="label", new_keys=["seg"], label_list=train_label_list, fs_healthy=fs_healthy, mb_healthy=mb_healthy, lesion_fading=fade
            ),
            mn.transforms.AsDiscreteD(keys="seg", to_onehot=2) if not fs_healthy and not mb_healthy else mn.transforms.IdentityD(keys="label"),
            custom_cc.CCSynthSeg(label_key='label', image_key='image', coreg_keys=['seg','healthy']),
            custom.RemapSegToLabel(in_key="seg", out_key="label"),
            mn.transforms.RandSpatialCropD(
                keys=["image", "label", "seg", "healthy"], roi_size=(ptch, ptch, ptch), random_size=False
            ) if not lowres else mn.transforms.IdentityD(keys="label"),
            mn.transforms.OneOf(
                transforms=[
                    custom.RandomSkullStrip(
                        label_key="healthy",
                        image_key=["image","label"],
                        out_key="mask",
                        channels_to_use=targetlab[1:] if fs_healthy else [0],
                        dilate_prob=0.3,
                        erode_prob=0.3,
                    ),
                    mn.transforms.IdentityD(keys=["image"]),
                ],
                weights=[0.3, 0.7],
            ),
            mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
            mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
            mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.ResizeD(keys=["image", "label"], spatial_size=(ptch, ptch, ptch)) if not lowres else mn.transforms.IdentityD(keys="label"),
            mn.transforms.ToTensorD(dtype=torch.float32, keys="image"),
            mn.transforms.ToTensorD(dtype=torch.float32, keys="label"),
            mn.transforms.DeleteItemsD(keys=["healthy","seg","healthy_meta_dict","seg_meta_dict"]),
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