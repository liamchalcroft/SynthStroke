import monai as mn
import glob
import os
import csv
import shutil
from monai.networks.nets import UNet
import torch
from fsl.wrappers import flirt, bet
import numpy as np
from tqdm import tqdm
import argparse


fslab = [0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,77,80,192,786]
targetlab = list(range(len(fslab)))
labmap = {}
for i, l in enumerate(fslab):
    labmap[l] = i

def clip_ct(img):
    img[img<0] = 0
    img[img>80] = 80
    return img


def run_model(args, device):
    if args.healthy:
        o_ch = len(fslab)
    elif args.mb:
        o_ch = 6
    else:
        o_ch = 2
    model = UNet(
        spatial_dims=3,
        in_channels=1,  # needs to be equal to img channels + label channels
        out_channels=o_ch,
        channels=[32, 64, 128, 256, 320, 320],
        strides=[2, 2, 2, 2, 2],
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=1,
        act="PRELU",
        norm="INSTANCE",
        dropout=0,
        bias=True,
        adn_ordering="NDA",
    ).to(device)

    window = mn.inferers.SlidingWindowInferer(
        [args.patch, args.patch, args.patch],
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
        sigma_scale=0.125,
        cval=0.0,
        sw_device=None,
        device=None,
        progress=False,
        cache_roi_weight_map=False,
    )

    # find files and load with preprocessing etc
    if args.files.split('.')[-1] == 'txt':
        files = np.loadtxt(args.files, dtype=str)
    else:
        files = glob.glob(args.files, recursive=True)
    load = mn.transforms.Compose(transforms=[
        mn.transforms.LoadImageD(keys=["img"], allow_missing_keys=True),
        mn.transforms.EnsureChannelFirstD(keys=["img"], allow_missing_keys=True),
        mn.transforms.ToTensorD(keys=["img","seg"], device=device, allow_missing_keys=True),
        mn.transforms.LambdaD(keys=["img"], func=clip_ct, allow_missing_keys=True) if args.ct else mn.transforms.LambdaD(keys=["img"], func=lambda x: x, allow_missing_keys=True)
        ])
    preproc = mn.transforms.Compose(transforms=[
        mn.transforms.OrientationD(keys=["img","seg"], axcodes="RAS", allow_missing_keys=True),
        mn.transforms.SpacingD(keys=["img","seg"], pixdim=2 if args.lowres else 1, allow_missing_keys=True),
        mn.transforms.HistogramNormalizeD(keys="img"),
        mn.transforms.NormalizeIntensityD(keys="img",nonzero=False, channel_wise=True, allow_missing_keys=True),
    ])
    postproc = mn.transforms.Compose(transforms=[
        mn.transforms.Activations(softmax=True),
        mn.transforms.AsDiscrete(argmax=True),
    ])
    if args.tta:
        flips = [
            mn.transforms.Flip(spatial_axis=ax) for ax in [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
        ]

    # load model weights and put everything on device
    weights = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights['net'])
    model.eval()

    os.makedirs(args.savedir, exist_ok=True)
    # loop of: infer, post-process, save
    for f in tqdm(files, total=len(files)):
        try:
            if 'label-L_desc-T1lesion_mask' in f:
                f = f.replace('label-L_desc-T1lesion_mask', 'T1w')
            elif 'masked_1mm_native_Lesion_binary_swc1':
                f = f.replace('masked_1mm_native_Lesion_binary_swc1', '1mm_')
            if not args.norm:
                f = f.replace('1mm_', '')
            if not os.path.exists(f):
                if '.nii.gz' in f:
                    f = f.replace('.nii.gz', '.nii')   
                else:
                    f = f.replace('.nii', '.nii.gz')
            if '/ARC/' in f:
                f = f.replace('T1w.nii.gz', 'T1w_flirt.nii.gz')
                f = f.replace('FLAIR.nii.gz', 'FLAIR_flirt.nii.gz')
            shutil.copyfile(f, os.path.join(args.savedir, f.split('/')[-1]))
            f = os.path.join(args.savedir, f.split('/')[-1])
            if args.align is not None:
                flirt(f, args.align, applyxfm=True)
            if args.bet:
                bet(f, os.path.join(args.savedir, 'ss_'+f.split('/')[-1]), mask=True, robust=True, seg=True)
                batch = {'img': os.path.join(args.savedir, 'ss_'+f.split('/')[-1].replace('.nii','.nii.gz'))}
            else:
                batch = {'img': f}
            batch = load(batch)
            batch = preproc(batch)
            img = batch['img']
            with torch.no_grad():
                pred = window(img[None], model)[0]
                if args.tta:
                    for flip in flips:
                        pred += flip(window(flip(img)[None], model)[0])
                    pred /= len(flips)
            pred.applied_operations = img.applied_operations
            pred_dict = {}
            pred_dict["img"] = pred
            with mn.transforms.utils.allow_missing_keys_mode(preproc):
                inverted_pred = preproc.inverse(pred_dict)
            pred = inverted_pred["img"]
            pred = postproc(pred)            
            lesion = pred == 5 if args.mb else pred == 1
            mn.transforms.SaveImage(output_dir=args.savedir, output_postfix='reslice', separate_folder=False, 
                                    print_log=False, resample=False, dtype=np.int16)(img)
            mn.transforms.SaveImage(output_dir=args.savedir, output_postfix='pred', separate_folder=False, 
                                    print_log=False, resample=False, dtype=np.int16)(pred)
            mn.transforms.SaveImage(output_dir=args.savedir, output_postfix='lesion', separate_folder=False, 
                                    print_log=False, resample=False, dtype=np.int16)(lesion)
        except Exception as e:
            print("Error occurred in file {}: {}".format(f, e))


def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", type=str, help="Path to trained model weights.")
    parser.add_argument("--tta", default=False, action="store_true")
    parser.add_argument("--healthy", default=False, action="store_true")
    parser.add_argument("--lowres", default=False, action="store_true")
    parser.add_argument("--mb", default=False, action="store_true")
    parser.add_argument("--patch", type=int, default=128, help="Isotropic patch size for inference.")
    parser.add_argument("--savedir", type=str, help="Path to save prediction outputs")
    parser.add_argument("--files", type=str, help="Regex string to find files on disk, or path to txt list.")
    parser.add_argument("--norm", default=False, action="store_true", help="Predict on MNI-normalised images.")
    parser.add_argument("--bet", default=False, action="store_true", help="Predict on (robust-)BET skullstripped images.")
    parser.add_argument("--ct", default=False, action="store_true", help="If CT image, clip intensity of image when copying to save directory.")
    parser.add_argument("--align", type=str, default=None, help="Rigid-align and crop to template file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args, device


def main():
    args, device = set_up()
    run_model(args, device)


if __name__ == "__main__":
    main()