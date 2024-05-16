import monai as mn
import glob
import os
from monai.networks.nets import UNet
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
from scipy import ndimage

def run(args, device):
    # find files and load with preprocessing etc
    if args.gt.split('.')[-1] == 'txt':
        files = np.loadtxt(args.gt, dtype=str)
    else:
        files = glob.glob(args.gt, recursive=True)
    preproc = mn.transforms.Compose(transforms=[
        mn.transforms.LoadImageD(keys=["pred","gt"], allow_missing_keys=True),
        mn.transforms.EnsureChannelFirstD(keys=["pred","gt"], allow_missing_keys=True),
        mn.transforms.SpacingD(keys=["pred","gt"], pixdim=1, allow_missing_keys=True),
        mn.transforms.ToTensorD(keys=["pred","gt"],device=device, allow_missing_keys=True),
    ])

    os.makedirs('/'.join(args.save.split('/')[:-1]), exist_ok=True)
    
    all_metrics = []
    for f in tqdm(files, total=len(files)):
        if not args.norm:
            f = f.replace('1mm_', '')
        trgt = os.path.join(args.pred, f.split('/')[-1].split('.nii')[0]+'_pred.nii.gz')
        if 'ISLES' in f:
            f = glob.glob('/'.join(f.split('/')[:-2])+'/*OT*/*.nii')
            assert len(f) == 1
            f = f[0]
        if 'label-L_desc-T1lesion_mask' in trgt:
            trgt = trgt.replace('label-L_desc-T1lesion_mask', 'T1w')
        elif 'masked_1mm_native_Lesion_binary_swc1':
            trgt = trgt.replace('masked_1mm_native_Lesion_binary_swc1', '1mm_')
        batch = preproc({'pred': trgt, 'gt': f})

        pred = torch.nan_to_num(batch['pred'][None])
        if pred.max() > 1:
            trgt = 4 if args.mb else 22
            pred = pred==trgt # needs double checking
            pred = pred.int()
        gt = torch.nan_to_num(batch['gt'][None])

        pred = torch.cat([1.-pred.float(), pred.float()], dim=1)
        gt = torch.cat([1.-gt.float(), gt.float()], dim=1)

        current = {'File': f.split('/')[-1]}

        current['dice'] = mn.metrics.compute_dice(pred, gt, include_background=False).item()
        current['hd95'] = mn.metrics.compute_hausdorff_distance(pred, gt, include_background=False, percentile=95).item()
        cfx = mn.metrics.get_confusion_matrix(pred, gt, include_background=False)
        for mtrc in ["sensitivity", "specificity", "precision", 
                     "negative predictive value", "miss rate", 
                     "fall out", "false discovery rate", 
                     "false omission rate", "prevalence threshold", 
                     "threat score", "accuracy", "balanced accuracy", 
                     "f1 score", "matthews correlation coefficient", 
                     "fowlkes mallows index", "informedness", "markedness"]:
            current[mtrc] = mn.metrics.compute_confusion_matrix_metric(mtrc, cfx).item()
        current['Lesion Volume'] = int(gt[0,1].sum())
        current['Lesion Count'] = int(ndimage.label(gt[0,1].int().cpu().numpy())[1])

        all_metrics.append(current)
        df = pd.DataFrame.from_dict(all_metrics)
        df.to_csv(args.save)


def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save", type=str, help="Path to save csv results")
    parser.add_argument("--pred", type=str, help="Path to find prediction outputs")
    parser.add_argument("--gt", type=str, help="Regex string to find files on disk, or path to txt list.")
    parser.add_argument("--lowres", default=False, action="store_true")
    parser.add_argument("--mb", default=False, action="store_true")
    parser.add_argument("--norm", default=False, action="store_true", help="Benchmark on MNI-normalised images.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args, device


def main():
    args, device = set_up()
    run(args, device)


if __name__ == "__main__":
    main()