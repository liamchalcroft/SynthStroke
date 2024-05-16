import monai as mn
import glob
import os
from monai.networks.nets import UNet
import torch
from preprocess import get_loaders, fslab, fsnames
import preprocess_baseline
from tqdm import tqdm
import wandb
import argparse
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from custom import DiceCEL2Loss
from contextlib import nullcontext
import logging
logging.getLogger("monai").setLevel(logging.ERROR)
logging.getLogger("monai.apps").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore', '.*pixdim*.', )


def compute_dice(y_pred, y, eps=1e-8):
    y_pred = torch.flatten(y_pred)
    y = torch.flatten(y)
    y = y.float()
    intersect = (y_pred * y).sum(-1)
    denominator = (y_pred * y_pred).sum(-1) + (y * y).sum(-1)
    return 2 * (intersect / denominator.clamp(min=eps))


def run_model(args, device, train_loader, val_loader):
    dim = 2 if args.twod else 3
    model = UNet(
        spatial_dims=dim,
        in_channels=1,
        out_channels=len(fslab) if args.fshealthy else 6 if args.mbhealthy else 2,
        channels=[32, 64, 128, 256, 320, 320],
        strides=[2, 2, 2, 2, 2],
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=1,
        act="PRELU",
        norm="INSTANCE",
        dropout=args.dropout,
        bias=True,
        adn_ordering="NDA",
    ).to(device)

    if args.resume or args.resume_best:
        ckpts = glob.glob(os.path.join(args.logdir, args.name, 'checkpoint.pt' if args.resume else 'checkpoint_best.pt'))
        if len(ckpts) == 0:
            args.resume = False
            print('\nNo checkpoints found. Beginning from epoch #0')
        else:
            checkpoint = torch.load(ckpts[0], map_location=device)
            print('\nResuming from epoch #{} with WandB ID {}'.format(checkpoint['epoch'], checkpoint["wandb"]))
    print()

    wandb.init(
        project="your-project-here",
        entity="your-entity-here",
        save_code=True,
        name=args.name,
        settings=wandb.Settings(start_method="fork"),
        resume="must" if args.resume else None,
        id=checkpoint["wandb"] if args.resume else None,
    )
    if not args.resume or args.resume_best:
        wandb.config.update(args)
    wandb.watch(model)

    if args.mix_real:
        import random
        train_rl_loader, val_rl_loader = preprocess_baseline.get_loaders(args.batch_size, device, args.patch, args.local_paths, args.lowres)
        def chunk(indices, size):
            return torch.split(torch.tensor(indices), size)

        class MyBatchSampler(torch.utils.data.Sampler):
            def __init__(self, a_indices, b_indices, batch_size): 
                self.a_indices = a_indices
                self.b_indices = b_indices
                self.batch_size = batch_size
            
            def __iter__(self):
                random.shuffle(self.a_indices)
                random.shuffle(self.b_indices)
                a_batches  = chunk(self.a_indices, self.batch_size)
                b_batches = chunk(self.b_indices, self.batch_size)
                all_batches = list(a_batches + b_batches)
                all_batches = [batch.tolist() for batch in all_batches]
                random.shuffle(all_batches)
                return iter(all_batches)
        
        new_dataset = torch.utils.data.ConcatDataset((train_loader.dataset, train_rl_loader.dataset))
        a_len = train_loader.__len__()
        ab_len = a_len + train_rl_loader.__len__()
        a_indices = list(range(a_len))
        b_indices = list(range(a_len, ab_len))
        batch_sampler = MyBatchSampler(a_indices, b_indices, train_loader.batch_size)
        train_loader = torch.utils.data.DataLoader(new_dataset,  batch_sampler=batch_sampler)
        
        new_dataset = torch.utils.data.ConcatDataset((val_loader.dataset, val_rl_loader.dataset))
        a_len = val_loader.__len__()
        ab_len = a_len + val_rl_loader.__len__()
        a_indices = list(range(a_len))
        b_indices = list(range(a_len, ab_len))
        batch_sampler = MyBatchSampler(a_indices, b_indices, val_loader.batch_size)
        val_loader = torch.utils.data.DataLoader(new_dataset,  batch_sampler=batch_sampler)

    print()
    print(f"Training with {len(train_loader.dataset)} samples.")
    print(f"Validating with {len(val_loader.dataset)} samples.")

    crit = DiceCEL2Loss(
        include_background=False,
        to_onehot_y=False,
        sigmoid=False,
        softmax=True,
        other_act=None,
        squared_pred=False,
        jaccard=False,
        reduction="mean",
        smooth_nr=1e-05,
        smooth_dr=1e-05,
        batch=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
        l2_epochs=args.l2,
    )

    class WandBID:
        def __init__(self, wandb_id):
            self.wandb_id = wandb_id

        def state_dict(self):
            return self.wandb_id

    class Epoch:
        def __init__(self, epoch):
            self.epoch = epoch

        def state_dict(self):
            return self.epoch
        
    class Metric:
        def __init__(self, metric):
            self.metric = metric

        def state_dict(self):
            return self.metric
        
    lab_dict = {k:v for k,v in zip(list(range(len(fsnames))),fsnames)} if args.fshealthy else\
            {0:"Background", 1:"Gray Matter", 2:"Gray/White PV", 3:"White Matter", 4:"CSF", 5:"Stroke lesion"} if args.mbhealthy else\
            {0:"Background", 1:"Stroke lesion"}
        
    try:
        opt = torch.optim.AdamW(model.parameters(), args.lr, foreach=torch.cuda.is_available())
    except:
        opt = torch.optim.AdamW(model.parameters(), args.lr)
    # Try to load most recent weight
    if args.resume or args.resume_best:
        model.load_state_dict(checkpoint["net"])
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"] + 1
        metric_best = checkpoint["metric"]
    else:
        start_epoch = 0
        metric_best = 0

    # override learning rate stuff
    def lambda1(epoch):
        return (1 - (epoch+start_epoch) / args.epochs) ** 0.9
    for param_group in opt.param_groups:
            param_group['lr'] = lambda1(0) * args.lr
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1])
        
    train_iter = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        if args.amp:
            ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
            scaler = torch.cuda.amp.GradScaler()
        else:
            ctx = nullcontext()
        progress_bar = tqdm(range(args.epoch_length), total=args.epoch_length, ncols=60)
        progress_bar.set_description(f"[Training] Epoch {epoch}")
        if train_iter is None:
            train_iter = iter(train_loader)
        for step in progress_bar:
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with ctx:
                logits = model(images)
                loss = crit(logits, labels)
                assert loss.isnan().sum() == 0, "NaN found in loss!"
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                opt.step()
            epoch_loss += loss.sum().item()
            wandb.log({"train/loss": loss.sum().item()})
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            dice_metric = []
            plots = []
            ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if args.amp else nullcontext()
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader) if not args.mix_real else None, ncols=60)
            progress_bar.set_description(f"[Validation] Epoch {epoch}")
            with torch.no_grad():
                for val_step, batch in progress_bar:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    opt.zero_grad(set_to_none=True)
                    with ctx:
                        logits = model(images)
                        probs = torch.softmax(logits, dim=1)
                        dice_metric.append(compute_dice(y_pred=probs[:,-1], y=labels[:,-1]).mean().cpu().item()) # lesion only
                    if val_step < 5:
                        plots.append(wandb.Image(images[0,0,...,images.size(-1)//2].cpu().float(),
                                                 masks={
                                                     "predictions": {"mask_data": probs[0].argmax(0).cpu()[...,probs.size(-1)//2], "class_labels": lab_dict},
                                                     "ground truth": {"mask_data": labels[0].argmax(0).cpu()[...,labels.size(-1)//2], "class_labels": lab_dict}
                                                 }             
                                    ))
                    elif val_step == 5:  
                        wandb.log({"val/examples": plots})
            metric = np.nanmean(dice_metric)
            wandb.log({"val/dice": metric})

            if metric > metric_best:
                metric_best = metric
                torch.save(
                    {
                        "net": model.state_dict(),
                        "opt": opt.state_dict(),
                        "lr": lr_scheduler.state_dict(),
                        "wandb": WandBID(wandb.run.id).state_dict(),
                        "epoch": Epoch(epoch).state_dict(),
                        "metric": Metric(metric_best).state_dict()
                    },
                    os.path.join(args.logdir, args.name,'checkpoint_best.pt'.format(epoch)))
            torch.save(
                {
                    "net": model.state_dict(),
                    "opt": opt.state_dict(),
                    "lr": lr_scheduler.state_dict(),
                    "wandb": WandBID(wandb.run.id).state_dict(),
                    "epoch": Epoch(epoch).state_dict(),
                    "metric": Metric(metric_best).state_dict()
                },
                os.path.join(args.logdir, args.name,'checkpoint.pt'.format(epoch)))

def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--epoch_length", type=int, default=100, help="Number of iterations per epoch.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout ratio.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--val_interval", type=int, default=2, help="Validation interval.")
    parser.add_argument("--lesion_weight", type=float, default=2, help="Weighting of lesion in CE loss. All other classes have unit weighting.")
    parser.add_argument("--mix_real", default=False, action="store_true", help="Used for mixing Synth & real training. Currently can only apply in 50:50 ratio.")
    parser.add_argument("--l2", type=int, default=None, help="Number of epochs to use L2 loss before moving to Dice/CE.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--patch", type=int, default=128, help="Patch size for cropping.")
    parser.add_argument("-a", "--amp", default=False, action="store_true")
    parser.add_argument("--logdir", type=str, default="./", help="Path to saved outputs")
    parser.add_argument("--fade", default=False, action="store_true", help="Fade lesions for inhomogeneity.")
    parser.add_argument("-dbg", "--debug", default=False, action="store_true")
    parser.add_argument("-res", "--resume", default=False, action="store_true")
    parser.add_argument("--resume_best", default=False, action="store_true")
    parser.add_argument("-hlt", "--fshealthy", default=False, action="store_true", help="Use FreeSurfer healthy labels alongside lesion.")
    parser.add_argument("-mb", "--mbhealthy", default=False, action="store_true", help="Use MultiBrain healthy labels alongside lesion.")
    parser.add_argument("--baseline", default=False, action="store_true", help="Train baseline with real images.")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--test_run", default=False, action="store_true", help="Run single iteration per epoch for quick debug.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Train with un-cropped 2D images.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.baseline:
        train_loader, val_loader = preprocess_baseline.get_loaders(args.batch_size, device, args.patch, args.lowres)
    else:
        train_loader, val_loader = get_loaders(args.batch_size, args.fshealthy, args.mbhealthy, 
                                                                                      device, args.fade, args.lowres, args.patch)

    return args, device, train_loader, val_loader


def main():
    args, device, train_loader, val_loader = set_up()
    if args.debug:
        saver1 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="img",
            separate_folder=False,
        )
        saver2 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="label",
            separate_folder=False,
        )
        for i, batch in enumerate(val_loader):
            if i > 5:
                break
            else:
                print('Image: ', batch['image'].shape, 'min={}'.format(batch['image'].min()), 'max={}'.format(batch['image'].max()))
                print('Label: ', batch['label'].shape, 'min={}'.format(batch['label'].min()), 'max={}'.format(batch['label'].max()))
                saver1(
                    torch.Tensor(batch["image"][0].cpu().float()),
                )
                saver2(
                    torch.Tensor(torch.argmax(batch["label"][0], dim=0)[None].cpu().float()),
                )
        print("Debug finished and samples saved.")
        exit()
    run_model(args, device, train_loader, val_loader)


if __name__ == "__main__":
    main()