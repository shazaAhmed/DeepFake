import argparse
import json
import os
from collections import defaultdict

from sklearn.metrics import log_loss
from torch import topk

from training import losses
from training.datasets.classifier_dataset import DeepFakeClassifierDataset
from training.losses import WeightedLosses
from training.tools.config import load_config
from training.tools.utils import create_optimizer, AverageMeter
from training.transforms.albu import IsotropicResize
from training.zoo import classifiers

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur

from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tensorboardX import SummaryWriter

from apex import amp

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
# CUDNN is a GPU-accelerated library of primitives for deep neural networks.
# it provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.
# when setting torch.backends.cudnn.benchmark to true it allows us to enable the inbuilt cudnn auto-tuner
#to find the best algorithm to use for our hardware.
torch.backends.cudnn.benchmark = True


def create_train_transforms(size=300):
    # defining an augmentation pipeline
    # this will return a transform function that will perform image augmentation.
    return Compose([
        # Decrease Jpeg, WebP compression of an image
        # with the quality_lower parameter as the lower bound on the image quality
        # and the quality_upper as the upper bound on the image quality
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        # used to apply Gaussian noise to the input picture
        # with p as the probability of applying the transform
        GaussNoise(p=0.1),
        # used to blur the input image using a Gaussian filter with a random kernel size
        # with the blur_limit as the maximum Gaussian kernel size for blurring the input image
        GaussianBlur(blur_limit=3, p=0.05),
        # flips the input image horizontally around the y-axis
        HorizontalFlip(),
        # Select one of transforms to apply
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        # Pad side of the image / max if side is less than desired number
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        # Select one of the following transforms to apply:
        # RandomBrightnessContrast: used to randomly change brightness and contrast of the input image
        # FancyPCA: Augment RGB image using FancyPCA
        # HueSaturationValue: Randomly change hue, saturation and value of the input image
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        # this converts the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.
        ToGray(p=0.2),
        # this randomly apply affine transforms: translate, scale and rotate the input.
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )


def create_val_transforms(size=300):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])


def main():
    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
    arg('--workers', type=int, default=6, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='classifier_')
    arg('--data-dir', type=str, default="/mnt/sota/datasets/deepfake")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--crops-dir', type=str, default='crops')
    arg('--label-smoothing', type=float, default=0.01)
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg('--freeze-epochs', type=int, default=0)
    arg("--local_rank", default=0, type=int)
    arg("--seed", default=777, type=int)
    arg("--padding-part", default=3, type=int)
    arg("--opt-level", default='O1', type=str)
    arg("--test_every", type=int, default=1)
    arg("--no-oversample", action="store_true")
    arg("--no-hardcore", action="store_true")
    arg("--only-changed-frames", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        # Initializes the default distributed process group
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    conf = load_config(args.config)
    model = classifiers.__dict__[conf['network']](encoder=conf['encoder'])

    model = model.cuda()
    if args.distributed:
        # Recursively traverse module and its children to replace all instances of
        # batch norms with sync batch norm
        # a Synchronized Batch Normalization (SyncBN) is a type of batch normalization used for multi-GPU training.
        # Standard batch normalization only normalizes the data within each device (GPU). 
        # SyncBN normalizes the input within the whole mini-batch.
        model = convert_syncbn_model(model)
    ohem = conf.get("ohem_samples", None)
    reduction = "mean"
    if ohem:
        reduction = "none"
    loss_fn = []
    weights = []
    for loss_name, weight in conf["losses"].items():
        loss_fn.append(losses.__dict__[loss_name](reduction=reduction).cuda())
        weights.append(weight)
    loss = WeightedLosses(loss_fn, weights)
    loss_functions = {"classifier_loss": loss}
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)
    bce_best = 100
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']

    data_train = DeepFakeClassifierDataset(mode="train",
                                           oversample_real=not args.no_oversample,
                                           fold=args.fold,
                                           padding_part=args.padding_part,
                                           hardcore=not args.no_hardcore,
                                           crops_dir=args.crops_dir,
                                           data_path=args.data_dir,
                                           label_smoothing=args.label_smoothing,
                                           folds_csv=args.folds_csv,
                                           transforms=create_train_transforms(conf["size"]),
                                           normalize=conf.get("normalize", None))
    data_val = DeepFakeClassifierDataset(mode="val",
                                         fold=args.fold,
                                         padding_part=args.padding_part,
                                         crops_dir=args.crops_dir,
                                         data_path=args.data_dir,
                                         folds_csv=args.folds_csv,
                                         transforms=create_val_transforms(conf["size"]),
                                         normalize=conf.get("normalize", None))
    val_data_loader = DataLoader(data_val, batch_size=batch_size * 2, num_workers=args.workers, shuffle=False,
                                 pin_memory=False)
    os.makedirs(args.logdir, exist_ok=True)
    # The SummaryWriter class creates an event file in a given directory and add summaries and events to it. 
    # The class updates the file contents asynchronously.
    summary_writer = SummaryWriter(args.logdir + '/' + conf.get("prefix", args.prefix) + conf['encoder'] + "_" + str(args.fold))
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
                if not args.zero_score:
                    bce_best = checkpoint.get('bce_best', 0)
            print("=> loaded checkpoint '{}' (epoch {}, bce_best {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['bce_best']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.from_zero:
        start_epoch = 0
    current_epoch = start_epoch

    if conf['fp16']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale='dynamic')

    snapshot_name = "{}{}_{}_{}".format(conf.get("prefix", args.prefix), conf['network'], conf['encoder'], args.fold)
    # the difference between DistributedDataParallel and DataParallel
    # DataParallel is single-process, multi-thread, and only works on a single machine,
    # while DistributedDataParallel is multi-process and works for both single- and multi- machine training.
    # DataParallel is usually slower than DistributedDataParallel
    if args.distributed:
        #this enables multiprocess distributed data parallel training
        model = DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = DataParallel(model).cuda()
    data_val.reset(1, args.seed)
    max_epochs = conf['optimizer']['schedule']['epochs']
    for epoch in range(start_epoch, max_epochs):
        data_train.reset(epoch, args.seed)
        train_sampler = None
        if args.distributed:
            # this restricts data loading to a subset of the dataset
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
            train_sampler.set_epoch(epoch)
        if epoch < args.freeze_epochs:
            print("Freezing encoder!!!")
            model.module.encoder.eval()
            for p in model.module.encoder.parameters():
                p.requires_grad = False
        else:
            model.module.encoder.train()
            for p in model.module.encoder.parameters():
                p.requires_grad = True

        train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=args.workers,
                                       shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                       drop_last=True)

        train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                    args.local_rank, args.only_changed_frames)
        model = model.eval()

        if args.local_rank == 0:
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': bce_best,
            }, args.output_dir + '/' + snapshot_name + "_last")
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': bce_best,
            }, args.output_dir + snapshot_name + "_{}".format(current_epoch))
            if (epoch + 1) % args.test_every == 0:
                bce_best = evaluate_val(args, val_data_loader, bce_best, model,
                                        snapshot_name=snapshot_name,
                                        current_epoch=current_epoch,
                                        summary_writer=summary_writer)
        current_epoch += 1


def evaluate_val(args, data_val, bce_best, model, snapshot_name, current_epoch, summary_writer):
    print("Test phase")
    model = model.eval()

    bce, probs, targets = validate(model, data_loader=data_val)
    if args.local_rank == 0:
        summary_writer.add_scalar('val/bce', float(bce), global_step=current_epoch)
        if bce < bce_best:
            print("Epoch {} improved from {} to {}".format(current_epoch, bce_best, bce))
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'bce_best': bce,
                }, args.output_dir + snapshot_name + "_best_dice")
            bce_best = bce
            with open("predictions_{}.json".format(args.fold), "w") as f:
                json.dump({"probs": probs, "targets": targets}, f)
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'bce_best': bce_best,
        }, args.output_dir + snapshot_name + "_last")
        print("Epoch: {} bce: {}, bce_best: {}".format(current_epoch, bce, bce_best))
    return bce_best


def validate(net, data_loader, prefix=""):
    probs = defaultdict(list)
    targets = defaultdict(list)

    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = sample["image"].cuda()
            img_names = sample["img_name"]
            labels = sample["labels"].cuda().float()
            out = net(imgs)
            labels = labels.cpu().numpy()
            preds = torch.sigmoid(out).cpu().numpy()
            for i in range(out.shape[0]):
                video, img_id = img_names[i].split("/")
                probs[video].append(preds[i].tolist())
                targets[video].append(labels[i].tolist())
    data_x = []
    data_y = []
    for vid, score in probs.items():
        score = np.array(score)
        lbl = targets[vid]

        score = np.mean(score)
        lbl = np.mean(lbl)
        data_x.append(score)
        data_y.append(lbl)
    y = np.array(data_y)
    x = np.array(data_x)
    fake_idx = y > 0.1
    real_idx = y < 0.1
    fake_loss = log_loss(y[fake_idx], x[fake_idx], labels=[0, 1])
    real_loss = log_loss(y[real_idx], x[real_idx], labels=[0, 1])
    print("{}fake_loss".format(prefix), fake_loss)
    print("{}real_loss".format(prefix), real_loss)

    return (fake_loss + real_loss) / 2, probs, targets


def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                local_rank, only_valid):
    losses = AverageMeter()
    fake_losses = AverageMeter()
    real_losses = AverageMeter()
    max_iters = conf["batches_per_epoch"]
    print("training epoch {}".format(current_epoch))
    model.train()
    pbar = tqdm(enumerate(train_data_loader), total=max_iters, desc="Epoch {}".format(current_epoch), ncols=0)
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in pbar:
        imgs = sample["image"].cuda()
        labels = sample["labels"].cuda().float()
        out_labels = model(imgs)
        if only_valid:
            valid_idx = sample["valid"].cuda().float() > 0
            out_labels = out_labels[valid_idx]
            labels = labels[valid_idx]
            if labels.size(0) == 0:
                continue

        fake_loss = 0
        real_loss = 0
        fake_idx = labels > 0.5
        real_idx = labels <= 0.5

        ohem = conf.get("ohem_samples", None)
        if torch.sum(fake_idx * 1) > 0:
            fake_loss = loss_functions["classifier_loss"](out_labels[fake_idx], labels[fake_idx])
        if torch.sum(real_idx * 1) > 0:
            real_loss = loss_functions["classifier_loss"](out_labels[real_idx], labels[real_idx])
        if ohem:
            fake_loss = topk(fake_loss, k=min(ohem, fake_loss.size(0)), sorted=False)[0].mean()
            real_loss = topk(real_loss, k=min(ohem, real_loss.size(0)), sorted=False)[0].mean()

        loss = (fake_loss + real_loss) / 2
        losses.update(loss.item(), imgs.size(0))
        fake_losses.update(0 if fake_loss == 0 else fake_loss.item(), imgs.size(0))
        real_losses.update(0 if real_loss == 0 else real_loss.item(), imgs.size(0))

        optimizer.zero_grad()
        pbar.set_postfix({"lr": float(scheduler.get_lr()[-1]), "epoch": current_epoch, "loss": losses.avg,
                          "fake_loss": fake_losses.avg, "real_loss": real_losses.avg})

        if conf['fp16']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        torch.cuda.synchronize()
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * max_iters)
        if i == max_iters - 1:
            break
    pbar.close()
    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)


if __name__ == '__main__':
    main()
