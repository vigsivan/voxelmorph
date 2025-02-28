#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import numpy as np
import torch
from torch import nn
from monai.losses import GlobalMutualInformationLoss
from nets import VxmSemisupervised
from voxelmorph.torch.losses import Grad, Dice
import torchio as tio
from einops import rearrange, repeat
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


# import voxelmorph with pytorch backend
os.environ["VXM_BACKEND"] = "pytorch"
import voxelmorph as vxm  # nopep8

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument(
    "--img-list", required=True, help="line-seperated list of training files"
)
parser.add_argument("--img-prefix", help="optional input image file prefix")
parser.add_argument("--img-suffix", help="optional input image file suffix")
parser.add_argument("--seg-prefix", help="optional input seg file prefix")
parser.add_argument("--seg-suffix", help="optional input seg file suffix")
parser.add_argument("--atlas", help="atlas filename (default: data/atlas_norm.npz)")
parser.add_argument(
    "--model-dir", default="models", help="model output directory (default: models)"
)
parser.add_argument(
    "--multichannel",
    action="store_true",
    help="specify that data has multiple channels",
)

# training parameters
parser.add_argument(
    "--gpu", default="0", help="GPU ID number(s), comma-separated (default: 0)"
)
parser.add_argument("--batch-size", type=int, default=1, help="batch size (default: 1)")
parser.add_argument(
    "--epochs", type=int, default=1500, help="number of training epochs (default: 1500)"
)
parser.add_argument(
    "--steps-per-epoch",
    type=int,
    default=100,
    help="frequency of model saves (default: 100)",
)
parser.add_argument("--load-model", help="optional model file to initialize with")
parser.add_argument(
    "--initial-epoch", type=int, default=0, help="initial epoch number (default: 0)"
)
parser.add_argument(
    "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)"
)
parser.add_argument(
    "--cudnn-nondet",
    action="store_true",
    help="disable cudnn determinism - might slow down training",
)

# network architecture parameters
parser.add_argument(
    "--enc",
    type=int,
    nargs="+",
    help="list of unet encoder filters (default: 16 32 32 32)",
)
parser.add_argument(
    "--dec",
    type=int,
    nargs="+",
    help="list of unet decorder filters (default: 32 32 32 32 32 16 16)",
)
parser.add_argument('--labels', required=True, help='label list (npy format) to use in dice loss')
parser.add_argument(
    "--int-steps", type=int, default=7, help="number of integration steps (default: 7)"
)
parser.add_argument(
    "--int-downsize",
    type=int,
    default=2,
    help="flow downsample factor for integration (default: 2)",
)
parser.add_argument(
    "--bidir", action="store_true", help="enable bidirectional cost function"
)

# loss hyperparameters
parser.add_argument(
    "--image-loss",
    default="mse",
    help="image reconstruction loss - can be mse or ncc (default: mse)",
)
parser.add_argument(
    "--lambda",
    type=float,
    dest="weight",
    default=0.01,
    help="weight of deformation loss (default: 0.01)",
)
parser.add_argument('--grad-loss-weight', type=float, default=0.01,
                    help='weight of gradient loss (lamba) (default: 0.01)')
parser.add_argument('--dice-loss-weight', type=float, default=0.01,
                    help='weight of dice loss (gamma) (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

# load and prepare training data
train_imgs = vxm.py.utils.read_file_list(
    args.img_list, prefix=args.img_prefix, suffix=args.img_suffix
)
train_segs = vxm.py.utils.read_file_list(args.img_list, prefix=args.seg_prefix,
                                         suffix=args.seg_suffix)
assert len(train_imgs) > 0, "Could not find any training data."

train_labels = np.load(args.labels)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas != None:
    "This script does not properly deal with atlas. Use a different script!"

generator = vxm.generators.semisupervised(
    train_imgs, train_segs, labels=train_labels, atlas_file=args.atlas, downsize=1)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(",")
nb_gpus = len(gpus)
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, (
    "Batch size (%d) should be a multiple of the nr of gpus (%d)"
    % (args.batch_size, 1)
)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# otherwise configure new model
model = VxmSemisupervised(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    bidir=bidir,
    int_steps=args.int_steps,
    int_downsize=args.int_downsize,
)


# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
# if args.image_loss == "ncc":
#     image_loss_func = vxm.losses.NCC().loss
# elif args.image_loss == "mse":
#     image_loss_func = vxm.losses.MSE().loss
# else:
#     raise ValueError(
#         'Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss
#     )
image_loss_func = GlobalMutualInformationLoss()

def transform(img, seg):
    subject = tio.Subject({
        "image": tio.ScalarImage(tensor=rearrange(img, 'b d w h f -> (b f) d w h')),
        "seg": tio.LabelMap(tensor=rearrange(seg, 'b d w h f -> (b f) d w h'))
    })
    transform = tio.Compose([
        tio.RandomAffine(scales=0, degrees=15),
        tio.RandomElasticDeformation(),
        tio.RandomAnisotropy(),
        tio.RandomGamma(),
    ])
    transformed_subject = transform(subject)
    img, seg = transformed_subject["image"].data, transformed_subject["seg"].data
    img, seg = [repeat(i, 'b d w h -> b d w h f', f=1) for i in (img, seg)]
    return img.numpy(), seg.numpy()


losses = [image_loss_func, 
          Grad('l2', loss_mult=args.int_downsize).loss, 
          Dice().loss]
weights = [1, args.grad_loss_weight, args.dice_loss_weight]

log_dir = Path.home()/"vxm_tb"
log_dir.mkdir(exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, "%04d.pt" % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)

        src_vol, trg_vol, src_seg = inputs
        src_vol, src_seg = transform(src_vol, src_seg)
        inputs = (src_vol, trg_vol, src_seg)

        inputs = [
            torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3)
            for d in inputs
        ]
        y_true = [
            torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3)
            for d in y_true
        ]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        global_step = (epoch * args.steps_per_epoch) + (step+1)
        for loss_v, loss_n in zip(loss_list, ("mi", "l2", "dice")):
            writer.add_scalar(loss_n, loss_v, global_step=global_step)

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = "Epoch %d/%d" % (epoch + 1, args.epochs)
    time_info = "%.4f sec/step" % np.mean(epoch_step_time)
    losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = "loss: %.4e  (%s)" % (np.mean(epoch_total_loss), losses_info)
    print(" - ".join((epoch_info, time_info, loss_info)), flush=True)

# final model save
model.save(os.path.join(model_dir, "%04d.pt" % args.epochs))
writer.close()
