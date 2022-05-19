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
from typing import Callable
from einops import repeat
from functools import partial

transform = tio.OneOf([
    tio.RandomAnisotropy(),
    tio.RandomBiasField(),
])

# import voxelmorph with pytorch backend
os.environ["VXM_BACKEND"] = "pytorch"
import voxelmorph as vxm  # nopep8

def pairgen(
    imgpairs,
    segpairs,
    batch_size=1,
    np_var='vol',
    pad_shape=None,
    resize_factor=1,
    add_feat_axis=True,
    transform: Callable = transform,
):
    """
    Generator for random volume loading pairwise.

    Parameters:
        vol_names: list of paths.
        batch_size: Batch size. Default is 1.
        segs: Loads corresponding segmentations (list). Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    if batch_size != 1:
        raise ValueError("Batch size not equal to 1 not supported!")
    if add_feat_axis:
        reshape = partial(repeat, pattern="d h w -> b d h w f", b=1, f=1)
    else:
        reshape = partial(repeat, pattern="d h w -> b d h w", b=1)

    def process_pairs(image_pair, seg_pair):
        image_arr_pair = [0, 0]
        seg_arr_pair = [0, 0]
        for pair_index in range(2):
            image, seg = image_pair[pair_index], seg_pair[pair_index]
            subject = tio.Subject({
                "image": tio.ScalarImage(image),
                "seg": tio.LabelMap(seg)
            })
            subject = transform(subject) # TODO
            image_arr_pair[pair_index] = reshape(subject["image"].data.squeeze().numpy())
            seg_arr_pair[pair_index] = reshape(subject["seg"].data.squeeze().numpy())

        # image_arr = np.concatenate(image_arr_pair, axis=0)
        # seg_arr = np.concatenate(seg_arr_pair, axis=0)
        return (image_arr_pair, seg_arr_pair)

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(imgpairs), size=batch_size)
        src_images, trg_images, src_segs, trg_segs = [], [], [], []
        for i in indices:
            (src_image, trg_image), (src_seg, trg_seg) = process_pairs(imgpairs[i], segpairs[i])
            src_images.append(src_image)
            trg_images.append(trg_image)
            src_segs.append(src_seg)
            trg_segs.append(trg_seg)

        yield tuple([
            np.concatenate(src_images, axis=0),
            np.concatenate(src_segs, axis=0),
            np.concatenate(trg_images, axis=0),
            np.concatenate(trg_segs, axis=0),
        ])
            
def semisupervised(img_pairs, seg_pairs, labels, atlas_file=None, downsize=1):
    """
    Generator for semi-supervised registration training using ground truth segmentations.
    Scan-to-atlas training can be enabled by providing the atlas_file argument. 

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        seg_names: List of corresponding seg files to load, or list of preloaded volumes.
        labels: Array of discrete label values to use in training.
        atlas_file: Atlas npz file for scan-to-atlas training. Default is None.
        downsize: Downsize factor for segmentations. Default is 2.
    """
    # configure base generator
    gen = pairgen(img_pairs, seg_pairs, np_var='vol')
    zeros = None

    # internal utility to generate downsampled prob seg from discrete seg
    def split_seg(seg):
        prob_seg = np.zeros((*seg.shape[:4], len(labels)))
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = seg[0, ..., 0] == label
        return prob_seg[:, ::downsize, ::downsize, ::downsize, :]

    while True:
        # load source vol and seg
        src_vol, src_seg, trg_vol, trg_seg = next(gen)
        src_seg = split_seg(src_seg)
        trg_seg = split_seg(trg_seg)

        # cache zeros
        if zeros is None:
            shape = src_vol.shape[1:-1]
            zeros = np.zeros((1, *shape, len(shape)))

        invols = [src_vol, trg_vol, src_seg]
        outvols = [trg_vol, zeros, trg_seg]
        yield (invols, outvols)


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters

parser.add_argument('--pairs', required=True, help='line-seperated list of training files')
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

img_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
seg_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)

assert len(img_pairs) > 0, 'Could not find any training data.'

# load labels file
train_labels = np.load(args.labels)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas != None:
    "This script does not properly deal with atlas. Use a different script!"

# generator (scan-to-scan unless the atlas cmd argument was provided)
generator = semisupervised(
    img_pairs, seg_pairs, labels=train_labels, atlas_file=args.atlas)

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


losses = [image_loss_func, 
          Grad('l2', loss_mult=args.int_downsize).loss, 
          Dice().loss]
weights = [1, args.grad_loss_weight, args.dice_loss_weight]


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