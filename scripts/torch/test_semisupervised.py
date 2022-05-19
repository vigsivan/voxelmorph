#!/usr/bin/env python

"""
Example script for testing quality of trained VxmDense models. This script iterates over a list of
images pairs, registers them, propagates segmentations via the deformation, and computes the dice
overlap. Example usage is:

    test.py  \
        --model model.h5  \
        --pairs pairs.txt  \
        --img-suffix /img.nii.gz  \
        --seg-suffix /seg.nii.gz

Where pairs.txt is a text file with line-by-line space-seperated registration pairs.
This script will most likely need to be customized to fit your data.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""
#%%

import os
import argparse
import time
import einops
import numpy as np
import voxelmorph as vxm
import torch
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import torchio as tio

from monai.losses import GlobalMutualInformationLoss
from nets import VxmSemisupervised
from voxelmorph.torch.losses import Grad, Dice

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--model', required=True, help='VxmDense model file')
parser.add_argument('--pairs', required=True, help='path to list of image pairs to register')
parser.add_argument('--img-suffix', help='input image file suffix')
parser.add_argument('--seg-suffix', help='input seg file suffix')
parser.add_argument('--img-prefix', help='input image file prefix')
parser.add_argument('--seg-prefix', help='input seg file prefix')
parser.add_argument('--labels', help='optional label list to compute dice for (in npy format)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()
args.int_steps = 7
args.int_downsize=2

# class Args:
#     gpu=0
#     model="/home/vsivan/voxelmorph/scripts/torch/models_cropped_torch/1480.pt"
#     pairs="/home/vsivan/t1t2_test_pairs.txt"
#     img_suffix=None
#     seg_suffix=None
#     img_prefix="/home/vsivan/cropped_reshaped/"
#     seg_prefix="/home/vsivan/cropped_reshaped_seg/"
#     labels="/home/vsivan/labels.npy"
#     int_steps=7
#     int_downsize=2
#     multichannel=False

# args = Args()

#%%

# sanity check on input pairs
if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
    print('Error: Must provide a differing file suffix and/or prefix for images and segs.')
    exit(1)

pairs = args.pairs
img_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
seg_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)

# device handling
device = "cuda"

# load seg labels if provided
labels = np.load(args.labels) if args.labels else None

# check if multi-channel data
add_feat_axis = not args.multichannel

# keep track of all dice scores
reg_times = []
dice_means = []
mse = vxm.losses.MSE().loss
mi = vxm.losses.MutualInformation().loss
save_dir = Path("/home/vsivan/warped_with_torch_aug/")
save_dir2 = Path("/home/vsivan/warped_with_torch_aug_target_seg/")
save_dir3 = Path("/home/vsivan/warped_with_torch_aug_moved_seg/")
save_dir4 = Path("/home/vsivan/warped_with_torch_aug_fixed_seg/")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir2, exist_ok=True)
os.makedirs(save_dir3, exist_ok=True)
os.makedirs(save_dir4, exist_ok=True)

#%%

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = True

# unet architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

inshape = nib.load(img_pairs[0][0]).shape

# otherwise configure new model
model = VxmSemisupervised(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    bidir=False,
    int_steps=args.int_steps,
    int_downsize=args.int_downsize,
)

device="cpu"
checkpoint = torch.load(args.model, map_location="cpu")
grid_buffers = [key for key in checkpoint.keys() if key.endswith('.grid')]
for key in grid_buffers:
    checkpoint.pop(key)

# prepare the model for training and send to device
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()



# %%
dice_losses = []
for img_pair, seg_pair in zip(img_pairs, seg_pairs):
    moving, target = [Path(i).name for i in img_pair]

    image_tios = [tio.ScalarImage(i) for i in img_pair]
    seg_tios = [tio.ScalarImage(i) for i in img_pair]

    images = [tio.ScalarImage(i).data.unsqueeze(-1).to(device).float().permute(0, 4, 1, 2, 3)
            for i in img_pair]
    segs = [tio.LabelMap(i).data.unsqueeze(-1).to(device).float().permute(0, 4, 1, 2, 3)
            for i in seg_pair]

    warped_image, deformation_field, warped_seg = model(images[0], images[1], segs[0])
    dice = vxm.py.utils.dice(warped_seg.detach().cpu().numpy(), segs[1].detach().cpu().numpy(), labels=labels)
    warped_tio = tio.ScalarImage(tensor=einops.rearrange(warped_image, 'b f d w h -> (b f) d w h').detach().cpu())
    warped_tio.affine = image_tios[0].affine

    unmoved_tio = tio.ScalarImage(tensor=einops.rearrange(images[1], 'b f d w h -> (b f) d w h').detach().cpu())
    unmoved_tio.affine = image_tios[1].affine
    
    warped_tio.save(save_dir/Path(img_pair[0]).name)
    unmoved_tio.save(save_dir/Path(img_pair[1]).name)

    unmoved_seg_tio = tio.LabelMap(tensor=einops.rearrange(segs[1], 'b f d w h -> (b f) d w h').detach().cpu())
    unmoved_seg_tio.affine = seg_tios[1].affine
    unmoved_seg_tio.save(save_dir2/Path(img_pair[1]).name)

    moved_seg_tio = tio.LabelMap(tensor=einops.rearrange(warped_seg, 'b f d w h -> (b f) d w h').detach().cpu())
    moved_seg_tio.affine = seg_tios[0].affine
    moved_seg_tio.save(save_dir3/Path(img_pair[0]).name)

    og_seg_tio = tio.LabelMap(tensor=einops.rearrange(segs[0], 'b f d w h -> (b f) d w h').detach().cpu())
    og_seg_tio.affine = seg_tios[0].affine
    og_seg_tio.save(save_dir4/Path(img_pair[0]).name)

    dice_losses.append(dice.item())
    print(f"{moving}\t{target}\tDice: {dice}")

print(f"Average dice: {round(np.mean(dice_losses),4)}")
# %%
