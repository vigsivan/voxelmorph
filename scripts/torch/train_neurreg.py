#!/usr/bin/env python
"""
Semisupervised variant in PyTorch
"""
#%%
import os
import random
import argparse
import time
import numpy as np
from tensorflow.python.keras.backend import int_shape
import torch

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

#%%
debug = True # FIXME
if debug:
    from types import SimpleNamespace

    args = SimpleNamespace(
        img_list = "./processed_images.txt",
        img_prefix = "/media/vsivan/Untitled/processed/processed_t1_images/",
        img_suffix = None,
        seg_prefix = "/media/vsivan/Untitled/processed/processed_t1_segs/",
        seg_suffix = None,
        labels = "./labels.npy",
        atlas  = None,
        model_dir = "./models_neurreg",
        enc = [16, 32, 32, 32],
        dec=[32, 32, 32, 32, 32, 16, 16],
        int_steps=7,
        int_downsize=1,
        gpu='0',
        batch_size=1,
        epochs=1500,
        steps_per_epoch=100,
        load_model = None,
        initial_epoch = 0,
        lr = 1e-4,
        cudnn_nondet = None,
        image_loss="mse",
        grad_loss_weight=.01,
        dice_loss_weight=.01
    )
else:
    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
    parser.add_argument('--img-prefix', help='optional input image file prefix')
    parser.add_argument('--img-suffix', help='optional input image file suffix')
    parser.add_argument('--seg-prefix', help='input sef file prefix')
    parser.add_argument('--seg-suffix', help='input sef file suffix')
    parser.add_argument('--labels', required=True, help="ground truth needed for DICE loss")
    parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--grad-loss-weight', type=float, default=0.01,
                        help='weight of gradient loss (lamba) (default: 0.01)')
    parser.add_argument('--dice-loss-weight', type=float, default=0.01,
                        help='weight of dice loss (gamma) (default: 0.01)')

    args = parser.parse_args()


#%%
if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
    print('Error: Must provide a differing file suffix and/or prefix for images and segs.')
    exit(1)

# load and prepare training data
train_imgs = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)

train_segs = vxm.py.utils.read_file_list(args.img_list, prefix=args.seg_prefix,
                                          suffix=args.seg_suffix)

assert len(train_imgs) > 0, 'Could not find any training data.'
#%%

train_labels = np.load(args.labels)

# generator (scan-to-scan unless the atlas cmd argument was provided)
generator = vxm.generators.neurreg(
    train_imgs, train_segs, labels=train_labels, atlas_file=args.atlas)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

#%%
# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.NeurRegModel.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.NeurRegModel(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#%%
# TODO: change losses + weights to reflect Neurreg I/O
# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# losses
num_pixels = np.prod(inshape)
supervised_losses = [vxm.losses.EPE(num_pixels, 3).loss, image_loss_func, vxm.losses.Dice().loss]
weights = [1, 10, 10]

semisupervised_losses = [image_loss_func, vxm.losses.Dice().loss]
weights = [10, 10]

#%%
# FIXME: remove this once you're done testing!!
from torch import nn

inputs, semisupervised_true = next(generator)
inputs[:-1] = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs[:-1]]
inputs[-1] = torch.from_numpy(inputs[-1]).float().unsqueeze(0).to(device)

#%%
semisupervised_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in semisupervised_true]

# %%
outputs = model(*inputs)
supervised_true, y_supervised, y_semisupervised = outputs
loss = 0
loss_list = []

for i, loss_function in enumerate(supervised_losses):
    curr_loss = loss_function(
        supervised_true[i], y_supervised[i]
    )
    loss += curr_loss
    loss_list.append(curr_loss)

for i, loss_function in enumerate(semisupervised_losses):
    curr_loss = loss_function(
        semisupervised_true[i], y_semisupervised[i]
    )
    loss += curr_loss
    loss_list.append(curr_loss)
#%%
# l0 = supervised_losses[0](supervised_true[0], y_supervised[0])
# l1 = supervised_losses[1](supervised_true[1], y_supervised[1])
# l2 = supervised_losses[2](supervised_true[2], y_supervised[2])
# for n, loss_function in enumerate(supervised_losses):
#     curr_loss = loss_function(supervised_true[n], y_supervised[n])
#     loss += curr_loss
#     loss_list.append(curr_loss)
# %%
# 
# vxm_dense = vxm.networks.VxmDense(
#         inshape=inshape,
#         nb_unet_features=[enc_nf, dec_nf],
#         int_steps=args.int_steps,
#         int_downsize=args.int_downsize).to(device)
# #%%
# conv_w_softmax = conv_w_softmax = nn.Sequential(
#         nn.Conv3d(vxm_dense.unet_model.final_nf+1, 1, kernel_size=3, padding=1), 
#         nn.Softmax()).to(device)

# #%%

# source, target, source_seg, displacement_field = inputs
# src2trg_image, pre_int_flow_trg = vxm_dense(source, target, False)
# pos_flow_trg = vxm_dense.integrate(pre_int_flow_trg)
# # seg_flow_trg = vxm.layers.ResizeTransform(1/2, 3)(pos_flow_trg)
# src2trg_seg = vxm_dense.transformer(source_seg, pos_flow_trg)

# # %%
# src2_trg_seg_concat = torch.cat((vxm_dense.unet_output, src2trg_seg), dim=1)
# src2_trg_seg_boosted = conv_w_softmax(src2_trg_seg_concat)
# # src2trg_seg_resized = vxm.layers.ResizeTransform(
# #     2, 3)(src2_trg_seg_boosted)

# # y_semisupervised = [src2trg_image, src2trg_seg_resized]
#%%
# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, _ = next(generator)
        inputs[:-1] = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs[:-1]]
        inputs[-1] = torch.from_numpy(inputs[-1]).float().unsqueeze(0)

        # run inputs through the model to produce a warped image and flow field
        supervised_true, y_supervised, y_semisupervised = model(*inputs)

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
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
