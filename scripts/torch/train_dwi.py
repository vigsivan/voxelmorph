#!/usr/bin/env python

"""
"""

from email.generator import Generator
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jvp
from monai.losses import GlobalMutualInformationLoss
from nets import VxmSemisupervised, VxmDense, VxmDti
from voxelmorph.torch.losses import Grad, Dice
import torchio as tio
from einops import rearrange, repeat
from typing import Callable, List, Optional, Tuple
from einops import repeat
from functools import partial
import typer

transform = tio.OneOf([
    tio.RandomElasticDeformation(),
    tio.RandomGamma(),
    tio.RandomBiasField(),
])

# import voxelmorph with pytorch backend
os.environ["VXM_BACKEND"] = "pytorch"
import voxelmorph as vxm  # nopep8

def triplet_gen(
    triplets_list: List[List[Path]],
    batch_size=1,
    add_feat_axis=True,
    anat_transform: Optional[Callable] = transform,
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
    
    if batch_size > 1:
        raise ValueError("Currently do not support batch size greater than 1")

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(triplets_list), size=batch_size)
        anat_images, dwi_images, dti_images = [], [], []
        for i in indices:
            anat, dwi, dti = triplets_list[i]
            anat, dwi, dti = [tio.ScalarImage(j) for j in (anat, dwi, dti)]
            if anat_transform is not None:
                anat = anat_transform(anat)

            anat, dwi, dti = [reshape(j.data.squeeze().numpy()) for j in (anat, dwi, dti)]
            anat_images.append(anat)
            dwi_images.append(dwi)
            dti_images.append(dti)

        yield tuple([
            np.concatenate(anat_images, axis=0),
            np.concatenate(dwi_images, axis=0),
            np.concatenate(dti_images, axis=0),
        ])
            
def dti_gen(triplets_list):
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
    gen = triplet_gen(triplets_list)
    zeros = None

    while True:
        # load source vol and seg
        anat, dwi, dti = next(gen)

        # cache zeros
        if zeros is None:
            shape = anat.shape[1:-1]
            zeros = np.zeros((1, *shape, len(shape)))

        invols = [dwi, anat, dti]
        outvols = [anat, zeros, dti]
        yield (invols, outvols)


def compute_jacobian(deformation_field: torch.Tensor, device: str):
    phi_X = deformation_field[:,2,:,:,:] ## IS x,y,z ok ?
    phi_Y = deformation_field[:,1,:,:,:]
    phi_Z = deformation_field[:,0,:,:,:]

    B,C,D,H,W = deformation_field.shape 
        
    phi_X_dx = torch.zeros(phi_X.shape).to(device)
    phi_X_dx[:,:,:,0:W-1] = phi_X[:,:,:,1:W] - phi_X[:,:,:,0:W-1]
    phi_X_dy = torch.zeros(phi_X.shape).to(device)
    phi_X_dy[:,:,0:H-1,:] = phi_X[:,:,1:H,:] - phi_X[:,:,0:H-1,:]
    phi_X_dz = torch.zeros(phi_X.shape).to(device)
    phi_X_dz[:,0:D-1,:,:] = phi_X[:,1:D,:,:] - phi_X[:,0:D-1,:,:]
    
    phi_Y_dx = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dx[:,:,:,0:W-1] = phi_Y[:,:,:,1:W] - phi_Y[:,:,:,0:W-1]
    phi_Y_dy = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dy[:,:,0:H-1,:] = phi_Y[:,:,1:H,:] - phi_Y[:,:,0:H-1,:]
    phi_Y_dz = torch.zeros(phi_Y.shape).to(device)
    phi_Y_dz[:,0:D-1,:,:] = phi_Y[:,1:D,:,:] - phi_Y[:,0:D-1,:,:]
    
    phi_Z_dx = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dx[:,:,:,0:W-1] = phi_Z[:,:,:,1:W] - phi_Z[:,:,:,0:W-1]
    phi_Z_dy = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dy[:,:,0:H-1,:] = phi_Z[:,:,1:H,:] - phi_Z[:,:,0:H-1,:]
    phi_Z_dz = torch.zeros(phi_Z.shape).to(device)
    phi_Z_dz[:,0:D-1,:,:] = phi_Z[:,1:D,:,:] - phi_Z[:,0:D-1,:,:]

    num_pixels = D*H*W
    
    jac = torch.zeros(B,num_pixels, C,C).to(device)
    
    jac[:,:,0,0] = phi_X_dx.reshape(B,num_pixels)
    jac[:,:,0,1] = phi_X_dy.reshape(B,num_pixels)
    jac[:,:,0,2] = phi_X_dz.reshape(B,num_pixels)
    
    jac[:,:,1,0] = phi_Y_dx.reshape(B,num_pixels)
    jac[:,:,1,1] = phi_Y_dy.reshape(B,num_pixels)
    jac[:,:,1,2] = phi_Y_dz.reshape(B,num_pixels)
    
    jac[:,:,2,0] = phi_Z_dx.reshape(B,num_pixels)
    jac[:,:,2,1] = phi_Z_dy.reshape(B,num_pixels)
    jac[:,:,2,2] = phi_Z_dz.reshape(B,num_pixels)

    return jac

def get_model_optimizers_losses(
    inshape: Tuple[int,...],
    device: str,
    learning_rate: float=1e-4,
    grad_loss_weight: float = 1.,
    dti_loss_weight: float = .1,
):
    # unet architecture
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    int_steps = 7
    int_downsize=1

    # otherwise configure new model
    model = VxmDti(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=False,
        int_steps=int_steps,
        int_downsize=int_downsize,
    )

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    image_loss_func = GlobalMutualInformationLoss()

    losses = [image_loss_func, Grad('l2', loss_mult=1).loss, image_loss_func]
    weights = [1, grad_loss_weight, dti_loss_weight]

    return {
        "model": model,
        "optimizer": optimizer,
        "losses": losses,
        "weights": weights
    }

def training_loop(
    total_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    model_dir: Path,
    generator: Generator,
    losses: List[Callable],
    weights: List[float],
    steps_per_epoch: int=100,
    initial_epoch: int=0
):
    for epoch in range(initial_epoch, total_epochs):

        # save model checkpoint
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "%04d.pt" % epoch))

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for step in range(steps_per_epoch):

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
            flow_jac = compute_jacobian(y_pred[-1], device)
            u,_,vh = torch.svd(flow_jac.squeeze())
            rotation = torch.bmm(u, vh)
            # TODO: get the rotation matrix
            breakpoint()

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
        epoch_info = "Epoch %d/%d" % (epoch + 1, total_epochs)
        time_info = "%.4f sec/step" % np.mean(epoch_step_time)
        losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = "loss: %.4e  (%s)" % (np.mean(epoch_total_loss), losses_info)
        print(" - ".join((epoch_info, time_info, loss_info)), flush=True)

    # final model save
    model.save(os.path.join(model_dir, "%04d.pt" % total_epochs))

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = True
app = typer.Typer()

@app.command()
def main(
    train_list: Path,
    model_dir: Path,
    root_dir: Optional[Path]=None,
    total_epochs: int=8000,
    device: str="cuda",
):
    with open(train_list, 'r') as f:
        triplets = [i.strip().split(" ") for i in f.readlines()]

    if root_dir is not None:
        triplets = [[root_dir/p for p in triplet] for triplet in triplets]

    model_dir.mkdir(exist_ok=True)

    generator = dti_gen(triplets)
    inshape = next(generator)[0][0].shape[1:-1]

    training_loop(
        total_epochs=total_epochs,
        model_dir=model_dir,
        generator=generator,
        device=device,
        **get_model_optimizers_losses(inshape, device)
    )

app()