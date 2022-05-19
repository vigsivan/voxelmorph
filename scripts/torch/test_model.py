from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import torch
import typer
import voxelmorph

from torch import nn

from nets import VxmDense, VxmSemisupervised

app = typer.Typer()

class Model(Enum):
    vxm="vxm"
    semisupervised="semisupervised"

class Device(Enum):
    cpu="cpu"
    cuda="cuda"

torch.backends.cudnn.deterministic = True

# unet architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

@app.command()
def on_pairs(model_path: Path,
             pairs: Path=Path("/home/vsivan/t1t2_pairs.txt"),
             savename: Optional[str]=None,
             savefolder: Optional[Path]=Path.home,
             model: Model = Model.semisupervised,
             device: Device = Device.cpu):

    if savefolder is not None and not savefolder.is_dir():
        raise ValueError("Save Folder directory must be provided")
    


    checkpoint = torch.load(model_path, map_location=device)
    grid_buffers = [key for key in checkpoint.keys() if key.endswith('.grid')]
    for key in grid_buffers:
        checkpoint.pop(key)

def get_model(model: Model, inshape: Tuple[int,...]) -> nn.Module:
    if model == Model.semisupervised:
        model = VxmSemisupervised(
            inshape=inshape,
            bidir=False,
        )
    else:
        model = VxmDense()