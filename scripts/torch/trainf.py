from enum import Enum
from typing import Optional
from pathlib import Path
import typer

from data import ImageSegPairs

app = typer.Typer()

class Lossfn(str, Enum):
    mse = "mse",
    ncc = "ncc",
    mi = "mi"

@app.command()
def semisupervised(
    images_dir: Path,
    segs_dir: Path,
    train_files: Path,
    checkpoint_dir: Path,
    val_files: Optional[Path]=None,
    weights: Optional[Path]=None,
    device: str="gpu",
    image_loss: Lossfn = Lossfn.mi,
    batch_size: int=1,
    grad_loss_weight: float = .01,
    dice_loss_weight: float = .01
):

    train_dataset = ImageSegPairs(images_dir, segs_dir, train_files)
    if val_files is not None:
        val_dataset = ImageSegPairs(images_dir, segs_dir, val_files)

    print(len(train_dataset))
    checkpoint_dir.mkdir(exist_ok=True)

if __name__ == "__main__":
    app()