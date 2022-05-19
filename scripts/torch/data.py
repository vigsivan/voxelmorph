from pathlib import Path
from typing import Optional, Callable, Tuple

import torch
import torchio as tio
from torch.utils.data import Dataset

class ImageSegPairs(Dataset):
    def __init__(self, images_dir: Path, segs_dir: Path, train_files: Path, transform: Optional[Callable]=None):
        
        assert images_dir.is_dir() and segs_dir.is_dir() and train_files.exists(), \
        f"""One or more of the following not found:
            - {images_dir} <-- {"found" if images_dir.is_dir() else "NOT found"}
            - {segs_dir} <-- {"found" if segs_dir.is_dir() else "NOT found"}
            - {train_files} <-- {"found" if train_files.exists() else "NOT found"}
        """

        with open(train_files, 'r') as f:
            subjects = [i.strip() for i in f.readlines()]

        for sub in subjects:
            if not (images_dir/sub).exists():
                raise ValueError(f"{images_dir/sub} not found!")
            if not (segs_dir/sub).exists():
                raise ValueError(f"{segs_dir/sub} not found!")

        self.subjects = subjects
        self.images_dir = images_dir
        self.segs_dir = segs_dir
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sub = self.subjects[index]
        image = tio.ScalarImage(self.images_dir/sub)
        seg = tio.LabelMap(self.segs_dir/sub)

        if self.transform is not None:
            sub = tio.Subject({"image": image, "seg": seg})
            sub = self.transform(sub)
            image, seg = sub["image"], sub["seg"]

        return (image.data, seg.data)