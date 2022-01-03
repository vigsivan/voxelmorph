# from dipy.align import affine_registration, rigid, center_of_mass
from dipy.align.imaffine import transform_centers_of_mass
from dipy.io.image import load_nifti, save_nifti
from argparse import ArgumentParser
import os
import numpy as np
from pathlib import Path
from tqdm import trange

parser = ArgumentParser(description="Applies affine transform using dipy before doing the learning.")
parser.add_argument("--img-dir", required=True, help="Directory containing the images that we wish to affinely register.")
parser.add_argument("--out-dir", required=False, default="registered_images")
parser.add_argument("--seg-dir", required=False, help="If provided, the segmentation images are transformed using the same transform. The output dir is out-dir + `seg'.")
parser.add_argument("--template", required=False, default=None, help="If provided, this image will be used as the template image")

args = parser.parse_args()
if args.seg_dir: args.seg_dir = Path(args.seg_dir)

seg_save_dir = None if not args.seg_dir else Path(args.out_dir + "_seg")
args.out_dir = Path(args.out_dir)
os.makedirs(name= args.out_dir, exist_ok=True)
if seg_save_dir: os.makedirs(name= seg_save_dir, exist_ok=True)

images = [Path(args.img_dir) / i for i in os.listdir(args.img_dir)]
template_fname = args.template if args.template else images[0]

transform = transform_centers_of_mass

print(f"Using {template_fname} as the template.")

template, template_affine = load_nifti(template_fname)

for i in trange(1, len(images)):
    image = images[i]
    moving, moving_affine = load_nifti(image)
    com = transform(template, template_affine, moving, moving_affine)
    transformed = com.transform(moving)

    save_name = args.out_dir / image.name
    save_nifti(save_name, transformed, template_affine)

    if args.seg_dir:
        seg_name = args.seg_dir / image.name
        seg_image, _ = load_nifti(str(seg_name))
        transformed_seg = com.transform(seg_image)
        save_nifti(seg_save_dir/image.name, transformed_seg, template_affine)