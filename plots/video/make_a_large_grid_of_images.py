import sys
sys.path.append('../../../')
import constants as cnst
import os
from torchvision.utils import save_image
import torch
from PIL import Image
import numpy as np
import glob




images = []
image_files = sorted(glob.glob(cnst.generated_random_image_root_dir + '/mesh*.png'))
# for i in range(len(image_files)):
#     image_files[i] = image_files[i].replace('/meshes', '/images').replace('/mesh', '/')
n_row = 12
n_col = 6
for i in range(int(n_row*n_col)):
    img_file = image_files[i]
    np_img = np.array(Image.open(img_file)).transpose((2, 0, 1))[None, ...]
    np_img = np_img.astype('float32')/255
    images.append(np_img)

images = np.concatenate(images, axis=0)
images = torch.from_numpy(images)

save_image(images, os.path.join(cnst.random_imagesdestination_dir, 'stitched.png'), nrow=n_row)