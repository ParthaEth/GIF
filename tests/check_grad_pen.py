import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from model import StyledGenerator, Discriminator
from torchvision import utils
import my_utils
from dataset_loaders import FFHQ, fast_image_reshape
from torchvision import transforms


normalization_file_path = '/is/cluster/work/pghosh/gif1.0/FFHQ_dynamicfit_normalization.npz'
condition_dim = 159
conditional_discrim = True
resolution = 64
step_max = int(np.log2(resolution) - 2)
ckpt = torch.load('/is/cluster/work/pghosh/gif1.0/checkpoint/14/088000_0.5965608333333333.model')

discriminator = nn.DataParallel(Discriminator(conditional=conditional_discrim, condition_dim=condition_dim,
                                              from_rgb_activate=True)).cuda()
discriminator.load_state_dict(ckpt['discriminator'])

params_dir = '/raid/data/pghosh/face_gan_data/flame_dynamic/params/'
data_root = '/raid/data/pghosh/face_gan_data/FFHQ/multiscale/'
generic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True), ])

dataset = FFHQ(real_img_root=data_root, params_dir=params_dir, generic_transform=generic_transform,
               pose_cam_from_yao=True, resolution=128, normalization_file_path=normalization_file_path)
dataset = iter(dataset)

resolutions = [4 * 2 ** _step for _step in range(step_max + 1)]

img, flm_prm = next(dataset)
flm_prm = torch.from_numpy(flm_prm[0]).cuda()[None, :]

real_image = img.cuda()[None, :]
real_image_list = [fast_image_reshape(real_image, height, height) for height in resolutions]

discriminator.eval()
with torch.no_grad():
    decissions = discriminator(in_img_list=real_image_list, condition=flm_prm,  step=step_max, alpha=0.5965608333333333)

import ipdb; ipdb.set_trace()