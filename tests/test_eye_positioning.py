import sys
sys.path.append('../')
import os
from model import Blur
from dataset_loaders import FFHQ, fast_image_reshape
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import numpy as np
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.eye_centering import position_to_given_location
import torch

generic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),])


params_dir = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_flame_params_camera_corrected.npy'
data_root = '/is/cluster/pghosh/face_gan_data/FFHQ/multiscale/'
rendered_flame_root = '/is/cluster/work/pghosh/gif1.0/DECA_inferred/deca_rendered_with_texture.lmdb'
normalization_file_path = None
flame_param_est = None

# import ipdb; ipdb.set_trace()
list_bad_images = np.load('/is/cluster/work/pghosh/gif1.0/DECA_inferred/b_box_stats.npz')['bad_images']
dataset = FFHQ(real_img_root=data_root, rendered_flame_root=rendered_flame_root, params_dir=params_dir,
               generic_transform=generic_transform, pose_cam_from_yao=False,
               rendered_flame_as_condition=True, resolution=256,
               normalization_file_path=normalization_file_path, debug=True, random_crop=False, get_normal_images=True,
               flame_version='DECA', list_bad_images=list_bad_images)

show_img_at_res = 256


overlay_visualizer = OverLayViz(full_neck=False, add_random_noise_to_background=False, inside_mouth_faces=True,
                                background_img=None, texture_pattern_name='MEAN_TEXTURE_WITH_CHKR_BOARD',
                                flame_version='DECA', image_size=256)
deca_flame = overlay_visualizer.deca.flame

example_out_dir = '/is/cluster/work/pghosh/gif1.0/eye_cntr_images'

normalized_eye_cntr_L = []
normalized_eye_cntr_R = []
for indx in range(50):
    # i = int(random.randint(0, 60_000))
    i = indx

    fig, ax1 = plt.subplots(1, 1)
    img, flm_rndr, flm_lbl, index = dataset.__getitem__(i, bypass_valid_indexing=False)
    img = img[None, ...]

    img = fast_image_reshape(img, height_out=show_img_at_res, width_out=show_img_at_res)
    img = img[0]
    ax1.imshow((img.numpy().transpose((1, 2, 0)) + 1)/2)

    flame_batch = torch.from_numpy(flm_lbl[0][None, ...]).cuda()

    flame_batch = position_to_given_location(deca_flame, flame_batch)

    # import ipdb; ipdb.set_trace()
    shape, expression, pose = (flame_batch[:, 0:100, ], flame_batch[:, 100:150], flame_batch[:, 150:156])
    vertices, l_m2d, _ = deca_flame(shape_params=shape, expression_params=expression, pose_params=pose)
    # l_m2d[:, :, 1] *= -1
    # vertices[:, :, 1] *= -1
    eye_left_3d = vertices[:, 4051]
    eye_right_3d = vertices[:, 4597]

    eye_R_2d = (eye_left_3d[:, :2] + flame_batch[:, 157:159]) * flame_batch[:, 156]
    eye_L_2d = (eye_right_3d[:, :2] + flame_batch[:, 157:159]) * flame_batch[:, 156]

    eye_R_2d[:, 1] *= -1
    eye_L_2d[:, 1] *= -1

    normalized_eye_cntr_L.append(eye_L_2d[:, 0:2])
    normalized_eye_cntr_R.append(eye_R_2d[:, 0:2])

    eye_R_2d = eye_R_2d * show_img_at_res / 2 + show_img_at_res / 2
    eye_L_2d = eye_L_2d * show_img_at_res / 2 + show_img_at_res / 2

    # import ipdb; ipdb.set_trace()

    # l_m2d = l_m2d[0]
    # l_m2d = (l_m2d[:, :2] + flame_batch[:, 157:159]) * flame_batch[:, 156]
    # l_m2d[:, 1] *= -1
    #
    # l_m2d = l_m2d*show_img_at_res / 2 + show_img_at_res / 2
    #
    # l_m2d = l_m2d.cpu().detach().numpy()
    # plt.plot(l_m2d[:, 0], l_m2d[:, 1], 'g*')

    eye_R_2d = eye_R_2d.cpu().detach().numpy()
    eye_L_2d = eye_L_2d.cpu().detach().numpy()

    plt.plot(eye_R_2d[:, 0], eye_R_2d[:, 1], 'b*')
    plt.plot(eye_L_2d[:, 0], eye_L_2d[:, 1], 'r*')

    plt.savefig(os.path.join(example_out_dir, str(i)+'.png'))

normalized_eye_cntr_L = torch.cat(normalized_eye_cntr_L, dim=0)
print(f'Mean left_eye_cntr: {normalized_eye_cntr_L.mean(dim=0)}, std: {normalized_eye_cntr_L.std(dim=0)}')

normalized_eye_cntr_R = torch.cat(normalized_eye_cntr_R, dim=0)
print(f'Mean right_eye_cntr: {normalized_eye_cntr_R.mean(dim=0)}, std: {normalized_eye_cntr_R.std(dim=0)}')